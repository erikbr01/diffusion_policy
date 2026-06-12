"""Per-timestep policy-embedding extraction from a diffusion U-Net.

Shared by ``scripts/eval_save_rollouts.py`` (sim rollouts) and the DROID
offline extractors (``scripts/droid_extract_embeddings.py``); moved here so
there is exactly one implementation of the hook/noising machinery.

Embedding layer names follow ``{bottleneck|decoder|encoder}[_{plan|plan8|exec}][_t{T}]``:

    hook        bottleneck = model.mid_modules[-1], decoder = up_modules[-1][1],
                encoder = down_modules[-1][1]; output is global-avg-pooled.
    action      UNet action input: omitted = random noise, plan = full predicted
                plan, exec = action[0] tiled, plan8 = first 8 steps zero-padded.
    t           diffusion timestep: omitted = average over all inference steps,
                _t{T} = single timestep T (0 = final denoising step).

The default ``bottleneck_plan_t0`` conditions the UNet bottleneck on the actual
action plan at the final denoising step.

The special layer ``obs_encoder`` skips the UNet pass entirely and returns the
policy's conditioning vector: the obs-encoder features for image policies, the
flattened raw obs window for low-dim policies.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
    DiffusionUnetHybridImagePolicy,
)
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy

DEFAULT_LAYER = "bottleneck_plan_t0"
OBS_ENCODER_LAYER = "obs_encoder"

_LAYER_RE = re.compile(
    r"^(?P<hook>bottleneck|decoder|encoder)"
    r"(?:_(?P<action>plan8|plan|exec))?"
    r"(?:_t(?P<t>\d+))?$"
)


def parse_layer(layer: str) -> Tuple[str, Optional[str], Optional[int]]:
    """Parse a layer name into (hook, action, t_single)."""
    m = _LAYER_RE.match(layer)
    if m is None:
        raise ValueError(
            f"Unknown layer {layer!r}. Format: "
            f"{{bottleneck|decoder|encoder}}[_{{plan|plan8|exec}}][_t{{T}}], "
            f"or the special layer {OBS_ENCODER_LAYER!r}"
        )
    t = m.group("t")
    return m.group("hook"), m.group("action"), None if t is None else int(t)


def get_hook_module(model, hook: str):
    """Return the U-Net module to hook for a given hook name."""
    if hook == "bottleneck":
        return model.mid_modules[-1]
    if hook == "decoder":
        return model.up_modules[-1][1]
    if hook == "encoder":
        return model.down_modules[-1][1]
    raise ValueError(f"Unknown hook: {hook}")


def build_noise_schedule(
    n_infer: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    betas = torch.linspace(1e-4, 0.02, n_infer, device=device)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    return alphas_cumprod.sqrt(), (1.0 - alphas_cumprod).sqrt()


def build_action_batch(
    action_arr: Optional[np.ndarray],
    action_type: Optional[str],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Return clean actions (B, horizon, action_dim), or None for random noise."""
    if action_type is None:
        return None
    assert action_arr is not None
    B, horizon, action_dim = action_arr.shape
    if action_type == "plan":
        clean = action_arr
    elif action_type == "exec":
        clean = np.tile(action_arr[:, 0:1, :], (1, horizon, 1))
    elif action_type == "plan8":
        clean = np.concatenate(
            [action_arr[:, :8, :], np.zeros((B, horizon - 8, action_dim), dtype=np.float32)],
            axis=1,
        )
    else:
        raise ValueError(f"Unknown action type: {action_type}")
    return torch.as_tensor(clean, dtype=torch.float32, device=device)


def lowdim_global_cond(policy, obs_batch: List, device: torch.device) -> torch.Tensor:
    """Flattened raw obs window, as in the low-dim policy's conditioning."""
    obs_arr = np.stack(obs_batch, axis=0)
    obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device)
    return obs_t.reshape(len(obs_batch), -1)


def image_global_cond(policy, obs_batch: List, device: torch.device) -> torch.Tensor:
    """Obs-encoder features, mirroring DiffusionUnetHybridImagePolicy.predict_action.

    ``obs_batch`` is a list of obs dicts, each ``{key: (To, ...) float32}`` with
    rgb keys as CHW images in [0, 1] (the policy's predict_action format).
    """
    keys = obs_batch[0].keys()
    obs_dict = {
        k: torch.as_tensor(
            np.stack([o[k] for o in obs_batch]), dtype=torch.float32, device=device
        )
        for k in keys
    }
    with torch.no_grad():
        obs_dict = policy._center_crop_external(obs_dict)
        nobs = policy.normalizer.normalize(obs_dict)
        To = policy.n_obs_steps
        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = policy.obs_encoder(this_nobs)
        return nobs_features.reshape(len(obs_batch), -1)


def extract_batch(
    policy,
    global_cond_b: torch.Tensor,
    action_batch: Optional[np.ndarray],
    action_type: Optional[str],
    hook_name: str,
    noise_step_indices: torch.Tensor,
    sqrt_alpha: torch.Tensor,
    sqrt_one_minus_alpha: torch.Tensor,
    device: torch.device,
    batch_seed: int,
) -> np.ndarray:
    """Return a (B, D) embedding for one batch of rollout timesteps.

    ``global_cond_b`` is the (B, cond_dim) UNet conditioning: flattened raw
    observations for low-dim policies, obs-encoder features for image
    policies — mirroring each policy's own conditioning path.
    """
    model = policy.model
    B = len(global_cond_b)
    N = len(noise_step_indices)

    global_cond = global_cond_b.to(device).repeat_interleave(N, dim=0)

    rng = torch.Generator(device=device)
    rng.manual_seed(batch_seed)
    base_noise = torch.randn(
        B, policy.horizon, policy.action_dim, generator=rng, device=device
    ).repeat_interleave(N, dim=0)

    t_indices = noise_step_indices.repeat(B)
    a = sqrt_alpha[t_indices].view(B * N, 1, 1)
    s = sqrt_one_minus_alpha[t_indices].view(B * N, 1, 1)

    clean_actions = build_action_batch(action_batch, action_type, device)
    if clean_actions is not None:
        noisy_actions = a * clean_actions.repeat_interleave(N, dim=0) + s * base_noise
    else:
        noisy_actions = s * base_noise

    captured = []
    hook_module = get_hook_module(model, hook_name)

    def _hook(module, inp, out):
        feat = out[0] if isinstance(out, tuple) else out
        captured.append(feat.detach().float().mean(dim=-1).cpu())  # global avg pool

    handle = hook_module.register_forward_hook(_hook)
    with torch.no_grad():
        model(noisy_actions, t_indices, global_cond=global_cond)
    handle.remove()

    return captured[0].reshape(B, N, -1).mean(dim=1).numpy().astype(np.float32)


class PolicyEmbeddingExtractor:
    """Stateful wrapper: parse the layer once, then extract per batch.

    ``obs_is_dict`` tells the caller the obs format the extractor expects:
    obs dicts (image policies) vs (To, obs_dim) arrays (low-dim policies).
    ``needs_actions`` is True when the layer conditions on the action plan
    (``plan``/``plan8``/``exec``); pass (B, horizon, action_dim) plans then.

    The ``obs_encoder`` layer returns the conditioning vector itself (no UNet
    pass): obs-encoder features for image policies, the flattened raw obs
    window for low-dim policies.
    """

    def __init__(self, policy, layer: str, device: torch.device) -> None:
        if isinstance(policy, DiffusionUnetLowdimPolicy):
            self._make_global_cond = lowdim_global_cond
            self.obs_is_dict = False
        elif isinstance(policy, DiffusionUnetHybridImagePolicy):
            self._make_global_cond = image_global_cond
            self.obs_is_dict = True
        else:
            raise NotImplementedError(
                f"Policy embeddings are supported for DiffusionUnetLowdimPolicy and "
                f"DiffusionUnetHybridImagePolicy, got {type(policy).__name__}."
            )
        self.policy = policy
        self.layer = layer
        self.device = device
        self.is_obs_encoder = layer == OBS_ENCODER_LAYER
        if self.is_obs_encoder:
            self.hook_name = None
            self.action_type = None
            self.needs_actions = False
            return
        self.hook_name, self.action_type, t_single = parse_layer(layer)
        self.needs_actions = self.action_type is not None

        n_infer = policy.num_inference_steps
        if t_single is not None:
            self._noise_step_indices = torch.tensor(
                [t_single], dtype=torch.long, device=device
            )
        else:
            self._noise_step_indices = torch.linspace(
                0, n_infer - 1, n_infer, dtype=torch.long, device=device
            )
        self._sqrt_alpha, self._sqrt_one_minus_alpha = build_noise_schedule(
            n_infer, device
        )

    def extract(
        self,
        obs_batch: List,
        action_batch: Optional[np.ndarray],
        batch_seed: int,
    ) -> np.ndarray:
        """Return (B, D) embeddings for one batch of timesteps."""
        global_cond_b = self._make_global_cond(self.policy, obs_batch, self.device)
        if self.is_obs_encoder:
            return global_cond_b.detach().cpu().numpy().astype(np.float32)
        return extract_batch(
            self.policy,
            global_cond_b,
            action_batch,
            self.action_type,
            self.hook_name,
            self._noise_step_indices,
            self._sqrt_alpha,
            self._sqrt_one_minus_alpha,
            self.device,
            batch_seed=batch_seed,
        )
