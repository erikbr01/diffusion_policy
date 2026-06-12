"""Policy backends for DROID real-robot inference.

All backends share the same submit / get / poll interface so
DROIDInferenceRunner is backend-agnostic:

  WebSocketPolicy  — openpi WebSocket server (pi0 / pi0.5)
  HttpPolicy       — diffusion checkpoints served by ``policy_server.py``
  LocalPolicy      — diffusion checkpoint loaded in-process (no server)

Heavy/optional imports (openpi_client, requests, torch) happen lazily so the
module imports cleanly everywhere.
"""

from __future__ import annotations

import io
import json
import logging
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Optional, Tuple

import numpy as np

from diffusion_policy.real_world.droid.obs_processing import (
    ObsSpec,
    ObsWindowBuffer,
    obs_spec_from_cfg,
    obs_spec_from_shape_meta,
    resolve_key_map,
)

logger = logging.getLogger(__name__)

DEFAULT_STATE_KEYS: Tuple[str, ...] = ("joint_position", "gripper_position")


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class PolicyBase(ABC):
    """Uniform inference interface: submit (non-blocking) → get (blocking)."""

    @abstractmethod
    def reset(self) -> None:
        """Called at the start of each episode."""

    @abstractmethod
    def submit(self, obs_dict: Dict, instruction: str = "") -> None:
        """Kick off inference asynchronously. Does not block."""

    @abstractmethod
    def get(self) -> np.ndarray:
        """Block until the last submitted inference completes.

        Returns the action chunk, shape (T, action_dim).
        """

    def poll(self) -> Optional[np.ndarray]:
        """Return the action chunk if inference is already done, else None."""
        if not self.pending and self._has_result():
            return self.get()
        return None

    @property
    @abstractmethod
    def pending(self) -> bool:
        """True while a submitted call has not yet completed."""

    def _has_result(self) -> bool:
        return False


class _FuturePolicy(PolicyBase):
    """Shared single-worker-executor future plumbing."""

    def __init__(self, thread_name: str) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_name)
        self._future: Optional[Future] = None

    def reset(self) -> None:
        self._future = None

    def _submit_fn(self, fn, *args) -> None:
        self._future = self._executor.submit(fn, *args)

    def get(self) -> np.ndarray:
        if self._future is None:
            raise RuntimeError("call submit() before get()")
        chunk = self._future.result()  # re-raises if inference failed
        self._future = None
        return chunk

    @property
    def pending(self) -> bool:
        return self._future is not None and not self._future.done()

    def _has_result(self) -> bool:
        return self._future is not None and self._future.done()


# ---------------------------------------------------------------------------
# WebSocket backend (openpi / pi0 / pi0.5)
# ---------------------------------------------------------------------------

class WebSocketPolicy(_FuturePolicy):
    """openpi WebSocket policy client with async submit/get.

    The WebSocket call is blocking I/O, so it runs in a single background
    worker so the control loop stays live.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        external_camera: str = "ext1",
        wrist_key: str = "wrist_image",
        ext_key: Optional[str] = None,
    ) -> None:
        super().__init__("droid_ws_policy")
        self._host = host
        self._port = port
        self._ext_key = ext_key or (
            "exterior_image_1_left" if external_camera == "ext1" else "exterior_image_2_left"
        )
        self._wrist_key = wrist_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openpi_client import websocket_client_policy

            self._client = websocket_client_policy.WebsocketClientPolicy(
                self._host, self._port
            )
        return self._client

    def submit(self, obs_dict: Dict, instruction: str = "") -> None:
        self._submit_fn(self._infer, self._build_request(obs_dict, instruction))

    def _build_request(self, obs_dict: Dict, instruction: str) -> Dict:
        from openpi_client import image_tools

        req: Dict = {
            "observation/joint_position": np.asarray(
                obs_dict["joint_position"], dtype=np.float32
            ),
            "observation/gripper_position": np.asarray(
                obs_dict["gripper_position"], dtype=np.float32
            ),
            "prompt": instruction,
        }
        ext_img = obs_dict.get(self._ext_key)
        wrist_img = obs_dict.get(self._wrist_key)
        if ext_img is not None:
            req["observation/exterior_image_1_left"] = image_tools.resize_with_pad(
                ext_img, 224, 224
            )
        if wrist_img is not None:
            req["observation/wrist_image_left"] = image_tools.resize_with_pad(
                wrist_img, 224, 224
            )
        return req

    def _infer(self, request: Dict) -> np.ndarray:
        result = self._get_client().infer(request)
        return np.asarray(result["actions"], dtype=np.float32)  # (T, action_dim)


# ---------------------------------------------------------------------------
# HTTP backend (policy_server.py + diffusion checkpoints)
# ---------------------------------------------------------------------------

class HttpPolicy(_FuturePolicy):
    """Client for the diffusion policy server in ``policy_server.py``.

    On first submit it fetches ``/meta`` to learn the checkpoint's obs layout:
    image policies get full obs-window assembly (resize + CHW float, matching
    training preprocessing) client-side; lowdim policies get a concatenated
    state window from ``state_keys``.

    Wire format: POST /infer with np.savez bytes ({key: (1, To, ...)}) →
    np.save bytes of the action chunk (1, T, action_dim).
    """

    def __init__(
        self,
        url: str = "http://127.0.0.1:5001",
        state_keys: Tuple[str, ...] = DEFAULT_STATE_KEYS,
        obs_key_overrides: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__("droid_http_policy")
        self._url = url.rstrip("/")
        self._state_keys = tuple(state_keys)
        self._obs_key_overrides = dict(obs_key_overrides or {})
        self._timeout = timeout
        self._meta: Optional[Dict] = None
        self._window: Optional[ObsWindowBuffer] = None
        self._state_history: list = []

    # ------------------------------------------------------------------

    def _fetch_meta(self, obs_dict: Dict) -> None:
        import requests

        resp = requests.get(f"{self._url}/meta", timeout=self._timeout)
        resp.raise_for_status()
        self._meta = resp.json()
        if self._meta["policy_type"] == "image":
            spec = obs_spec_from_shape_meta(
                self._meta["shape_meta"],
                n_obs_steps=self._meta["n_obs_steps"],
                horizon=self._meta["horizon"],
            )
            key_map = resolve_key_map(spec, obs_dict.keys(), self._obs_key_overrides)
            self._window = ObsWindowBuffer(spec, key_map)
            logger.info("policy server: image policy, obs map %s", key_map)
        else:
            logger.info(
                "policy server: lowdim policy, state keys %s", self._state_keys
            )

    def reset(self) -> None:
        super().reset()
        if self._window is not None:
            self._window.reset()
        self._state_history = []

    def submit(self, obs_dict: Dict, instruction: str = "") -> None:
        if self._meta is None:
            self._fetch_meta(obs_dict)

        if self._window is not None:  # image policy
            self._window.push(obs_dict)
            arrays = {k: v[None] for k, v in self._window.window().items()}
        else:  # lowdim policy: (1, To, obs_dim)
            state = np.concatenate(
                [np.atleast_1d(np.asarray(obs_dict[k], dtype=np.float32))
                 for k in self._state_keys]
            )
            To = int(self._meta["n_obs_steps"])
            self._state_history.append(state)
            window = self._state_history[-To:]
            while len(window) < To:
                window.insert(0, window[0])
            arrays = {"obs": np.stack(window)[None]}

        buf = io.BytesIO()
        np.savez(buf, **arrays)
        self._submit_fn(self._post, buf.getvalue())

    def _post(self, data: bytes) -> np.ndarray:
        import requests

        resp = requests.post(
            f"{self._url}/infer",
            data=data,
            headers={"Content-Type": "application/octet-stream"},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        action = np.load(io.BytesIO(resp.content))  # (1, T, action_dim)
        return action[0]


# ---------------------------------------------------------------------------
# Local backend (checkpoint loaded in-process)
# ---------------------------------------------------------------------------

class LocalPolicy(_FuturePolicy):
    """Run a diffusion checkpoint in-process (image or lowdim policy).

    Useful when the robot workstation has the GPU, and for offline smoke
    tests (``--dry-run``). Inference runs in a single background worker so
    the async runner modes behave as with a remote server.
    """

    def __init__(
        self,
        checkpoint: str,
        device: str = "cpu",
        state_keys: Tuple[str, ...] = DEFAULT_STATE_KEYS,
        obs_key_overrides: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__("droid_local_policy")
        import torch

        from diffusion_policy.common.trak_util import get_policy_from_checkpoint
        from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
            DiffusionUnetHybridImagePolicy,
        )

        logger.info("loading checkpoint %s on %s", checkpoint, device)
        self._torch = torch
        self._device = torch.device(device)
        self.policy, self.cfg = get_policy_from_checkpoint(
            checkpoint, device=self._device
        )
        self._is_image = isinstance(self.policy, DiffusionUnetHybridImagePolicy)
        self._state_keys = tuple(state_keys)
        self._obs_key_overrides = dict(obs_key_overrides or {})
        self._window: Optional[ObsWindowBuffer] = None
        self._spec: Optional[ObsSpec] = None
        if self._is_image:
            self._spec = obs_spec_from_cfg(self.cfg)
        self._state_history: list = []

    def reset(self) -> None:
        super().reset()
        if self._window is not None:
            self._window.reset()
        self._state_history = []

    def submit(self, obs_dict: Dict, instruction: str = "") -> None:
        if self._is_image:
            if self._window is None:
                key_map = resolve_key_map(
                    self._spec, obs_dict.keys(), self._obs_key_overrides
                )
                logger.info("local policy obs map: %s", key_map)
                self._window = ObsWindowBuffer(self._spec, key_map)
            self._window.push(obs_dict)
            batch = {
                k: self._torch.as_tensor(v[None], device=self._device)
                for k, v in self._window.window().items()
            }
        else:
            state = np.concatenate(
                [np.atleast_1d(np.asarray(obs_dict[k], dtype=np.float32))
                 for k in self._state_keys]
            )
            To = int(self.cfg.n_obs_steps)
            self._state_history.append(state)
            window = self._state_history[-To:]
            while len(window) < To:
                window.insert(0, window[0])
            batch = {
                "obs": self._torch.as_tensor(
                    np.stack(window)[None], device=self._device
                )
            }
        self._submit_fn(self._infer, batch)

    def _infer(self, batch: Dict) -> np.ndarray:
        with self._torch.no_grad():
            result = self.policy.predict_action(batch)
        return result["action"].detach().cpu().numpy()[0]  # (T, action_dim)
