"""HTTP inference server for diffusion-policy checkpoints (Flask).

Loads a checkpoint (image or lowdim) and serves ``predict_action`` over HTTP.
Run it on the GPU machine; the robot workstation talks to it through
``HttpPolicy`` (see ``policy_client.py``), which assembles the obs windows
client-side using ``/meta``.

Endpoints
---------
GET  /health   {"status": "ok", "device": ...}
GET  /meta     {"policy_type": "image"|"lowdim", "n_obs_steps", "horizon",
                "n_action_steps", "action_dim", "shape_meta" (image only)}
POST /infer    body: np.savez bytes, one array per obs key, each (1, To, ...);
               lowdim policies use the single key "obs" = (1, To, obs_dim).
               response: np.save bytes of the action chunk (1, T, action_dim).

Usage:
    python -m diffusion_policy.real_world.droid.policy_server \
        --checkpoint /path/to/latest.ckpt --device cuda:0 --port 5001

Pass --compile to torch.compile the obs_encoder/model sub-modules (same helper
``train_diffusion_unet_hybrid_workspace.py`` uses for training). This pays the
compilation cost (tens of seconds) once at startup via a warmup call with
dummy inputs, rather than on the first real /infer request. Only worthwhile
if requests reuse a fixed obs window shape, since a new shape triggers
recompilation (dynamic=False).
"""

from __future__ import annotations

import argparse
import io
import logging

logger = logging.getLogger(__name__)


def build_app(
    checkpoint: str,
    device: str,
    embed_layer: str | None = None,
    num_inference_steps: int | None = None,
    compile: bool = False,
):
    """Load the checkpoint and return the Flask app."""
    import time

    import numpy as np
    import torch
    from flask import Flask, request
    from omegaconf import OmegaConf

    from diffusion_policy.common.ddp_util import compile_policy
    from diffusion_policy.common.trak_util import get_policy_from_checkpoint
    from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
        DiffusionUnetHybridImagePolicy,
    )

    logger.info("loading checkpoint: %s", checkpoint)
    policy, cfg = get_policy_from_checkpoint(checkpoint, device=device)
    is_image = isinstance(policy, DiffusionUnetHybridImagePolicy)
    shape_meta = None
    if is_image:
        sm = cfg.task.shape_meta if "task" in cfg else cfg.shape_meta
        shape_meta = OmegaConf.to_container(sm, resolve=True)

    if num_inference_steps is not None:
        policy.num_inference_steps = num_inference_steps
        logger.info("overriding num_inference_steps: %d", num_inference_steps)

    if compile:
        compile_policy(policy)
        logger.info("compiling (warmup call, this can take ~10s)...")
        n_obs_steps = int(cfg.n_obs_steps)
        if is_image:
            dummy_batch = {
                key: torch.rand(1, n_obs_steps, *spec["shape"], device=device)
                for key, spec in shape_meta["obs"].items()
            }
        else:
            dummy_batch = {"obs": torch.rand(1, n_obs_steps, policy.obs_dim, device=device)}
        t0 = time.time()
        with torch.no_grad():
            policy.predict_action(dummy_batch)
        logger.info("compile warmup done in %.1fs", time.time() - t0)

    extractor = None
    if embed_layer is not None:
        from diffusion_policy.common.policy_embeddings import PolicyEmbeddingExtractor
        extractor = PolicyEmbeddingExtractor(policy, embed_layer, torch.device(device))
        logger.info("embedding layer: %s", embed_layer)

    logger.info("ready on %s (%s policy)", device, "image" if is_image else "lowdim")

    app = Flask(__name__)

    @app.get("/health")
    def health():
        return {"status": "ok", "device": device, "embed_layer": embed_layer, "compiled": compile}

    @app.get("/meta")
    def meta():
        return {
            "policy_type": "image" if is_image else "lowdim",
            "n_obs_steps": int(cfg.n_obs_steps),
            "horizon": int(cfg.horizon),
            "n_action_steps": int(cfg.n_action_steps),
            "action_dim": int(policy.action_dim),
            "shape_meta": shape_meta,
            "embed_layer": embed_layer,
        }

    @app.post("/infer")
    def infer():
        arrays = np.load(io.BytesIO(request.data), allow_pickle=False)
        batch = {
            key: torch.as_tensor(arrays[key], dtype=torch.float32, device=device)
            for key in arrays.files
        }
        with torch.no_grad():
            result = policy.predict_action(batch)
        action = result["action"].detach().cpu().numpy()  # (1, T, action_dim)

        buf = io.BytesIO()
        if extractor is not None:
            obs_list = [{k: arrays[k][0] for k in arrays.files}]
            # extractor needs (B, horizon, action_dim); pad short chunk with zeros
            h = int(cfg.horizon)
            if action.shape[1] < h:
                pad = np.zeros(
                    (action.shape[0], h - action.shape[1], action.shape[2]),
                    dtype=np.float32,
                )
                action_full = np.concatenate([action, pad], axis=1)
            else:
                action_full = action
            embedding = extractor.extract(obs_list, action_batch=action_full, batch_seed=0)
            np.savez(buf, action=action, embedding=embedding)
        else:
            np.savez(buf, action=action)
        return buf.getvalue(), 200, {"Content-Type": "application/octet-stream"}

    return app


def serve(
    checkpoint: str,
    device: str,
    host: str = "127.0.0.1",
    port: int = 5001,
    embed_layer: str | None = None,
    num_inference_steps: int | None = None,
    compile: bool = False,
) -> None:
    app = build_app(
        checkpoint,
        device,
        embed_layer=embed_layer,
        num_inference_steps=num_inference_steps,
        compile=compile,
    )
    # threaded=False keeps the policy on a single thread (one request at a time).
    app.run(host=host, port=port, threaded=False, use_reloader=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="HTTP diffusion-policy server")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--embed-layer", default=None,
                        help="Layer to extract embeddings from (e.g. bottleneck_plan_t0). "
                             "When set, /infer returns action + embedding.")
    parser.add_argument("--num-inference-steps", type=int, default=None,
                        help="Override diffusion steps at inference (e.g. 16 for DDIM). "
                             "Defaults to checkpoint value (100 for DDPM).")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the obs_encoder/model sub-modules and warm up "
                             "the compiled graph at startup (adds ~10s to startup, speeds up "
                             "steady-state /infer calls).")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    serve(args.checkpoint, args.device, args.host, args.port,
          embed_layer=args.embed_layer, num_inference_steps=args.num_inference_steps,
          compile=args.compile)


if __name__ == "__main__":
    main()
