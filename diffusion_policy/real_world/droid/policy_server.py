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
"""

from __future__ import annotations

import argparse
import io
import logging

logger = logging.getLogger(__name__)


def build_app(checkpoint: str, device: str):
    """Load the checkpoint and return the Flask app."""
    import numpy as np
    import torch
    from flask import Flask, request
    from omegaconf import OmegaConf

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
    logger.info("ready on %s (%s policy)", device, "image" if is_image else "lowdim")

    app = Flask(__name__)

    @app.get("/health")
    def health():
        return {"status": "ok", "device": device}

    @app.get("/meta")
    def meta():
        return {
            "policy_type": "image" if is_image else "lowdim",
            "n_obs_steps": int(cfg.n_obs_steps),
            "horizon": int(cfg.horizon),
            "n_action_steps": int(cfg.n_action_steps),
            "action_dim": int(policy.action_dim),
            "shape_meta": shape_meta,
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
        np.save(buf, action)
        return buf.getvalue(), 200, {"Content-Type": "application/octet-stream"}

    return app


def serve(checkpoint: str, device: str, host: str = "127.0.0.1", port: int = 5001) -> None:
    app = build_app(checkpoint, device)
    # threaded=False keeps the policy on a single thread (one request at a time).
    app.run(host=host, port=port, threaded=False, use_reloader=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="HTTP diffusion-policy server")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    serve(args.checkpoint, args.device, args.host, args.port)


if __name__ == "__main__":
    main()
