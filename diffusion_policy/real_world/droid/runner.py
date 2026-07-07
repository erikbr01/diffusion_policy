"""DROID real-robot inference runner (control loop only; no persistence).

Three inference modes
---------------------
SYNC
    Inference blocks the control loop. At every open_loop_horizon boundary,
    submit + get() before executing the next chunk.

ASYNC_CHUNK
    At the start of each chunk, submit inference for the *next* chunk
    immediately. Execute the current chunk at control_hz while inference runs
    in the background; block for the new chunk only at the chunk boundary.

ASYNC_STREAMING
    A dedicated background thread calls submit → get in a loop, always using
    the latest observation, throttled to at most policy_hz (uncapped if
    policy_hz is None — resubmits back-to-back as fast as the policy
    allows). The foreground control loop reads from a lock-protected buffer
    at control_hz; if the buffer is empty (policy slower than the loop),
    applies the StaleActionPolicy.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from diffusion_policy.real_world.droid.env import DROIDInferenceEnv
from diffusion_policy.real_world.droid.policy_client import PolicyBase

logger = logging.getLogger(__name__)


class InferenceMode(enum.Enum):
    SYNC = "sync"
    ASYNC_CHUNK = "async_chunk"
    ASYNC_STREAMING = "async_streaming"


class StaleActionPolicy(enum.Enum):
    HOLD_LAST = "hold_last"
    ZERO_ON_STALE = "zero_on_stale"


@dataclass
class EpisodeRecord:
    episode_idx: int
    instruction: str
    n_steps: int
    n_new_chunks: int
    n_stale_steps: int
    success: Optional[bool]  # real-robot success is annotated externally
    avg_chunk_latency_ms: float
    chunk_latencies_ms: List[float] = field(default_factory=list)
    # Per-step embeddings if the policy returned them; None otherwise.
    # Shape (n_steps, D) — each step gets the embedding from its chunk.
    embeddings: Optional[np.ndarray] = None
    # Leading steps taken before the first policy result arrived
    # (streaming mode only). n_steps/embeddings already exclude these;
    # the caller must drop the same count from its own step recording
    # (e.g. env.pop_episode_data()) to stay aligned.
    n_warmup_steps: int = 0


class DROIDInferenceRunner:
    """Run one or many rollouts with a configurable inference mode.

    The runner is a pure control loop: episode persistence is the caller's
    job (drain ``env.pop_episode_data()`` after each ``run_episode``).
    """

    def __init__(
        self,
        env: DROIDInferenceEnv,
        policy: PolicyBase,
        mode: InferenceMode = InferenceMode.ASYNC_CHUNK,
        stale_action_policy: StaleActionPolicy = StaleActionPolicy.HOLD_LAST,
        open_loop_horizon: int = 4,
        warmup_steps: int = 15,
        warmup_horizon: int = 16,
        max_timesteps: int = 600,
        control_hz: float = 15.0,
        policy_hz: Optional[float] = None,
    ) -> None:
        self.env = env
        self.policy = policy
        self.mode = mode
        self.stale_action_policy = stale_action_policy
        self.open_loop_horizon = open_loop_horizon
        self.warmup_steps = warmup_steps
        self.warmup_horizon = warmup_horizon
        self.max_timesteps = max_timesteps
        self.control_hz = control_hz
        # ASYNC_STREAMING only: caps how often the background thread submits
        # a new inference request, decoupled from control_hz. None (default)
        # re-submits back-to-back, as fast as the policy allows — matches
        # prior behavior.
        self.policy_hz = policy_hz

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_episode(
        self,
        instruction: str = "",
        episode_idx: int = 0,
        stop_event: Optional[threading.Event] = None,
    ) -> EpisodeRecord:
        """Run a single rollout and return a summary record.

        If *stop_event* is set mid-episode the control loop exits early and
        returns whatever was collected so far.
        """
        obs = self.env.reset()
        self.policy.reset()

        if self.mode == InferenceMode.SYNC:
            return self._run_sync(obs, instruction, episode_idx, stop_event)
        if self.mode == InferenceMode.ASYNC_CHUNK:
            return self._run_async_chunk(obs, instruction, episode_idx, stop_event)
        return self._run_async_streaming(obs, instruction, episode_idx, stop_event)

    # ------------------------------------------------------------------
    # SYNC mode
    # ------------------------------------------------------------------

    def _run_sync(
        self,
        obs: dict,
        instruction: str,
        episode_idx: int,
        stop_event: Optional[threading.Event] = None,
    ) -> EpisodeRecord:
        dt = 1.0 / self.control_hz
        chunk: Optional[np.ndarray] = None
        chunk_idx = 0
        n_chunks = 0
        latencies: List[float] = []
        total_steps = 0

        for _ in range(self.max_timesteps):
            if stop_event is not None and stop_event.is_set():
                break
            t0 = time.time()

            if chunk is None or chunk_idx >= self.open_loop_horizon or chunk_idx >= len(chunk):
                t_inf = time.time()
                self.policy.submit(obs, instruction)
                chunk = self.policy.get()
                latencies.append((time.time() - t_inf) * 1000)
                chunk_idx = 0
                n_chunks += 1

            action = chunk[chunk_idx]
            chunk_idx += 1
            obs, _, _, _ = self.env.step(action)
            total_steps += 1

            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

        return self._make_record(
            episode_idx, instruction, total_steps, n_chunks, 0, latencies
        )

    # ------------------------------------------------------------------
    # ASYNC_CHUNK mode
    # ------------------------------------------------------------------

    def _run_async_chunk(
        self,
        obs: dict,
        instruction: str,
        episode_idx: int,
        stop_event: Optional[threading.Event] = None,
    ) -> EpisodeRecord:
        dt = 1.0 / self.control_hz
        n_chunks = 0
        latencies: List[float] = []
        embedding_rows: List[np.ndarray] = []

        # Submit the first chunk immediately.
        t_submit = time.time()
        self.policy.submit(obs, instruction)

        total_steps = 0
        # Absolute deadline clock — avoids drift from incremental per-step sleeps.
        t_next = time.time()
        while total_steps < self.max_timesteps:
            if stop_event is not None and stop_event.is_set():
                break
            # Block for the pending chunk (should already be done when we arrive).
            chunk = self.policy.get()
            latencies.append((time.time() - t_submit) * 1000)
            n_chunks += 1
            chunk_embedding = getattr(self.policy, "last_embedding", None)

            # Use a longer horizon during warmup so the policy has time to
            # commit to an action before switching to the reactive steady-state.
            effective_horizon = (
                self.warmup_horizon if total_steps < self.warmup_steps
                else self.open_loop_horizon
            )
            horizon = min(effective_horizon, len(chunk), self.max_timesteps - total_steps)

            for i in range(horizon):
                if stop_event is not None and stop_event.is_set():
                    break
                # Sleep until the next absolute deadline, then step.
                now = time.time()
                if t_next > now:
                    time.sleep(t_next - now)
                t_next += dt

                obs, _, _, _ = self.env.step(chunk[i])
                total_steps += 1
                if chunk_embedding is not None:
                    embedding_rows.append(chunk_embedding)

                # Submit on the first step of the chunk so inference runs in
                # the background during steps 1..horizon-1.
                if i == 0:
                    t_submit = time.time()
                    self.policy.submit(obs, instruction)

        embeddings = np.stack(embedding_rows) if embedding_rows else None
        return self._make_record(
            episode_idx, instruction, total_steps, n_chunks, 0, latencies, embeddings
        )

    # ------------------------------------------------------------------
    # ASYNC_STREAMING mode
    # ------------------------------------------------------------------

    def _run_async_streaming(
        self,
        obs: dict,
        instruction: str,
        episode_idx: int,
        stop_event: Optional[threading.Event] = None,
    ) -> EpisodeRecord:
        dt = 1.0 / self.control_hz

        # Shared state between foreground loop and inference thread.
        _lock = threading.Lock()
        _obs_ref: List[dict] = [obs]
        _instr_ref: List[str] = [instruction]
        _buffer: List[Optional[np.ndarray]] = [None]
        _chunk_step: List[int] = [0]
        _last_action: List[Optional[np.ndarray]] = [None]
        _chunk_embedding: List[Optional[np.ndarray]] = [None]
        _last_embedding: List[Optional[np.ndarray]] = [None]
        _stop = threading.Event()
        latencies: List[float] = []
        n_chunks_ref: List[int] = [0]
        # One entry per executed step; None for steps taken before the first
        # policy result arrived. Trimmed (along with those leading steps)
        # once the episode ends — see n_warmup_steps below.
        embedding_rows: List[Optional[np.ndarray]] = []

        min_submit_period = 1.0 / self.policy_hz if self.policy_hz else 0.0

        def _inference_thread() -> None:
            while not _stop.is_set():
                with _lock:
                    curr_obs = _obs_ref[0]
                    curr_instr = _instr_ref[0]
                t0 = time.time()
                try:
                    self.policy.submit(curr_obs, curr_instr)
                    new_chunk = self.policy.get()
                except Exception as e:  # noqa: BLE001 — keep the loop alive
                    logger.warning("inference error: %s", e)
                    continue
                lat_ms = (time.time() - t0) * 1000
                new_embedding = getattr(self.policy, "last_embedding", None)
                with _lock:
                    _buffer[0] = new_chunk
                    _chunk_step[0] = 0  # latest beats stale: drop unexecuted tail
                    _chunk_embedding[0] = new_embedding
                    n_chunks_ref[0] += 1
                    latencies.append(lat_ms)
                # Throttle re-submission so a fast policy doesn't replace the
                # in-flight chunk faster than policy_hz — caps plan churn
                # independent of control_hz.
                remaining = min_submit_period - (time.time() - t0)
                if remaining > 0:
                    time.sleep(remaining)

        thread = threading.Thread(target=_inference_thread, daemon=True, name="droid_infer")
        thread.start()

        n_stale = 0
        total_steps = 0
        action_dim = self.env.action_dim

        try:
            for _ in range(self.max_timesteps):
                if stop_event is not None and stop_event.is_set():
                    break
                t0 = time.time()

                with _lock:
                    chunk = _buffer[0]
                    idx = _chunk_step[0]
                    embedding = _chunk_embedding[0]

                if chunk is not None and idx < len(chunk):
                    action = chunk[idx]
                    with _lock:
                        _chunk_step[0] += 1
                    _last_action[0] = action
                    _last_embedding[0] = embedding
                else:
                    # Buffer exhausted or not yet ready.
                    if (
                        self.stale_action_policy == StaleActionPolicy.HOLD_LAST
                        and _last_action[0] is not None
                    ):
                        action = _last_action[0]
                    else:
                        action = np.zeros(action_dim, dtype=np.float32)
                    embedding = _last_embedding[0]
                    if embedding is not None:
                        # Buffer momentarily empty after the policy was
                        # already active — a real stale step, not warmup.
                        n_stale += 1

                embedding_rows.append(embedding)

                new_obs, _, _, _ = self.env.step(action)
                total_steps += 1

                with _lock:
                    _obs_ref[0] = new_obs

                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)
        finally:
            _stop.set()
            thread.join(timeout=5.0)

        # Drop the leading steps taken before the first policy result
        # arrived — no embedding exists for them, so they can't be kept
        # aligned with the rest of the episode. n_warmup_steps tells the
        # caller to drop the same count from its own step recording.
        n_warmup_steps = next(
            (i for i, r in enumerate(embedding_rows) if r is not None),
            len(embedding_rows),
        )
        active_rows = embedding_rows[n_warmup_steps:]
        embeddings = np.stack(active_rows) if active_rows else None
        return self._make_record(
            episode_idx, instruction, total_steps - n_warmup_steps, n_chunks_ref[0],
            n_stale, latencies, embeddings, n_warmup_steps,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_record(
        episode_idx: int,
        instruction: str,
        n_steps: int,
        n_chunks: int,
        n_stale: int,
        latencies: List[float],
        embeddings: Optional[np.ndarray] = None,
        n_warmup_steps: int = 0,
    ) -> EpisodeRecord:
        avg_lat = float(np.mean(latencies)) if latencies else 0.0
        return EpisodeRecord(
            episode_idx=episode_idx,
            instruction=instruction,
            n_steps=n_steps,
            n_new_chunks=n_chunks,
            n_stale_steps=n_stale,
            success=None,
            avg_chunk_latency_ms=avg_lat,
            chunk_latencies_ms=list(latencies),
            embeddings=embeddings,
            n_warmup_steps=n_warmup_steps,
        )
