"""robo_dataset-backed replacements for the Robomimic replay datasets.

These read a canonical robo_dataset store (VideoStore by default — H.264 MP4
per episode for cameras, zarr lz4 for lowdim/action) produced by
``python -m policy_doctor.scripts.convert_dataset`` from any supported source
(robomimic HDF5, MimicGen merged HDF5, RoboCasa LeRobot v2, DROID), and emit
batches identical to RobomimicReplayLowdimDataset / RobomimicReplayImageDataset:

  lowdim:  {"obs": (T, D_obs) float32, "action": (T, Da) float32}
  image:   {"obs": {rgb: (To, H, W, C) uint8, lowdim: (To, D) float32},
            "action": (T, Da) float32}

The store holds *training-ready* actions: the converter applies the
axis_angle → rotation_6d transform for abs-action datasets at conversion time,
so ``abs_action`` here only selects the normalizer family (it never
re-transforms data).

Dataset splits and curation reuse the exact helpers the replay datasets use
(get_dataset_masks / build_combined_curation_masks), so train/val/holdout
membership is bit-identical given the same ``dataset_path`` (the original
HDF5 — still needed for MH quality-tier splits) and seed.

Pass ``worker_init_fn=robo_store_worker_init_fn`` to DataLoader (the training
workspaces do this automatically when the dataset exposes
``dataloader_worker_init_fn``) so file handles and video decoders are
re-opened per worker after fork.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from robo_dataset.dataset import TrajectoryDataset
from robo_dataset.store_factory import open_store

from robo_dataset.normalization import KeyNormalizer, fit_key

from diffusion_policy.common.normalize_util import (
    array_to_stats,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    robomimic_abs_action_only_normalizer_from_stat,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.sampler import (
    build_combined_curation_masks,
    get_dataset_masks,
)
from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseLowdimDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


def robo_store_worker_init_fn(worker_id: int) -> None:
    """DataLoader worker init: limit thread pools + reopen store handles."""
    threadpool_limits(1)
    info = torch.utils.data.get_worker_info()
    store = getattr(info.dataset, "store", None)
    if store is not None:
        store.on_worker_init(worker_id)


def _validate_action_schema(schema, abs_action: bool, store_path: str) -> None:
    """Loudly reject a store whose action layout contradicts the abs_action flag.

    The schema travels with the store (written at conversion time), so a
    delta-action store passed to an abs-action training config — or vice
    versa — fails here with a precise message instead of training silently
    on mis-normalized actions.
    """
    pos_fields = schema.fields_of_kind("pos")
    rot_fields = schema.fields_of_kind("rot")
    if abs_action:
        bad_rot = [f for f in rot_fields if f.representation != "rotation_6d"]
        if bad_rot:
            raise ValueError(
                f"abs_action=True but store {store_path!r} action schema "
                f"{schema.describe()} has rot field(s) "
                f"{[f.name for f in bad_rot]} not in rotation_6d — this looks "
                "like a store converted without source.abs_action=true."
            )
        bad_pos = [f for f in pos_fields if f.representation == "delta"]
        if bad_pos:
            raise ValueError(
                f"abs_action=True but store {store_path!r} action schema "
                f"{schema.describe()} declares delta position field(s) "
                f"{[f.name for f in bad_pos]}."
            )
    else:
        abs_pos = [f for f in pos_fields if f.representation == "abs"]
        if abs_pos:
            raise ValueError(
                f"abs_action=False but store {store_path!r} action schema "
                f"{schema.describe()} declares ABS position field(s) "
                f"{[f.name for f in abs_pos]} — identity action normalization "
                "would be wrong. Use the delta-action store or set "
                "abs_action=true."
            )


def _verify_store_matches_hdf5(store, store_path: str, dataset_path: str) -> None:
    """Loudly reject a mask-source HDF5 that does not describe the store.

    Split masks (incl. MH quality tiers read from the HDF5's mask/ groups)
    are arrays over episode indices; if the HDF5 and the store disagree on
    episode count, ordering or lengths, every split/curation decision lands
    on the wrong episodes — silently. Episode boundaries are compared
    exactly (shape reads only; cheap)."""
    import h5py

    from diffusion_policy.common.hdf5_robomimic_layout import (
        sorted_robomimic_demo_keys,
    )

    with h5py.File(dataset_path, "r") as f:
        demo_keys = sorted_robomimic_demo_keys(f["data"])
        hdf5_ends = np.cumsum(
            [f[f"data/{dk}/actions"].shape[0] for dk in demo_keys]
        ).astype(np.int64)
    if len(hdf5_ends) != store.n_episodes or not np.array_equal(
        hdf5_ends, store.episode_ends
    ):
        raise ValueError(
            f"dataset_path {dataset_path!r} ({len(hdf5_ends)} episodes, "
            f"{int(hdf5_ends[-1]) if len(hdf5_ends) else 0} frames) does not "
            f"describe store {store_path!r} ({store.n_episodes} episodes, "
            f"{store.n_frames} frames) — split and curation masks would be "
            "assigned to the wrong episodes. Re-convert the store from this "
            "HDF5 or point dataset_path at the store's true source."
        )


# Default normalization specs (robo_dataset.normalization) — the
# robomimic/diffusion-policy conventions, declared once. Override per key via
# the datasets' ``normalization`` parameter, e.g.
#   ++task.dataset.normalization.action=zscore
#   ++task.dataset.normalization.obs=minmax_p99
_ABS_ACTION_SPEC = {"per_kind": {"pos": "minmax"}, "default": "none"}
_DELTA_ACTION_SPEC = "none"
_LOWDIM_OBS_CONCAT_SPEC = "max_abs"
_IMAGE_SPEC = {"method": "fixed_range", "in_range": [0.0, 1.0]}


def _to_dp_normalizer(kn: KeyNormalizer) -> SingleFieldLinearNormalizer:
    """Bridge a robo_dataset KeyNormalizer into dp's checkpoint container."""
    stats = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in kn.stats.items()
        if k in ("min", "max", "mean", "std")
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=np.asarray(kn.scale, dtype=np.float32),
        offset=np.asarray(kn.offset, dtype=np.float32),
        input_stats_dict=stats,
    )


def _fit(arr, spec, schema=None) -> SingleFieldLinearNormalizer:
    """Fit one key with robo_dataset.normalization and wrap for dp."""
    return _to_dp_normalizer(fit_key(arr, spec, schema=schema))


def _action_normalizer(action, abs_action, action_schema, use_legacy_normalizer,
                       override_spec=None):
    """Action normalizer, fitted by robo_dataset.normalization.

    Defaults: abs actions → pos fields minmax / rest identity (schema-driven);
    delta actions → identity. Pre-schema abs stores fall back to the legacy
    hardcoded robomimic slicing (with a warning)."""
    if override_spec is not None:
        return _fit(action, override_spec, schema=action_schema)
    if not abs_action:
        return _fit(action, _DELTA_ACTION_SPEC)
    if use_legacy_normalizer:
        return _fit(action, "max_abs")
    if action_schema is not None:
        return _fit(action, _ABS_ACTION_SPEC, schema=action_schema)
    import warnings
    warnings.warn(
        "Store carries no action schema — guessing single/dual-arm abs-action "
        "layout from the action dim (legacy robomimic heuristic). Re-convert "
        "the store with a schemas: block to make the layout explicit.",
        stacklevel=2,
    )
    stat = array_to_stats(action)
    if stat["mean"].shape[-1] > 10:
        return robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
    return robomimic_abs_action_only_normalizer_from_stat(stat)


class _LazyStoreKey:
    """Lazy per-key array view: slicing/indexing translates to ranged
    store reads, so ``replay_buffer[image_key][frame_idx]`` decodes ONE
    frame — never the whole dataset."""

    def __init__(self, store, key: str):
        self._store = store
        self._key = key

    @property
    def shape(self) -> tuple:
        return (self._store.n_frames,) + tuple(self._store.frame_shape(self._key))

    @property
    def dtype(self):
        return self._store.frame_dtype(self._key)

    def __len__(self) -> int:
        return self._store.n_frames

    def __getitem__(self, idx):
        n = self._store.n_frames
        if isinstance(idx, slice):
            start, stop, step = idx.indices(n)
            out = self._store.get_sequence(self._key, start, stop)
            return out[::step] if step != 1 else out
        if isinstance(idx, (int, np.integer)):
            i = int(idx) + n if idx < 0 else int(idx)
            return self._store.get_sequence(self._key, i, i + 1)[0]
        # fancy indexing: fetch per unique frame (fine for small index sets)
        idx = np.asarray(idx)
        return np.stack([self[int(i)] for i in idx])

    def __array__(self, dtype=None):
        arr = self._store.get_sequence(self._key, 0, self._store.n_frames)
        return arr.astype(dtype) if dtype is not None else np.asarray(arr)


class _StoreReplayView:
    """ReplayBuffer-compatible read view over a TrajectoryStore.

    The attribution stack and the behavior-graphs data layer consume the
    legacy datasets through ``dataset.replay_buffer`` — episode_ends, keys()
    and per-key array access. ``view[key]`` returns a lazy array view whose
    slicing maps to ranged store reads (single-frame image fetches decode a
    single frame); ``np.asarray(view[key])`` or full slices materialize the
    whole key.
    """

    def __init__(self, store):
        self._store = store

    @property
    def episode_ends(self) -> np.ndarray:
        return self._store.episode_ends

    @property
    def n_episodes(self) -> int:
        return self._store.n_episodes

    @property
    def n_steps(self) -> int:
        return self._store.n_frames

    def keys(self):
        return list(self._store.keys)

    def __contains__(self, key) -> bool:
        return key in self._store.keys

    def __getitem__(self, key: str) -> _LazyStoreKey:
        if key not in self._store.keys:
            raise KeyError(
                f"No key {key!r} in store (have: {sorted(self._store.keys)})")
        return _LazyStoreKey(self._store, key)


class _RoboStoreMixin:
    """Shared store opening, split/curation masks and validation/holdout views."""

    def _setup_store_and_masks(
        self,
        store_path: str,
        store_format: str,
        dataset_path: Optional[str],
        horizon: int,
        pad_before: int,
        pad_after: int,
        keys: List[str],
        key_first_k: Optional[Dict[str, int]],
        seed: int,
        val_ratio: float,
        max_train_episodes: Optional[int],
        dataset_mask_kwargs: Dict[str, Any],
        sample_curation_config: Optional[str],
        holdout_selection_config: Optional[str],
        sample_mask_drop_threshold: float,
        verify_curation_dataset_identity: bool,
        store_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        store = open_store(store_path, format=store_format, **(store_kwargs or {}))

        # Self-describing stores: validate the action layout against the
        # abs_action flag before anything trains on it.
        self.action_schema = store.schemas.get("action")
        if self.action_schema is not None:
            self.action_schema.validate_dim(
                int(store.frame_shape("action")[0]), context=f"action @ {store_path}")
            _validate_action_schema(
                self.action_schema, bool(getattr(self, "abs_action", False)), store_path)

        # Same split helper as the replay datasets. dataset_path (the original
        # HDF5) is required for MH quality-tier splits; the store path works
        # for i.i.d. splits (path-keyword detection).
        if dataset_path is not None and str(dataset_path).endswith((".hdf5", ".h5")):
            _verify_store_matches_hdf5(store, store_path, dataset_path)
        mask_path = dataset_path if dataset_path is not None else store_path
        train_mask, val_mask, holdout_mask = get_dataset_masks(
            dataset_path=mask_path,
            num_episodes=store.n_episodes,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes,
            seed=seed,
            **dataset_mask_kwargs,
        )

        episode_mask = train_mask
        sample_mask = None
        if sample_curation_config is not None or holdout_selection_config is not None:
            episode_mask, sample_mask = build_combined_curation_masks(
                episode_ends=store.episode_ends,
                train_mask=train_mask,
                holdout_mask=holdout_mask,
                sample_curation_config=sample_curation_config,
                holdout_selection_config=holdout_selection_config,
                sample_mask_drop_threshold=sample_mask_drop_threshold,
                verify_curation_dataset_identity=verify_curation_dataset_identity,
            )

        self.store = store
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.holdout_mask = holdout_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self._keys = keys
        self._key_first_k = key_first_k
        self._traj = TrajectoryDataset(
            store,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            keys=keys,
            key_first_k=key_first_k,
            episode_mask=episode_mask,
            sample_mask=sample_mask,
            sample_mask_drop_threshold=sample_mask_drop_threshold,
        )

    # Matches the replay datasets: validation/holdout views share the store
    # and drop the sample mask. key_first_k is KEPT (unlike the legacy val
    # sampler): frames past n_obs_steps are discarded by __getitem__ anyway,
    # and for H.264 stores re-decoding them dominates validation cost.
    def _masked_view(self, episode_mask: np.ndarray):
        view = copy.copy(self)
        view._traj = TrajectoryDataset(
            self.store,
            horizon=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=self._keys,
            key_first_k=self._key_first_k,
            episode_mask=episode_mask,
        )
        view.train_mask = episode_mask
        return view

    def get_validation_dataset(self):
        return self._masked_view(self.val_mask)

    def get_holdout_dataset(self):
        return self._masked_view(self.holdout_mask)

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(
            np.asarray(self.store.get_sequence("action", 0, self.store.n_frames))
        )

    def __len__(self) -> int:
        return len(self._traj)

    @property
    def dataloader_worker_init_fn(self):
        return robo_store_worker_init_fn

    # ── Legacy-surface compatibility (attribution / behavior_graphs) ─────────

    @property
    def replay_buffer(self) -> _StoreReplayView:
        """ReplayBuffer-shaped view (episode_ends / keys() / lazy [key])."""
        if not hasattr(self, "_replay_view"):
            self._replay_view = _StoreReplayView(self.store)
        return self._replay_view

    @property
    def sampler(self):
        """The underlying robo_dataset SequenceSampler (exposes .indices with
        the same (N, 4) layout as the legacy sampler)."""
        return self._traj._sampler


class RoboStoreLowdimDataset(_RoboStoreMixin, BaseLowdimDataset):
    """Lowdim dataset over a robo_dataset store.

    Emits {"obs": (T, sum(obs_key_dims)) float32, "action": (T, Da) float32}
    with obs keys concatenated in the order given — identical to
    RobomimicReplayLowdimDataset.
    """

    def __init__(
        self,
        store_path: str,
        obs_keys: List[str] = (
            "object",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ),
        dataset_path: Optional[str] = None,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        abs_action: bool = False,
        # Accepted for drop-in config compatibility; the store already holds
        # converted actions (the transform runs at conversion time).
        rotation_rep: str = "rotation_6d",
        use_legacy_normalizer: bool = False,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
        dataset_mask_kwargs: Dict[str, Any] = {},
        sample_curation_config: Optional[str] = None,
        holdout_selection_config: Optional[str] = None,
        sample_mask_drop_threshold: float = 1.0,
        verify_curation_dataset_identity: bool = True,
        store_format: str = "auto",
        store_kwargs: Optional[Dict[str, Any]] = None,
        # Per-key normalization overrides (robo_dataset.normalization specs):
        # {action: zscore, obs: minmax_p99, ...}; None = robomimic defaults.
        normalization: Optional[Dict[str, Any]] = None,
    ):
        obs_keys = list(obs_keys)
        self.obs_keys = obs_keys
        self.abs_action = abs_action
        self.use_legacy_normalizer = use_legacy_normalizer
        self.normalization = dict(normalization) if normalization else None
        self._setup_store_and_masks(
            store_path=store_path,
            store_format=store_format,
            dataset_path=dataset_path,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            keys=obs_keys + ["action"],
            key_first_k=None,
            seed=seed,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes,
            dataset_mask_kwargs=dict(dataset_mask_kwargs),
            sample_curation_config=sample_curation_config,
            holdout_selection_config=holdout_selection_config,
            sample_mask_drop_threshold=sample_mask_drop_threshold,
            verify_curation_dataset_identity=verify_curation_dataset_identity,
            store_kwargs=store_kwargs,
        )

    def _full_obs_array(self) -> np.ndarray:
        n = self.store.n_frames
        return np.concatenate(
            [self.store.get_sequence(k, 0, n).astype(np.float32) for k in self.obs_keys],
            axis=-1,
        )

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        overrides = self.normalization or {}

        action = np.asarray(
            self.store.get_sequence("action", 0, self.store.n_frames)
        ).astype(np.float32)
        normalizer["action"] = _action_normalizer(
            action, self.abs_action, self.action_schema,
            self.use_legacy_normalizer,
            override_spec=overrides.get("action"))

        normalizer["obs"] = _fit(
            self._full_obs_array(),
            overrides.get("obs", _LOWDIM_OBS_CONCAT_SPEC))
        return normalizer

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self._traj._sampler.sample_sequence(idx)
        obs = np.concatenate(
            [sample[k].astype(np.float32) for k in self.obs_keys], axis=-1
        )
        return {
            "obs": torch.from_numpy(obs),
            "action": torch.from_numpy(sample["action"].astype(np.float32)),
        }


def _task_episode_mask(
    store_path: str,
    task_filter: Optional[List[str]],
    max_episodes_per_task: Optional[int],
    n_episodes: int,
) -> Optional[np.ndarray]:
    if task_filter is None and max_episodes_per_task is None:
        return None
    import json, pathlib
    tasks_path = pathlib.Path(store_path) / "tasks.json"
    if not tasks_path.exists():
        raise FileNotFoundError(
            f"tasks.json not found at {tasks_path} — re-run conversion to generate it"
        )
    with open(tasks_path) as f:
        ep_tasks: dict = json.load(f)
    mask = np.zeros(n_episodes, dtype=bool)
    counts: dict = {}
    for ep_idx in range(n_episodes):
        task = ep_tasks.get(str(ep_idx), "")
        if task_filter is not None and task not in task_filter:
            continue
        if max_episodes_per_task is not None:
            if counts.get(task, 0) >= max_episodes_per_task:
                continue
            counts[task] = counts.get(task, 0) + 1
        mask[ep_idx] = True
    return mask


class RoboStoreImageDataset(_RoboStoreMixin, BaseImageDataset):
    """Image dataset over a robo_dataset store (cameras decoded from H.264).

    Emits the same batches as RobomimicReplayImageDataset: rgb keys stay
    uint8 (To, H, W, C) — the training workspace converts to float CHW on
    the GPU; lowdim obs and action are float32.
    """

    def __init__(
        self,
        shape_meta: dict,
        store_path: str,
        dataset_path: Optional[str] = None,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: Optional[int] = None,
        abs_action: bool = False,
        rotation_rep: str = "rotation_6d",  # compat; see RoboStoreLowdimDataset
        use_legacy_normalizer: bool = False,
        seed: int = 42,
        val_ratio: float = 0.0,
        dataset_mask_kwargs: Dict[str, Any] = {},
        sample_curation_config: Optional[str] = None,
        holdout_selection_config: Optional[str] = None,
        sample_mask_drop_threshold: float = 1.0,
        verify_curation_dataset_identity: bool = True,
        store_format: str = "auto",
        store_kwargs: Optional[Dict[str, Any]] = None,
        # Per-key normalization overrides (robo_dataset.normalization specs);
        # None = robomimic defaults (see _default_lowdim_spec / _IMAGE_SPEC).
        normalization: Optional[Dict[str, Any]] = None,
        task_filter: Optional[List[str]] = None,
        max_episodes_per_task: Optional[int] = None,
    ):
        rgb_keys, lowdim_keys = [], []
        for key, attr in shape_meta["obs"].items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                rgb_keys.append(key)
            elif obs_type == "low_dim":
                lowdim_keys.append(key)

        key_first_k = {}
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.abs_action = abs_action
        self.use_legacy_normalizer = use_legacy_normalizer
        self.normalization = dict(normalization) if normalization else None
        self._setup_store_and_masks(
            store_path=store_path,
            store_format=store_format,
            dataset_path=dataset_path,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            keys=rgb_keys + lowdim_keys + ["action"],
            key_first_k=key_first_k,
            seed=seed,
            val_ratio=val_ratio,
            max_train_episodes=None,
            dataset_mask_kwargs=dict(dataset_mask_kwargs),
            sample_curation_config=sample_curation_config,
            holdout_selection_config=holdout_selection_config,
            sample_mask_drop_threshold=sample_mask_drop_threshold,
            verify_curation_dataset_identity=verify_curation_dataset_identity,
            store_kwargs=store_kwargs,
        )
        if task_filter is not None or max_episodes_per_task is not None:
            task_mask = _task_episode_mask(
                store_path, task_filter, max_episodes_per_task, self.store.n_episodes
            )
            from robo_dataset.dataset import TrajectoryDataset
            combined = self.train_mask & task_mask
            self._traj = TrajectoryDataset(
                self.store,
                horizon=horizon,
                pad_before=pad_before,
                pad_after=pad_after,
                keys=rgb_keys + lowdim_keys + ["action"],
                key_first_k=key_first_k,
                episode_mask=combined,
            )
            self.train_mask = combined

    @staticmethod
    def _default_lowdim_spec(key: str) -> str:
        """Robomimic suffix conventions, made explicit: positions and joint
        positions are range-normalized; quaternions are already in [-1, 1]."""
        if key.endswith(("pos", "position", "positions", "qpos")):
            return "minmax"
        if key.endswith("quat"):
            return "none"
        raise RuntimeError(
            f"No default normalization for lowdim obs key {key!r} — set it "
            "explicitly via the dataset's normalization parameter."
        )

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        n = self.store.n_frames
        overrides = self.normalization or {}

        action = np.asarray(self.store.get_sequence("action", 0, n)).astype(np.float32)
        normalizer["action"] = _action_normalizer(
            action, self.abs_action, self.action_schema,
            self.use_legacy_normalizer,
            override_spec=overrides.get("action"))

        for key in self.lowdim_keys:
            spec = overrides.get(key, self._default_lowdim_spec(key))
            arr = np.asarray(self.store.get_sequence(key, 0, n)).astype(np.float32)
            normalizer[key] = _fit(arr, spec, schema=self.store.schemas.get(key))

        for key in self.rgb_keys:
            normalizer[key] = _fit(None, overrides.get(key, _IMAGE_SPEC))
        return normalizer

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self._traj._sampler.sample_sequence(idx)
        T_slice = slice(self.n_obs_steps)

        obs_dict = {}
        for key in self.rgb_keys:
            # uint8 HWC — permute + float conversion happens on the GPU.
            obs_dict[key] = sample[key][T_slice]
        for key in self.lowdim_keys:
            obs_dict[key] = sample[key][T_slice].astype(np.float32)

        return {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(sample["action"].astype(np.float32)),
        }
