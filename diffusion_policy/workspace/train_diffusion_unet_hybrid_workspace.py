if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import time
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.common.device_util import get_device, non_blocking_for
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.common.step_timer import StepTimer
from diffusion_policy.dataset.robomimic_replay_image_dataset import dataset_worker_init_fn

OmegaConf.register_new_resolver("eval", eval, replace=True)


class DataPrefetcher:
    """Overlaps CPU→GPU transfer of batch N+1 with GPU compute on batch N."""
    def __init__(self, loader, device, transform_fn=None):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.transform_fn = transform_fn
        self._preload()

    def _preload(self):
        try:
            self._next = next(self.loader)
        except StopIteration:
            self._next = None
            return
        with torch.cuda.stream(self.stream):
            self._next = dict_apply(self._next, lambda x: x.to(self.device, non_blocking=True))
            if self.transform_fn is not None:
                self._next = self.transform_fn(self._next)

    def __iter__(self):
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self._next
        if batch is None:
            raise StopIteration
        dict_apply(batch, lambda x: x.record_stream(torch.cuda.current_stream()) if x.is_cuda else None)
        self._preload()
        return batch

    def __len__(self):
        return len(self.loader)

class TrainDiffusionUnetHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        terminate_training = False

        # --- acceleration flags (DDP / TF32 / compile) ---
        rank        = int(os.environ.get("LOCAL_RANK", cfg.training.get("_ddp_rank", 0)))
        world_size  = int(os.environ.get("WORLD_SIZE",  cfg.training.get("_ddp_world_size", 1)))
        is_main     = (rank == 0)
        use_tf32    = bool(cfg.training.get("tf32", False))
        use_compile = bool(cfg.training.get("compile", False))

        if world_size > 1:
            from diffusion_policy.common.ddp_util import setup_ddp
            setup_ddp(rank=rank, world_size=world_size)

        if use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)

        # device: DDP uses one GPU per rank; single-GPU falls back to cfg
        if world_size > 1:
            device = torch.device(f"cuda:{rank}")
        else:
            device = get_device(cfg.training.device)

        # robo_dataset-backed datasets need per-worker store re-init after fork
        worker_init_fn = getattr(
            dataset, "dataloader_worker_init_fn", dataset_worker_init_fn)

        # train dataloader: DistributedSampler for DDP, standard otherwise
        if world_size > 1:
            from torch.utils.data.distributed import DistributedSampler
            train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
            train_dl_kw = {k: v for k, v in cfg.dataloader.items() if k != "shuffle"}
            train_dataloader = DataLoader(dataset, sampler=train_sampler,
                worker_init_fn=worker_init_fn, **train_dl_kw)
        else:
            train_sampler = None
            train_dataloader = DataLoader(dataset,
                worker_init_fn=worker_init_fn, **cfg.dataloader)

        normalizer = dataset.get_normalizer()

        # configure validation dataset (no sampler — rank 0 runs full val)
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset,
            worker_init_fn=worker_init_fn, **cfg.val_dataloader)

        # GPU image transforms (robo_dataset tier 3): uint8 HWC → float32 CHW
        # in [0,1], applied inside the prefetch CUDA stream (train) or inline
        # (val). With training.gpu_random_crop=true, train batches additionally
        # get a per-timestep RandomCrop to policy.external_crop_shape — this
        # replaces the in-encoder CropRandomizer (set policy.crop_shape=null,
        # policy.external_crop_shape=[h,w]); the policy center-crops full-size
        # obs at validation/inference itself.
        rgb_keys = list(getattr(dataset, 'rgb_keys', []))
        if not rgb_keys:
            import warnings
            warnings.warn(
                f"{type(dataset).__name__} exposes no rgb_keys — no GPU "
                "uint8→float conversion will run. Correct only for datasets "
                "that already emit float images (e.g. pusht); uint8-emitting "
                "datasets MUST define .rgb_keys or training sees 0-255 "
                "values.", stacklevel=2)

        # external-crop config consistency: random crop must run exactly once.
        _gpu_rc = bool(cfg.training.get('gpu_random_crop', False))
        _ext_crop = OmegaConf.select(cfg, 'policy.external_crop_shape')
        _enc_crop = OmegaConf.select(cfg, 'policy.crop_shape')
        if _gpu_rc and _ext_crop is None:
            raise ValueError(
                "training.gpu_random_crop=true requires "
                "policy.external_crop_shape=[h, w] (the GPU transform's crop "
                "size and the encoder input size).")
        if _gpu_rc and _enc_crop is not None:
            raise ValueError(
                "training.gpu_random_crop=true with policy.crop_shape set "
                "would crop twice — set policy.crop_shape: null.")
        if not _gpu_rc and _ext_crop is not None:
            raise ValueError(
                "policy.external_crop_shape is set but "
                "training.gpu_random_crop=false — training would silently run "
                "with NO crop augmentation (the policy center-crops "
                "deterministically). Enable training.gpu_random_crop or use "
                "the in-encoder crop (policy.crop_shape).")

        # horizon / n_obs_steps must agree across dataset, top level and policy
        for name, ds_val, cfg_val in (
            ("horizon", getattr(dataset, 'horizon', None), cfg.task.dataset.horizon),
            ("n_obs_steps", getattr(dataset, 'n_obs_steps', None), cfg.n_obs_steps),
        ):
            if ds_val is not None and int(ds_val) != int(cfg_val):
                raise ValueError(
                    f"dataset.{name}={ds_val} != cfg {name}={cfg_val} — "
                    "config copies drifted apart.")
        if int(cfg.task.dataset.horizon) != int(cfg.policy.horizon) or                 int(cfg.n_obs_steps) != int(cfg.policy.n_obs_steps):
            raise ValueError(
                f"task.dataset.horizon={cfg.task.dataset.horizon} / "
                f"policy.horizon={cfg.policy.horizon}, "
                f"n_obs_steps={cfg.n_obs_steps} / "
                f"policy.n_obs_steps={cfg.policy.n_obs_steps} — "
                "config copies drifted apart.")

        gpu_image_transform = None   # val / eval batches
        train_gpu_transform = None   # train batches (+ optional random crop)
        if rgb_keys:
            from robo_dataset.transforms import (
                ComposeGPU, ImageToTensorDevice, RandomCrop)
            val_compose = ComposeGPU([ImageToTensorDevice(rgb_keys)])
            train_tfms = [ImageToTensorDevice(rgb_keys)]
            if cfg.training.get('gpu_random_crop', False):
                crop_shape = tuple(cfg.policy.external_crop_shape)
                train_tfms.append(RandomCrop(
                    rgb_keys, size=crop_shape, pad=0, per_timestep=True))
            train_compose = ComposeGPU(train_tfms)

            def gpu_image_transform(batch, _compose=val_compose):
                batch['obs'] = _compose(batch['obs'])
                return batch

            def train_gpu_transform(batch, _compose=train_compose):
                batch['obs'] = _compose(batch['obs'])
                return batch

        getattr(self.model, "module", self.model).set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # training duration: num_steps overrides num_epochs when set
        # max_global_steps is kept as a backward-compatible alias
        import math
        _num_steps = cfg.training.get("num_steps", None) or getattr(cfg.training, "max_global_steps", None)
        _steps_per_epoch = len(train_dataloader)
        num_epochs_to_run = (
            math.ceil(_num_steps / _steps_per_epoch)
            if _num_steps is not None
            else cfg.training.num_epochs
        )

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                (_num_steps if _num_steps is not None else _steps_per_epoch * num_epochs_to_run)
                // cfg.training.gradient_accumulate_every),
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env (main rank only to avoid parallel sim instances)
        env_runner: BaseImageRunner = None
        if is_main:
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir)
            assert isinstance(env_runner, BaseImageRunner)

        # configure logging (main rank only)
        wandb_run = None
        if is_main:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update({"output_dir": self.output_dir}, allow_val_change=True)

        # configure checkpoint (main rank only)
        topk_manager = None
        if is_main:
            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, 'checkpoints'),
                **cfg.checkpoint.topk
            )

        # device transfer
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # compile before DDP wrap (torch 2.x recommendation)
        if use_compile:
            from diffusion_policy.common.ddp_util import compile_policy
            compile_policy(self.model)

        # DDP wrap
        if world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(self.model, device_ids=[rank])

        # save batch for sampling
        train_sampling_batch = None

        step_timer = StepTimer()

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            num_epochs_to_run = 2
            _num_steps = None
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with (JsonLogger(log_path) if is_main else open(os.devnull, 'w')) as json_logger:
            for local_epoch_idx in range(num_epochs_to_run):
                if terminate_training:
                    break
                step_log = dict()
                # ========= train for this epoch ==========
                if train_sampler is not None:
                    train_sampler.set_epoch(self.epoch)
                train_losses = list()
                prefetcher = DataPrefetcher(train_dataloader, device, transform_fn=train_gpu_transform) if device.type == 'cuda' else None
                _iter = prefetcher if prefetcher is not None else train_dataloader
                with tqdm.tqdm(_iter, desc=f"Training epoch {self.epoch}",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        step_timer.reset()
                        # device transfer (skipped for CUDA — DataPrefetcher handles it async)
                        with step_timer.time("data_transfer"):
                            if prefetcher is None:
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=non_blocking_for(device)))
                                if train_gpu_transform is not None:
                                    batch = train_gpu_transform(batch)
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        with step_timer.time("forward_backward"):
                            raw_loss = getattr(self.model, "module", self.model).compute_loss(batch)
                            loss = raw_loss / cfg.training.gradient_accumulate_every
                            loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            with step_timer.time("optimizer_step"):
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                lr_scheduler.step()

                        # update ema
                        if cfg.training.use_ema:
                            with step_timer.time("ema_update"):
                                from diffusion_policy.common.ddp_util import unwrap_model, ema_safe_model
                                raw = unwrap_model(self.model)
                                with ema_safe_model(raw):
                                    ema.step(raw)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }
                        step_log.update(step_timer.to_log_dict())

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            if is_main:
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)
                            self.global_step += 1
                            if _num_steps is not None and self.global_step >= _num_steps:
                                terminate_training = True
                                break

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch (main rank only) ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                if is_main:
                    policy.eval()

                    # run rollout
                    if (self.epoch % cfg.training.rollout_every) == 0 or terminate_training:
                        _t0 = time.perf_counter()
                        runner_log = env_runner.run(policy)
                        step_log['timer/rollout'] = time.perf_counter() - _t0
                        # log all
                        step_log.update(runner_log)

                    # run validation
                    if (self.epoch % cfg.training.val_every) == 0 or terminate_training:
                        _t0 = time.perf_counter()
                        with torch.no_grad():
                            val_losses = list()
                            with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                                for batch_idx, batch in enumerate(tepoch):
                                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=non_blocking_for(device)))
                                    if gpu_image_transform is not None:
                                        batch = gpu_image_transform(batch)
                                    loss = getattr(self.model, "module", self.model).compute_loss(batch)
                                    val_losses.append(loss)
                                    if (cfg.training.max_val_steps is not None) \
                                        and batch_idx >= (cfg.training.max_val_steps-1):
                                        break
                            if len(val_losses) > 0:
                                val_loss = torch.mean(torch.tensor(val_losses)).item()
                                # log epoch average validation loss
                                step_log['val_loss'] = val_loss
                        step_log['timer/validation'] = time.perf_counter() - _t0

                    # run diffusion sampling on a training batch
                    if (self.epoch % cfg.training.sample_every) == 0 or terminate_training:
                        _t0 = time.perf_counter()
                        with torch.no_grad():
                            # sample trajectory from training set, and evaluate difference
                            batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=non_blocking_for(device)))
                            obs_dict = batch['obs']
                            gt_action = batch['action']

                            result = policy.predict_action(obs_dict)
                            pred_action = result['action_pred']
                            mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                            step_log['train_action_mse_error'] = mse.item()
                            del batch
                            del obs_dict
                            del gt_action
                            del result
                            del pred_action
                            del mse
                        step_log['timer/sampling'] = time.perf_counter() - _t0

                    # checkpoint
                    if (self.epoch % cfg.training.checkpoint_every) == 0 or terminate_training:
                        # checkpointing
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint()
                        if cfg.checkpoint.save_last_snapshot:
                            self.save_snapshot()

                        # sanitize metric names
                        metric_dict = dict()
                        for key, value in step_log.items():
                            new_key = key.replace('/', '_')
                            metric_dict[new_key] = value

                        # We can't copy the last checkpoint here
                        # since save_checkpoint uses threads.
                        # therefore at this point the file might have been empty!
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                if is_main:
                    policy.train()

                    # end of epoch
                    # log of last step is combined with validation and rollout
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
                # Release CUDA caching-allocator blocks retained by the prior
                # epoch's DataPrefetcher stream (record_stream pins blocks
                # until the stream catches up; recreating the prefetcher each
                # epoch can otherwise let those pins accumulate as a slow
                # VRAM creep across hundreds of epochs).
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if terminate_training or (
                    _num_steps is not None and self.global_step >= _num_steps
                ):
                    break

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
