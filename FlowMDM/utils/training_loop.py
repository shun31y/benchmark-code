import copy
import functools
import json
import os
import time
from types import SimpleNamespace
import numpy as np

import blobfile as bf
import torch
from torch.optim import AdamW

from diffusion import logger
from diffusion.fp16_util import MixedPrecisionTrainer
from utils import dist_util
from diffusion.resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from runners import eval
from data_loaders.get_data import get_dataset_loader
from diffusion.diffusion_wrappers import DiffusionWrapper_FlowMDM
from utils.history_text import build_history_text_embeddings

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

#torch.backends.cuda.matmul.allow_tf32 = True
#torch.backends.cudnn.allow_tf32 = True

class TrainLoop:
    def __init__(self, args, train_platform, model, diffusion, data):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        # num_steps is treated as "additional steps to run in this launch".
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_motion_loader, self.eval_gt_data = None, None, None
        if args.dataset in ['humanml'] and args.eval_during_training:
            eval_args = copy.copy(args)
            if not hasattr(eval_args, "transition_length"):
                eval_args.transition_length = 60
            eval_guidance = getattr(args, "guidance_param", 1.0)
            eval_diffusion = DiffusionWrapper_FlowMDM(
                SimpleNamespace(guidance_param=eval_guidance), diffusion, model
            )
            self.eval_gt_data = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=False,
                                                   split=args.eval_split,
                                                   load_mode='gt', protocol=args.protocol, pose_rep=args.pose_rep,)
            self.eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
            eval_split = args.eval_split if args.eval_split in eval.EVAL_FILES[args.dataset] else 'test'
            eval_file = eval.EVAL_FILES[args.dataset][eval_split]
            if args.eval_num_samples > 0:
                with open(eval_file, "r") as f:
                    all_eval_items = json.load(f)
                if args.eval_num_samples < len(all_eval_items):
                    rng = np.random.RandomState(args.seed)
                    idx = rng.choice(len(all_eval_items), size=args.eval_num_samples, replace=False)
                    idx = np.sort(idx)
                    eval_subset = [all_eval_items[i] for i in idx]
                    eval_file_subset = os.path.join(self.save_dir, "eval_subset_train.json")
                    with open(eval_file_subset, "w") as f:
                        json.dump(eval_subset, f)
                    eval_file = eval_file_subset
                    print(f"Training-time eval subset: {len(eval_subset)}/{len(all_eval_items)} from [{eval.EVAL_FILES[args.dataset][eval_split]}]")
            max_motion_length = self.eval_gt_data.dataset.opt.max_motion_length
            precomputed_root = os.path.join(self.save_dir, "evaluation_precomputed_train")
            self.eval_motion_loader = lambda rep: eval.get_mdm_loader(
                eval_args, model, eval_diffusion, args.eval_batch_size, eval_file, self.eval_gt_data.dataset,
                max_motion_length=max_motion_length,
                precomputed_folder=os.path.join(precomputed_root, f"{rep:02d}"),
                scenario="",
            )
        self.use_ddp = False
        self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                ), strict=False
            )

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
            for motion, cond in tqdm(self.data):
                if self.step >= self.num_steps:
                    break
                if not (not self.lr_anneal_steps or self.global_step() < self.lr_anneal_steps):
                    break

                motion = motion.to(self.device)
                cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}

                self.run_step(motion, cond)
                if self.step % self.log_interval == 0:
                    data_to_report = { k: v for k,v in logger.get_current().name2val.items() if k not in ['loss', 'step', 'samples'] and '_q' not in k}
                    self.train_platform.report_data(data=data_to_report, iteration=self.step, group_name='Loss')
                    log_message = 'step[{}]: '.format(self.global_step())
                    for k,v in data_to_report.items():
                        if k not in ["param_norm", "grad_norm"]:
                            log_message += f'{k}={v:0.5f} '
                    for k,v in logger.get_current().name2val.items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.global_step(), v))

                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=self.step, group_name='Loss')

                if self.step % self.save_interval == 0:
                    self.save()
                    self.model.eval()
                    self.evaluate()
                    self.model.train()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
            if self.step >= self.num_steps:
                break
            if not (not self.lr_anneal_steps or self.global_step() < self.lr_anneal_steps):
                break
        # Save the last checkpoint if it wasn't already saved.
        if self.step > 0 and self.step % self.save_interval != 0:
            self.save()
            self.evaluate()

    def evaluate(self):
        if not self.args.eval_during_training:
            return
        start_eval = time.time()
        if self.eval_wrapper is not None:
            print('Running evaluation loop: [Should take about 90 min]')
            log_file = os.path.join(self.save_dir, f'eval_humanml_{self.global_step():09d}.log')
            diversity_times = 300
            mean_dict, _ = eval.evaluation(
                self.eval_wrapper,
                [self.eval_gt_data],
                self.eval_motion_loader,
                ['motion'],
                [log_file],
                replication_times=self.args.eval_rep_times,
                diversity_times=diversity_times,
                extrapolation=False,
            )
            print(mean_dict)
            motion_metrics = mean_dict.get('motion', {})
            for metric_name, metric_value in motion_metrics.items():
                if isinstance(metric_value, (list, tuple)) and len(metric_value) == 2:
                    self.train_platform.report_scalar(
                        name=metric_name, value=metric_value[0], iteration=self.step, group_name='Eval'
                    )
                    self.train_platform.report_scalar(
                        name=f'{metric_name}_ci', value=metric_value[1], iteration=self.step, group_name='Eval'
                    )
                else:
                    self.train_platform.report_scalar(
                        name=metric_name, value=metric_value, iteration=self.step, group_name='Eval'
                    )


        end_eval = time.time()
        print(f'Evaluation time: {round(end_eval-start_eval)/60}min')


    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.step += 1
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            if self.args.use_history_text:
                micro_cond['y']['text_embeddings'] = build_history_text_embeddings(
                    self.ddp_model,
                    micro_cond['y'],
                    self.args.history_current_weight,
                )

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond,
                dataset=self.data.dataset
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            loss.backward()

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = min(1.0, self.global_step() / self.lr_anneal_steps)
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step)
        logger.logkv("samples", self.step * self.global_batch)

    def global_step(self):
        return self.step + self.resume_step


    def ckpt_file_name(self):
        return f"model{self.global_step():09d}.pt"


    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
            for e in clip_weights:
                del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{self.global_step():09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
