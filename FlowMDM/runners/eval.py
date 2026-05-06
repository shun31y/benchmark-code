from utils.parser_util import evaluation_parser
from utils.fixseed import fixseed
from datetime import datetime
from data_loaders.model_motion_loaders import get_mdm_loader
from data_loaders.humanml.utils.metrics import *
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from collections import OrderedDict
from data_loaders.humanml.scripts.motion_process import *
from data_loaders.humanml.utils.utils import *
from utils.model_util import load_model
import os
import re

from diffusion import logger
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from diffusion.diffusion_wrappers import DiffusionWrapper_FlowMDM as DiffusionWrapper

torch.multiprocessing.set_sharing_strategy('file_system')

from utils.metrics import evaluate_jerk, evaluate_matching_score, evaluate_fid, evaluate_diversity, get_metric_statistics, generate_plot_PJ


EVAL_FILES = {
    "humanml": {
        "test": "./dataset/humanml_test_set.json",
        "anaphora": "./dataset/HumanML3D/humanml_test_set_anaphora.json",
        "anaphora_fullgen": "./dataset/HumanML3D/humanml_test_set_anaphora_fullgen.json",
        "extrapolation": "./dataset/humanml_extrapolation.json",
    },
    "babel": {
        "val": "./dataset/babel_val_set.json",
        "extrapolation": "./dataset/babel_extrapolation.json",
    },
}
SCENARIOS = {
    "humanml": ["short", "medium", "long", "all"],
    "babel": ["in-distribution", "out-of-distribution"]
}
SEQ_LENS = {
    "humanml": (70, 200),
    "babel": (30, 200),
}
BATCH_SIZE = 32


def _safe_suffix(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _resolved_eval_file(args, eval_file=None):
    if eval_file:
        return eval_file
    return getattr(args, "resolved_eval_file", "") or getattr(args, "eval_file", "")


def is_anaphora_fullgen_eval(args, eval_file=None):
    resolved_eval_file = _resolved_eval_file(args, eval_file)
    if getattr(args, "scenario", "") == "anaphora_fullgen":
        return True
    return "anaphora_fullgen" in os.path.basename(resolved_eval_file)


def is_anaphora_eval(args, eval_file=None):
    if getattr(args, "target_segment_from_id", False):
        return True
    if getattr(args, "scenario", "") in {"anaphora", "anaphora_fullgen"}:
        return True
    resolved_eval_file = _resolved_eval_file(args, eval_file)
    return "anaphora" in os.path.basename(resolved_eval_file)


def resolve_eval_targets(args, eval_file):
    if is_anaphora_fullgen_eval(args, eval_file) and args.eval_target == "motion_and_transition":
        raise ValueError("scenario=anaphora_fullgen does not support transition evaluation")
    if args.eval_target == "motion_only":
        return ['motion']
    if args.eval_target == "motion_and_transition":
        return ['motion', 'transition']
    if is_anaphora_eval(args, eval_file):
        args.target_segment_from_id = True
        return ['motion']
    return ['motion', 'transition']


def method_suffix(args, eval_file=None):
    suffix = ""
    if getattr(args, "use_history_text", False):
        suffix += f"_histText_w{args.history_current_weight:g}"
    if is_anaphora_fullgen_eval(args, eval_file):
        suffix += "_anaFullGenTargetOnly"
    elif is_anaphora_eval(args, eval_file):
        suffix += "_anaTarget"
    if getattr(args, "eval_target", "auto") != "auto":
        suffix += f"_{args.eval_target}"
    if getattr(args, "eval_file", ""):
        suffix += f"_{_safe_suffix(os.path.splitext(os.path.basename(args.eval_file))[0])}"
    return suffix



def update_metrics_dict(all_metrics_dict, metric, current_metrics_dict):
    for key, item in current_metrics_dict.items():
        if key not in all_metrics_dict[metric]:
            all_metrics_dict[metric][key] = [item]
        else:
            all_metrics_dict[metric][key] += [item]

# print to both stdout and file
def pprint(f, *args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=f, flush=True)


def compute_gt_fid_stats(eval_wrapper, gt_loader):
    gt_motion_embeddings = []
    with torch.no_grad():
        for batch in gt_loader:
            motions, m_lens = batch[4:6]
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions,
                m_lens=m_lens,
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    return calculate_activation_statistics(gt_motion_embeddings)
    
def evaluation(eval_wrapper, gt_loaders, eval_motion_loader, targets, log_files, replication_times, diversity_times, extrapolation=False):
    assert len(targets) == len(log_files), f'len(targets)={len(targets)} != len(log_files)={len(log_files)}'
    files = [open(log_file, 'w') for log_file in log_files]

    targets_metrics = {}
    for target in targets:
        if "transition" in target.lower():
            targets_metrics[target] = OrderedDict({
                                'FID': OrderedDict({}),
                                'Diversity': OrderedDict({}),
                                'PeakJerk': OrderedDict({}),
                                'AUJ': OrderedDict({}),
                                }) 
        else:
            targets_metrics[target] = OrderedDict({
                                'MatchingScore': OrderedDict({}),
                                'R_precision': OrderedDict({}),
                                'FID': OrderedDict({}),
                                'Diversity': OrderedDict({}),
                                'MultiModality': OrderedDict({})})

    targets_plots = {
        target: OrderedDict({
            'PeakJerk': OrderedDict({}),
        }) for target in targets
    }

    gt_fid_stats = {}
    for target, gt_loader, f in zip(targets, gt_loaders, files):
        if "FID" in targets_metrics[target]:
            pprint(f, f'Computing GT FID cache for [{target}]...')
            gt_fid_stats[target] = compute_gt_fid_stats(eval_wrapper, gt_loader)
            pprint(f, f'GT FID cache ready for [{target}]')

    for replication in range(replication_times):
        # we use the same generated motions for both targets (motion and transition)
        motion_loader = eval_motion_loader(replication)[0]
        for target, gt_loader, f in zip(targets, gt_loaders, files):
            motion_loader.dataset.switch_target(target)
            motion_loaders = {
                f'{target}': motion_loader
            }
            all_metrics = targets_metrics[target]
            all_plots = targets_plots[target]
            
            pprint(f, f'==================== [{target}] Replication {replication} ====================')
            
            pprint(f, f'Time: {datetime.now()}')

            if "FID" in all_metrics or "Diversity" in all_metrics or "MatchingScore" in all_metrics or "R_precision" in all_metrics:
                compute_precision = "R_precision" in all_metrics and not extrapolation
                compute_matching = "MatchingScore" in all_metrics and not extrapolation
                mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(eval_wrapper, motion_loaders, f, compute_precision=compute_precision, compute_matching=compute_matching)
                if compute_matching:
                    update_metrics_dict(all_metrics, 'MatchingScore', mat_score_dict)
                if compute_precision:
                    update_metrics_dict(all_metrics, 'R_precision', R_precision_dict)

            if "FID" in all_metrics:
                fid_score_dict = evaluate_fid(eval_wrapper, gt_loader, acti_dict, f, gt_stats=gt_fid_stats.get(target))
                update_metrics_dict(all_metrics, 'FID', fid_score_dict)

            if "Diversity" in all_metrics:
                div_score_dict = evaluate_diversity(acti_dict, f, diversity_times)
                update_metrics_dict(all_metrics, 'Diversity', div_score_dict)
                
            if "Jerk" in all_metrics or "AUJ" in all_metrics:
                jerk_score_dict, auj_score_dict, auj_plot_values = evaluate_jerk(motion_loaders)
                update_metrics_dict(all_metrics, 'PeakJerk', jerk_score_dict)
                update_metrics_dict(all_metrics, 'AUJ', auj_score_dict)
                update_metrics_dict(all_plots, 'PeakJerk', auj_plot_values)
            pprint(f, f'!!! DONE !!!')


    mean_dict = {}
    for target, f in zip(targets, files):
        all_metrics = targets_metrics[target]
        for metric_name, metric_dict in all_metrics.items():
            pprint(f, '========== %s Summary ==========' % metric_name)
            for model_name, values in metric_dict.items():
                if model_name not in mean_dict:
                    mean_dict[model_name] = {}
                mean, conf_interval = get_metric_statistics(np.array(values), replication_times)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    pprint(f, f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    mean_dict[model_name][metric_name] = [float(mean), float(conf_interval)]
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    pprint(f, line)
                    for i in range(len(mean)):
                        mean_dict[model_name][metric_name + '_top' + str(i+1)] = [float(mean[i]), float(conf_interval[i])]
        f.close()

    # handle plot values. Keep mean across repetitions
    plots_dict = {}
    for target, all_plots in targets_plots.items():
        plots_dict[target] = {}
        for metric_name, metric_dict in all_plots.items():
            plots_dict[target][metric_name] = {}
            for model_name, values in metric_dict.items():
                if model_name not in plots_dict[target][metric_name]:
                    plots_dict[target][metric_name][model_name] = {}
                plots_dict[target][metric_name][model_name] = np.mean(np.array(values), axis=0).tolist()
    
    return mean_dict, plots_dict

def get_log_filename(args, target, only_method=False, eval_file=None):
    assert target in ['motion', 'transition']

    folder = os.path.join(os.path.dirname(args.model_path), "evaluation")
    os.makedirs(folder, exist_ok=True)
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    log_file = os.path.join(folder, '{}_{}_{}'.format(target.capitalize(), name, niter))
    if args.guidance_param != 1.:
        log_file += f'_gscale{args.guidance_param}'
    log_file += method_suffix(args, eval_file)
    log_file += f'_{args.eval_mode}'
    log_file += f'_transLen{args.transition_length}' if not only_method else ''
    
    if args.extrapolation:
        log_file += '_extrapolation'
    if not only_method and args.scenario is not None and args.scenario not in ["", "test", "val"]:
        # it will be a targetted experiment --> in-distribution, out-distribution, etc.
        log_file += f'_{args.scenario}'
    return log_file + f'_s{args.seed}.log'

def get_log_filename_precomputed(args, target, only_method=False):
    # only_method --> only arguments related to the method, not the evaluation
    # this is to create the folder for precomputed motions (reused for different evaluations)
    assert target in ['motion', 'transition']

    folder = os.path.join(args.model_path, "evaluation")
    os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, '{}'.format(target))
    log_file += f'_transLen{args.transition_length}' if not only_method else ''
    if args.extrapolation:
        log_file += '_extrapolation'
    return log_file + f'_s{args.seed}.log'

def get_summary_filename(args, eval_file=None):
    # this is SPECIFIC TO AN EVALUATION
    folder = os.path.join(os.path.dirname(args.model_path), "evaluations_summary")
    os.makedirs(folder, exist_ok=True)
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    suffix = "_transLen{}".format(args.transition_length)
    suffix += method_suffix(args, eval_file)
    log_file = os.path.join(folder,
                            '{}_{}_{}{}'.format(niter, args.eval_mode, args.seed, suffix))
    if args.extrapolation:
        log_file += '_extrapolation'
    if args.scenario is not None and args.scenario not in ["", "test", "val"]:
        # it will be a targetted experiment --> in-distribution, out-distribution, etc.
        log_file += f'_{args.scenario}'
    return log_file + ".json"

def run(args, shuffle=True):
    
    fixseed(args.seed)
    drop_last = False
    if args.eval_batch_size != 32:
        raise ValueError("eval_batch_size must be 32 because R-precision computation assumes fixed batches of 32")
    args.batch_size = args.eval_batch_size
    args.generation_batch_size = max(1, int(args.generation_batch_size))

    print(f'Eval mode [{args.eval_mode}]')
    print(f'Eval metric batch size [{args.eval_batch_size}]')
    print(f'Eval generation batch size [{args.generation_batch_size}]')
    if args.eval_mode == 'debug':
        diversity_times = 30
        replication_times = 1  # about 3 Hrs
    elif args.eval_mode == 'fast':
        diversity_times = 300
        replication_times = 5
    elif args.eval_mode == 'final':
        diversity_times = 300
        replication_times = 10 # about 12 Hrs
    else:
        raise ValueError()
    
    if args.scenario == "short": # not enough samples in the other scenarios <300
        diversity_times = 175
    elif args.scenario == "medium":
        diversity_times = 275


    logger.configure()

    logger.log("creating data loader...")
    dist_util.setup_dist(args.device)
    print("Creating model and diffusion...")
    model, diffusion = load_model(args, dist_util.dev())
    diffusion = DiffusionWrapper(args, diffusion, model)

    split = 'test' if args.dataset != "babel" else "val" # no test set for babel
    if args.eval_file:
        eval_file = args.eval_file
    elif args.extrapolation:
        eval_file = EVAL_FILES[args.dataset]['extrapolation']
    elif args.scenario == "anaphora_fullgen" and args.dataset == "humanml":
        eval_file = EVAL_FILES[args.dataset]['anaphora_fullgen']
    elif args.scenario == "anaphora" and args.dataset == "humanml":
        eval_file = EVAL_FILES[args.dataset]['anaphora']
    else:
        eval_file = EVAL_FILES[args.dataset][split]
    if not os.path.exists(eval_file):
        raise FileNotFoundError(f"Evaluation file was not found: {eval_file}")
    args.resolved_eval_file = eval_file

    min_seq_len, max_seq_len = SEQ_LENS[args.dataset]
    args.target_segment_from_id = args.target_segment_from_id or is_anaphora_eval(args, eval_file)
    precomputed_folder = get_log_filename(args, "motion", only_method=True, eval_file=eval_file).replace(".log", "").replace("evaluation", "evaluation_precomputed")
    eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())

    print("Loading motion GT...")
    motion_gt_load_mode = 'gt'
    motion_gt_num_frames = (min_seq_len, max_seq_len)
    if args.dataset == "humanml" and is_anaphora_fullgen_eval(args, eval_file):
        motion_gt_load_mode = 'gt_anaphora_fullgen'
        motion_gt_num_frames = (100, 100)
    motion_gt_loader = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=motion_gt_num_frames,
        num_workers=0,
        split=split,
        load_mode=motion_gt_load_mode,
        shuffle=shuffle,
        drop_last=drop_last,
        cropping_sampler=False,
        eval_file=eval_file,
    )
    print(f"Done! Motion dataset size: {len(motion_gt_loader.dataset)}")
    targets = resolve_eval_targets(args, eval_file)
    gt_loaders = [motion_gt_loader]
    eval_reference_dataset = motion_gt_loader.dataset
    if 'transition' in targets:
        if is_anaphora_fullgen_eval(args, eval_file):
            raise ValueError("scenario=anaphora_fullgen must not construct a transition GT loader")
        print("Loading transitions GT...")
        transition_gt_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=(args.transition_length, args.transition_length), num_workers=0, split=split, load_mode='gt_transitions', shuffle=shuffle, drop_last=drop_last, cropping_sampler=True)
        print(f"Done! Transition dataset size: {len(transition_gt_loader.dataset)}")
        gt_loaders.append(transition_gt_loader)
        eval_reference_dataset = transition_gt_loader.dataset
    log_files = [get_log_filename(args, t, eval_file=eval_file) for t in targets]

    eval_motion_loader = lambda rep: get_mdm_loader(args,
            model, diffusion, args.batch_size, eval_file, eval_reference_dataset,
            max_motion_length=max_seq_len,
            precomputed_folder=os.path.join(precomputed_folder, f"{rep:02d}",),
            scenario=args.scenario, # to evaluate a subset of them
        )

    print("="*10, "Motion evaluation", "="*10)
    for mode, log_file in zip(targets, log_files):
        print(f'[{mode}] Will save to log file {log_file}')
    mean_dict, plots_dict = evaluation(eval_wrapper, gt_loaders, eval_motion_loader, targets, log_files, replication_times, diversity_times, extrapolation=args.extrapolation)
    
    # store to folder "evaluations_summary"
    log_file = get_summary_filename(args, eval_file=eval_file)
    plot_file = log_file.replace(".json", "_plots.json")
    print(f'Saving summary to {log_file}')
    # store mean_dict
    import json
    with open(log_file, 'w') as f:
        json.dump(mean_dict, f, default=str)
    # store plots_dict
    with open(plot_file, 'w') as f:
        json.dump(plots_dict, f, default=str)

    if 'transition' in targets:
        try:
            generate_plot_PJ(plot_file)
        except Exception as exc:
            print(f"Skipping transition PeakJerk plot generation: {exc}")

    return mean_dict

if __name__ == '__main__':
    args = evaluation_parser()
    run(args)
