import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import dist_util
import  numpy as np
from data_loaders.tensors import lengths_to_mask
import os
import json
import re
from data_loaders.amass.babel import get_tokens
from data_loaders.humanml.utils.get_opt import get_opt
from data_loaders.humanml.data.dataset import process_tokens
from os.path import join as pjoin

ANAPHORA_SEGMENT_RE = re.compile(r"__ana_seg(\d+)_")
ANAPHORA_FULLGEN_SCENARIO = "anaphora_fullgen"
ANAPHORA_FULLGEN_SEGMENT_LENGTH = 100
ANAPHORA_FULLGEN_NUM_SEQS = 6


def resolve_anaphora_target_segment_idx(sample_id, *, strict=False):
    match = ANAPHORA_SEGMENT_RE.search(str(sample_id))
    if match is None:
        if strict:
            raise ValueError(f"Could not resolve target segment from sample id: {sample_id}")
        return None
    target_idx = int(match.group(1)) - 1
    if target_idx < 0:
        raise ValueError(f"Invalid anaphora target segment in sample id: {sample_id}")
    return target_idx


def is_anaphora_fullgen_eval(args, eval_file=None):
    if getattr(args, "scenario", "") == ANAPHORA_FULLGEN_SCENARIO:
        return True
    if eval_file is None:
        eval_file = getattr(args, "resolved_eval_file", "") or getattr(args, "eval_file", "")
    return ANAPHORA_FULLGEN_SCENARIO in os.path.basename(str(eval_file))


def is_anaphora_eval(args, eval_file=None):
    if getattr(args, "target_segment_from_id", False):
        return True
    if getattr(args, "scenario", "") in {"anaphora", ANAPHORA_FULLGEN_SCENARIO}:
        return True
    if eval_file is None:
        eval_file = getattr(args, "resolved_eval_file", "") or getattr(args, "eval_file", "")
    return "anaphora" in os.path.basename(str(eval_file))


def validate_anaphora_fullgen_kwargs(sample_id, lengths):
    if len(lengths) != ANAPHORA_FULLGEN_NUM_SEQS:
        raise ValueError(
            f"Anaphora fullgen sample {sample_id} must contain {ANAPHORA_FULLGEN_NUM_SEQS} segments, "
            f"got {len(lengths)}"
        )
    if any(length != ANAPHORA_FULLGEN_SEGMENT_LENGTH for length in lengths):
        raise ValueError(
            f"Anaphora fullgen sample {sample_id} must have lengths="
            f"{[ANAPHORA_FULLGEN_SEGMENT_LENGTH] * ANAPHORA_FULLGEN_NUM_SEQS}, got {lengths}"
        )


def pad_sample_with_zeros(sample, max_len=250):
    # pad inp, change lenghts, and pad is transition
    seq_len, n_feats = sample.shape
    len_to_pad = max_len - seq_len
    np.zeros_like(sample)
    sample_padding = np.zeros((len_to_pad, n_feats))
    sample = np.concatenate((sample, sample_padding))
    return sample


def _lengths_to_list(lengths):
    if torch.is_tensor(lengths):
        return [int(v) for v in lengths.tolist()]
    return [int(v) for v in lengths]


def build_batched_model_kwargs(samples):
    has_history_text = any("history_text" in sample["y"] for sample in samples)
    batched = {"y": {"all_texts": [], "all_lengths": [], "text": []}}
    if has_history_text:
        batched["y"]["history_text"] = []

    for sample in samples:
        texts = list(sample["y"]["text"])
        lengths = _lengths_to_list(sample["y"]["lengths"])
        batched["y"]["all_texts"].append(texts)
        batched["y"]["all_lengths"].append(lengths)
        batched["y"]["text"].append(" -- ".join(texts))
        if has_history_text:
            if "history_text" not in sample["y"]:
                sample_id = sample["y"].get("id", "<unknown>")
                raise ValueError(f"history_text is missing for sample {sample_id} while batching evaluation generation")
            batched["y"]["history_text"].append(sample["y"]["history_text"])

    return batched


class CompMDMGeneratedDataset(Dataset):

    def load_model_kwargs_dataset(self, eval_file, scenario=""):
        import json
        with open(eval_file, 'r') as f:
            all_model_kwargs = json.load(f)
            print(f"loaded {eval_file}", len(all_model_kwargs))

            # convert all "lengths" to torch
            final_model_kwargs = []
            for i, raw_kwargs in enumerate(all_model_kwargs):
                sample_id = raw_kwargs.get('id', i)
                idx = int(sample_id) if str(sample_id).isdigit() else i
                kwargs = {"y": raw_kwargs}
                if scenario != "" and scenario is not None and "scenario" in kwargs['y'] and kwargs['y']['scenario'] != scenario:
                    continue # skip this one

                kwargs['y']['lengths'] = torch.tensor([int(v) for v in kwargs['y']['lengths']])
                if len(kwargs['y']['text']) != len(kwargs['y']['lengths']):
                    raise ValueError(
                        f"text/length count mismatch for sample {sample_id}: "
                        f"{len(kwargs['y']['text'])} != {len(kwargs['y']['lengths'])}"
                    )
                if kwargs['y']['lengths'].max().item() <= 0:
                    raise ValueError(f"Sample {sample_id} has no positive-length segments")
                kwargs['y']['mask'] = lengths_to_mask(kwargs['y']['lengths'], kwargs['y']['lengths'].max()).unsqueeze(1).unsqueeze(1)
                final_model_kwargs.append((idx, kwargs))
            
            assert len(final_model_kwargs) > 0, f"No model kwargs found for this scenario: {scenario}"
            print(f"loaded {len(final_model_kwargs)} model kwargs for scenario {scenario if scenario != '' else '> all <'}")

        return final_model_kwargs
    
    def process_tokens(self, tokens):
        return process_tokens(tokens, self.opt.max_text_len, self.w_vectorizer)

    def __init__(self, args, model, diffusion, mm_num_samples, mm_num_repeats, eval_file):

        dataloader = self.load_model_kwargs_dataset(eval_file)
        #assert mm_num_samples < len(dataset)
        use_ddim = False  # FIXME - hardcoded
        clip_denoised = False  # FIXME - hardcoded
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )

        generated_motion = []
        mm_generated_motions = []
        num_seqs = 32 # ALWAYS
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(len(dataloader), mm_num_samples // num_seqs +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        model.eval()


        with torch.no_grad():
            for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)):

                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):

                    sample = sample_fn(
                        motion.shape,
                        clip_denoised=clip_denoised,
                        model_kwargs=model_kwargs,
                        #skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                        progress=False,
                    )

                    if t == 0:
                        sub_dicts = [{'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
                                    'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    } for bs_i in range(num_seqs)]
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [{'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                                        'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                        } for bs_i in range(num_seqs)]

                if is_mm:
                    mm_generated_motions += [{
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'mm_motions': mm_motions[bs_i::num_seqs],  # collect all 10 repeats from the (32*10) generated motions
                                    } for bs_i in range(num_seqs)]

        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption = data['motion'], data['length'], data['caption']
        
        if self.dataset_name != "babel": # babel already takes care of its de)normalization itself
            normed_motion = motion
            denormed_motion = normed_motion * self.std + self.mean
            renormed_motion = (denormed_motion - self.mean_for_eval) / self.std_for_eval  # according to T2M norms
            motion = renormed_motion
            # This step is needed because T2M evaluators expect their norm convention
        
        tokens = get_tokens(caption)
        word_embeddings, pos_one_hots, sent_len, tokens = self.process_tokens(tokens)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, tokens

    def inv_transform(self, data):
        return data * self.std_for_eval + self.mean_for_eval
    
    def switch_target(self, target):
        assert target in ['motion', ], "Only motion eval target is available for non-unfolding dataset"

class CompMDMUnfoldingGeneratedDataset(CompMDMGeneratedDataset):

    def __init__(self, args, model, diffusion, max_motion_length, eval_file, w_vectorizer=None, opt=None, precomputed_folder=None, scenario="", generation_batch_size=1):
        self.dataset_name = args.dataset
        self.w_vectorizer = w_vectorizer
        self.opt = opt
        assert self.dataset_name == "babel" or self.dataset_name == "humanml", "Only babel and humanml are supported"
        if self.dataset_name == "humanml":
            self.mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
            self.std = np.load(pjoin(opt.data_root, 'Std.npy'))
            self.mean_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))

        dataloader = self.load_model_kwargs_dataset(eval_file, scenario=scenario)
        self.max_motion_length = max_motion_length
        generation_batch_size = max(1, int(generation_batch_size))

        # Will be changed later by the evaluation script for each copy of this dataset
        self.step_to_eval = 1
        self.transition = False

        generated_transitions = []
        generated_motion = []

        os.makedirs(precomputed_folder, exist_ok=True)
        with torch.no_grad():
            for chunk_start in tqdm(range(0, len(dataloader), generation_batch_size)):
                chunk = dataloader[chunk_start:chunk_start + generation_batch_size]
                chunk_results = [None] * len(chunk)
                pending = []

                for local_idx, (i, model_kwargs) in enumerate(chunk):
                    sample_id = model_kwargs['y'].get('id', i)
                    lengths = _lengths_to_list(model_kwargs['y']['lengths'])
                    fullgen_eval = is_anaphora_fullgen_eval(args, eval_file)
                    if fullgen_eval:
                        validate_anaphora_fullgen_kwargs(sample_id, lengths)

                    precomputed_file_kwargs = os.path.join(precomputed_folder, f'{i:02d}_kwargs.json')
                    precomputed_file_pt = os.path.join(precomputed_folder, f'{i:02d}.pt')
                    precomputed_file_npy = os.path.join(precomputed_folder, f'{i:02d}.npy')

                    if os.path.exists(precomputed_file_kwargs):
                        loaded_kwargs = json.load(open(precomputed_file_kwargs, 'r'))
                        kwargs = loaded_kwargs if "y" in loaded_kwargs else {"y": loaded_kwargs}
                        assert kwargs['y']['lengths'] == model_kwargs['y']['lengths'].tolist()
                        assert kwargs['y']['text'] == model_kwargs['y']['text']
                        assert kwargs['y'].get('id', sample_id) == sample_id
                        if fullgen_eval:
                            assert kwargs['y'].get('scenario') == model_kwargs['y']['scenario']
                        if "history_text" in model_kwargs['y']:
                            assert kwargs['y'].get('history_text') == model_kwargs['y']['history_text']

                        if os.path.exists(precomputed_file_pt):
                            unfolded = torch.load(precomputed_file_pt, map_location='cpu')
                        elif os.path.exists(precomputed_file_npy):
                            unfolded = np.load(precomputed_file_npy, allow_pickle=True).item()
                            rots = torch.tensor(unfolded["rots"])
                            transl = torch.tensor(unfolded["transl"])

                            from utils.rotation_conversions import matrix_to_axis_angle
                            rots = matrix_to_axis_angle(rots)

                            from data_loaders.amass.tools.smpl import smpl_data_to_matrix_and_trans
                            smpl_data = {
                                "poses": rots,
                                "trans": transl,
                            }
                            smpl_data = smpl_data_to_matrix_and_trans(smpl_data, nohands=True)
                            from data_loaders.amass.transforms import SlimSMPLTransform
                            transform = SlimSMPLTransform(batch_size=32, name='SlimSMPLTransform', ename='smplnh', normalization=True)
                            features = transform.rots2rfeats(smpl_data)
                            unfolded = features.permute(1, 0).unsqueeze(1).unsqueeze(0).cpu()
                            torch.save(unfolded, precomputed_file_pt)
                        else:
                            raise AssertionError("Precomputed file not found")

                        chunk_results[local_idx] = (i, model_kwargs, sample_id, unfolded, fullgen_eval)
                    else:
                        pending.append((local_idx, i, model_kwargs, sample_id, precomputed_file_kwargs, precomputed_file_pt, fullgen_eval))

                if pending:
                    assert model is not None and diffusion is not None, "Model and diffusion must be provided for evaluation if precomputed files are not available"
                    model.eval()
                    batched_model_kwargs = build_batched_model_kwargs([item[2] for item in pending])
                    batched_unfolded = diffusion.p_sample_loop(
                        clip_denoised=False,
                        model_kwargs=batched_model_kwargs,
                        progress=False,
                    )
                    for batch_idx, (local_idx, i, model_kwargs, sample_id, precomputed_file_kwargs, precomputed_file_pt, fullgen_eval) in enumerate(pending):
                        total_length = sum(_lengths_to_list(model_kwargs['y']['lengths']))
                        unfolded = batched_unfolded[batch_idx:batch_idx+1, ..., :total_length].detach().cpu()
                        torch.save(unfolded, precomputed_file_pt)
                        kwargs_to_save = {
                            'id': sample_id,
                            'scenario': model_kwargs['y'].get('scenario'),
                            'lengths': model_kwargs['y']['lengths'].tolist(),
                            'text': model_kwargs['y']['text'],
                        }
                        if "history_text" in model_kwargs['y']:
                            kwargs_to_save['history_text'] = model_kwargs['y']['history_text']
                        json.dump(kwargs_to_save, open(precomputed_file_kwargs, 'w'))
                        chunk_results[local_idx] = (i, model_kwargs, sample_id, unfolded, fullgen_eval)

                for result in chunk_results:
                    if result is None:
                        raise RuntimeError("Missing generated or precomputed motion for an evaluation sample")
                    i, model_kwargs, sample_id, unfolded, fullgen_eval = result

                    if fullgen_eval and unfolded.shape[-1] != ANAPHORA_FULLGEN_SEGMENT_LENGTH * ANAPHORA_FULLGEN_NUM_SEQS:
                        raise ValueError(
                            f"Anaphora fullgen sample {sample_id} must generate "
                            f"{ANAPHORA_FULLGEN_SEGMENT_LENGTH * ANAPHORA_FULLGEN_NUM_SEQS} frames, "
                            f"got {unfolded.shape[-1]}"
                        )

                    start = 0
                    num_seqs = len(model_kwargs['y']['lengths'])
                    strict_target = is_anaphora_eval(args, eval_file)
                    target_segment_idx = resolve_anaphora_target_segment_idx(sample_id, strict=strict_target)
                    if target_segment_idx is not None and target_segment_idx >= num_seqs:
                        raise ValueError(
                            f"Target segment index {target_segment_idx} out of range for sample {sample_id} with {num_seqs} segments"
                        )
                    for bs_i in range(num_seqs):
                        segment_length = int(model_kwargs['y']['lengths'][bs_i].item())
                        end = start + segment_length
                        if target_segment_idx is not None and bs_i != target_segment_idx:
                            start = end
                            continue
                        if segment_length <= 0:
                            raise ValueError(f"Target segment {bs_i} for sample {sample_id} has zero length")
                        motion_slice = unfolded[..., start:end].squeeze().permute(1, 0).cpu().numpy()
                        assert motion_slice.shape[0] == segment_length, f'{motion_slice.shape[0]} != {segment_length}'

                        generated_motion.append({
                            'motion': pad_sample_with_zeros(motion_slice, self.max_motion_length),
                            'length': segment_length,
                            'caption': model_kwargs['y']['text'][bs_i],
                        })
                        start = end

                    if target_segment_idx is None:
                        l_margin = (args.transition_length // 2)
                        r_margin = args.transition_length - l_margin
                        mid = 0
                        for bs_i in range(num_seqs - 1):
                            mid += int(model_kwargs['y']['lengths'][bs_i].item())
                            motion_slice = unfolded[..., mid - l_margin : mid + r_margin].squeeze().permute(1, 0).cpu().numpy()
                            assert motion_slice.shape[0] == args.transition_length

                            generated_transitions.append({
                                'motion': motion_slice,
                                'length': args.transition_length,
                                'caption': model_kwargs['y']['text'][bs_i],
                            })

        self.generated_inbetween = generated_motion
        self.generated_transitions = generated_transitions
        
        self.switch_target('motion') # 'motion' or 'transition'

    def switch_target(self, target):
        """
        Switches between 'motion' and 'transition' targets. In 'motion' target, the dataset returns the full motion segment. 
        In 'transition' target, the dataset returns only the transition part of the motion
        """
        assert target in ['motion', 'transition']
        if target == 'transition' and len(self.generated_transitions) == 0:
            raise ValueError("transition target is not available for target-segment-only evaluation")
        self.target = target
        self.generated_motion = self.generated_inbetween if self.target == 'motion' else self.generated_transitions
