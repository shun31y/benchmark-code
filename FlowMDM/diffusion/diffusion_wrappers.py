
import torch
import torch as th
from copy import deepcopy
from utils.history_text import build_history_text_embeddings


class DiffusionWrapper_FlowMDM():
    def __init__(self, args, diffusion, model):
        self.model = model
        self.diffusion = diffusion
        self.guidance_param = args.guidance_param
        self.use_history_text = getattr(args, "use_history_text", False)
        self.history_current_weight = getattr(args, "history_current_weight", 1.0)

    @staticmethod
    def _normalize_segment_lengths(lengths):
        if torch.is_tensor(lengths):
            return [int(v) for v in lengths.tolist()]
        return [int(v) for v in lengths]

    def _prepare_eval_batch(self, model_kwargs, device):
        model_kwargs = deepcopy(model_kwargs)
        y = model_kwargs["y"]

        if "all_lengths" in y and "all_texts" in y:
            all_lengths = [self._normalize_segment_lengths(lengths) for lengths in y["all_lengths"]]
            all_texts = [list(texts) for texts in y["all_texts"]]
            joined_text = list(y.get("text", [" -- ".join(texts) for texts in all_texts]))
        else:
            segment_lengths = self._normalize_segment_lengths(y["lengths"])
            all_lengths = [segment_lengths]
            all_texts = [list(y["text"])]
            joined_text = [" -- ".join(y["text"])]

        bs = len(all_lengths)
        total_lengths = [sum(lengths) for lengths in all_lengths]
        nframes = max(total_lengths)
        max_segments = max(len(lengths) for lengths in all_lengths)

        mask = th.zeros((bs, nframes), device=device, dtype=th.bool)
        conditions_mask = th.zeros((max_segments, nframes, bs), device=device, dtype=th.bool)
        pos_pe_abs = th.zeros((bs, nframes), device=device, dtype=th.float32)
        pe_bias = th.full((bs, nframes, nframes), float('-inf'), device=device, dtype=th.float32)

        for batch_idx, seg_lengths in enumerate(all_lengths):
            total_len = total_lengths[batch_idx]
            mask[batch_idx, :total_len] = True
            start = 0
            for seg_idx, length in enumerate(seg_lengths):
                pos_pe_abs[batch_idx, start:start+length] = torch.arange(length, device=device, dtype=torch.float32)
                pe_bias[batch_idx, start:start+length, start:start+length] = 0
                conditions_mask[seg_idx, start:start+length, batch_idx] = True
                start += length

        y["all_lengths"] = all_lengths
        y["all_texts"] = all_texts
        y["text"] = joined_text
        y["mask"] = mask
        y["lengths"] = th.tensor(total_lengths, device=device, dtype=th.int64)
        y["conditions_mask"] = conditions_mask
        y["pos_pe_abs"] = pos_pe_abs
        y["pe_bias"] = pe_bias
        y["scale"] = th.ones(bs, device=device) * self.guidance_param
        if "text_embeddings" in y and torch.is_tensor(y["text_embeddings"]):
            y["text_embeddings"] = y["text_embeddings"].to(device)
        if self.use_history_text:
            y["text_embeddings"] = build_history_text_embeddings(
                self.model,
                y,
                self.history_current_weight,
            )
        return model_kwargs, bs, nframes

    def p_sample_loop(
        self,
        model_kwargs=None, # list of dicts
        **kwargs,
    ):
        final = None
        for i, sample in enumerate(self.p_sample_loop_progressive(
            model_kwargs=model_kwargs,
            **kwargs,
        )):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        noise=None,
        model_kwargs=None, # list of dicts
        device=None,
        progress=False,
        **kwargs,
    ):
        if device is None:
            device = next(self.model.parameters()).device
        model_kwargs, bs, nframes = self._prepare_eval_batch(model_kwargs, device)
        shape = (bs, self.model.njoints, self.model.nfeats, nframes)
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        indices = list(range(self.diffusion.num_timesteps))[::-1]
        
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for t in indices:
                
            with th.no_grad():

                t = th.tensor([t] * shape[0], device=device)
                out = self.diffusion.p_sample(
                    self.model,
                    img,
                    t,
                    model_kwargs=model_kwargs,
                    **kwargs,
                )

                yield out
                img = out["sample"]
