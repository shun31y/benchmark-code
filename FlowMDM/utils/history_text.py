import torch


def validate_history_current_weight(current_weight):
    if not 0.0 <= current_weight <= 1.0:
        raise ValueError(f"history_current_weight must be in [0, 1], got {current_weight}")


def validate_text_embeddings_shape(text_embeddings, expected_shape, context="text_embeddings"):
    if not torch.is_tensor(text_embeddings):
        raise ValueError(f"{context} must be a torch.Tensor, got {type(text_embeddings).__name__}")
    actual_shape = tuple(text_embeddings.shape)
    if actual_shape != tuple(expected_shape):
        raise ValueError(f"{context} shape mismatch: expected {tuple(expected_shape)}, got {actual_shape}")


def build_history_text_embeddings(model, y, current_weight):
    """
    Build CLIP text embeddings mixed with cumulative textual history.

    Single-text mode returns [B, 512].
    Multi-text mode returns [I, B, 512], matching FlowMDM conditions_mask.
    """
    validate_history_current_weight(current_weight)
    if "history_text" not in y:
        raise ValueError("history_text is required when --use_history_text is enabled")

    if "all_texts" in y:
        return _build_multi_text_embeddings(model, y, current_weight)
    return _build_single_text_embeddings(model, y, current_weight)


def _clip_dim(model):
    return int(getattr(model, "clip_dim", 512))


def _validate_text(value, context):
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string, got {type(value).__name__}")
    if value.strip() == "":
        raise ValueError(f"{context} must not be empty")


def _normalize_single_histories(history_text, batch_size):
    if not isinstance(history_text, (list, tuple)):
        raise ValueError("history_text must be a list")
    if batch_size == 1 and all(isinstance(x, str) for x in history_text):
        return [list(history_text)]
    if len(history_text) != batch_size:
        raise ValueError(f"history_text batch size mismatch: expected {batch_size}, got {len(history_text)}")
    return history_text


def _normalize_multi_histories(history_text, batch_size):
    if not isinstance(history_text, (list, tuple)):
        raise ValueError("history_text must be a list")
    if batch_size == 1 and len(history_text) == 1 and _looks_like_segment_histories(history_text[0]):
        return history_text
    if batch_size == 1 and _looks_like_segment_histories(history_text):
        return [list(history_text)]
    if len(history_text) != batch_size:
        raise ValueError(f"history_text batch size mismatch: expected {batch_size}, got {len(history_text)}")
    return history_text


def _looks_like_segment_histories(value):
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        return False
    return all(_looks_like_history(item) for item in value)


def _looks_like_history(value):
    return isinstance(value, (list, tuple)) and len(value) > 0 and all(isinstance(item, str) for item in value)


def _validate_history(history, current_text, context):
    _validate_text(current_text, f"{context}.text")
    if not isinstance(history, (list, tuple)):
        raise ValueError(f"{context}.history_text must be a list of strings")
    if len(history) == 0:
        raise ValueError(f"{context}.history_text must not be empty")
    for idx, text in enumerate(history):
        _validate_text(text, f"{context}.history_text[{idx}]")
    if history[-1] != current_text:
        raise ValueError(
            f"{context}.history_text[-1] must match current text: {history[-1]!r} != {current_text!r}"
        )


def _mix_history_embeddings(model, history, current_weight):
    encoded = model.encode_text(list(history))
    if encoded.ndim != 2 or encoded.shape[1] != _clip_dim(model):
        raise ValueError(
            f"encode_text returned invalid shape for history_text: expected [H, {_clip_dim(model)}], "
            f"got {tuple(encoded.shape)}"
        )
    if len(history) == 1:
        return encoded[-1]
    return current_weight * encoded[-1] + (1.0 - current_weight) * encoded[:-1].mean(dim=0)


def _build_single_text_embeddings(model, y, current_weight):
    texts = y.get("text")
    if not isinstance(texts, (list, tuple)):
        raise ValueError("y['text'] must be a list of strings in single-text history mode")
    histories = _normalize_single_histories(y["history_text"], len(texts))

    embeddings = []
    for batch_idx, (text, history) in enumerate(zip(texts, histories)):
        context = f"sample[{batch_idx}]"
        _validate_history(history, text, context)
        embeddings.append(_mix_history_embeddings(model, history, current_weight))

    out = torch.stack(embeddings, dim=0)
    expected_shape = (len(texts), _clip_dim(model))
    if "text_embeddings" in y:
        validate_text_embeddings_shape(y["text_embeddings"], expected_shape)
    validate_text_embeddings_shape(out, expected_shape, "history_text_embeddings")
    return out


def _build_multi_text_embeddings(model, y, current_weight):
    all_texts = y.get("all_texts")
    if not isinstance(all_texts, (list, tuple)) or len(all_texts) == 0:
        raise ValueError("y['all_texts'] must be a non-empty list in multi-text history mode")

    batch_size = len(all_texts)
    histories_by_sample = _normalize_multi_histories(y["history_text"], batch_size)
    max_segments = max(len(texts) for texts in all_texts)
    clip_dim = _clip_dim(model)
    device = next(model.parameters()).device
    out = torch.zeros((max_segments, batch_size, clip_dim), device=device, dtype=torch.float32)

    for batch_idx, (texts, histories) in enumerate(zip(all_texts, histories_by_sample)):
        if not isinstance(texts, (list, tuple)) or len(texts) == 0:
            raise ValueError(f"all_texts[{batch_idx}] must be a non-empty list of strings")
        if not isinstance(histories, (list, tuple)):
            raise ValueError(f"history_text[{batch_idx}] must be a list of per-segment histories")
        if len(histories) != len(texts):
            raise ValueError(
                f"history_text[{batch_idx}] segment count mismatch: expected {len(texts)}, got {len(histories)}"
            )
        for seg_idx, (text, history) in enumerate(zip(texts, histories)):
            context = f"sample[{batch_idx}].segment[{seg_idx}]"
            _validate_history(history, text, context)
            out[seg_idx, batch_idx] = _mix_history_embeddings(model, history, current_weight)

    expected_shape = (max_segments, batch_size, clip_dim)
    if "text_embeddings" in y:
        validate_text_embeddings_shape(y["text_embeddings"], expected_shape)
    validate_text_embeddings_shape(out, expected_shape, "history_text_embeddings")
    return out
