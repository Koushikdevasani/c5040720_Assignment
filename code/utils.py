# src/utils.py
import os
import yaml
import numpy as np
from PIL import Image
import tensorflow as tf


def load_config(path="config.yaml"):
    """Load YAML config from path."""
    with open(path) as f:
        return yaml.safe_load(f)


def make_image_processor(image_size):
    """
    Return a function that loads/resizes an image and normalizes it to [-1, +1].
    Accepts either a file path (str) or a NumPy array / PIL Image.
    """
    def proc(img):
        if isinstance(img, str):
            im = Image.open(img).convert("RGB")
        else:
            im = Image.fromarray(img) if not isinstance(img, Image.Image) else img
            im = im.convert("RGB")
        im = im.resize((image_size, image_size))
        arr = np.array(im).astype(np.float32) / 255.0
        # Normalize to [-1, +1] which is suitable for the lightweight CNN encoder
        arr = (arr - 0.5) * 2.0
        return arr
    return proc


def prepare_dataset(cfg, split="train", keep_small=False):
    """
    Load the HuggingFace dataset specified in cfg['dataset']['hf_name'],
    tokenize captions with a BERT tokenizer, and process images + reasoning tags.

    Returns:
      processed: list of dicts with keys
        'images'     : (seq_len, H, W, C)
        'input_ids'  : (seq_len, T_cap)
        'reason_ids' : (seq_len, T_reason)
      tokenizer: the HF tokenizer (AutoTokenizer)
    """
    # Lazy imports to avoid heavy import at module load time
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset(cfg['dataset']['hf_name'], split=split)
    if keep_small:
        ds = ds.select(range(min(256, len(ds))))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    image_size = cfg['dataset']['image_size']
    seq_len = cfg['dataset']['seq_len']
    max_cap_len = cfg['dataset']['max_caption_len']
    max_reason_len = int(cfg['dataset'].get('max_reason_len', 32))
    image_proc = make_image_processor(image_size)

    # keys to probe for frames, captions, reasoning tags
    frame_keys = ["frames", "images", "frames_paths", "image_paths", "imgs"]
    caption_keys = ["captions", "descriptions", "texts", "story"]
    reason_keys = [
        "reasoning_tags",
        "reasoning",
        "logic_tags",
        "tags",
        "reason_labels",
        "annotations"
    ]

    def map_example(example):
        # Frames
        frames = None
        for k in frame_keys:
            if k in example:
                frames = example[k]
                break
        if frames is None:
            frames = []

        # Captions
        captions = None
        for k in caption_keys:
            if k in example:
                captions = example[k]
                break
        if captions is None:
            captions = []

        # Reasoning tags (per frame or per story)
        reasons = None
        for k in reason_keys:
            if k in example:
                reasons = example[k]
                break
        if reasons is None:
            # fallback: no explicit tags; use empty strings
            reasons = []

        # Ensure list-like
        if not isinstance(captions, (list, tuple)):
            captions = [captions] if captions is not None else []
        if not isinstance(reasons, (list, tuple)):
            reasons = [reasons] if reasons is not None else []

        # Truncate to seq_len
        frames = frames[:seq_len]
        captions = captions[:seq_len]
        reasons = reasons[:seq_len]

        # If reasons shorter than captions, pad with empty strings
        while len(reasons) < len(captions):
            reasons.append("")

        # Process images
        imgs = []
        for f in frames:
            try:
                imgs.append(image_proc(f))
            except Exception:
                # if loading fails, append a zero image
                imgs.append(np.zeros((image_size, image_size, 3), dtype=np.float32))
        # pad frames if fewer than seq_len
        while len(imgs) < seq_len:
            imgs.append(np.zeros((image_size, image_size, 3), dtype=np.float32))

        # Tokenize captions (story text)
        tok_caps = tokenizer(
            list(captions),
            padding='max_length',
            truncation=True,
            max_length=max_cap_len,
            return_tensors="np"
        )
        input_ids = tok_caps['input_ids']          # (n_caps, max_cap_len)

        # Tokenize reasoning tags as text strings
        # e.g. "cause: X, effect: Y, relation: in_front_of"
        # This keeps things flexible while still allowing sequence modeling.
        tok_reasons = tokenizer(
            list(reasons),
            padding='max_length',
            truncation=True,
            max_length=max_reason_len,
            return_tensors="np"
        )
        reason_ids = tok_reasons['input_ids']      # (n_reasons, max_reason_len)

        # Pad rows if fewer than seq_len
        if input_ids.shape[0] < seq_len:
            pad_rows = np.zeros((seq_len - input_ids.shape[0], max_cap_len), dtype=np.int32)
            input_ids = np.vstack([input_ids, pad_rows])

        if reason_ids.shape[0] < seq_len:
            pad_rows = np.zeros((seq_len - reason_ids.shape[0], max_reason_len), dtype=np.int32)
            reason_ids = np.vstack([reason_ids, pad_rows])

        return {
            "images": np.stack(imgs).astype(np.float32),      # (seq_len, H, W, C)
            "input_ids": input_ids.astype(np.int32),          # (seq_len, T_cap)
            "reason_ids": reason_ids.astype(np.int32)         # (seq_len, T_reason)
        }

    processed = []
    for ex in ds:
        processed.append(map_example(ex))
    return processed, tokenizer


def generator_from_processed(processed_list, cfg):
    """Yield tuples (images, input_ids, reason_ids) for tf.data.Dataset.from_generator."""
    def gen():
        for ex in processed_list:
            yield ex['images'], ex['input_ids'], ex['reason_ids']
    return gen


def make_tf_dataset(processed_list, cfg, shuffle=True):
    """
    Build a batched tf.data.Dataset from processed_list.

    Yields: (batch_images, batch_input_ids, batch_reason_ids) with shapes:
      batch_images     : (B, seq_len, H, W, C)
      batch_input_ids  : (B, seq_len, T_cap)
      batch_reason_ids : (B, seq_len, T_reason)
    """
    seq_len = cfg['dataset']['seq_len']
    max_cap_len = cfg['dataset']['max_caption_len']
    max_reason_len = int(cfg['dataset'].get('max_reason_len', 32))
    batch_size = cfg['dataset']['batch_size']
    image_size = cfg['dataset']['image_size']

    out_types = (tf.float32, tf.int32, tf.int32)
    out_shapes = (
        (seq_len, image_size, image_size, 3),
        (seq_len, max_cap_len),
        (seq_len, max_reason_len)
    )

    ds = tf.data.Dataset.from_generator(
        generator_from_processed(processed_list, cfg),
        output_types=out_types,
        output_shapes=out_shapes
    )
    if shuffle:
        ds = ds.shuffle(1024)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
