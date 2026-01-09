# src/train.py
import os
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from src.utils import load_config, prepare_dataset, make_tf_dataset, ensure_dir
from src.model import build_multimodal_model


def collate_for_training(batch_images, batch_input_ids, bos_id=101):
    """
    Prepare dec_input and dec_target for teacher-forcing training.

    Args:
      batch_images   : (B, seq_len, H, W, C)
      batch_input_ids: (B, seq_len, T_cap)  - caption token ids

    Returns:
      images      : (B, seq_len, H, W, C)
      captions_seq: (B, seq_len, T_cap)
      dec_input   : (B, T_cap)
      dec_target  : (B, T_cap)
    """
    target = batch_input_ids[:, -1, :]   # last caption in the sequence (B, T)
    dec_input = np.concatenate(
        [np.full((target.shape[0], 1), bos_id, dtype=np.int32), target[:, :-1]],
        axis=1
    )  # (B, T)
    return batch_images, batch_input_ids, dec_input, target


def train(cfg, keep_small=False):
    # 1) Load dataset
    print("Loading dataset (this may take a while)...")
    processed, tokenizer = prepare_dataset(cfg, split="train", keep_small=keep_small)
    print("Total processed examples:", len(processed))
    ds = make_tf_dataset(processed, cfg, shuffle=True)

    # 2) Build models
    models = build_multimodal_model(cfg)
    # full_model expects [images_seq, captions_seq, reasons_seq, dec_input]
    model = models['full_model']
    visual_enc = models['visual_enc']

    # 3) Optimizer & hyperparams
    lr = float(cfg['training'].get('lr', 1e-4))
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # 4) Prepare saving
    ensure_dir(cfg['training']['save_dir'])
    ckpt_prefix = os.path.join(cfg['training']['save_dir'], "ckpt")

    # 5) Training step (no @tf.function for easier debugging)
    def train_step(images_np, captions_seq_np, reasons_seq_np, dec_input_np, dec_target_np):
        """
        Single training step.

        images_np      : (B, S, H, W, C)
        captions_seq_np: (B, S, T_cap)
        reasons_seq_np : (B, S, T_reason)
        dec_input_np   : (B, T_cap)
        dec_target_np  : (B, T_cap)
        """
        # Convert to TF tensors with expected dtype
        images = tf.convert_to_tensor(images_np, dtype=tf.float32)
        captions_seq = tf.convert_to_tensor(captions_seq_np, dtype=tf.int32)
        reasons_seq = tf.convert_to_tensor(reasons_seq_np, dtype=tf.int32)
        dec_input = tf.convert_to_tensor(dec_input_np, dtype=tf.int32)
        dec_target = tf.convert_to_tensor(dec_target_np, dtype=tf.int32)

        # Defensive: ensure dec_input/dec_target are rank 2 (B, T)
        if tf.rank(dec_input) == 3 and tf.shape(dec_input)[-1] == 1:
            dec_input = tf.squeeze(dec_input, axis=-1)
        if tf.rank(dec_target) == 3 and tf.shape(dec_target)[-1] == 1:
            dec_target = tf.squeeze(dec_target, axis=-1)

        with tf.GradientTape() as tape:
            # model outputs (logits, img_pred)
            logits, img_pred = model(
                [images, captions_seq, reasons_seq, dec_input],
                training=True
            )

            # TEXT LOSS: per-token sparse categorical crossentropy
            per_token_loss = tf.keras.losses.sparse_categorical_crossentropy(
                dec_target, logits, from_logits=True
            )  # (B, T)
            pad_id = int(cfg['model'].get('pad_token_id', 0))
            mask = tf.cast(tf.not_equal(dec_target, pad_id), tf.float32)  # (B, T)
            per_token_loss = per_token_loss * mask
            loss_text = tf.reduce_sum(per_token_loss) / (tf.reduce_sum(mask) + 1e-8)

            # IMAGE LOSS: encode true last image and compute MSE
            # This aligns img_pred with the visual encoder space, which
            # can later be used as a conditioning for a diffusion decoder.
            last_images = images[:, -1, :, :, :]  # (B, H, W, C)
            target_img_feat = tf.stop_gradient(
                visual_enc(last_images, training=False)
            )  # (B, feat_dim)
            loss_img = tf.reduce_mean(tf.square(target_img_feat - img_pred))

            # Joint loss: text + image feature consistency
            loss = loss_text + 0.5 * loss_img

        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(
            grads, float(cfg['training'].get('grad_clip', 1.0))
        )
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return float(loss.numpy()), float(loss_text.numpy()), float(loss_img.numpy())

    # 6) Training loop
    epochs = int(cfg['training'].get('epochs', 1))
    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        prog = tqdm(ds)
        for batch in prog:
            # Dataset now yields (images, input_ids, reason_ids)
            images_batch, input_ids_batch, reason_ids_batch = batch
            images_np = images_batch.numpy()
            input_ids_np = input_ids_batch.numpy()
            reasons_np = reason_ids_batch.numpy()

            imgs, caps_seq, dec_input, dec_target = collate_for_training(
                images_np,
                input_ids_np,
                bos_id=int(cfg['model'].get('bos_token_id', 101))
            )

            loss, loss_text, loss_img = train_step(
                imgs, caps_seq, reasons_np, dec_input, dec_target
            )

            if step % int(cfg['training'].get('log_interval', 50)) == 0:
                prog.set_description(
                    f"step {step} loss={loss:.4f} text={loss_text:.4f} img={loss_img:.4f}"
                )
            step += 1

        # save weights each epoch (Keras-compatible .weights.h5)
        model.save_weights(f"{ckpt_prefix}_epoch{epoch+1}.weights.h5")
        print(f"Saved checkpoint: {ckpt_prefix}_epoch{epoch+1}.weights.h5")

    print("Training complete.")
    return model, models, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use small subset for fast debugging"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, keep_small=args.small)
