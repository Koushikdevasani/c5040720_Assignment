# src/model.py
import tensorflow as tf
from tensorflow.keras import layers, Model


def build_visual_encoder(image_size=224, feat_dim=512, backbone='simple_cnn'):
    """
    Lightweight CNN visual encoder that replaces ResNet50.
    Produces a fixed feature vector of dimension feat_dim.
    Designed to encode each frame so the temporal module can reason over sequences.
    """
    inp = tf.keras.Input(shape=(image_size, image_size, 3), name='image_input')
    x = inp

    # --- Stem ---
    x = layers.Conv2D(32, 3, strides=1, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(2)(x)

    # --- Block 1 ---
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(2)(x)

    # --- Block 2 ---
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(2)(x)

    # --- Block 3 ---
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # --- Projection layer ---
    x = layers.Dense(feat_dim, activation='relu', name='proj_dense')(x)

    return Model(inputs=inp, outputs=x, name='visual_encoder_cnn')


def build_text_encoder(vocab_size, embed_dim=300, hidden_dim=512, max_len=64, name_prefix="text"):
    """
    Encodes a caption into a fixed-size hidden vector (LSTM final hidden state).
    Used for both narrative text and (optionally) reasoning tags when treated as text.
    mask_zero=False to avoid mask propagation issues with TimeDistributed.
    """
    inp = tf.keras.Input(shape=(max_len,), dtype='int32', name=f'{name_prefix}_input')
    x = layers.Embedding(vocab_size, embed_dim, mask_zero=False, name=f'{name_prefix}_embed')(inp)
    _, state_h, _ = layers.LSTM(hidden_dim, return_state=True, name=f'{name_prefix}_lstm')(x)
    return Model(inputs=inp, outputs=state_h, name=f'{name_prefix}_encoder')


def build_reasoning_encoder(vocab_size,
                            embed_dim=256,
                            hidden_dim=512,
                            max_len=32):
    """
    Encodes a sequence of reasoning tags for a single timestep.

    In the proposal, reasoning tags capture:
      - causal labels,
      - object relations,
      - scene transitions, etc.

    Here we model them as a token sequence (e.g. comma-separated tags tokenized
    with the same tokenizer as captions) and encode via an LSTM.
    """
    return build_text_encoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        max_len=max_len,
        name_prefix="reason"
    )


def build_multimodal_model(cfg):
    """
    Build the multimodal temporal model that:
      - Encodes image sequences, narrative text, and reasoning tags per timestep.
      - Aggregates over time with an LSTM to capture story-level context.
      - Outputs:
          (1) text logits for next-caption decoding,
          (2) an image feature prediction that can be used to condition
              a diffusion-based image decoder for the next frame.

    Inputs:
      images_seq   : (B, S, H, W, C)
      captions_seq : (B, S, T_cap)
      reason_seq   : (B, S, T_reason)
      dec_input    : (B, T_cap)

    Outputs:
      logits  : (B, T_cap, vocab_size)
      img_pred: (B, image_feat_dim)
    """
    seq_len = cfg['dataset']['seq_len']
    image_size = cfg['dataset']['image_size']
    max_cap_len = cfg['dataset']['max_caption_len']
    max_reason_len = int(cfg['dataset'].get('max_reason_len', 32))

    # --- Build encoders ---
    visual_enc = build_visual_encoder(
        image_size=image_size,
        feat_dim=cfg['model']['image_feat_dim']
    )

    # Narrative text encoder
    text_enc = build_text_encoder(
        vocab_size=cfg['model']['vocab_size'],
        embed_dim=cfg['model']['text_embed_dim'],
        hidden_dim=cfg['model']['text_hidden_dim'],
        max_len=max_cap_len,
        name_prefix="caption"
    )

    # Reasoning-tag encoder
    reason_hidden_dim = int(cfg['model'].get('reason_hidden_dim',
                                             cfg['model']['text_hidden_dim']))
    reason_enc = build_reasoning_encoder(
        vocab_size=cfg['model']['vocab_size'],   # share vocab with captions
        embed_dim=int(cfg['model'].get('reason_embed_dim',
                                       cfg['model']['text_embed_dim'])),
        hidden_dim=reason_hidden_dim,
        max_len=max_reason_len
    )

    # --- Inputs ---
    images_in = tf.keras.Input(
        shape=(seq_len, image_size, image_size, 3),
        name='images_seq'
    )
    captions_in = tf.keras.Input(
        shape=(seq_len, max_cap_len),
        dtype='int32',
        name='captions_seq'
    )
    reasons_in = tf.keras.Input(
        shape=(seq_len, max_reason_len),
        dtype='int32',
        name='reasons_seq'
    )

    # --- Encode each timestep ---
    td_visual = layers.TimeDistributed(visual_enc, name='td_visual')(images_in)
    td_text = layers.TimeDistributed(text_enc, name='td_text')(captions_in)
    td_reason = layers.TimeDistributed(reason_enc, name='td_reason')(reasons_in)

    # --- Fuse modalities per frame ---
    fused = layers.Concatenate(axis=-1, name='fuse_vis_text_reason')(
        [td_visual, td_text, td_reason]
    )

    fused = layers.TimeDistributed(
        layers.Dense(cfg['model']['multimodal_dim'], activation='relu'),
        name='td_proj'
    )(fused)

    # --- Temporal LSTM over story sequence ---
    temporal_out, state_h, state_c = layers.LSTM(
        cfg['model']['temporal_hidden_dim'],
        return_sequences=True,
        return_state=True,
        name='temporal_lstm'
    )(fused)

    # Last timestep context represents full story-level reasoning state
    context = temporal_out[:, -1, :]  # (B, temporal_hidden_dim)

    # ---------------------------------------------------------------------
    # Text Decoder: predict next caption conditioned on story context
    # ---------------------------------------------------------------------
    dec_input = tf.keras.Input(
        shape=(max_cap_len,),
        dtype='int32',
        name='dec_input'
    )
    dec_emb = layers.Embedding(
        cfg['model']['vocab_size'],
        cfg['model']['text_embed_dim'],
        mask_zero=False,
        name='dec_embed'
    )(dec_input)

    # Tile context across dec sequence length and concatenate
    ctx_tile = layers.RepeatVector(max_cap_len, name='dec_ctx_repeat')(context)
    dec_lstm_in = layers.Concatenate(name='dec_concat')([dec_emb, ctx_tile])

    dec_lstm_out = layers.LSTM(
        cfg['model']['text_decoder_hidden'],
        return_sequences=True,
        name='dec_lstm'
    )(dec_lstm_in)

    logits = layers.TimeDistributed(
        layers.Dense(cfg['model']['vocab_size']),
        name='dec_logits'
    )(dec_lstm_out)

    text_decoder_model = tf.keras.Model(
        inputs=[images_in, captions_in, reasons_in, dec_input],
        outputs=logits,
        name='text_decoder'
    )

    # ---------------------------------------------------------------------
    # Image feature predictor:
    #   produces a representation aligned with the visual encoder space
    #   that can later be used as conditioning for a diffusion image decoder.
    # ---------------------------------------------------------------------
    img_pred = layers.Dense(
        cfg['model']['temporal_hidden_dim'],
        activation='relu',
        name='img_pred_hidden'
    )(context)
    img_pred = layers.Dense(
        cfg['model']['image_feat_dim'],
        name='img_pred'
    )(img_pred)

    # Final multimodal model
    full_model = tf.keras.Model(
        inputs=[images_in, captions_in, reasons_in, dec_input],
        outputs=[logits, img_pred],
        name='multimodal_model'
    )

    return {
        "full_model": full_model,
        "visual_enc": visual_enc,
        "text_enc": text_enc,
        "reason_enc": reason_enc,
        "text_decoder": text_decoder_model
    }
