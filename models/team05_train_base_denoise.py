"""
   Base Image Denoising (noise level=50)
  - attention U-Net architecture (3 encoder levels, 64-128-256-512)
  - 96×96 patch size (CPU-feasible; larger patches impractical on CPU)
  - image-disjoint train/val split
  - deterministic validation noise
  - 5 fixed val crops, 8 train patches per image
  - sigmoid output
  
"""

import os
import warnings
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
)

# ── Create log directory ──
Path("logs_v7").mkdir(parents=True, exist_ok=True)

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler("logs_v7/train.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ============== BASIC SETUP ==============
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
tf.keras.utils.set_random_seed(42)

CONFIG = {
    "patch_size": 96,
    "noise_std": 50 / 255.0,
    "batch_size": 32,
    "epochs": 50,
    "validation_split": 0.2,
    "learning_rate": 1e-4,
    "min_lr": 1e-6,
    "weight_decay": 1e-5,              # AdamW weight decay
    "folder_path": "/home/ubuntu/data_lsdir_x3/train",
    "model_path": "ntire_unet_v7.keras",
    "use_mixed_precision": True,
    "clip_noisy_to_01": True,
    "cache_in_memory": False,
    "shuffle_buf": 4000,

    # 8 random patches per image
    "patches_per_image_train": 8,

    # 5 fixed val crops
    "val_crops_per_image": 5,
    "val_noise_seed": 12345,
}

AUTOTUNE = tf.data.AUTOTUNE


# ============== FILE LIST + SPLIT (image-disjoint) ==============

def list_image_files(folder_path: str):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    files = [str(p) for p in Path(folder_path).rglob("*")
             if p.is_file() and p.suffix.lower() in exts]
    if not files:
        raise ValueError(f"No image files found in: {folder_path}")
    files = sorted(files)
    log.info("Found %d image files.", len(files))
    return files


def split_train_val_files(files, val_split: float, seed: int = 42):
    n = len(files)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(val_split * n))
    val_idx = set(idx[:n_val])
    train_files = [files[i] for i in range(n) if i not in val_idx]
    val_files   = [files[i] for i in range(n) if i in val_idx]
    return train_files, val_files


# ============== LOSS ==============

def l1_loss(y_true, y_pred):
    """L1 loss — the dominant choice among NTIRE 2025 top-15 teams.

    Used by: Pixel Purifiers (rank 5), cipher_vision (rank 8),
    AKDT (rank 17), and others. Winner SRC-B alternated L1/L2.
    """
    return tf.reduce_mean(tf.abs(y_true - y_pred))


# ============== DATA PIPELINE ==============

def decode_rgb(path: tf.Tensor) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def augment(clean: tf.Tensor) -> tf.Tensor:
    """8-way geometric augmentation: flips + 90° rotations.

    Used by virtually every top-15 team in NTIRE 2025.
    Pixel Purifiers: "horizontal flipping, vertical flipping,
    and rotations of 90°, 180°, and 270°"
    """
    clean = tf.image.random_flip_left_right(clean)
    clean = tf.image.random_flip_up_down(clean)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    clean = tf.image.rot90(clean, k=k)
    return clean


def add_noise(clean: tf.Tensor, noise_std: float, clip_to_01: bool):
    noise = tf.random.normal(tf.shape(clean), mean=0.0, stddev=noise_std,
                             dtype=tf.float32)
    noisy = clean + noise
    if clip_to_01:
        noisy = tf.clip_by_value(noisy, 0.0, 1.0)
    return noisy, clean


# ---- Texture scoring via Laplacian variance ----
# Laplacian kernel for edge/texture detection
_LAPLACIAN_KERNEL = tf.constant(
    [[0,  1, 0],
     [1, -4, 1],
     [0,  1, 0]], dtype=tf.float32
)
# Reshape to [3, 3, 1, 1] for depthwise conv
_LAPLACIAN_KERNEL_4D = tf.reshape(_LAPLACIAN_KERNEL, [3, 3, 1, 1])


def _texture_score(patch: tf.Tensor) -> tf.Tensor:
    """Compute texture score for a patch using Laplacian variance.
    Higher score = more edges/texture = harder to denoise."""
    gray = tf.reduce_mean(patch, axis=-1, keepdims=True)  # [H, W, 1]
    gray = tf.expand_dims(gray, axis=0)                    # [1, H, W, 1]
    lap = tf.nn.conv2d(gray, _LAPLACIAN_KERNEL_4D, strides=[1,1,1,1],
                       padding="SAME")
    return tf.math.reduce_variance(lap)


def _find_highest_texture_patch(img: tf.Tensor, patch: int):
    """Search a 4x4 grid around the image center for the patch
    with the highest Laplacian variance (most texture/edges).
    Grid covers the middle ~50% of the image."""
    shape = tf.shape(img)
    h, w = shape[0], shape[1]

    # Define center region: middle 50% of valid patch positions
    max_y = h - patch
    max_x = w - patch
    quarter_y = max_y // 4
    quarter_x = max_x // 4

    # Grid spans from 25% to 75% of valid range (centered)
    y_start = quarter_y
    y_end = max_y - quarter_y
    x_start = quarter_x
    x_end = max_x - quarter_x

    ys = tf.cast(tf.linspace(tf.cast(y_start, tf.float32),
                             tf.cast(y_end, tf.float32), 4), tf.int32)
    xs = tf.cast(tf.linspace(tf.cast(x_start, tf.float32),
                             tf.cast(x_end, tf.float32), 4), tf.int32)

    # Safety clamp: ensure no position exceeds valid range
    ys = tf.clip_by_value(ys, 0, max_y)
    xs = tf.clip_by_value(xs, 0, max_x)

    best_score = tf.constant(-1.0, tf.float32)
    best_y = tf.constant(0, tf.int32)
    best_x = tf.constant(0, tf.int32)

    for yi in tf.range(4):
        for xi in tf.range(4):
            y = ys[yi]
            x = xs[xi]
            candidate = tf.image.crop_to_bounding_box(img, y, x, patch, patch)
            score = _texture_score(candidate)
            is_better = score > best_score
            best_score = tf.where(is_better, score, best_score)
            best_y = tf.where(is_better, y, best_y)
            best_x = tf.where(is_better, x, best_x)

    return tf.image.crop_to_bounding_box(img, best_y, best_x, patch, patch)


# ---- TRAIN: 1 center + 1 highest-texture + 6 random crops per image ----
def make_random_patches(img: tf.Tensor, patch: int, n_patches: int):
    """Extract per image:
      - 1 center crop (guaranteed central content)
      - 1 highest-texture crop (hardest patch — Laplacian variance)
      - (n_patches - 2) random crops (diversity)

    Inspired by Pixel Purifiers (rank 5) Hard Dataset Mining strategy.
    """
    shape = tf.shape(img)
    h, w = shape[0], shape[1]
    ok = tf.logical_and(h >= patch, w >= patch)

    def _crop_n():
        patches = tf.TensorArray(dtype=tf.float32, size=n_patches,
                                 dynamic_size=False)
        # Patch 0: center crop
        yc = (h - patch) // 2
        xc = (w - patch) // 2
        center = tf.image.crop_to_bounding_box(img, yc, xc, patch, patch)
        patches = patches.write(0, center)

        # Patch 1: highest-texture crop
        hard_patch = _find_highest_texture_patch(img, patch)
        patches = patches.write(1, hard_patch)

        # Patches 2..n_patches-1: random crops
        for i in tf.range(2, n_patches):
            p = tf.image.random_crop(img, size=[patch, patch, 3])
            patches = patches.write(i, p)
        return patches.stack()

    def _dummy():
        return tf.zeros([n_patches, patch, patch, 3], tf.float32)

    return tf.cond(ok, _crop_n, _dummy)


# ---- VAL: 5 fixed crops per image (deterministic) ----
def make_5_fixed_crops(img: tf.Tensor, patch: int):
    shape = tf.shape(img)
    h, w = shape[0], shape[1]
    ok = tf.logical_and(h >= patch, w >= patch)

    def _crops():
        y0 = tf.constant(0, tf.int32)
        x0 = tf.constant(0, tf.int32)
        y1 = h - patch
        x1 = w - patch
        yc = (h - patch) // 2
        xc = (w - patch) // 2

        tl = tf.image.crop_to_bounding_box(img, y0, x0, patch, patch)
        tr = tf.image.crop_to_bounding_box(img, y0, x1, patch, patch)
        bl = tf.image.crop_to_bounding_box(img, y1, x0, patch, patch)
        br = tf.image.crop_to_bounding_box(img, y1, x1, patch, patch)
        cc = tf.image.crop_to_bounding_box(img, yc, xc, patch, patch)

        return tf.stack([tl, tr, bl, br, cc], axis=0)

    def _dummy():
        return tf.zeros([5, patch, patch, 3], tf.float32)

    return tf.cond(ok, _crops, _dummy)


def make_train_dataset(train_files, patch, noise_std, batch_size,
                       clip_to_01, patches_per_image, shuffle_buf=4000,
                       cache_in_memory=False):
    n = len(train_files)
    ds = tf.data.Dataset.from_tensor_slices(train_files)
    ds = ds.shuffle(buffer_size=min(shuffle_buf, n),
                    reshuffle_each_iteration=True)
    ds = ds.map(decode_rgb, num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda img: make_random_patches(img, patch, patches_per_image),
                num_parallel_calls=AUTOTUNE)
    ds = ds.unbatch()
    ds = ds.filter(lambda x: tf.reduce_max(x) > 0)
    ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())

    if cache_in_memory:
        ds = ds.cache()

    ds = ds.shuffle(buffer_size=min(shuffle_buf * patches_per_image,
                                    n * patches_per_image),
                    reshuffle_each_iteration=True)
    ds = ds.map(lambda x: add_noise(x, noise_std, clip_to_01),
                num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)

    total_patches = n * patches_per_image
    return ds, total_patches


def make_val_dataset(val_files, patch, noise_std, batch_size,
                     clip_to_01, cache_in_memory=False):
    n = len(val_files)
    base_seed = int(CONFIG.get("val_noise_seed", 12345))

    def add_noise_stateless(clean, img_idx, crop_id):
        s0 = tf.cast(img_idx, tf.int32) + tf.constant(base_seed, tf.int32)
        s1 = tf.cast(crop_id, tf.int32) + tf.constant(1000, tf.int32)
        noise = tf.random.stateless_normal(
            tf.shape(clean),
            seed=tf.stack([s0, s1]),
            mean=0.0,
            stddev=tf.cast(noise_std, tf.float32),
            dtype=tf.float32
        )
        noisy = clean + noise
        if clip_to_01:
            noisy = tf.clip_by_value(noisy, 0.0, 1.0)
        return noisy, clean

    ds = tf.data.Dataset.from_tensor_slices(val_files)
    ds = ds.enumerate()
    ds = ds.map(lambda img_idx, path: (img_idx, decode_rgb(path)),
                num_parallel_calls=AUTOTUNE)

    def to_5_items(img_idx, img):
        crops    = make_5_fixed_crops(img, patch)
        crop_ids = tf.range(5, dtype=tf.int32)
        img_idx5 = tf.fill([5], tf.cast(img_idx, tf.int32))
        return tf.data.Dataset.from_tensor_slices((img_idx5, crop_ids, crops))

    ds = ds.flat_map(to_5_items)
    ds = ds.filter(lambda img_idx, crop_id, crop: tf.reduce_max(crop) > 0)
    ds = ds.apply(tf.data.experimental.ignore_errors())

    if cache_in_memory:
        ds = ds.cache()

    ds = ds.map(lambda img_idx, crop_id, crop:
                add_noise_stateless(crop, img_idx, crop_id),
                num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)

    total_patches = n * 5
    return ds, total_patches


# ============== ATTENTION U-NET ==============

def conv_block(x, filters, dropout=0.0):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    if dropout and dropout > 0:
        x = layers.Dropout(dropout)(x)
    return x


def attention_gate(x, g, filters):
    theta_x = layers.Conv2D(filters, 1, padding="same")(x)
    theta_x = layers.BatchNormalization()(theta_x)
    phi_g = layers.Conv2D(filters, 1, padding="same")(g)
    phi_g = layers.BatchNormalization()(phi_g)
    add_xg = layers.Add()([theta_x, phi_g])
    act_xg = layers.Activation("gelu")(add_xg)
    psi = layers.Conv2D(1, 1, padding="same")(act_xg)
    psi = layers.BatchNormalization()(psi)
    psi = layers.Activation("sigmoid")(psi)
    return layers.Multiply()([x, psi])


def build_attention_unet(input_shape=(96, 96, 3)):
    """Same architecture as v5 — this is proven to work at ~30 dB.
    No risky changes. sigmoid output for stable training."""
    inputs = layers.Input(shape=input_shape)

    c1 = conv_block(inputs, 64, dropout=0.05)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 128, dropout=0.05)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 256, dropout=0.10)
    p3 = layers.MaxPooling2D()(c3)

    b = conv_block(p3, 512, dropout=0.15)

    u3 = layers.UpSampling2D()(b)
    c3_att = attention_gate(c3, u3, 256)
    u3 = layers.Concatenate()([c3_att, u3])
    c4 = conv_block(u3, 256, dropout=0.10)

    u2 = layers.UpSampling2D()(c4)
    c2_att = attention_gate(c2, u2, 128)
    u2 = layers.Concatenate()([c2_att, u2])
    c5 = conv_block(u2, 128, dropout=0.05)

    u1 = layers.UpSampling2D()(c5)
    c1_att = attention_gate(c1, u1, 64)
    u1 = layers.Concatenate()([c1_att, u1])
    c6 = conv_block(u1, 64, dropout=0.05)

    outputs = layers.Conv2D(3, 1, activation="sigmoid", padding="same",
                            dtype="float32")(c6)

    return Model(inputs, outputs, name="Attention_UNet_v7")


# ============== METRICS ==============

@tf.function
def psnr_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


# ============== TRAINING ==============

def train():
    log.info("=" * 60)
    log.info("v7 — Conservative NTIRE-informed improvements")
    log.info("Changes: L1 loss, 8-way augment, AdamW")
    log.info("Kept: v5 architecture, 96x96 patches, sigmoid output")
    log.info("=" * 60)

    if CONFIG["use_mixed_precision"]:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        log.info("Mixed precision enabled")

    files = list_image_files(CONFIG["folder_path"])
    train_files, val_files = split_train_val_files(
        files, CONFIG["validation_split"], seed=42
    )

    log.info("Train images: %d", len(train_files))
    log.info("Val images  : %d (image-disjoint)", len(val_files))

    train_ds, train_patches = make_train_dataset(
        train_files=train_files,
        patch=CONFIG["patch_size"],
        noise_std=CONFIG["noise_std"],
        batch_size=CONFIG["batch_size"],
        clip_to_01=CONFIG["clip_noisy_to_01"],
        patches_per_image=CONFIG["patches_per_image_train"],
        shuffle_buf=CONFIG["shuffle_buf"],
        cache_in_memory=CONFIG["cache_in_memory"],
    )

    val_ds, val_patches = make_val_dataset(
        val_files=val_files,
        patch=CONFIG["patch_size"],
        noise_std=CONFIG["noise_std"],
        batch_size=CONFIG["batch_size"],
        clip_to_01=CONFIG["clip_noisy_to_01"],
        cache_in_memory=CONFIG["cache_in_memory"],
    )

    log.info("Train patches (approx): %d", train_patches)
    log.info("Val patches   (exact) : %d", val_patches)

    log.info("Building model...")
    model = build_attention_unet(
        (CONFIG["patch_size"], CONFIG["patch_size"], 3)
    )
    model.summary(print_fn=log.info)

    # ── AdamW + L1 (the NTIRE 2025 standard) ──
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"],
        ),
        loss=l1_loss,
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"),
                 psnr_metric],
    )

    callbacks = [
        ModelCheckpoint(
            CONFIG["model_path"],
            monitor="val_psnr_metric",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_psnr_metric",
            mode="max",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=CONFIG["min_lr"],
            verbose=1
        ),
        CSVLogger("logs_v7/history.csv"),
    ]

    log.info("Training for up to %d epochs...", CONFIG["epochs"])
    log.info("Loss: L1 (NTIRE standard)")
    log.info("Optimizer: AdamW (weight_decay=%.1e)", CONFIG["weight_decay"])
    log.info("Augmentation: 8-way (flips + 90° rotations)")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG["epochs"],
        callbacks=callbacks,
        verbose=1,
    )

    model.save(CONFIG["model_path"])
    log.info("Model saved: %s", CONFIG["model_path"])
    log.info("Best val_loss: %.6f", min(history.history["val_loss"]))

    val_psnr_keys = [k for k in history.history.keys() if "val_psnr" in k]
    if val_psnr_keys:
        best_psnr = max(history.history[val_psnr_keys[0]])
        log.info("Best val_PSNR: %.2f dB", best_psnr)

    log.info("=" * 60)
    log.info("NEXT STEPS FOR MORE PSNR (from NTIRE report):")
    log.info("  1. Upgrade TTA to 8 transforms (currently 4) → +0.1-0.3 dB")
    log.info("  2. Increase inference patch overlap to 20-50%% → +0.05-0.1 dB")
    log.info("  3. Average outputs from multiple patch sizes → +0.05 dB")
    log.info("  4. If GPU becomes available: try Restormer architecture")
    log.info("=" * 60)


if __name__ == "__main__":
    train()
