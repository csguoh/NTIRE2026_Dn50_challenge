"""
05_train_base_residual.py
===========================
Trains the residual directly from precomputed .npz cache.
Run 05_precompute_den1_for_residual_training.py first to build the cache.

Pipeline:
  Load (den1, clean) from .npz files
  → augment (train only: flip + rot90)
  → wrapper: den2 = clip(den1 + res_model(den1), 0, 1)
  → MSE loss vs clean
  → save res_model
"""

import os
import warnings
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, Callback
)

# ── Logging ───────────────────────────────────────────────────────────────────
Path("logs_refiner_fast").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler("logs_refiner_fast/train.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

# ── GPU setup ─────────────────────────────────────────────────
def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        log.info("GPU memory growth enabled on %d device(s).", len(gpus))
    else:
        log.warning("No GPU detected — running on CPU.")

configure_gpu()

tf.keras.utils.set_random_seed(42)
AUTOTUNE = tf.data.AUTOTUNE

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG = {
    "cache_dir":            "/home/ubuntu/den1_cache",
    "wrapper_ckpt_path":    "ntire_residual_v7.1_wrapper.keras",
    "residual_model_path":  "ntire_refiner_v7.1.keras",

    "patch_size":           96,

    # Training — A100 tuned
    "batch_size":           64,
    "epochs":               50,
    "learning_rate":        1e-4,
    "shuffle_buf":          16000,

    # 85% LSDIR / 15% texture mixing
    "lsdir_weight":         0.85,
    "texture_weight":       0.15,

    "use_mixed_precision":  True,
}

# ── Mixed precision ───────────────────────────────────────────────────────────
if CONFIG["use_mixed_precision"]:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    log.info("Mixed precision: mixed_float16")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATASET — load from .npz cache
# ══════════════════════════════════════════════════════════════════════════════

def augment_pair(den1, clean):
    """Apply identical random augmentation to both den1 and clean."""
    # Stack so both get the same random transform
    stacked = tf.concat([den1, clean], axis=-1)   # (H, W, 6)
    stacked = tf.image.random_flip_left_right(stacked)
    stacked = tf.image.random_flip_up_down(stacked)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    stacked = tf.image.rot90(stacked, k=k)
    den1_aug  = stacked[:, :, :3]
    clean_aug = stacked[:, :, 3:]
    return den1_aug, clean_aug


def load_npz_as_ds(npz_dir, is_train=True, shuffle_buf=8000):
    """
    Load all .npz files in npz_dir as a tf.data.Dataset of (den1, clean) pairs.
    Each .npz contains arrays of shape (N, 96, 96, 3) — unbatched to individual patches.
    """
    npz_files = sorted(Path(npz_dir).glob("*.npz"))
    if not npz_files:
        return None, 0

    npz_paths = [str(p) for p in npz_files]
    total_patches = 0

    # Count total patches for steps_per_epoch
    # Sample first file to get N per file
    sample = np.load(npz_paths[0])
    n_per_file = sample["den1"].shape[0]
    total_patches = len(npz_paths) * n_per_file
    log.info("  %s: %d files × %d patches = %d total",
             npz_dir, len(npz_paths), n_per_file, total_patches)

    def _load_npz(path):
        """Load one .npz and return (den1_patches, clean_patches)."""
        def _np_load(p):
            p = p.numpy().decode("utf-8")
            data = np.load(p)
            den1  = data["den1"].astype(np.float32)   # (N, 96, 96, 3)
            clean = data["clean"].astype(np.float32)  # (N, 96, 96, 3)
            return den1, clean

        den1, clean = tf.py_function(
            _np_load, [path], [tf.float32, tf.float32]
        )
        den1.set_shape([None, CONFIG["patch_size"], CONFIG["patch_size"], 3])
        clean.set_shape([None, CONFIG["patch_size"], CONFIG["patch_size"], 3])
        return tf.data.Dataset.from_tensor_slices((den1, clean))

    ds = tf.data.Dataset.from_tensor_slices(npz_paths)

    if is_train:
        ds = ds.shuffle(buffer_size=min(shuffle_buf, len(npz_paths)),
                        reshuffle_each_iteration=True)

    ds = ds.flat_map(_load_npz)
    ds = ds.filter(lambda d, c: tf.reduce_max(c) > 0)

    if is_train:
        ds = ds.map(augment_pair, num_parallel_calls=AUTOTUNE)
        ds = ds.shuffle(buffer_size=shuffle_buf, reshuffle_each_iteration=True)

    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds, total_patches


def make_train_dataset(cache_dir, batch_size, shuffle_buf):
    """
    Build training dataset from cache.
    Mixes LSDIR (85%) and texture (15%) using sample_from_datasets.
    """
    lsdir_dir   = Path(cache_dir) / "train" / "lsdir"
    texture_dir = Path(cache_dir) / "train" / "texture"

    lsdir_ds, lsdir_patches = load_npz_as_ds(
        str(lsdir_dir), is_train=True, shuffle_buf=shuffle_buf
    )
    if lsdir_ds is None:
        raise ValueError(f"No LSDIR train cache found at {lsdir_dir}")

    tex_ds, tex_patches = load_npz_as_ds(
        str(texture_dir), is_train=True, shuffle_buf=shuffle_buf
    )

    if tex_ds is not None and tex_patches > 0:
        ds = tf.data.Dataset.sample_from_datasets(
            [lsdir_ds, tex_ds],
            weights=[CONFIG["lsdir_weight"], CONFIG["texture_weight"]],
            rerandomize_each_iteration=True,
        )
        log.info("Mixed: %.0f%% LSDIR + %.0f%% texture",
                 CONFIG["lsdir_weight"]*100, CONFIG["texture_weight"]*100)
        effective_total = int(lsdir_patches / CONFIG["lsdir_weight"])
    else:
        ds = lsdir_ds
        log.info("LSDIR only (no texture cache found)")
        effective_total = lsdir_patches

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)
    return ds, effective_total


def make_val_dataset(cache_dir, batch_size):
    """
    Build validation dataset from cache.
    Concatenates LSDIR val + texture val — every image contributes exactly once.
    """
    lsdir_dir   = Path(cache_dir) / "val" / "lsdir"
    texture_dir = Path(cache_dir) / "val" / "texture"

    lsdir_ds, lsdir_patches = load_npz_as_ds(
        str(lsdir_dir), is_train=False
    )
    if lsdir_ds is None:
        raise ValueError(f"No LSDIR val cache found at {lsdir_dir}")

    tex_ds, tex_patches = load_npz_as_ds(
        str(texture_dir), is_train=False
    )

    if tex_ds is not None and tex_patches > 0:
        ds = lsdir_ds.concatenate(tex_ds)
        total_val = lsdir_patches + tex_patches
        log.info("Val: %d LSDIR + %d texture = %d total",
                 lsdir_patches, tex_patches, total_val)
    else:
        ds = lsdir_ds
        total_val = lsdir_patches
        log.info("Val: %d LSDIR only", lsdir_patches)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds, total_val


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

def conv_block(x, filters, dropout=0.0):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    return x


def build_unet_residual_refiner(input_shape=(96, 96, 3)):
    """UNet residual refiner — tanh output in [-1,1]."""
    inputs = layers.Input(shape=input_shape)

    c1 = conv_block(inputs, 32,  dropout=0.05)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1,  64,  dropout=0.05)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2,  128, dropout=0.10)
    p3 = layers.MaxPooling2D()(c3)

    b  = conv_block(p3,  256, dropout=0.15)

    u3 = layers.UpSampling2D()(b)
    u3 = layers.Concatenate()([u3, c3])
    c4 = conv_block(u3,  128, dropout=0.10)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.Concatenate()([u2, c2])
    c5 = conv_block(u2,  64,  dropout=0.05)

    u1 = layers.UpSampling2D()(c5)
    u1 = layers.Concatenate()([u1, c1])
    c6 = conv_block(u1,  32,  dropout=0.05)

    residual_out = layers.Conv2D(
        3, 1, activation="tanh", padding="same", dtype="float32"
    )(c6)

    return Model(inputs, residual_out, name="UNet_ResidualRefiner_v7.1")


def build_wrapper(res_model, patch_size=96):
    """
    den2 = clip(den1 + res_model(den1), 0, 1)
    All ops forced to float32 — safe under mixed precision.
    """
    den1_in  = layers.Input(shape=(patch_size, patch_size, 3), name="den1_input")
    den1_f32 = layers.Lambda(
        lambda x: tf.cast(x, tf.float32), name="cast_f32"
    )(den1_in)
    r_pred = res_model(den1_f32)
    den2   = layers.Add()([den1_f32, r_pred])
    den2   = layers.Lambda(
        lambda x: tf.clip_by_value(tf.cast(x, tf.float32), 0.0, 1.0),
        name="clip01", dtype="float32",
    )(den2)
    return Model(den1_in, den2, name="ResidualWrapper")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CALLBACK — save res_model (not wrapper) on val_loss improvement
# ══════════════════════════════════════════════════════════════════════════════

class SaveResidualModelCallback(Callback):
    def __init__(self, res_model, save_path, monitor="val_loss"):
        super().__init__()
        self.res_model = res_model
        self.save_path = save_path
        self.monitor   = monitor
        self.best      = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get(self.monitor, float("inf"))
        if current < self.best:
            self.best = current
            self.res_model.save(self.save_path)
            log.info("Saved res_model — val_loss improved to %.6f → %s",
                     current, self.save_path)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train():
    log.info("=" * 60)
    log.info("Residual Refiner v7.1 — Fast Training from Cache")
    log.info("=" * 60)

    cache_dir  = CONFIG["cache_dir"]
    batch_size = CONFIG["batch_size"]
    patch_size = CONFIG["patch_size"]

    # ── Datasets ──────────────────────────────────────────────────────────────
    log.info("Building training dataset from cache...")
    train_ds, effective_train = make_train_dataset(
        cache_dir, batch_size, CONFIG["shuffle_buf"]
    )

    log.info("Building validation dataset from cache...")
    val_ds, val_patches = make_val_dataset(cache_dir, batch_size)

    steps_per_epoch  = max(1, effective_train // batch_size)
    validation_steps = max(1, (val_patches + batch_size - 1) // batch_size)

    log.info("effective_train_patches : %d", effective_train)
    log.info("steps_per_epoch         : %d", steps_per_epoch)
    log.info("val_patches             : %d", val_patches)
    log.info("validation_steps        : %d", validation_steps)

    # ── Models ─────────────────────────────────────────────────────────────────
    log.info("Building residual model...")
    res_model = build_unet_residual_refiner((patch_size, patch_size, 3))
    res_model.summary(print_fn=log.info)

    log.info("Building wrapper...")
    train_model = build_wrapper(res_model, patch_size)
    train_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG["learning_rate"]),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )

    # ── Callbacks ──────────────────────────────────────────────────────────────
    callbacks = [
        SaveResidualModelCallback(
            res_model  = res_model,
            save_path  = CONFIG["residual_model_path"],
            monitor    = "val_loss",
        ),
        ModelCheckpoint(
            CONFIG["wrapper_ckpt_path"],
            monitor="val_loss", save_best_only=True, verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss", patience=15,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5,
            min_lr=1e-6, verbose=1,
        ),
        CSVLogger("logs_refiner_fast/history.csv"),
    ]

    # ── Train ──────────────────────────────────────────────────────────────────
    log.info("Training: den2 = clip(den1 + r_pred, 0, 1) vs clean — MSE")
    history = train_model.fit(
        train_ds,
        validation_data  = val_ds,
        epochs           = CONFIG["epochs"],
        steps_per_epoch  = steps_per_epoch,
        validation_steps = validation_steps,
        callbacks        = callbacks,
        verbose          = 1,
    )

    # ── Save + summary ─────────────────────────────────────────────────────────
    res_model.save(CONFIG["residual_model_path"])
    log.info("Final res_model saved: %s", CONFIG["residual_model_path"])

    best   = min(history.history["val_loss"])
    first  = history.history["val_loss"][0]
    log.info("Epoch-1 val_loss : %.6f", first)
    log.info("Best    val_loss : %.6f", best)
    log.info("Improvement      : %.1f%%",
             (first - best) * 100 / (first + 1e-12))
    log.info("")
    log.info("INFERENCE:")
    log.info("  den1  = base_model(noisy)")
    log.info("  r     = res_model(den1)")
    log.info("  final = clip(den1 + r, 0, 1)")


if __name__ == "__main__":
    train()
