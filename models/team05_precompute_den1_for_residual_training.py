"""
Run ONCE before training. Computes den1 = base_model(noisy) for every patch
and saves (den1, clean) pairs to disk as .npz files.

This eliminates the base_model inference bottleneck during training.
After precomputation, training loads directly from .npz files on GPU.

Output structure:
  cache_dir/
    train/
      lsdir_000000.npz   # keys: den1 (4,96,96,3), clean (4,96,96,3)
      lsdir_000001.npz
      ...
      texture_000000.npz # keys: den1 (1,96,96,3), clean (1,96,96,3)
      ...
    val/
      lsdir_000000.npz   # keys: den1 (1,96,96,3), clean (1,96,96,3)
      ...
      texture_000000.npz
      ...
    splits.npz           # saves train/val file indices for reproducibility


"""

import os
import warnings
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

# ── Logging ───────────────────────────────────────────────────────────────────
Path("logs_precompute").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler("logs_precompute/precompute.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

# ── GPU setup ────────────────────────
def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        log.info("GPU memory growth enabled on %d device(s).", len(gpus))
    else:
        log.warning("No GPU detected.")

configure_gpu()

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG = {
    "lsdir_folder":         "/home/ubuntu/data_lsdir_x3/train",
    "texture_folder":       "/home/ubuntu/NTIRE_textures",
    "base_model_path":      "ntire_unet_v7.keras",
    "cache_dir":            "/home/ubuntu/den1_cache",

    "patch_size":           96,
    "noise_std":            50 / 255.0,
    "clip_noisy_to_01":     True,

    "patches_per_lsdir":    4,     # 2 high-texture + 2 random
    "val_split_lsdir":      0.10,
    "val_split_texture":    0.10,
    "split_seed":           42,
    "val_noise_seed":       12345,

    "inference_batch_size": 256,   # base model forward pass batch size
    "use_mixed_precision":  True,
}

AUTOTUNE = tf.data.AUTOTUNE

# ── Mixed precision ───────────────────────────────────────────────────────────
if CONFIG["use_mixed_precision"]:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    log.info("Mixed precision: mixed_float16")

# ── File helpers ──────────────────────────────────────────────────────────────
def list_image_files(folder):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    files = sorted([
        str(p) for p in Path(folder).rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    ])
    if not files:
        raise ValueError(f"No images found in: {folder}")
    log.info("Found %d files in %s", len(files), folder)
    return files

def split_files(files, val_split, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    n_val = int(round(val_split * len(files)))
    val_idx = set(idx[:n_val])
    train = [files[i] for i in range(len(files)) if i not in val_idx]
    val   = [files[i] for i in range(len(files)) if i in val_idx]
    return train, val

# ── Crop helpers (pure numpy — run outside tf.data for precompute) ────────────

def get_random_patches_np(img, patch, n_patches):
    """Returns (n_patches, patch, patch, 3) float32 — pure random crops."""
    h, w, _ = img.shape
    if h < patch or w < patch:
        img = _resize_np(img, max(h, patch), max(w, patch))
        h, w, _ = img.shape
    max_y = h - patch
    max_x = w - patch
    rng = np.random.default_rng()
    patches = []
    for _ in range(n_patches):
        y = rng.integers(0, max_y + 1) if max_y > 0 else 0
        x = rng.integers(0, max_x + 1) if max_x > 0 else 0
        patches.append(img[y:y+patch, x:x+patch])
    return np.stack(patches, axis=0).astype(np.float32)

def get_center_patch_np(img, patch):
    """Returns (1, patch, patch, 3) float32."""
    h, w, _ = img.shape
    if h < patch or w < patch:
        img = _resize_np(img, max(h, patch), max(w, patch))
        h, w, _ = img.shape
    yc = (h - patch) // 2
    xc = (w - patch) // 2
    return img[yc:yc+patch, xc:xc+patch][np.newaxis].astype(np.float32)

def _resize_np(img, new_h, new_w):
    import cv2
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

def load_image_np(path):
    """Load image as float32 [0,1] RGB."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0

def add_noise_deterministic_np(clean, noise_std, seed):
    """Deterministic noise keyed on seed integer."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_std, clean.shape).astype(np.float32)
    noisy = clean + noise
    if CONFIG["clip_noisy_to_01"]:
        noisy = np.clip(noisy, 0.0, 1.0)
    return noisy

def add_noise_random_np(clean, noise_std):
    noise = np.random.normal(0, noise_std, clean.shape).astype(np.float32)
    noisy = clean + noise
    if CONFIG["clip_noisy_to_01"]:
        noisy = np.clip(noisy, 0.0, 1.0)
    return noisy

# ── Base model inference ──────────────────────────────────────────────────────
@tf.function
def _run_base_model(base_model, batch):
    den1 = base_model(batch, training=False)
    return tf.clip_by_value(tf.cast(den1, tf.float32), 0.0, 1.0)


# ── Main precompute functions ─────────────────────────────────────────────────
def precompute_split(files, out_dir, base_model, patch, n_patches_fn,
                     is_val=False, val_noise_seed=12345, prefix="img"):
    """
    Batched precompute: accumulates patches from many images, runs GPU
    inference on full batches of 256, then saves per-image .npz files.
    This keeps the GPU busy instead of doing 4-patch micro-batches.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    batch_size = CONFIG["inference_batch_size"]
    noise_std  = CONFIG["noise_std"]
    n_files    = len(files)

    log.info("Processing %d files → %s", n_files, out_dir)

    # Accumulation buffers
    buf_noisy  = []   # accumulated noisy patches
    buf_clean  = []   # accumulated clean patches
    buf_idx    = []   # which file index each patch belongs to
    buf_nper   = []   # how many patches per file (to split results back)

    def _flush(force=False):
        """Run base model on accumulated buffer, save per-image .npz files."""
        if not buf_noisy:
            return
        if not force and len(buf_noisy) < batch_size:
            return

        noisy_all = np.concatenate(buf_noisy, axis=0)   # (Total, P, P, 3)
        clean_all = np.concatenate(buf_clean, axis=0)

        # GPU inference in one big batch
        den1_all = []
        for s in range(0, len(noisy_all), batch_size):
            b = tf.constant(noisy_all[s:s+batch_size], dtype=tf.float32)
            den1_all.append(_run_base_model(base_model, b).numpy())
        den1_all = np.concatenate(den1_all, axis=0)

        # Split back per image and save
        offset = 0
        for file_i, n_per in zip(buf_idx, buf_nper):
            den1_img  = den1_all[offset:offset+n_per]
            clean_img = clean_all[offset:offset+n_per]
            offset   += n_per
            out_path  = Path(out_dir) / f"{prefix}_{file_i:06d}.npz"
            np.savez_compressed(
                out_path,
                den1=den1_img.astype(np.float32),
                clean=clean_img.astype(np.float32),
            )
            if file_i % 500 == 0:
                log.info("  [%d/%d] saved %s  patches=%d",
                         file_i, n_files, out_path.name, n_per)

        buf_noisy.clear()
        buf_clean.clear()
        buf_idx.clear()
        buf_nper.clear()

    for i, path in enumerate(files):
        out_path = Path(out_dir) / f"{prefix}_{i:06d}.npz"

        # Skip already computed
        if out_path.exists():
            if i % 2000 == 0:
                log.info("  [%d/%d] skipping (already exists)", i, n_files)
            continue

        try:
            img = load_image_np(path)
        except Exception as e:
            log.warning("  [%d/%d] failed to load %s: %s", i, n_files, path, e)
            continue

        clean_patches = n_patches_fn(img, patch)
        if clean_patches is None or len(clean_patches) == 0:
            continue

        if is_val:
            noisy_patches = np.stack([
                add_noise_deterministic_np(clean_patches[j], noise_std,
                                           seed=val_noise_seed + i * 100 + j)
                for j in range(len(clean_patches))
            ], axis=0)
        else:
            noisy_patches = np.stack([
                add_noise_random_np(clean_patches[j], noise_std)
                for j in range(len(clean_patches))
            ], axis=0)

        buf_noisy.append(noisy_patches)
        buf_clean.append(clean_patches)
        buf_idx.append(i)
        buf_nper.append(len(clean_patches))

        # Flush when buffer has enough patches for a full GPU batch
        total_buffered = sum(x.shape[0] for x in buf_noisy)
        if total_buffered >= batch_size:
            _flush(force=True)

    # Flush remaining
    _flush(force=True)
    log.info("Done: %s", out_dir)


def main():
    log.info("=" * 60)
    log.info("Precomputing den1 cache")
    log.info("=" * 60)

    # Load base model
    log.info("Loading base model: %s", CONFIG["base_model_path"])
    base_model = tf.keras.models.load_model(CONFIG["base_model_path"], compile=False)
    base_model.trainable = False
    log.info("Base model loaded. Params: %d", base_model.count_params())

    patch = CONFIG["patch_size"]

    # ── LSDIR ──────────────────────────────────────────────────────────────────
    lsdir_files = list_image_files(CONFIG["lsdir_folder"])
    lsdir_train, lsdir_val = split_files(
        lsdir_files, CONFIG["val_split_lsdir"], CONFIG["split_seed"]
    )
    log.info("LSDIR — train: %d  val: %d", len(lsdir_train), len(lsdir_val))

    # ── Texture ────────────────────────────────────────────────────────────────
    tex_train, tex_val = [], []
    tex_path = CONFIG.get("texture_folder", "")
    if tex_path and os.path.isdir(tex_path):
        tex_files = list_image_files(tex_path)
        tex_train, tex_val = split_files(
            tex_files, CONFIG["val_split_texture"], CONFIG["split_seed"]
        )
        log.info("Texture — train: %d  val: %d", len(tex_train), len(tex_val))
    else:
        log.info("No texture dataset.")

    # Save split info so training script uses identical splits
    cache_dir = Path(CONFIG["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_dir / "splits.npz",
        lsdir_train=lsdir_train, lsdir_val=lsdir_val,
        tex_train=tex_train,     tex_val=tex_val,
    )
    log.info("Saved splits to %s/splits.npz", cache_dir)

    n_patches = CONFIG["patches_per_lsdir"]

    # ── Precompute LSDIR train ─────────────────────────────────────────────────
    log.info("\n--- LSDIR TRAIN ---")
    precompute_split(
        lsdir_train,
        out_dir     = str(cache_dir / "train" / "lsdir"),
        base_model  = base_model,
        patch       = patch,
        n_patches_fn= lambda img, p: get_random_patches_np(img, p, n_patches),
        is_val      = False,
        prefix      = "lsdir",
    )

    # ── Precompute LSDIR val ───────────────────────────────────────────────────
    log.info("\n--- LSDIR VAL ---")
    precompute_split(
        lsdir_val,
        out_dir     = str(cache_dir / "val" / "lsdir"),
        base_model  = base_model,
        patch       = patch,
        n_patches_fn= lambda img, p: get_center_patch_np(img, p),
        is_val      = True,
        val_noise_seed = CONFIG["val_noise_seed"],
        prefix      = "lsdir",
    )

    # ── Precompute texture train ───────────────────────────────────────────────
    if tex_train:
        log.info("\n--- TEXTURE TRAIN ---")
        precompute_split(
            tex_train,
            out_dir     = str(cache_dir / "train" / "texture"),
            base_model  = base_model,
            patch       = patch,
            n_patches_fn= lambda img, p: get_center_patch_np(img, p),
            is_val      = False,
            prefix      = "texture",
        )

    # ── Precompute texture val ─────────────────────────────────────────────────
    if tex_val:
        log.info("\n--- TEXTURE VAL ---")
        precompute_split(
            tex_val,
            out_dir     = str(cache_dir / "val" / "texture"),
            base_model  = base_model,
            patch       = patch,
            n_patches_fn= lambda img, p: get_center_patch_np(img, p),
            is_val      = True,
            val_noise_seed = CONFIG["val_noise_seed"],
            prefix      = "texture",
        )

    # ── Summary ────────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Precomputation complete!")
    log.info("Cache directory: %s", cache_dir)

    # Count files
    for split in ["train/lsdir", "train/texture", "val/lsdir", "val/texture"]:
        d = cache_dir / split
        if d.exists():
            n = len(list(d.glob("*.npz")))
            log.info("  %s: %d .npz files", split, n)

    log.info("Now run: python3 train_v7.1_refiner_fast.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
