import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import imageio
import numpy as np
import sys

# --------------------------------------------------
# Add Restormer repo to Python path
# --------------------------------------------------

sys.path.append("./Restormer")
from basicsr.models.archs.restormer_arch import Restormer


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class DenoiseDataset(Dataset):

    def __init__(self, noisy_dir, clean_dir, patch_size=128):

        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.files = sorted(os.listdir(noisy_dir))

        self.patch_size = patch_size
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        name = self.files[idx]

        noisy = imageio.imread(os.path.join(self.noisy_dir, name))
        clean = imageio.imread(os.path.join(self.clean_dir, name))

        h, w, _ = noisy.shape

        x = np.random.randint(0, h - self.patch_size)
        y = np.random.randint(0, w - self.patch_size)

        noisy = noisy[x:x+self.patch_size, y:y+self.patch_size]
        clean = clean[x:x+self.patch_size, y:y+self.patch_size]

        noisy = self.to_tensor(noisy)
        clean = self.to_tensor(clean)

        return noisy, clean


# --------------------------------------------------
# DATASET PATHS
# --------------------------------------------------
# IMPORTANT:
# Change these paths to where your datasets are stored.

clean_dir = "./datasets/DIV2K_train_HR"        # <-- path to clean DIV2K images
noisy_dir = "./datasets/DIV2K_train_noise50"   # <-- path to noisy images generated using add_noise.py


# --------------------------------------------------
# MODEL PATHS
# --------------------------------------------------

initial_weights = "./model_zoo/team13_initialweights.pth"   # Restormer sigma50 pretrained weights
save_path = "./model_zoo/13_Restormer.pth"                  # best fine-tuned model

os.makedirs("model_zoo", exist_ok=True)


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

print("Using device:", device)


# --------------------------------------------------
# DATASET LOADER
# --------------------------------------------------

dataset = DenoiseDataset(noisy_dir, clean_dir, patch_size=128)

train_loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4
)

print("Dataset size:", len(dataset))


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = Restormer(
    inp_channels=3,
    out_channels=3,
    dim=48,
    num_blocks=[4,6,6,8],
    num_refinement_blocks=4,
    heads=[1,2,4,8],
    ffn_expansion_factor=2.66,
    bias=False,
    LayerNorm_type='BiasFree'
)

model = model.to(device)

print("Restormer model initialized")


# --------------------------------------------------
# LOAD INITIAL WEIGHTS (sigma50 pretrained)
# --------------------------------------------------

if os.path.exists(initial_weights):

    print("Loading initial Restormer weights...")

    checkpoint = torch.load(initial_weights, map_location=device)

    if "params" in checkpoint:
        checkpoint = checkpoint["params"]

    model.load_state_dict(checkpoint, strict=False)

    print("Initial weights loaded successfully")

else:

    print("WARNING: Initial weights not found:", initial_weights)


# --------------------------------------------------
# LOSS + OPTIMIZER
# --------------------------------------------------

criterion = nn.L1Loss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-5,
    weight_decay=1e-4
)


# --------------------------------------------------
# PSNR FUNCTION
# --------------------------------------------------

def psnr(img1, img2):

    mse = torch.mean((img1 - img2) ** 2, dim=[1,2,3])
    psnr = 10 * torch.log10(1 / mse)

    return torch.mean(psnr)


# --------------------------------------------------
# TRAINING LOOP
# --------------------------------------------------

epochs = 50
best_loss = float("inf")

print("Starting training...")

for epoch in range(epochs):

    model.train()

    total_loss = 0
    total_psnr = 0

    for i, (noisy, clean) in enumerate(train_loader):

        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()

        output = model(noisy)

        loss = criterion(output, clean)

        loss.backward()
        optimizer.step()

        batch_psnr = psnr(output.detach(), clean)

        total_loss += loss.item()
        total_psnr += batch_psnr.item()

        if i % 20 == 0:

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Batch {i}/{len(train_loader)} | "
                f"Loss {loss.item():.4f} | "
                f"PSNR {batch_psnr:.2f}"
            )

    avg_loss = total_loss / len(train_loader)
    avg_psnr = total_psnr / len(train_loader)

    print("\nEpoch Summary")
    print("Average Loss:", avg_loss)
    print("Average PSNR:", avg_psnr)

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(model.state_dict(), save_path)

        print("New best model saved →", save_path)


print("Training complete")