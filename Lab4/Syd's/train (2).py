import os
import time

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import (
    load_coco_captions,
    attach_embed_indices,
    COCORetrievalDataset,
    build_image_transform,
)
from model import ResNet50ImageEncoder, CLIPLoss


def train_clip(
    image_encoder,
    train_loader,
    val_loader,
    train_text_embs,
    val_text_embs,
    criterion,
    optimizer,
    device,
    epochs=5,
    print_every=50,
):
    train_losses = []
    val_losses = []

    start = time.time()

    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch+1}/{epochs} =====")

        image_encoder.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        # ---------------- TRAIN ----------------
        for batch_idx, batch in enumerate(train_loader):
            imgs = batch["image"].to(device, non_blocking=True)
            idxs = batch["embed_idx"].to(device)

            text_batch = train_text_embs[idxs]

            optimizer.zero_grad()
            img_emb = image_encoder(imgs)
            loss = criterion(img_emb, text_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % print_every == 0:
                batches_total = len(train_loader)
                progress = (batch_idx + 1) / batches_total * 100
                elapsed = (time.time() - epoch_start) / 60

                print(
                    f"  [Batch {batch_idx+1}/{batches_total}] "
                    f"({progress:.1f}%) Loss: {loss.item():.4f} "
                    f"Elapsed: {elapsed:.1f} min"
                )

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # ---------------- VALIDATION ----------------
        image_encoder.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                imgs = batch["image"].to(device, non_blocking=True)
                idxs = batch["embed_idx"].to(device)

                txt_emb = val_text_embs[idxs]
                img_emb = image_encoder(imgs)
                loss = criterion(img_emb, txt_emb)

                val_loss += loss.item()

                if (batch_idx + 1) % print_every == 0:
                    print(
                        f"  [Val Batch {batch_idx+1}/{len(val_loader)}] "
                        f"Val Loss: {loss.item():.4f}"
                    )

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")
        print(f"Epoch {epoch+1} time: {(time.time() - epoch_start)/60:.2f} min")

    total_time = (time.time() - start) / 60
    print(f"\nTotal training time: {total_time:.2f} minutes\n")

    return train_losses, val_losses, total_time


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---------- PATHS ----------
    # COCO images/captions: local, fast
    IMG_TRAIN = "/content/coco2014/coco2014/images/train2014"
    IMG_VAL   = "/content/coco2014/coco2014/images/val2014"

    CAP_TRAIN = "/content/coco2014/annotations/captions_train2014.json"
    CAP_VAL   = "/content/coco2014/annotations/captions_val2014.json"


    # Cached text embeddings: in Drive, persistent
    CACHE_DIR = "/content/drive/MyDrive/elec475_lab4/cache"

    # ---------- HYPERPARAMETERS (UPDATED) ----------
    BATCH_SIZE    = 128
    EPOCHS        = 5        
    LR            = 5e-5       # was 1e-4
    WEIGHT_DECAY  = 0.05       # was 1e-2
    TEMPERATURE   = 0.07

    # -------- LOAD CAPTIONS --------
    train_samples = load_coco_captions(CAP_TRAIN, IMG_TRAIN)
    val_samples   = load_coco_captions(CAP_VAL,   IMG_VAL)

    attach_embed_indices(train_samples)
    attach_embed_indices(val_samples)

    print("Unique train images:", len(set(s["image_id"] for s in train_samples)))
    print("Unique val images:",   len(set(s["image_id"] for s in val_samples)))
    print("Train caption samples:", len(train_samples))
    print("Val caption samples:",   len(val_samples))

    # -------- DATASETS + LOADERS --------
    transform = build_image_transform()

    train_dataset = COCORetrievalDataset(train_samples, transform=transform)
    val_dataset   = COCORetrievalDataset(val_samples,   transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # -------- LOAD CACHED TEXT EMBEDDINGS --------
    train_cache = torch.load(os.path.join(CACHE_DIR, "train_text_embeds.pt"))
    val_cache   = torch.load(os.path.join(CACHE_DIR, "val_text_embeds.pt"))

    train_text_embs = train_cache["text_embeddings"].to(device)
    val_text_embs   = val_cache["text_embeddings"].to(device)

    # -------- MODEL + LOSS + OPTIMIZER --------
    image_encoder = ResNet50ImageEncoder(
        embed_dim=512,
        proj_hidden_dim=1024,
    ).to(device)

    criterion = CLIPLoss(temperature=TEMPERATURE)

    optimizer = torch.optim.AdamW(
        image_encoder.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # -------- TRAIN --------
    train_losses, val_losses, total_time = train_clip(
        image_encoder,
        train_loader,
        val_loader,
        train_text_embs,
        val_text_embs,
        criterion,
        optimizer,
        device,
        epochs=EPOCHS,
        print_every=50,
    )

    # -------- OUTPUT DIRECTORY (IN DRIVE PROJECT FOLDER) --------
    OUTPUT = "/content/drive/MyDrive/elec475_lab4"

    os.makedirs(OUTPUT, exist_ok=True)

    # -------- SAVE LOSS PLOT --------
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CLIP Training & Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/loss_curves.png")
    print("Saved outputs/loss_curves.png")

    # -------- SAVE MODEL --------
    torch.save(image_encoder.state_dict(), f"{OUTPUT}/image_encoder_clip_resnet50.pt")
    print("Saved outputs/image_encoder_clip_resnet50.pt")

if __name__ == "__main__":
    main()
