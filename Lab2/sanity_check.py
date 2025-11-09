import os
import random
from typing import Tuple

import torch
from PIL import Image, ImageDraw

from datamodule import get_datasets


ROOT = os.path.dirname(os.path.abspath(__file__))


def draw_point(img: Image.Image, u: float, v: float, color=(255, 0, 0)) -> Image.Image:
    r = 3
    draw = ImageDraw.Draw(img)
    draw.ellipse((u - r, v - r, u + r, v + r), fill=color)
    return img


def main():
    train_ds, test_ds = get_datasets(ROOT, augment=False, normalize_imagenet=False)

    print(f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")

    # Inspect a few random samples from train and test
    for name, ds in [("train", train_ds), ("test", test_ds)]:
        print(f"\n[{name}] Checking 3 samples:")
        for _ in range(3):
            idx = random.randrange(len(ds))
            x, uv = ds[idx]
            print(f"idx={idx}, tensor={tuple(x.shape)}, uv={uv.tolist()}")

    # Save a couple of visualizations to disk
    vis_dir = os.path.join(ROOT, "_vis_check")
    os.makedirs(vis_dir, exist_ok=True)

    for name, ds in [("train", train_ds), ("test", test_ds)]:
        for i in range(2):
            x, uv = ds[i]
            img = (x * 255.0).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            pil = Image.fromarray(img)
            pil = draw_point(pil, float(uv[0]), float(uv[1]))
            out_path = os.path.join(vis_dir, f"{name}_{i}.png")
            pil.save(out_path)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()


