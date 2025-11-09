import os
import re
from typing import Callable, Optional, Tuple, List

from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


LABEL_LINE_RE = re.compile(r"^\s*([^,\s]+)\s*,\s*\"\((\d+)\,\s*(\d+)\)\"\s*$")


class OxfordPetNosesDataset(Dataset):
    """
    Custom Dataset for oxford-iiit-pet-noses.

    Each label line format:
        filename,"(u, v)"
    where (u, v) are pixel coordinates in the original image width/height space.

    This dataset will:
      - Load image from images_dir / filename
      - Resize image to (output_size, output_size)
      - Geometrically transform the coordinate (u, v) accordingly
      - Return (image_tensor: FloatTensor[3, H, W], target_uv: FloatTensor[2])

    Transforms:
      - Two-stage: geometric -> tensor/color/normalize
      - Provide an optional `geom_transform` callable that takes (img, (u, v)) -> (img, (u, v))
    """

    def __init__(
        self,
        images_dir: str,
        labels_file: str,
        output_size: int = 227,
        geom_transform: Optional[Callable[[Image.Image, Tuple[float, float]], Tuple[Image.Image, Tuple[float, float]]]] = None,
        to_tensor_normalize: Optional[Callable[[Image.Image], Tensor]] = None,
    ) -> None:
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"images_dir not found: {images_dir}")
        if not os.path.isfile(labels_file):
            raise FileNotFoundError(f"labels_file not found: {labels_file}")

        self.images_dir = images_dir
        self.labels_file = labels_file
        self.output_size = int(output_size)
        self.geom_transform = geom_transform
        self.to_tensor_normalize = to_tensor_normalize

        self.samples: List[Tuple[str, float, float]] = self._read_labels(labels_file)

    @staticmethod
    def _read_labels(labels_file: str) -> List[Tuple[str, float, float]]:
        samples: List[Tuple[str, float, float]] = []
        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                m = LABEL_LINE_RE.match(line)
                if not m:
                    # Try to be tolerant of potential spaces without quotes
                    # Fallback parsing: filename,(u, v)
                    try:
                        name_part, coord_part = line.split(",", 1)
                        coord_part = coord_part.strip().strip('"').strip()
                        coord_part = coord_part.strip("()")
                        u_str, v_str = coord_part.split(",")
                        filename = name_part.strip()
                        u = float(u_str)
                        v = float(v_str)
                        samples.append((filename, u, v))
                        continue
                    except Exception as e:
                        raise ValueError(f"Could not parse label line: {line}") from e
                filename = m.group(1)
                u = float(m.group(2))
                v = float(m.group(3))
                samples.append((filename, u, v))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _resize_and_scale_point(self, img: Image.Image, u: float, v: float) -> Tuple[Image.Image, Tuple[float, float]]:
        orig_w, orig_h = img.size
        img = img.resize((self.output_size, self.output_size), Image.BILINEAR)
        scale_u = self.output_size / float(orig_w)
        scale_v = self.output_size / float(orig_h)
        u2 = u * scale_u
        v2 = v * scale_v
        return img, (u2, v2)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        filename, u, v = self.samples[idx]
        img_path = os.path.join(self.images_dir, filename)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")

        # First: deterministic resize to model's input size and scale coordinates
        img, (u, v) = self._resize_and_scale_point(img, u, v)

        # Optional geometric augmentation (must also update (u,v))
        if self.geom_transform is not None:
            img, (u, v) = self.geom_transform(img, (u, v))

        # Convert to tensor + normalize
        if self.to_tensor_normalize is None:
            # Default: simple [0,1] scaling
            x = TF.to_tensor(img)  # FloatTensor in [0,1]
        else:
            x = self.to_tensor_normalize(img)

        target = torch.tensor([u, v], dtype=torch.float32)
        return x, target


