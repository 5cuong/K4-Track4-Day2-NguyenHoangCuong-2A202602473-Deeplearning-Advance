"""Deterministic DeepWeeds split loading, validation, transforms, and batching."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

NUM_CLASSES = 9
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXPECTED_TOTAL = 17_509
SPLIT_FILES = {
    "train": "train_subset{fold}.csv",
    "val": "val_subset{fold}.csv",
    "test": "test_subset{fold}.csv",
}


def load_split(labels_dir: str | Path, fold: int = 0):
    """Read an author-provided fold unchanged and return train, validation, and test frames."""
    if fold not in range(5):
        raise ValueError("fold must be in 0..4; the required assignment fold is 0.")
    root = Path(labels_dir)
    frames = []
    for split, pattern in SPLIT_FILES.items():
        path = root / pattern.format(fold=fold)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {split} split CSV: {path}")
        frame = pd.read_csv(path)
        missing = {"Filename", "Label"} - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        frames.append(frame)
    return tuple(frames)


def _index_images(images_dir: Path) -> dict[str, list[Path]]:
    """Index image basenames once, allowing archives with an extra images/ prefix."""
    index: dict[str, list[Path]] = {}
    if images_dir.exists():
        for path in images_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                index.setdefault(path.name, []).append(path)
    return index


def _resolve_image(filename: str, images_dir: Path, index: dict[str, list[Path]]) -> Path | None:
    direct = images_dir / filename
    if direct.is_file():
        return direct
    basename = Path(filename).name
    matches = index.get(basename, [])
    return matches[0] if len(matches) == 1 else None


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict[str, Any]:
    """Enforce the published fold, no overlap, expected proportions, and image availability."""
    frames = {"train": train_df, "val": val_df, "test": test_df}
    root = Path(images_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")

    names: dict[str, set[str]] = {}
    per_class: dict[str, dict[str, int]] = {}
    for split, frame in frames.items():
        missing_cols = {"Filename", "Label"} - set(frame.columns)
        if missing_cols:
            raise ValueError(f"{split} is missing columns: {sorted(missing_cols)}")
        if frame["Filename"].isna().any() or frame["Label"].isna().any():
            raise ValueError(f"{split} contains missing filenames or labels.")
        filenames = frame["Filename"].astype(str)
        if filenames.duplicated().any():
            duplicates = filenames[filenames.duplicated()].head(5).tolist()
            raise ValueError(f"{split} contains duplicate filenames: {duplicates}")
        labels = pd.to_numeric(frame["Label"], errors="raise").astype(int)
        if not labels.between(0, NUM_CLASSES - 1).all():
            raise ValueError(f"{split} has labels outside 0..{NUM_CLASSES - 1}.")
        names[split] = set(filenames)
        counts = labels.value_counts().reindex(range(NUM_CLASSES), fill_value=0)
        per_class[split] = {
            CLASS_NAMES[i]: int(counts.loc[i]) for i in range(NUM_CLASSES)
        }

    overlap = {
        "train_val": sorted(names["train"] & names["val"]),
        "train_test": sorted(names["train"] & names["test"]),
        "val_test": sorted(names["val"] & names["test"]),
    }
    nonempty = {key: value[:10] for key, value in overlap.items() if value}
    if nonempty:
        raise ValueError(f"Split filename intersections must be empty: {nonempty}")

    sizes = {key: len(value) for key, value in names.items()}
    if sum(sizes.values()) != EXPECTED_TOTAL:
        raise ValueError(
            f"Fold must contain exactly {EXPECTED_TOTAL:,} unique images; got {sizes} "
            f"(sum={sum(sizes.values()):,})."
        )
    expected = {"train": 0.60, "val": 0.20, "test": 0.20}
    ratios = {key: value / EXPECTED_TOTAL for key, value in sizes.items()}
    deviations = {key: abs(ratios[key] - expected[key]) for key in expected}
    if any(value > 0.0101 for value in deviations.values()):
        raise ValueError(
            "Split proportions differ from the published 60/20/20 split by more than "
            f"about one percentage point: {ratios}."
        )

    image_index = _index_images(root)
    missing_images = []
    ambiguous = []
    for split, frame in frames.items():
        for filename in frame["Filename"].astype(str):
            matches = image_index.get(Path(filename).name, [])
            resolved = _resolve_image(filename, root, image_index)
            if resolved is None:
                (ambiguous if len(matches) > 1 else missing_images).append(filename)
    if missing_images or ambiguous:
        raise FileNotFoundError(
            f"Image lookup failed: missing={missing_images[:10]}, ambiguous={ambiguous[:10]}"
        )

    result = {
        "n": sizes,
        "ratios": ratios,
        "per_class": per_class,
        "overlap": {key: len(value) for key, value in overlap.items()},
        "total_unique": sum(sizes.values()),
        "images_dir": str(root.resolve()),
    }
    print("Split sizes:", result["n"])
    print("Split proportions:", {k: round(v, 4) for k, v in ratios.items()})
    print("Per-class counts:", result["per_class"])
    print("Overlap counts:", result["overlap"])
    print("Unique total:", result["total_unique"])
    return result


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Build train augmentation or deterministic validation preprocessing."""
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    if img_size < 32:
        raise ValueError("img_size must be at least 32 pixels.")
    normalization = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if not train:
        return transforms.Compose([
            transforms.Resize(img_size + 32, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalization,
        ])

    if aug not in {"basic", "color", "trivial", "randaug"}:
        raise ValueError(f"Unknown augmentation recipe: {aug}")
    operations = [
        transforms.RandomResizedCrop(
            img_size, scale=(0.70, 1.0), interpolation=InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if aug == "color":
        operations.append(transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03
        ))
    elif aug == "trivial":
        operations.append(transforms.TrivialAugmentWide(
            interpolation=InterpolationMode.BICUBIC
        ))
    elif aug == "randaug":
        operations.append(transforms.RandAugment(
            num_ops=2, magnitude=7, interpolation=InterpolationMode.BICUBIC
        ))
    operations.extend([transforms.ToTensor(), normalization])
    return transforms.Compose(operations)


class DeepWeedsDataset(Dataset):
    """Image dataset returning (normalized image, integer label, original filename)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        required = {"Filename", "Label"}
        if required - set(df.columns):
            raise ValueError(f"DataFrame must contain {sorted(required)}.")
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self._image_index = _index_images(self.images_dir)
        self._paths = []
        for filename in self.df["Filename"].astype(str):
            path = _resolve_image(filename, self.images_dir, self._image_index)
            if path is None:
                raise FileNotFoundError(f"Could not uniquely resolve image {filename!r}.")
            self._paths.append(path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        with Image.open(self._paths[i]) as image:
            image = image.convert("RGB")
            tensor = self.transform(image) if self.transform is not None else image
        return tensor, int(row["Label"]), str(row["Filename"])


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    """Create a reproducible DataLoader; evaluation loaders retain CSV order."""
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative.")
    if sampler not in {None, "balanced"}:
        raise ValueError("sampler must be None or 'balanced'.")
    if sampler is not None and not train:
        raise ValueError("A sampler may only be used for the training split.")
    dataset = DeepWeedsDataset(df, images_dir, transform)
    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed() % (2**63 - 1))

    sampler_obj = None
    shuffle = bool(train)
    if sampler == "balanced":
        labels = pd.to_numeric(df["Label"], errors="raise").astype(int).to_numpy()
        counts = pd.Series(labels).value_counts().to_dict()
        weights = torch.as_tensor([1.0 / counts[int(y)] for y in labels], dtype=torch.double)
        sampler_obj = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True, generator=generator
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler_obj is None else False,
        sampler=sampler_obj,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=bool(train and len(dataset) >= batch_size),
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
