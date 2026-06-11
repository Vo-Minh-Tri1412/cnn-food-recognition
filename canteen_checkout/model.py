from __future__ import annotations

import torch
from torch import nn
from torchvision import models, transforms


def build_classifier(num_classes: int, pretrained: bool = True, arch: str = "mobilenet_v3_small") -> nn.Module:
    if arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported architecture: {arch}")


def train_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def eval_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def resolve_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(path, model, class_names, image_size: int, metadata: dict | None = None, arch: str = "mobilenet_v3_small") -> None:
    payload = {
        "state_dict": model.state_dict(),
        "class_names": list(class_names),
        "image_size": image_size,
        "arch": arch,
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_checkpoint(path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    arch = checkpoint.get("arch", "mobilenet_v3_small")
    model = build_classifier(num_classes=len(class_names), pretrained=False, arch=arch)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    image_size = int(checkpoint.get("image_size", 224))
    return model, class_names, image_size, checkpoint
