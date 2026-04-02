from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


def _resize_stem(weight: torch.Tensor) -> torch.Tensor:
    resized = F.interpolate(weight, size=(3, 3), mode="bilinear", align_corners=False)
    old_norm = weight.flatten(1).norm(dim=1, keepdim=True)
    new_norm = resized.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-8)
    return resized * (old_norm / new_norm).view(-1, 1, 1, 1)


@dataclass
class ResNetObservations:
    states: list[torch.Tensor]
    residuals: list[torch.Tensor]
    layer_channels: list[int]
    layer_names: list[str]
    grid_size: int

    @property
    def n_layers(self) -> int:
        return len(self.states)


class FrozenResNetObserver(nn.Module):
    """Frozen ResNet observer with explicit state and residual hooks."""

    def __init__(
        self,
        arch: str = "resnet18",
        pretrained: bool = True,
        num_classes: int = 10,
        grid_size: int = 4,
        weights_path: str | None = None,
    ) -> None:
        super().__init__()
        if arch not in {"resnet18", "resnet34", "resnet50"}:
            raise ValueError(f"Unsupported architecture: {arch}")

        self.arch = arch
        self.grid_size = grid_size
        self.model, self.blocks, self.flow_modules, self.layer_names = self._build_model(
            arch=arch,
            pretrained=pretrained,
            num_classes=num_classes,
            weights_path=weights_path,
        )
        self.layer_channels = [self._block_out_channels(block) for block in self.blocks]
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)

    def observe(self, x: torch.Tensor) -> ResNetObservations:
        states: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []
        handles: list[torch.utils.hooks.RemovableHandle] = []

        def make_state_hook():
            def hook(module, inputs, output):
                tensor = output[0] if isinstance(output, (tuple, list)) else output
                states.append(tensor.detach())
            return hook

        def make_flow_hook():
            def hook(module, inputs, output):
                tensor = output[0] if isinstance(output, (tuple, list)) else output
                residuals.append(tensor.detach())
            return hook

        for block in self.blocks:
            handles.append(block.register_forward_hook(make_state_hook()))
        for module in self.flow_modules:
            handles.append(module.register_forward_hook(make_flow_hook()))

        with torch.no_grad():
            self.model(x)

        for handle in handles:
            handle.remove()

        return ResNetObservations(
            states=states,
            residuals=residuals,
            layer_channels=self.layer_channels,
            layer_names=self.layer_names,
            grid_size=self.grid_size,
        )

    def forward(self, x: torch.Tensor) -> ResNetObservations:
        return self.observe(x)

    def _build_model(
        self,
        arch: str,
        pretrained: bool,
        num_classes: int,
        weights_path: str | None,
    ) -> tuple[nn.Module, list[nn.Module], list[nn.Module], list[str]]:
        weights = "IMAGENET1K_V1" if pretrained else None
        model: nn.Module = getattr(tvm, arch)(weights=weights)

        if weights_path:
            state_dict = torch.load(Path(weights_path), map_location="cpu", weights_only=False)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if unexpected:
                raise ValueError(f"Unexpected keys in weights_path: {unexpected}")
            if missing:
                # Allow classifier replacement below to leave fc keys missing.
                missing = [key for key in missing if not key.startswith("fc.")]
                if missing:
                    raise ValueError(f"Missing keys in weights_path: {missing}")

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        stem = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if pretrained:
            with torch.no_grad():
                stem.weight.copy_(_resize_stem(model.conv1.weight.data))
        model.conv1 = stem
        model.maxpool = nn.Identity()

        blocks: list[nn.Module] = []
        flow_modules: list[nn.Module] = []
        layer_names: list[str] = []
        for group_idx, layer_name in enumerate(["layer1", "layer2", "layer3", "layer4"], start=1):
            layer = getattr(model, layer_name)
            for block_idx, block in enumerate(layer, start=1):
                blocks.append(block)
                flow_modules.append(block.bn3 if hasattr(block, "bn3") else block.bn2)
                layer_names.append(f"group{group_idx}.block{block_idx}")
        return model, blocks, flow_modules, layer_names

    @staticmethod
    def _block_out_channels(block: nn.Module) -> int:
        if hasattr(block, "bn3"):
            return int(block.bn3.num_features)
        return int(block.bn2.num_features)
