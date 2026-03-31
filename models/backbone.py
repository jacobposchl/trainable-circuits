"""
Frozen backbone: pretrained network with dual hooks capturing the activation
trajectory T(x) and flow targets F(x) for Phase 1 training.

Design:
  - Trajectory T(x) = (h_1, ..., h_L): compressed block outputs (post-relu,
    post-skip addition). Input to the MetaEncoder.
  - Flow targets F(x) = (f_1, ..., f_L): compressed bn2/bn3 outputs (pre-skip,
    pre-relu). Capture the pure block contribution isolated from accumulated
    history. Reconstruction targets for InfoLoss.
  - Compression: AdaptiveMaxPool2d(G, G) -> Flatten -> Linear(C*G*G, D_flow).
    Grid-based max pooling preserves spatially-local peak activations where the
    block is doing something circuit-relevant, rather than averaging them away.
    Compression weights are fixed and initialized with a fixed seed for
    reproducibility across runs.
  - layer_dims = [D_flow] * L is exposed so MetaEncoder builds projectors of
    manageable size regardless of architecture.
  - ViT: CLS token used for trajectory; flow_targets mirrors trajectory
    (no bn2 split available for transformer blocks).

Supported architectures:
  ResNet family: resnet18, resnet34, resnet50, resnet101
  ViT family:    vit_b_16, vit_s_16 (requires timm)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class FrozenBackbone(nn.Module):
    def __init__(
        self,
        arch: str,
        num_classes: int,
        pretrained: bool = True,
        grid_size: int = 4,
        flow_dim: int = 256,
        trainable_stem: bool = False,
    ):
        """
        Args:
            arch: Architecture name (resnet18, resnet34, resnet50, vit_b_16, ...)
            num_classes: Number of output classes.
            pretrained: Whether to load pretrained ImageNet weights.
            grid_size: Grid size for AdaptiveMaxPool2d(G, G).
            flow_dim: Output dimension of each compression module.
            trainable_stem: For ResNets, keep only the CIFAR-adapted stem conv
                trainable while the rest of the backbone remains frozen.
        """
        super().__init__()

        self._hook_handles: list = []
        self._trajectory: list[torch.Tensor] = []
        self._flow_targets: list[torch.Tensor] = []
        self._grid_size = grid_size
        self._flow_dim = flow_dim
        self._is_vit = False
        self._trainable_stem = bool(trainable_stem and arch.startswith("resnet"))

        if arch.startswith("resnet"):
            self.model, self._block_modules, self._bn2_modules = _build_resnet(
                arch,
                num_classes,
                pretrained,
            )
        elif arch.startswith("vit"):
            self.model, self._block_modules = _build_vit(arch, num_classes, pretrained)
            self._bn2_modules = None
            self._is_vit = True
        else:
            raise ValueError(f"Unsupported architecture: {arch}. Use resnet* or vit*.")

        self._build_compression_modules()
        self._register_hooks()

        self.layer_dims: list[int] = [flow_dim] * len(self._block_modules)

        self.requires_grad_(False)
        if self._trainable_stem:
            self.model.conv1.requires_grad_(True)
        self.eval()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Run backbone forward. Populates self._trajectory and self._flow_targets.

        Returns:
            trajectory: list of L tensors, each [B, D_flow]
        """
        self._trajectory = []
        self._flow_targets = []
        with torch.set_grad_enabled(self._trainable_stem and torch.is_grad_enabled()):
            self.model(x)
        if self._is_vit:
            self._flow_targets = [tensor.detach() for tensor in self._trajectory]
        return list(self._trajectory)

    def train(self, mode: bool = True):
        """Force eval mode permanently, regardless of the requested mode."""
        return super().train(False)

    def _build_compression_modules(self):
        """
        Discover intermediate shapes with temporary hooks, then build fixed
        per-layer compression modules.
        """
        traj_shapes: list[tuple] = []
        flow_shapes: list[tuple] = []
        tmp_handles: list = []

        def _capture(shapes: list):
            def hook(module, inp, out):
                tensor = out[0] if isinstance(out, (tuple, list)) else out
                shapes.append(tuple(tensor.shape))
            return hook

        for module in self._block_modules:
            tmp_handles.append(module.register_forward_hook(_capture(traj_shapes)))

        if not self._is_vit and self._bn2_modules:
            for module in self._bn2_modules:
                tmp_handles.append(module.register_forward_hook(_capture(flow_shapes)))

        dummy = torch.zeros(1, 3, 32, 32)
        with torch.no_grad():
            self.model(dummy)

        for handle in tmp_handles:
            handle.remove()

        grid = self._grid_size
        out_dim = self._flow_dim

        saved_rng = torch.get_rng_state()
        torch.manual_seed(42)

        def _make_spatial_compressor(shape: tuple) -> nn.Module:
            channels = shape[1]
            return nn.Sequential(
                nn.AdaptiveMaxPool2d((grid, grid)),
                nn.Flatten(),
                nn.Linear(channels * grid * grid, out_dim, bias=False),
            )

        def _make_token_compressor(shape: tuple) -> nn.Module:
            width = shape[2]
            return nn.Linear(width, out_dim, bias=False)

        traj_mods: list[nn.Module] = []
        for shape in traj_shapes:
            if len(shape) == 4:
                traj_mods.append(_make_spatial_compressor(shape))
            elif len(shape) == 3:
                traj_mods.append(_make_token_compressor(shape))
            else:
                traj_mods.append(nn.Linear(shape[-1], out_dim, bias=False))

        self.traj_compressors = nn.ModuleList(traj_mods)
        self.traj_compressors.requires_grad_(False)

        if not self._is_vit and flow_shapes:
            flow_mods: list[nn.Module] = []
            for shape in flow_shapes:
                if len(shape) == 4:
                    flow_mods.append(_make_spatial_compressor(shape))
                else:
                    flow_mods.append(nn.Linear(shape[-1], out_dim, bias=False))
            self.flow_compressors = nn.ModuleList(flow_mods)
            self.flow_compressors.requires_grad_(False)
        else:
            self.flow_compressors = self.traj_compressors

        torch.set_rng_state(saved_rng)

    def _register_hooks(self):
        for idx, module in enumerate(self._block_modules):
            self._hook_handles.append(module.register_forward_hook(self._make_traj_hook(idx)))

        if not self._is_vit and self._bn2_modules:
            for idx, module in enumerate(self._bn2_modules):
                self._hook_handles.append(module.register_forward_hook(self._make_flow_hook(idx)))

    def _make_traj_hook(self, idx: int):
        def hook(module, inp, out):
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            if tensor.dim() == 3:
                tensor = tensor[:, 0]
            compressed = self.traj_compressors[idx](tensor)
            normalized = F.normalize(compressed, dim=-1)
            if not (self._trainable_stem and torch.is_grad_enabled()):
                normalized = normalized.detach()
            self._trajectory.append(normalized)
        return hook

    def _make_flow_hook(self, idx: int):
        def hook(module, inp, out):
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            compressed = self.flow_compressors[idx](tensor.detach())
            self._flow_targets.append(F.normalize(compressed, dim=-1).detach())
        return hook


def _resize_resnet_stem_weights(weight: torch.Tensor) -> torch.Tensor:
    """Resize pretrained 7x7 conv filters to a 3x3 CIFAR stem."""
    resized = F.interpolate(weight, size=(3, 3), mode="bilinear", align_corners=False)
    original_norm = weight.flatten(1).norm(dim=1, keepdim=True)
    resized_norm = resized.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-8)
    return resized * (original_norm / resized_norm).view(-1, 1, 1, 1)


def _build_resnet(
    arch: str,
    num_classes: int,
    pretrained: bool,
) -> tuple[nn.Module, list[nn.Module], list[nn.Module]]:
    weights = "IMAGENET1K_V1" if pretrained else None
    model: nn.Module = getattr(tvm, arch)(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    original_conv1 = model.conv1
    stem = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    if pretrained:
        with torch.no_grad():
            stem.weight.copy_(_resize_resnet_stem_weights(original_conv1.weight.data))
    model.conv1 = stem
    model.maxpool = nn.Identity()

    block_modules: list[nn.Module] = []
    bn_modules: list[nn.Module] = []
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, layer_name)
        for block in layer:
            block_modules.append(block)
            bn_modules.append(block.bn3 if hasattr(block, "bn3") else block.bn2)

    return model, block_modules, bn_modules


def _build_vit(
    arch: str,
    num_classes: int,
    pretrained: bool,
) -> tuple[nn.Module, list[nn.Module]]:
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "timm is required for ViT architectures. Install with: pip install timm"
        ) from exc

    model = timm.create_model(
        arch,
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=32,
    )
    return model, list(model.blocks)
