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
    Compression weights are FIXED (no_grad) and initialized with a fixed seed
    for reproducibility across runs.
  - Both T and F are L2-normalized after compression; all are detached.
  - layer_dims = [D_flow] * L is exposed so MetaEncoder builds projectors of
    manageable size (Linear(D_flow, projection_dim)) regardless of arch.
  - ViT: CLS token used for trajectory; flow_targets mirrors trajectory
    (no bn2 split available for transformer blocks).

Supported architectures (via torchvision):
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
    ):
        """
        Args:
            arch:       Architecture name (resnet18, resnet34, resnet50, vit_b_16, …)
            num_classes: Number of output classes (needed for model construction).
            pretrained: Whether to load pretrained ImageNet weights.
            grid_size:  G for AdaptiveMaxPool2d(G, G). Controls spatial resolution
                        of the compressed flow/trajectory vectors.
            flow_dim:   D_flow — output dimension of each compression module.
                        Becomes layer_dims[l] for all l.
        """
        super().__init__()

        self._hook_handles: list = []
        self._trajectory:   list[torch.Tensor] = []
        self._flow_targets: list[torch.Tensor] = []
        self._grid_size = grid_size
        self._flow_dim  = flow_dim
        self._is_vit    = False

        if arch.startswith("resnet"):
            self.model, self._block_modules, self._bn2_modules = (
                _build_resnet(arch, num_classes, pretrained)
            )
        elif arch.startswith("vit"):
            self.model, self._block_modules = _build_vit(arch, num_classes, pretrained)
            self._bn2_modules = None
            self._is_vit = True
        else:
            raise ValueError(
                f"Unsupported architecture: {arch}. Use resnet* or vit*."
            )

        # Two-pass: discover shapes, build fixed compression modules
        self._build_compression_modules()

        # Register the permanent dual hooks
        self._register_hooks()

        # Uniform layer_dims for MetaEncoder projectors
        self.layer_dims: list[int] = [flow_dim] * len(self._block_modules)

        # Freeze everything — backbone + compression modules are read-only
        self.requires_grad_(False)
        self.eval()

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Run backbone forward.  Populates self._trajectory and self._flow_targets.

        Returns:
            trajectory: list of L tensors, each [B, D_flow], L2-normalized
        """
        self._trajectory   = []
        self._flow_targets = []
        with torch.no_grad():
            self.model(x)
        # ViT: no bn2 split, so flow targets mirror trajectory
        if self._is_vit:
            self._flow_targets = list(self._trajectory)
        return list(self._trajectory)

    def train(self, mode: bool = True):
        """No-op: backbone stays in eval mode permanently."""
        return self

    # ------------------------------------------------------------------ #
    # Internal: compression module construction (two-pass)
    # ------------------------------------------------------------------ #

    def _build_compression_modules(self):
        """
        Step 1: register lightweight shape-capture hooks and run a dummy
                forward to learn C_l, H_l, W_l for every block.
        Step 2: build per-layer AdaptiveMaxPool2d + Flatten + Linear modules
                from the observed shapes.
        Step 3: remove the temporary hooks.

        Uses a fixed RNG seed so compression weights are identical across runs.
        """
        traj_shapes: list[tuple] = []
        flow_shapes: list[tuple] = []
        tmp_handles: list       = []

        def _capture(shapes: list):
            def hook(module, inp, out):
                t = out[0] if isinstance(out, (tuple, list)) else out
                shapes.append(tuple(t.shape))
            return hook

        for m in self._block_modules:
            tmp_handles.append(m.register_forward_hook(_capture(traj_shapes)))

        if not self._is_vit and self._bn2_modules:
            for m in self._bn2_modules:
                tmp_handles.append(m.register_forward_hook(_capture(flow_shapes)))

        dummy = torch.zeros(1, 3, 32, 32)
        with torch.no_grad():
            self.model(dummy)

        for h in tmp_handles:
            h.remove()

        G      = self._grid_size
        D_flow = self._flow_dim

        # Fixed seed → same random linear projection every instantiation
        saved_rng = torch.get_rng_state()
        torch.manual_seed(42)

        def _make_spatial_compressor(shape: tuple) -> nn.Module:
            """Build compressor for a 4-D [B, C, H, W] tensor."""
            C = shape[1]
            return nn.Sequential(
                nn.AdaptiveMaxPool2d((G, G)),
                nn.Flatten(),
                nn.Linear(C * G * G, D_flow, bias=False),
            )

        def _make_token_compressor(shape: tuple) -> nn.Module:
            """Build compressor for a ViT 3-D [B, N+1, D] tensor (CLS extracted in hook)."""
            D = shape[2]
            return nn.Linear(D, D_flow, bias=False)

        traj_mods: list[nn.Module] = []
        for shape in traj_shapes:
            if len(shape) == 4:
                traj_mods.append(_make_spatial_compressor(shape))
            elif len(shape) == 3:
                traj_mods.append(_make_token_compressor(shape))
            else:                    # already [B, D]
                D = shape[-1]
                traj_mods.append(nn.Linear(D, D_flow, bias=False))

        self.traj_compressors = nn.ModuleList(traj_mods)
        self.traj_compressors.requires_grad_(False)

        if not self._is_vit and flow_shapes:
            flow_mods: list[nn.Module] = []
            for shape in flow_shapes:
                if len(shape) == 4:
                    flow_mods.append(_make_spatial_compressor(shape))
                else:
                    D = shape[-1]
                    flow_mods.append(nn.Linear(D, D_flow, bias=False))
            self.flow_compressors = nn.ModuleList(flow_mods)
            self.flow_compressors.requires_grad_(False)
        else:
            # ViT: reuse the same compressors (flow_targets will equal trajectory)
            self.flow_compressors = self.traj_compressors

        torch.set_rng_state(saved_rng)

    # ------------------------------------------------------------------ #
    # Internal: permanent hook registration
    # ------------------------------------------------------------------ #

    def _register_hooks(self):
        # Trajectory: block outputs → compress → L2-norm → _trajectory
        for idx, module in enumerate(self._block_modules):
            handle = module.register_forward_hook(self._make_traj_hook(idx))
            self._hook_handles.append(handle)

        # Flow targets: bn2/bn3 outputs → compress → L2-norm → _flow_targets
        if not self._is_vit and self._bn2_modules:
            for idx, module in enumerate(self._bn2_modules):
                handle = module.register_forward_hook(self._make_flow_hook(idx))
                self._hook_handles.append(handle)

    def _make_traj_hook(self, idx: int):
        def hook(module, inp, out):
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            with torch.no_grad():
                if tensor.dim() == 3:   # ViT: [B, N+1, D] → CLS token
                    tensor = tensor[:, 0]
                compressed = self.traj_compressors[idx](tensor)
                self._trajectory.append(F.normalize(compressed, dim=-1).detach())
        return hook

    def _make_flow_hook(self, idx: int):
        def hook(module, inp, out):
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            with torch.no_grad():
                compressed = self.flow_compressors[idx](tensor)
                self._flow_targets.append(F.normalize(compressed, dim=-1).detach())
        return hook


# --------------------------------------------------------------------------- #
# Architecture builders
# --------------------------------------------------------------------------- #

def _build_resnet(
    arch: str, num_classes: int, pretrained: bool
) -> tuple[nn.Module, list[nn.Module], list[nn.Module]]:
    weights = "IMAGENET1K_V1" if pretrained else None
    model: nn.Module = getattr(tvm, arch)(weights=weights)

    # Replace classifier head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # CIFAR-10 (32×32): replace aggressive stem with a small 3×3 conv
    model.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    block_modules: list[nn.Module] = []
    bn2_modules:   list[nn.Module] = []

    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, layer_name)
        for block in layer:
            block_modules.append(block)
            # Last BN in the main branch, before the skip addition:
            #   BasicBlock  → bn2
            #   Bottleneck  → bn3
            if hasattr(block, "bn3"):
                bn2_modules.append(block.bn3)
            else:
                bn2_modules.append(block.bn2)

    return model, block_modules, bn2_modules


def _build_vit(
    arch: str, num_classes: int, pretrained: bool
) -> tuple[nn.Module, list[nn.Module]]:
    try:
        import timm
        model = timm.create_model(
            arch, pretrained=pretrained, num_classes=num_classes, img_size=32
        )
        hook_targets = list(model.blocks)
    except ImportError:
        raise ImportError(
            "timm is required for ViT architectures. "
            "Install with: pip install timm"
        )
    return model, hook_targets
