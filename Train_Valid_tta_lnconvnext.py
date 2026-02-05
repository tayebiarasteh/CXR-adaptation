"""
Created on Feb 5, 2026.
Train_Valid_tta_lnconvnext.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/

Test-time adaptation for HF DINOv3-ConvNeXt (and similar AutoModel backbones) using:
- Consistency between two light augmentations (radiology-safe)
- An anchor (teacher) loss to prevent drifting away from the source model
- A source-prior constraint (label marginal matching) enforced with a dual variable
"""

import os
import time
import copy
import torch
from torchvision import transforms

from config.serde import read_config

_EPS = 1e-6


def _consistency_mse(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """Mean squared error between two probability tensors (B,C)."""
    return torch.mean((p1 - p2) ** 2)


def _marginal_mse(p_batch: torch.Tensor, p_target: torch.Tensor) -> torch.Tensor:
    """MSE between batch marginal (C,) and target marginal (C,)."""
    p_bar = p_batch.mean(dim=0)
    return torch.mean((p_bar - p_target) ** 2)


def _is_layernorm_module(m: torch.nn.Module) -> bool:
    """
    DINOv3ConvNext uses custom LN: DINOv3ConvNextLayerNorm
    plus a final torch.nn.LayerNorm((768,))
    """
    if isinstance(m, torch.nn.LayerNorm):
        return True
    name = m.__class__.__name__.lower()
    return ("layernorm" in name) or (name.endswith("layer_norm"))


def _set_requires_grad(model: torch.nn.Module, flag: bool):
    for p in model.parameters():
        p.requires_grad = flag


def _unfreeze_layernorm_affine(model: torch.nn.Module):
    for m in model.modules():
        if _is_layernorm_module(m):
            for p in m.parameters():
                p.requires_grad = True


def _unfreeze_head(model: torch.nn.Module):
    if hasattr(model, "head") and isinstance(model.head, torch.nn.Module):
        for p in model.head.parameters():
            p.requires_grad = True


class TTA_Adaptation:
    """
    Practical TTA for your setting:

    - Updates only LayerNorm affine parameters by default (adapt_ln=True).
      (Optionally also update the classification head.)
    - Uses only *unlabeled* target TRAIN split for adaptation.
    - Evaluates on target TEST split afterwards (handled in main).

    Key difference vs before:
      *No entropy minimization.*
      We instead (i) keep predictions stable w.r.t. small perturbations,
      (ii) keep the adapted model close to the source model (anchor),
      (iii) prevent marginal collapse by matching source label priors.
    """

    def __init__(self, cfg_path, label_names=None):
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.label_names = label_names
        self.setup_cuda()

        # Radiology-safe TTA aug on *tensor* images:
        # - no horizontal flip
        # - small rotation/translation
        # - no color jitter (can break intensity semantics)
        self.tta_aug = transforms.Compose([
            transforms.RandomAffine(
                degrees=7,
                translate=(0.02, 0.02),
                scale=(0.98, 1.02)
            ),
        ])

    def setup_cuda(self, cuda_device_id=0):
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    @torch.no_grad()
    def _apply_aug_batch(self, x: torch.Tensor) -> torch.Tensor:
        xs = []
        for i in range(x.size(0)):
            xs.append(self.tta_aug(x[i]))
        return torch.stack(xs, dim=0)

    def _forward_logits(self, model, x: torch.Tensor) -> torch.Tensor:
        out = model(x)
        logits = model.head(out.pooler_output)  # HF DINOv3 ConvNeXt pattern
        return logits

    def adapt(
        self,
        model,
        adapt_loader,
        save_name: str,
        # optimizer
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        # losses
        w_consistency: float = 1.0,
        w_anchor: float = 1.0,
        # source-prior constraint (dual)
        source_priors=None,          # list/np/torch, shape (C,)
        eps_prior: float = 0.002,    # allowed marginal mismatch
        dual_lr: float = 0.05,
        max_lambda: float = 100.0,
        # adaptation knobs
        steps_per_batch: int = 1,
        adapt_ln: bool = True,
        adapt_head: bool = False,
        # logging
        log_every: int = 50,
    ):
        """
        Adapt model on unlabeled target data.

        Parameters
        ----------
        source_priors:
            Empirical label prevalence on SOURCE training data (MIMIC),
            in the SAME order as label_names, values in [0,1].
            Compute once from your MIMIC train CSV.
        eps_prior:
            Constraint threshold on marginal MSE.
        """
        if source_priors is None:
            raise ValueError("source_priors must be provided (shape: (num_classes,)).")

        model = model.to(self.device)
        model.train()

        # Frozen teacher copy for anchoring (prevents drift / collapse)
        teacher = copy.deepcopy(model).to(self.device)
        teacher.eval()
        _set_requires_grad(teacher, False)

        # Freeze all, then unfreeze chosen params in student
        _set_requires_grad(model, False)
        if adapt_ln:
            _unfreeze_layernorm_affine(model)
        if adapt_head:
            _unfreeze_head(model)

        trainable = [p for p in model.parameters() if p.requires_grad]
        if len(trainable) == 0:
            raise RuntimeError("No trainable parameters selected. Check adapt_ln/adapt_head and LN detection.")

        opt = torch.optim.AdamW(
            trainable,
            lr=float(lr),
            weight_decay=float(weight_decay),
            amsgrad=False
        )

        # Dual variable for marginal constraint
        lam = torch.tensor(0.0, device=self.device)

        # Priors to tensor
        if not torch.is_tensor(source_priors):
            source_priors = torch.tensor(source_priors, dtype=torch.float32)
        source_priors = source_priors.to(self.device).clamp(0.0, 1.0)

        start = time.time()
        for step, (img, _) in enumerate(adapt_loader):
            img = img.to(self.device)

            # Two augmented views for consistency
            with torch.no_grad():
                x1 = self._apply_aug_batch(img)
                x2 = self._apply_aug_batch(img)

                # Teacher prediction on the unaugmented image for anchor
                t_logits = self._forward_logits(teacher, img)
                t_prob = torch.sigmoid(t_logits).clamp(_EPS, 1.0 - _EPS)

            for _ in range(steps_per_batch):
                opt.zero_grad(set_to_none=True)

                # Student predictions on aug views
                s_logits1 = self._forward_logits(model, x1)
                s_logits2 = self._forward_logits(model, x2)
                p1 = torch.sigmoid(s_logits1).clamp(_EPS, 1.0 - _EPS)
                p2 = torch.sigmoid(s_logits2).clamp(_EPS, 1.0 - _EPS)

                # Student prediction on unaugmented image for anchor matching
                s_logits0 = self._forward_logits(model, img)
                p0 = torch.sigmoid(s_logits0).clamp(_EPS, 1.0 - _EPS)

                loss_cons = _consistency_mse(p1, p2)
                loss_anchor = torch.mean((p0 - t_prob) ** 2)

                # Marginal constraint on mean across the two views
                p_stack = torch.cat([p1, p2], dim=0)  # (2B,C)
                loss_prior = _marginal_mse(p_stack, source_priors)

                # Lagrangian objective
                loss = (w_consistency * loss_cons) + (w_anchor * loss_anchor) + lam * (loss_prior - eps_prior)
                loss.backward()
                opt.step()

                # Dual ascent (projected)
                with torch.no_grad():
                    lam = lam + float(dual_lr) * (loss_prior.detach() - eps_prior)
                    lam = torch.clamp(lam, 0.0, float(max_lambda))

            if (step % log_every) == 0:
                elapsed = time.time() - start
                print(
                    f"[TTA] step {step}/{len(adapt_loader)} | "
                    f"cons={loss_cons.item():.6f} | anchor={loss_anchor.item():.6f} | "
                    f"prior={loss_prior.item():.6f} (eps={eps_prior}) | lam={lam.item():.3f} | "
                    f"time={elapsed:.1f}s"
                )

        # Save adapted weights
        os.makedirs(os.path.join(self.params["target_dir"], self.params["network_output_path"]), exist_ok=True)
        out_path = os.path.join(self.params["target_dir"], self.params["network_output_path"], save_name)
        torch.save(model.state_dict(), out_path)
        print(f"[TTA] saved adapted model: {out_path}")

        return model
