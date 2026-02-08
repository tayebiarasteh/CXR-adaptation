"""
Created on Feb 5, 2026.
Train_Valid_tta_lnconvnext.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/

Test-time adaptation for HF DINOv3-ConvNeXt (and similar AutoModel backbones) using:
- Consistency between two light augmentations (radiology-safe)
- A teacher anchor loss to prevent drifting away from the source model
- A teacher-anchored marginal stability loss (batch marginal matching to teacher on same stream)

Notes:
- No entropy minimization
- No source priors
- Updates only LayerNorm affine params by default (optionally head)
- Supports hard max_steps and optional automatic early stopping based on loss plateaus
"""

import os
import time
import copy
import torch
from torchvision import transforms

from config.serde import read_config

_EPS = 1e-6


def _is_layernorm_module(m: torch.nn.Module) -> bool:
    # catches: torch.nn.LayerNorm, DINOv3ConvNextLayerNorm, ConvNextLayerNorm, etc.
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


def _consistency_mse(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    return torch.mean((p1 - p2) ** 2)


def _batch_marginal_mse(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    p, q: (N,C) probabilities
    compares per-class means across the batch (marginals),
    anchored to TEACHER on the same target stream
    """
    mp = p.mean(dim=0)
    mq = q.mean(dim=0)
    return torch.mean((mp - mq) ** 2)


class TTA_Adaptation:
    """
    Practical TTA for your setting:

    - Updates only LayerNorm affine parameters by default (adapt_ln=True).
      Optionally also update the classification head.
    - Uses only unlabeled target TRAIN split for adaptation.
    - Evaluates on target TEST split afterwards (handled in main).

    Loss:
      (i)  consistency under small aug
      (ii) teacher anchor on clean image
      (iii) teacher-anchored marginal stability
    """

    def __init__(self, cfg_path, label_names=None):
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.label_names = label_names
        self.setup_cuda()

        # Radiology-safe aug on tensor images:
        # no horizontal flip, mild geometric jitter
        self.tta_aug = transforms.Compose([
            transforms.RandomAffine(
                degrees=10,
                translate=(0.04, 0.04),
                scale=(0.95, 1.05),
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
        xs = [self.tta_aug(x[i]) for i in range(x.size(0))]
        return torch.stack(xs, dim=0)

    def _forward_logits(self, model, x: torch.Tensor) -> torch.Tensor:
        out = model(x)
        return model.head(out.pooler_output)  # HF DINOv3 ConvNeXt pattern

    def adapt(
        self,
        model,
        adapt_loader,
        save_name: str,
        # optimizer
        lr: float = 5e-6,
        weight_decay: float = 0.0,
        # losses
        w_consistency: float = 1.0,
        w_anchor: float = 0.5,
        w_marginal: float = 0.5,
        # adaptation knobs
        steps_per_batch: int = 1,
        adapt_ln: bool = True,
        adapt_head: bool = False,
        max_steps: int | None = 300,
        # logging
        log_every: int = 50,
        # automatic early stop (optional)
        auto_stop: bool = True,
        ema_beta: float = 0.9,
        min_delta: float = 1e-5,
        patience_logs: int = 5,
    ):
        """
        Adapt model on unlabeled target data.

        max_steps:
          Hard cap on adaptation iterations (None = full loader).

        auto_stop:
          Stops early if EMA of total loss does not improve by min_delta for patience_logs prints.
        """
        model = model.to(self.device)
        model.train()

        # Frozen teacher copy for anchoring
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
            amsgrad=False,
        )

        start = time.time()

        # early stop trackers
        ema_loss = None
        best_ema = None
        bad_logs = 0

        denom = int(max_steps) if max_steps is not None else len(adapt_loader)

        for step, (img, _) in enumerate(adapt_loader):
            if (max_steps is not None) and (step >= int(max_steps)):
                break

            img = img.to(self.device)

            with torch.no_grad():
                x1 = self._apply_aug_batch(img)
                x2 = self._apply_aug_batch(img)

                t_logits0 = self._forward_logits(teacher, img)
                t_prob0 = torch.sigmoid(t_logits0).clamp(_EPS, 1.0 - _EPS)

                # teacher on augmented views for marginal anchoring
                t_logits1 = self._forward_logits(teacher, x1)
                t_logits2 = self._forward_logits(teacher, x2)
                t_prob12 = torch.sigmoid(torch.cat([t_logits1, t_logits2], dim=0)).clamp(_EPS, 1.0 - _EPS)

            for _ in range(int(steps_per_batch)):
                opt.zero_grad(set_to_none=True)

                s_logits1 = self._forward_logits(model, x1)
                s_logits2 = self._forward_logits(model, x2)
                p1 = torch.sigmoid(s_logits1).clamp(_EPS, 1.0 - _EPS)
                p2 = torch.sigmoid(s_logits2).clamp(_EPS, 1.0 - _EPS)

                s_logits0 = self._forward_logits(model, img)
                p0 = torch.sigmoid(s_logits0).clamp(_EPS, 1.0 - _EPS)

                loss_cons = _consistency_mse(p1, p2)
                loss_anchor = torch.mean((p0 - t_prob0) ** 2)

                p12 = torch.cat([p1, p2], dim=0)  # (2B,C)
                loss_marginal = _batch_marginal_mse(p12, t_prob12)

                total_loss = (w_consistency * loss_cons) + (w_anchor * loss_anchor) + (w_marginal * loss_marginal)
                total_loss.backward()
                opt.step()

            # update EMA on the last total_loss computed
            tl = float(total_loss.detach().item())
            if ema_loss is None:
                ema_loss = tl
                best_ema = tl
            else:
                ema_loss = (ema_beta * ema_loss) + ((1.0 - ema_beta) * tl)

            if (step % int(log_every)) == 0:
                elapsed = time.time() - start
                print(
                    f"[TTA] step {step}/{denom} | "
                    f"cons={loss_cons.item():.6f} | anchor={loss_anchor.item():.6f} | "
                    f"marg={loss_marginal.item():.6f} | total={total_loss.item():.6f} | "
                    f"ema={ema_loss:.6f} | time={elapsed:.1f}s"
                )

                if auto_stop:
                    if best_ema is None or (best_ema - ema_loss) > float(min_delta):
                        best_ema = ema_loss
                        bad_logs = 0
                    else:
                        bad_logs += 1
                        if bad_logs >= int(patience_logs):
                            print(f"[TTA] early stop: ema plateau for {patience_logs} logs (min_delta={min_delta})")
                            break

        os.makedirs(os.path.join(self.params["target_dir"], self.params["network_output_path"]), exist_ok=True)
        out_path = os.path.join(self.params["target_dir"], self.params["network_output_path"], save_name)
        torch.save(model.state_dict(), out_path)
        print(f"[TTA] saved adapted model: {out_path}")

        return model
