"""GAN training script for EDNIG (PyTorch port).

Run on a CUDA-enabled machine:

    python -m pytorch_impl.train --data ./data/lol_dataset --img-size 512 --epochs 180

This mirrors the original paper recipe:
- Generator + Critic (WGAN-like)
- Each generator step is preceded by `critic_updates` critic steps
- Generator loss = 100 * (perceptual + 10 * L2) + 1 * (-critic_score)
- Optimizer: Adam with lr 1e-4, beta1=0.9, beta2=0.999, linear LR decay
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import LOLDataset
from .losses import PerceptualAndL2Loss, VGGPerceptual, wasserstein_loss
from .model import Discriminator, Generator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="./data/lol_dataset",
                   help="Root directory containing our485/ and eval15/")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--critic-updates", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-dir", type=str, default="./weights")
    p.add_argument("--save-every", type=int, default=1,
                   help="Update latest/best checkpoint every N epochs (default 1).")
    p.add_argument("--keep-numbered", action="store_true",
                   help="Also save numbered per-epoch snapshots (generator_epNNN.pt). "
                        "Disabled by default to keep disk usage minimal — only "
                        "ednig_generator_latest.pt and ednig_generator_best.pt are kept.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--resume-g", type=str, default="",
                   help="Optional path to generator .pt to resume from.")
    p.add_argument("--resume-d", type=str, default="",
                   help="Optional path to discriminator .pt to resume from.")
    p.add_argument("--device", type=str, default="cuda",
                   help="cuda or cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("[WARN] CUDA requested but not available, falling back to CPU.")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Data
    ds = LOLDataset(args.data, split="train", img_size=args.img_size, augment=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    print(f"[INFO] Training samples: {len(ds)} | iters/epoch: {len(loader)}")

    # Models
    G = Generator().to(device)
    D = Discriminator().to(device)
    if args.resume_g:
        G.load_state_dict(torch.load(args.resume_g, map_location=device))
        print(f"[INFO] Resumed G from {args.resume_g}")
    if args.resume_d:
        D.load_state_dict(torch.load(args.resume_d, map_location=device))
        print(f"[INFO] Resumed D from {args.resume_d}")

    # Optimizers (linear decay over training, similar to original)
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    decay_rate = (args.lr / args.epochs) * 0.5

    # Loss helpers
    vgg = VGGPerceptual().to(device).eval()
    p_l2_loss = PerceptualAndL2Loss(a=1.0, b=10.0, vgg=vgg).to(device)

    # Critic ground-truth labels
    label_real = torch.ones(args.batch_size, 1, device=device)
    label_fake = -torch.ones(args.batch_size, 1, device=device)

    g_weight, c_weight = 100.0, 1.0
    best_content = float("inf")
    train_start = datetime.now()

    for epoch in range(1, args.epochs + 1):
        # Linear LR decay
        lr_now = max(args.lr - decay_rate * (epoch - 1), 1e-6)
        for pg in opt_d.param_groups:
            pg["lr"] = lr_now
        for pg in opt_g.param_groups:
            pg["lr"] = lr_now

        t0 = time.time()
        d_losses, g_losses, c_losses = [], [], []
        G.train(); D.train()
        for step, (inp, tgt) in enumerate(loader):
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            with torch.no_grad():
                fake = G(inp)

            # ---- Critic updates ----
            for _ in range(args.critic_updates):
                opt_d.zero_grad(set_to_none=True)
                d_real = D(tgt)
                d_fake = D(fake.detach())
                loss_d = 0.5 * (
                    wasserstein_loss(d_real, label_real)
                    + wasserstein_loss(d_fake, label_fake)
                )
                loss_d.backward()
                opt_d.step()
                d_losses.append(loss_d.item())

            # ---- Generator update ----
            opt_g.zero_grad(set_to_none=True)
            fake = G(inp)
            d_on_fake = D(fake)
            loss_g_content = p_l2_loss(fake, tgt)
            # The critic outputs "real-ness" — generator wants D(fake) -> +1
            # Adversarial term equivalent to wasserstein_loss(D(fake), +1):
            loss_g_adv = wasserstein_loss(d_on_fake, label_real)
            loss_g = g_weight * loss_g_content + c_weight * loss_g_adv
            loss_g.backward()
            opt_g.step()
            g_losses.append(loss_g.item())
            c_losses.append(loss_g_content.item())

            if (step + 1) % 25 == 0:
                print(
                    f"  [ep {epoch}/{args.epochs}] step {step+1}/{len(loader)} | "
                    f"d_loss={sum(d_losses[-args.critic_updates:])/max(1,len(d_losses[-args.critic_updates:])):.4f} "
                    f"g_loss={loss_g.item():.4f} (content={loss_g_content.item():.4f} adv={loss_g_adv.item():.4f})"
                )

        dt = time.time() - t0
        mean_d = sum(d_losses) / max(1, len(d_losses))
        mean_g = sum(g_losses) / max(1, len(g_losses))
        print(
            f"[Epoch {epoch}/{args.epochs}] lr={lr_now:.6f} "
            f"D_loss={mean_d:.4f} G_loss={mean_g:.4f} time={dt:.1f}s"
        )

        # Only save when this epoch is the best so far (lowest mean content loss).
        mean_content = sum(c_losses) / max(1, len(c_losses))
        is_best = mean_content < best_content
        if is_best:
            best_content = mean_content
            now = datetime.now()
            elapsed = (now - train_start).total_seconds()
            torch.save(G.state_dict(), save_dir / "ednig_generator_best.pt")
            torch.save(D.state_dict(), save_dir / "ednig_discriminator_best.pt")

            # Sidecar metadata so the web app (and humans) can read it.
            info = {
                "model_name": "EDNIG (Encoder-Decoder Network with Illumination Guidance)",
                "checkpoint": "ednig_generator_best.pt",
                "saved_at": now.isoformat(timespec="seconds"),
                "training_started_at": train_start.isoformat(timespec="seconds"),
                "elapsed_seconds_so_far": int(elapsed),
                "elapsed_human": f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m",
                "epoch": epoch,
                "total_epochs_planned": args.epochs,
                "best_mean_content_loss": float(mean_content),
                "epoch_mean_g_loss": float(mean_g),
                "epoch_mean_d_loss": float(mean_d),
                "epoch_time_seconds": round(dt, 2),
                "learning_rate_at_save": lr_now,
                "hyperparameters": {
                    "img_size": args.img_size,
                    "batch_size": args.batch_size,
                    "critic_updates": args.critic_updates,
                    "lr_initial": args.lr,
                    "generator_loss_weight": g_weight,
                    "adversarial_loss_weight": c_weight,
                    "content_loss_formula": "1.0 * perceptual_vgg16_block3 + 10.0 * L2",
                    "optimizer": "Adam(betas=(0.9, 0.999), eps=1e-8) with linear LR decay",
                    "augmentation": {
                        "random_crop_scale": [0.5, 1.0],
                        "horizontal_flip_prob": 0.5,
                        "vertical_flip_prob": 0.2,
                        "rot90_prob": 0.3,
                        "photometric_jitter_prob": 0.15,
                    },
                },
                "architecture": {
                    "generator": "U-Net + SPP + Swish, 4-ch input (RGB + BCP illumination)",
                    "discriminator": "U-Net encoder + GAP + Dense + Sigmoid",
                },
                "device": str(device),
                "torch_version": torch.__version__,
                "dataset": {
                    "name": "LOL (Low-Light)",
                    "root": str(args.data),
                    "training_samples": len(ds),
                },
                "paper": {
                    "title": "Low-Light Enhancement via Encoder-Decoder Network with Illumination Guidance",
                    "venue": "ICCCE 2025, IEEE",
                    "authors": [
                        "Le-Anh Tran", "Chung Nguyen Tran", "Ngoc-Luu Nguyen",
                        "Nhan Cach Dang", "Jordi Carrabina", "David Castells-Rufas",
                        "Minh Son Nguyen",
                    ],
                    "arxiv": "https://arxiv.org/abs/2507.13360",
                },
                "translator": {
                    "name": "Tran Quoc Bao",
                    "id": "2480101829",
                    "note": "Bien soan ban tieng Viet va port PyTorch + Web app",
                },
            }
            with open(save_dir / "ednig_model_info.json", "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2, ensure_ascii=False)

            print(f"  [SAVED BEST] epoch={epoch} content={mean_content:.4f} elapsed={info['elapsed_human']}")
        # else: skip saving entirely -- only the best is kept.

    print("[DONE] Training complete.")


if __name__ == "__main__":
    main()
