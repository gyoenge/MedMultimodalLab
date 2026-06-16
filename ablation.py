"""Ablation study: UNI-only baseline vs STaRN (fusion + neighbor).

Runs LOOCV gene prediction for two methods under comparable conditions,
prints a comparison table, then generates side-by-side spatial visualizations
with genes selected by the PCC gap (STaRN − UNI-only).

Methods
-------
uni_only  UNI(1024) → MLP(256) → 250 genes          [no backbone, plain DataLoader]
starn     concat(STaRN(128), UNI(1024)) → MLP → 250  [pretrained backbone, neighbor loader]

Gene groups in visualization
-----------------------------
[Win]  STaRN wins most        (largest positive gap)
[Tie]  methods perform similarly (smallest |gap|)
[Lose] UNI-only wins          (largest negative gap)

Usage
-----
    cd /root/workspace/STaRN
    python ablation.py --ckpt checkpoints/epoch_099.pt
    python ablation.py --ckpt checkpoints/epoch_099.pt --top-k 3 --skip-eval
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.config import Config
from dataset.loader import HestRadiomicsDataset, _PersampleDataset, get_common_genes
from eval import (
    _patch_dataset_no_img,
    build_backbone,
    MLPGeneHead,
    run_fold,
    EVAL_DATA_ROOT,
    SAMPLE_IDS,
    N_GENES,
    GENE_CRITERIA,
    HEAD_HIDDEN_DIM,
    HEAD_DROPOUT,
    NUM_WORKERS,
)
from model.tabular import SummaryTableModel

# ── method definitions ────────────────────────────────────────────────────────

METHODS = [
    {
        "name":         "uni_only",
        "mode":         "uni_only",
        "label":        "UNI-only",
        "use_backbone": False,
        "use_neighbor": False,   # no spatial context needed; plain DataLoader
        "ckpt_dir":     Path("checkpoints/loocv_uni_only_no_neighbor"),
    },
    {
        "name":         "starn",
        "mode":         "fusion",
        "label":        "STaRN",
        "use_backbone": True,
        "use_neighbor": True,    # spatial + semantic neighbor context
        "ckpt_dir":     Path("checkpoints/loocv_fusion_neighbor"),
    },
]

OUT_DIR     = Path("figures/ablation")
INFER_BATCH = 512
CMAP        = "RdBu_r"
VMIN, VMAX  = -2.0, 2.0


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_method(
    method:   dict,
    backbone: Optional[SummaryTableModel],
    head:     MLPGeneHead,
    dataset:  HestRadiomicsDataset,
    device:   torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run full-dataset inference for one method.

    Uses a plain DataLoader (no neighbor sampling) so every spot is predicted
    independently. Row attention in the backbone still sees batch-mates but
    without explicit neighbor structure — acceptable for visualization.

    Returns:
        coords  (N, 2)
        gt      (N, G)
        pred    (N, G)
    """
    loader = DataLoader(
        dataset,
        batch_size=INFER_BATCH,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    coords_l, gt_l, pred_l = [], [], []
    for batch in loader:
        rad     = batch["radiomics"].to(device)
        uni_emb = batch["uni_emb"].to(device)
        coord   = batch["coord"]
        gt      = batch["st"]

        if method["use_backbone"] and backbone is not None:
            z    = backbone.encode(rad, coord.to(device))     # (B, hidden_dim)
            feat = torch.cat([z, uni_emb], dim=-1)            # (B, hidden+uni)
        else:
            feat = uni_emb                                     # (B, uni_dim)

        pred = head(feat)

        coords_l.append(coord.numpy())
        gt_l.append(gt.numpy())
        pred_l.append(pred.cpu().numpy())

    return (
        np.concatenate(coords_l, 0).astype(np.float32),
        np.concatenate(gt_l,     0).astype(np.float32),
        np.concatenate(pred_l,   0).astype(np.float32),
    )


def zscore(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)


# ── gene group selection ──────────────────────────────────────────────────────

def select_gene_groups_by_gap(
    pcc_uni:  np.ndarray,
    pcc_star: np.ndarray,
    k:        int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select k genes for each of three groups based on PCC gap (STaRN - UNI-only).

    win_idx  — STaRN wins most  (largest positive gap)
    tie_idx  — methods tied     (smallest |gap|, excluding win/lose)
    lose_idx — UNI-only wins    (largest negative gap, i.e. most negative gap)
    """
    gap = pcc_star - pcc_uni
    sorted_asc = np.argsort(gap)              # ascending gap
    win_idx  = sorted_asc[::-1][:k]           # top-k positive gap
    lose_idx = sorted_asc[:k]                 # top-k negative gap

    excluded = set(win_idx.tolist()) | set(lose_idx.tolist())
    remain   = np.array([i for i in range(len(gap)) if i not in excluded])
    tie_order = remain[np.argsort(np.abs(gap[remain]))]
    tie_idx  = tie_order[:k]

    return win_idx, tie_idx, lose_idx


# ── plotting ──────────────────────────────────────────────────────────────────

def _scatter(ax, coords, vals, title, s):
    sc = ax.scatter(
        coords[:, 0], coords[:, 1], c=vals,
        s=s, cmap=CMAP, vmin=VMIN, vmax=VMAX,
        linewidths=0, rasterized=True,
    )
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=7, pad=3)
    return sc


def plot_ablation_sample(
    sample_id:   str,
    coords:      np.ndarray,
    gt_z:        np.ndarray,
    uni_z:       np.ndarray,
    star_z:      np.ndarray,
    pcc_uni:     np.ndarray,
    pcc_star:    np.ndarray,
    gene_names:  list[str],
    ckpt_epoch:  int,
    n_top:       int,
    save_path:   Path,
):
    win_idx, tie_idx, lose_idx = select_gene_groups_by_gap(pcc_uni, pcc_star, n_top)

    sections = [
        ("[Win]",  win_idx),
        ("[Tie]",  tie_idx),
        ("[Lose]", lose_idx),
    ]
    n_rows = 1 + n_top * 3

    s      = float(np.clip(120_000 / max(len(coords), 1), 1.0, 50.0))
    x_span = float(np.ptp(coords[:, 0]))
    y_span = float(np.ptp(coords[:, 1]))
    col_w  = 3.5
    row_h  = col_w * (y_span / x_span) if x_span > 0 else col_w
    row_h  = float(np.clip(row_h, 2.0, 8.0))

    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(col_w * 3 + 1.0, row_h * n_rows),
        constrained_layout=True,
    )

    uni_mean_pcc  = float(pcc_uni.mean())
    star_mean_pcc = float(pcc_star.mean())
    fig.suptitle(
        f"{sample_id}  |  epoch {ckpt_epoch:03d}  |  {len(coords):,} spots\n"
        f"Mean PCC — UNI-only: {uni_mean_pcc:.4f}   STaRN: {star_mean_pcc:.4f}   "
        f"Δ={star_mean_pcc - uni_mean_pcc:+.4f}",
        fontsize=10, fontweight="bold",
    )

    # Column headers
    for ax, lbl in zip(axes[0], ["GT  (mean z-score)", "UNI-only  (mean)", "STaRN  (mean)"]):
        pass   # titles handled per-scatter call below

    # Row 0: mean z-score
    for ax, arr, lbl in zip(
        axes[0],
        [gt_z.mean(1),  uni_z.mean(1),  star_z.mean(1)],
        ["GT  (mean z-score, 250 genes)",
         f"UNI-only  (mean)  PCC={uni_mean_pcc:.3f}",
         f"STaRN     (mean)  PCC={star_mean_pcc:.3f}"],
    ):
        sc = _scatter(ax, coords, arr, lbl, s)
        plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label="z-score")

    # Gene rows
    row = 1
    gap = pcc_star - pcc_uni
    for sec_label, indices in sections:
        for gi in indices:
            name = gene_names[gi]
            pu   = pcc_uni[gi]
            ps   = pcc_star[gi]
            dg   = gap[gi]
            titles = [
                f"{sec_label} GT  |  {name}",
                f"{sec_label} UNI-only  PCC={pu:.3f}",
                f"{sec_label} STaRN     PCC={ps:.3f}  (Δ={dg:+.3f})",
            ]
            for ax, arr, lbl in zip(
                axes[row],
                [gt_z[:, gi], uni_z[:, gi], star_z[:, gi]],
                titles,
            ):
                sc = _scatter(ax, coords, arr, lbl, s)
                plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label="z-score")
            row += 1

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {save_path}")


# ── comparison table ──────────────────────────────────────────────────────────

def print_comparison_table(
    method_results: dict[str, list[float]],
    gene_names:     list[str],
    method_pergene: dict[str, list[np.ndarray]],
) -> None:
    sample_list = list(SAMPLE_IDS)
    col_w = 12

    print(f"\n{'=' * 70}")
    print("Ablation Results  —  LOOCV gene-wise PCC")
    print("─" * 70)
    header = f"{'Sample':<12}" + "".join(f"{m['label']:>{col_w}}" for m in METHODS)
    print(header)
    print("─" * 70)
    for fold, sid in enumerate(sample_list):
        row = f"{sid:<12}"
        for m in METHODS:
            pcc = method_results[m["name"]][fold]
            row += f"{pcc:>{col_w}.4f}"
        print(row)
    print("─" * 70)

    row = f"{'Mean':12}"
    for m in METHODS:
        mean = float(np.mean(method_results[m["name"]]))
        row += f"{mean:>{col_w}.4f}"
    print(row)

    row = f"{'Std':12}"
    for m in METHODS:
        vals = method_results[m["name"]]
        std  = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        row += f"{std:>{col_w}.4f}"
    print(row)

    # Delta column (STaRN - UNI-only)
    pccs_uni  = method_results["uni_only"]
    pccs_star = method_results["starn"]
    deltas    = [s - u for s, u in zip(pccs_star, pccs_uni)]
    print("─" * 70)
    print(f"{'Δ (STaRN−UNI)':12}" + "".join(
        f"{d:>{col_w}+.4f}" for d in deltas + [float(np.mean(deltas))]
    ))
    print(f"{'=' * 70}")

    # Top/bottom genes by mean PCC gap
    if "uni_only" in method_pergene and "starn" in method_pergene:
        mean_uni  = np.stack(method_pergene["uni_only"]).mean(0)
        mean_star = np.stack(method_pergene["starn"]).mean(0)
        gap       = mean_star - mean_uni

        top5 = np.argsort(gap)[::-1][:5]
        bot5 = np.argsort(gap)[:5]

        print("\nTop-5 genes where STaRN gains most over UNI-only:")
        for i in top5:
            print(f"  {gene_names[i]:20s}  UNI={mean_uni[i]:.4f}  STaRN={mean_star[i]:.4f}  Δ={gap[i]:+.4f}")
        print("\nTop-5 genes where UNI-only outperforms STaRN:")
        for i in bot5:
            print(f"  {gene_names[i]:20s}  UNI={mean_uni[i]:.4f}  STaRN={mean_star[i]:.4f}  Δ={gap[i]:+.4f}")
        print()


# ── LOOCV runner ──────────────────────────────────────────────────────────────

def run_ablation_eval(
    gene_names: list[str],
    backbone:   Optional[SummaryTableModel],
    cfg:        Config,
    device:     torch.device,
    ckpt_path:  Optional[Path],
    skip_eval:  bool,
) -> tuple[dict[str, list[float]], dict[str, list[np.ndarray]]]:
    """Run LOOCV for all methods; skip if checkpoints already exist and skip_eval=True."""
    sample_list   = list(SAMPLE_IDS)
    method_results: dict[str, list[float]]      = {m["name"]: [] for m in METHODS}
    method_pergene: dict[str, list[np.ndarray]] = {m["name"]: [] for m in METHODS}

    for m in METHODS:
        print(f"\n{'#' * 60}")
        print(f"# Method: {m['label']}  (mode={m['mode']}, neighbor={m['use_neighbor']})")
        print(f"{'#' * 60}")

        backbone_for_method = backbone if m["use_backbone"] else None

        for fold, val_id in enumerate(sample_list):
            train_ids    = [s for s in sample_list if s != val_id]
            fold_ckpt    = m["ckpt_dir"] / f"fold_{fold}_best.pt"

            if skip_eval and fold_ckpt.exists():
                saved = torch.load(fold_ckpt, map_location="cpu", weights_only=False)
                best_pcc = float(saved["val_genewise_pcc"])
                per_gene = np.array(saved["per_gene_pcc"])
                print(f"  [skip] fold {fold} val={val_id}  PCC={best_pcc:.4f}  (loaded from checkpoint)")
            else:
                best_pcc, per_gene = run_fold(
                    fold=fold,
                    val_id=val_id,
                    train_ids=train_ids,
                    gene_names=gene_names,
                    backbone=backbone_for_method,
                    cfg=cfg,
                    mode=m["mode"],
                    use_neighbor=m["use_neighbor"],
                    device=device,
                    save_dir=m["ckpt_dir"],
                    ckpt_path=ckpt_path if m["use_backbone"] else None,
                )

            method_results[m["name"]].append(best_pcc)
            if per_gene is not None:
                method_pergene[m["name"]].append(per_gene)

    return method_results, method_pergene


# ── visualization ─────────────────────────────────────────────────────────────

def run_ablation_viz(
    gene_names:  list[str],
    backbone:    Optional[SummaryTableModel],
    cfg:         Config,
    device:      torch.device,
    ckpt_epoch:  int,
    n_top:       int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_list = list(SAMPLE_IDS)

    for fold, val_id in enumerate(sample_list):
        print(f"\nFold {fold} | val={val_id}")

        # Load both fold checkpoints
        method_heads: dict[str, MLPGeneHead] = {}
        method_pcc:   dict[str, np.ndarray]  = {}

        all_loaded = True
        for m in METHODS:
            fold_ckpt_path = m["ckpt_dir"] / f"fold_{fold}_best.pt"
            if not fold_ckpt_path.exists():
                print(f"  [skip] missing {fold_ckpt_path}")
                all_loaded = False
                break

            saved      = torch.load(fold_ckpt_path, map_location=device, weights_only=False)
            in_dim     = cfg.uni_dim if m["mode"] == "uni_only" else cfg.hidden_dim + cfg.uni_dim
            head       = MLPGeneHead(
                in_dim=in_dim, hidden_dim=HEAD_HIDDEN_DIM,
                out_dim=len(gene_names), dropout=0.0,
            ).to(device)
            head.load_state_dict(saved["head"])
            head.eval()

            method_heads[m["name"]] = head
            method_pcc[m["name"]]   = np.array(saved["per_gene_pcc"])

        if not all_loaded:
            continue

        # Single dataset for all methods (same spots)
        dataset = HestRadiomicsDataset(
            sources=[(EVAL_DATA_ROOT, [val_id])],
            gene_names=gene_names,
        )
        print(f"  {len(dataset):,} spots")

        coords, gt_arr, pred_uni = infer_method(
            METHODS[0], None, method_heads["uni_only"], dataset, device
        )
        _, _, pred_star = infer_method(
            METHODS[1], backbone, method_heads["starn"], dataset, device
        )

        gt_z   = zscore(gt_arr)
        uni_z  = zscore(pred_uni)
        star_z = zscore(pred_star)

        save_path = OUT_DIR / f"{val_id}_ep{ckpt_epoch:03d}_ablation.png"
        plot_ablation_sample(
            sample_id=val_id,
            coords=coords,
            gt_z=gt_z,
            uni_z=uni_z,
            star_z=star_z,
            pcc_uni=method_pcc["uni_only"],
            pcc_star=method_pcc["starn"],
            gene_names=gene_names,
            ckpt_epoch=ckpt_epoch,
            n_top=n_top,
            save_path=save_path,
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="STaRN ablation: UNI-only vs STaRN")
    p.add_argument(
        "--ckpt", type=Path, default=None,
        help="Pretrained STaRN backbone checkpoint (required unless --skip-eval "
             "and fusion checkpoints already exist).",
    )
    p.add_argument("--top-k",     type=int,  default=5,
                   help="Genes per section in visualization (default 5).")
    p.add_argument("--skip-eval", action="store_true",
                   help="Skip LOOCV training if per-fold checkpoints already exist.")
    p.add_argument("--viz-only",  action="store_true",
                   help="Skip LOOCV entirely and jump straight to visualization "
                        "(all fold checkpoints must exist).")
    return p.parse_args()


def main():
    args   = _parse()
    cfg    = Config()
    device = torch.device(cfg.device)

    _patch_dataset_no_img()

    # Gene names — derived from data, not stored
    st_paths   = [EVAL_DATA_ROOT / "st" / f"{sid}.h5ad" for sid in SAMPLE_IDS]
    gene_names = get_common_genes(st_paths, k=N_GENES, criteria=GENE_CRITERIA)
    print(f"Selected {len(gene_names)} common genes.")

    # Backbone (needed for STaRN method)
    backbone:   Optional[SummaryTableModel] = None
    ckpt_epoch: int = -1

    if args.ckpt is not None:
        backbone, ckpt_epoch = build_backbone(cfg, args.ckpt, device)
        print(f"Backbone loaded: {args.ckpt}  (epoch {ckpt_epoch})")
    else:
        # Try to infer ckpt_epoch from existing starn checkpoints for viz titles
        starn_ckpt_dir = METHODS[1]["ckpt_dir"]
        for fold_ckpt in starn_ckpt_dir.glob("fold_*_best.pt"):
            try:
                saved = torch.load(fold_ckpt, map_location="cpu", weights_only=False)
                raw = saved.get("backbone_ckpt", None)
                if raw and Path(raw).exists():
                    ck = torch.load(raw, map_location="cpu")
                    ckpt_epoch = int(ck.get("epoch", -1))
                break
            except Exception:
                pass

    # ── eval ────────────────────────────────────────────────────────────────
    if not args.viz_only:
        if backbone is None and not (args.skip_eval):
            # Check whether starn fold checkpoints already exist
            starn_dir = METHODS[1]["ckpt_dir"]
            starn_exists = all(
                (starn_dir / f"fold_{i}_best.pt").exists()
                for i in range(len(SAMPLE_IDS))
            )
            if not starn_exists:
                raise SystemExit(
                    "--ckpt is required to run STaRN LOOCV training.\n"
                    "Use --skip-eval if STaRN checkpoints already exist, or "
                    "provide --ckpt path."
                )

        method_results, method_pergene = run_ablation_eval(
            gene_names=gene_names,
            backbone=backbone,
            cfg=cfg,
            device=device,
            ckpt_path=args.ckpt,
            skip_eval=args.skip_eval or args.viz_only,
        )
        print_comparison_table(method_results, gene_names, method_pergene)

    # ── viz ─────────────────────────────────────────────────────────────────
    if backbone is None:
        # Rebuild backbone for viz from stored backbone_ckpt ref in checkpoint
        starn_dir = METHODS[1]["ckpt_dir"]
        for i in range(len(SAMPLE_IDS)):
            p = starn_dir / f"fold_{i}_best.pt"
            if p.exists():
                saved = torch.load(p, map_location="cpu", weights_only=False)
                ref   = saved.get("backbone_ckpt")
                if ref and Path(ref).exists():
                    backbone, ckpt_epoch = build_backbone(cfg, Path(ref), device)
                    print(f"Backbone rebuilt from {ref}  (epoch {ckpt_epoch})")
                    break

    if backbone is None:
        print("\n[warn] No backbone available — skipping visualization for STaRN method.")
        return

    print(f"\nGenerating ablation visualizations → {OUT_DIR}/")
    run_ablation_viz(
        gene_names=gene_names,
        backbone=backbone,
        cfg=cfg,
        device=device,
        ckpt_epoch=ckpt_epoch,
        n_top=args.top_k,
    )
    print(f"\nDone — figures in {OUT_DIR}/")


if __name__ == "__main__":
    main()
