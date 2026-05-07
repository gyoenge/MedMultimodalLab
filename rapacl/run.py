from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True"
)

import os

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from rapacl.engines.trainer_utils import (
    set_seed,
    setup_ddp,
    cleanup_ddp,
    is_main_process,
    ddp_barrier,
)
from rapacl.engines.data_utils import build_dataset, build_loader
from rapacl.engines.trainer import run_stage1, run_stage2
from rapacl.model.rapacl import build_model
from rapacl.configs.default.radiomics_columns import RADIOMICS_FEATURES_NAMES

import rapacl.configs.default.train as train


def main():
    distributed, rank, local_rank, world_size = setup_ddp()

    set_seed(train.SEED + rank)

    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(train.DEVICE)

    if is_main_process():
        print(f"[INFO] distributed: {distributed}")
        print(f"[INFO] world_size: {world_size}")
        print(f"[INFO] device: {device}")

    trainset = build_dataset(train.TRAIN_SPLIT_CSV)
    valset = build_dataset(train.VAL_SPLIT_CSV)

    train_loader, train_sampler = build_loader(
        trainset,
        shuffle=True,
        drop_last=False,
        distributed=distributed,
    )

    val_loader, val_sampler = build_loader(
        valset,
        shuffle=False,
        drop_last=False,
        distributed=distributed,
    )

    if is_main_process():
        print(f"[INFO] train samples: {len(trainset)}")
        print(f"[INFO] val samples: {len(valset)}")

    num_genes = len(trainset.genes)
    num_radiomics_features = len(RADIOMICS_FEATURES_NAMES)

    if is_main_process():
        print(f"[INFO] num_genes: {num_genes}")
        print(f"[INFO] num_radiomics_features: {num_radiomics_features}")

    model = build_model(
        device=device,
        num_genes=num_genes,
        num_radiomics_features=num_radiomics_features,
    )

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    save_dir = os.path.join(
        train.OUTPUT_CHECKPOINT_DIR,
        "rapacl_baseline",
    )

    if is_main_process():
        os.makedirs(save_dir, exist_ok=True)

    ddp_barrier()

    stage1_ckpt_path = run_stage1(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=save_dir,
        train_sampler=train_sampler,
        is_distributed=distributed,
    )

    ddp_barrier()

    run_stage2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=save_dir,
        stage1_ckpt_path=stage1_ckpt_path,
        train_sampler=train_sampler,
        is_distributed=distributed,
    )

    cleanup_ddp()


if __name__ == "__main__":
    main()
