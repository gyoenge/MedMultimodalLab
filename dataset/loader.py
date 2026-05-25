from __future__ import annotations 

from pathlib import Path 
from typing import Optional, Sequence 

import h5py 
import torch 
import pandas as pd 
import scanpy as sc 
from PIL import Image 
import torch 
from torch.utils.data import (
    Dataset, 
    DataLoader, 
    Sampler, 
)
from torchvision import transforms 


class HESTRadiomicsDataset(Dataset):
    def __init__(
        self,
        dataroot: str | Path,
        sample_id: str,
        gene_names: Optional[Sequence[str]] = None,
        transform=None,
    ):
        self.root = Path(dataroot)
        self.sample_id = sample_id
        self.gene_names = gene_names

        self.patches_path = self.root / "patches" / f"{sample_id}.h5"
        self.st_path = self.root / "st" / f"{sample_id}.h5ad"
        self.radiomics_path = self.root / "radiomics" / f"{sample_id}.h5ad"

        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
        ])

        self.patches_h5 = None

        self._init_patches()
        self._init_st()
        self._init_radiomics()
        self._align_barcodes()

    def _init_patches(self):
        with h5py.File(self.patches_path, "r") as f:
            self.patch_keys = list(f.keys())

            self.patches_barcodes = [
                b.decode() if isinstance(b, bytes) else str(b)
                for b in f["barcode"][:].reshape(-1)
            ]

            if "coords" in f:
                self.patch_coords = f["coords"][:]
            elif "coord" in f:
                self.patch_coords = f["coord"][:]
            elif "spatial" in f:
                self.patch_coords = f["spatial"][:]
            else:
                self.patch_coords = None

        self.patch_barcode_to_idx = {
            b: i for i, b in enumerate(self.patches_barcodes)
        }

    def _init_st(self):
        self.st_adata = sc.read_h5ad(self.st_path)

        if self.gene_names is not None:
            self.st_adata = self.st_adata[:, list(self.gene_names)].copy()

        self.st_barcodes = list(self.st_adata.obs_names)
        self.st_barcode_to_idx = {
            b: i for i, b in enumerate(self.st_barcodes)
        }

        X = self.st_adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()

        self.st_matrix = torch.tensor(X, dtype=torch.float32)

        if "spatial" in self.st_adata.obsm:
            self.st_coords = self.st_adata.obsm["spatial"]
        elif {"coord_x", "coord_y"}.issubset(self.st_adata.obs.columns):
            self.st_coords = self.st_adata.obs[["coord_x", "coord_y"]].to_numpy()
        else:
            self.st_coords = None

    def _init_radiomics(self):
        self.radiomics_adata = sc.read_h5ad(self.radiomics_path)

        if "barcode" in self.radiomics_adata.obs.columns:
            self.radiomics_barcodes = list(self.radiomics_adata.obs["barcode"])
        else:
            self.radiomics_barcodes = list(self.radiomics_adata.obs_names)

        self.radiomics_barcode_to_idx = {
            b: i for i, b in enumerate(self.radiomics_barcodes)
        }

        X = self.radiomics_adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()

        self.radiomics_matrix = torch.tensor(X, dtype=torch.float32)

    def _align_barcodes(self):
        patch_set = set(self.patches_barcodes)
        st_set = set(self.st_barcodes)
        rad_set = set(self.radiomics_barcodes)

        self.valid_barcodes = sorted(list(patch_set & st_set & rad_set))

    def __len__(self):
        return len(self.valid_barcodes)

    def _open_patch_h5(self):
        if self.patches_h5 is None:
            self.patches_h5 = h5py.File(self.patches_path, "r")

    def __getitem__(self, idx):
        self._open_patch_h5()

        barcode = self.valid_barcodes[idx]

        patch_idx = self.patch_barcode_to_idx[barcode]
        st_idx = self.st_barcode_to_idx[barcode]
        radiomics_idx = self.radiomics_barcode_to_idx[barcode]

        # Lazy Loading of H5 
        if "img" in self.patches_h5:
            patch_arr = self.patches_h5["img"][patch_idx] 
        elif "imgs" in self.patches_h5:
            patch_arr = self.patches_h5["imgs"][patch_idx]
        elif "patches" in self.patches_h5:
            patch_arr = self.patches_h5["patches"][patch_idx]
        else:
            raise KeyError(f"Cannot find patch image key. Available keys: {list(self.patches_h5.keys())}")

        patch = Image.fromarray(patch_arr)
        patch = self.transform(patch)

        
        if self.patch_coords is not None:
            coord = torch.tensor(self.patch_coords[patch_idx], dtype=torch.float32)
        elif self.st_coords is not None:
            coord = torch.tensor(self.st_coords[st_idx], dtype=torch.float32)
        else:
            coord = torch.tensor([-1, -1], dtype=torch.float32)

        st = self.st_matrix[st_idx]
        radiomics = self.radiomics_matrix[radiomics_idx]

        return {
            "idx": idx,
            "barcode": barcode,
            "coord": coord,
            "patch": patch,
            "st": st,
            "radiomics": radiomics,
        }

    def __del__(self):
        if getattr(self, "patches_h5", None) is not None:
            self.patches_h5.close()


class InductiveBatchSampler(Sampler):
    """
    batch will contain: 
        - anchor 
        - spatial neighbors 
        - random globals 
    """

    # specify the sequence of indices/keys used in data loading.
    # A custom Sampler that yields a list of batch indices at a time can be passed as the batch_sampler argument.

    def __init__(self): 
        pass 


def build_loader(
        batch
    ) -> DataLoader:
        dataloader = DataLoader(
             
        )

        """
        dataset: Dataset[_T_co@DataLoader], 
        batch_size: int | None = 1, 
        shuffle: bool | None = None, 
        sampler: Sampler | Iterable | None = None, 
        batch_sampler: Sampler[List] | Iterable[List] | None = None, 
        num_workers: int = 0, 
        collate_fn: _collate_fn_t | None = None, 
        pin_memory: bool = False, 
        drop_last: bool = False, 
        timeout: float = 0, 
        worker_init_fn: _worker_init_fn_t | None = None, 
        multiprocessing_context: Any | None = None, 
        generator: Any | None = None, 
        *, 
        prefetch_factor: int | None = None, 
        persistent_workers: bool = False, 
        pin_memory_device: str = ""
        """

        return dataloader 