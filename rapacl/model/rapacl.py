from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn


class MMCLReconClsModel(nn.Module):
    """
    Train stage:
      - pathomics CLS <-> radiomics contrastive token: MMCL loss
      - radiomics contrastive token -> radiomics reconstruction
      - radiomics CLS token -> cell-type classification

    Eval stage:
      - freeze pathomics encoder + trained radiomics TransTab
      - train gene head using concat(pathomics CLS, radiomics contrastive)
    """

    def __init__(
        self,
        radiomics_model: nn.Module,
        pathomics_encoder: nn.Module,
        pathomics_proj: nn.Module,
        recon_head: nn.Module,
        cls_head: nn.Module,
        gene_head: nn.Module,
        feature_cols: list[str],
    ):
        super().__init__()
        self.radiomics_model = radiomics_model
        self.pathomics_encoder = pathomics_encoder
        self.pathomics_proj = pathomics_proj
        self.recon_head = recon_head
        self.cls_head = cls_head
        self.gene_head = gene_head
        self.feature_cols = feature_cols

    def _to_dataframe(self, radiomics: torch.Tensor | pd.DataFrame) -> pd.DataFrame:
        if isinstance(radiomics, pd.DataFrame):
            return radiomics
        return pd.DataFrame(radiomics.detach().cpu().numpy(), columns=self.feature_cols)

    def encode_radiomics(self, radiomics: torch.Tensor | pd.DataFrame) -> dict[str, torch.Tensor]:
        x_df = self._to_dataframe(radiomics)
        feat = self.radiomics_model.input_encoder(x_df)
        feat = self.radiomics_model.contrastive_token(**feat)  # appended after feature tokens
        feat = self.radiomics_model.cls_token(**feat)          # prepended before all tokens
        enc = self.radiomics_model.encoder(**feat)

        # Final sequence: [CLS, original feature tokens..., CONTRASTIVE]
        # In your current implementation, cls_token is applied after contrastive_token.
        # Therefore CLS = index 0, Contrastive = index -1.
        rad_cls_h = enc[:, 0, :]
        rad_contrast_h = enc[:, -1, :] ######################## 
        rad_contrast_z = self.radiomics_model.projection_head(rad_contrast_h)

        return {
            "rad_cls_h": rad_cls_h,
            "rad_contrast_h": rad_contrast_h,
            "rad_contrast_z": rad_contrast_z,
        }

    @torch.no_grad()
    def encode_pathomics_frozen(self, image: torch.Tensor) -> torch.Tensor:
        self.pathomics_encoder.eval()
        return self.pathomics_encoder(image)

    def encode_pathomics_projected(
        self,
        image: torch.Tensor,
        freeze_encoder: bool = True,
    ) -> dict[str, torch.Tensor]:
        if freeze_encoder:
            with torch.no_grad():
                path_cls = self.encode_pathomics_frozen(image)
        else:
            path_cls = self.pathomics_encoder(image)

        path_z = self.pathomics_proj(path_cls)
        return {"path_cls": path_cls, "path_z": path_z}

    def forward_pretrain(self, image: torch.Tensor, radiomics: torch.Tensor | pd.DataFrame):
        rad = self.encode_radiomics(radiomics)
        path = self.encode_pathomics_projected(image, freeze_encoder=True)
        pred_radiomics = self.recon_head(rad["rad_contrast_z"])
        pred_class_logits = self.cls_head(rad["rad_cls_h"])
        return {**rad, **path, "pred_radiomics": pred_radiomics, "pred_class_logits": pred_class_logits}

    # def forward_gene(self, image: torch.Tensor, radiomics: torch.Tensor | pd.DataFrame):
    #     rad = self.encode_radiomics(radiomics)
    #     path = self.encode_pathomics_projected(image, freeze_encoder=False)
    #     fused = torch.cat([path["path_cls"], rad["rad_contrast_z"]], dim=1)
    #     pred_gene = self.gene_head(fused)
    #     return {**rad, **path, "fused": fused, "pred_gene": pred_gene}
    def forward_gene(self, image, radiomics):
        rad = self.encode_radiomics(radiomics)
        path = self.encode_pathomics_projected(image, freeze_encoder=False)

        fused = torch.cat([path["path_z"], rad["rad_contrast_z"]], dim=1)
        pred_gene = self.gene_head(fused)

        return {**rad, **path, "fused": fused, "pred_gene": pred_gene}

