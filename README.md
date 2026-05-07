# RaPaCL-ST
RadiomicsFeature-Pathomics Contrastive Learning for Spatial Transcriptomics prediction. 

<p align="justify">
<b>RaPaCL-ST (RadiomicsFeature-Pathomics Contrastive Learning for Spatial Transcriptomics prediction)</b> is a multimodal representation learning framework designed to bridge handcrafted radiomics features and deep pathomics features derived from histopathology images, which aim to predict spatial gene expression value in patch-wise level. In this approach, radiomics features extracted from image patches serve as structured, interpretable signals, while deep learning based patch image encoders encode high-dimensional visual representations. RaPaCL leverages contrastive learning to align these two modalities in a shared latent space, encouraging consistency between radiomics-informed characteristics (e.g., texture, heterogeneity) and deep image embeddings. By doing so, the framework aims to enhance the biological relevance and interpretability of learned representations, ultimately improving downstream tasks such as spatial gene expression prediction and tumor characterization in whole-slide images. 
</p> 

![mainfig](mainfig_radiomics_retrieval_style.png)

---

## Description 

### Prepare Data & Run Baselines

Please refer to: `dataset/README.md` and `baselines/README.md`. 

### Run RaPaCL

```bash
python -m rapacl.run 
```

For configuration, check `rapacl/configs/`. 

---

