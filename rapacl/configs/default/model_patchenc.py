import os


BACKBONE = "densenet121"  # "densenet121", "resnet50", "uni", "conch"

PRETRAINED = True
DEVICE = "cuda:0"
PROJECT_DIR = os.path.join(os.path.expanduser("~"), "workspace", "RaPaCL")

### UNI 
UNI_VERSION = "vit_large_patch16_224"
UNI_CKPT_PATH = os.path.join(PROJECT_DIR, "checkpoints", "UNI", f"{UNI_VERSION}.bin") 
