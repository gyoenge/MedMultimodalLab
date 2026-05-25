from dataclasses import dataclass, field
from pathlib import Path 
from typing import Dict, Any 


@dataclass
class Config: 
    data_root: Path = Path(
        "/root/workspace/datasets/hest_radiomics/IDC_Others_Xenium_11/IDC/"
    ).resolve() 

    # @property 
    # def patches_root(self) -> Path: 
    #     return self.data_root / "patches/"
    
    # @property 
    # def st_root(self) -> Path: 
    #     return self.data_root / "st/"
    
    # @property 
    # def radiomics_root(self) -> Path:
    #     return self.data_root / "radiomics/"
    
    # batch_size: int = 256 
    # num_workers: int = 8

    # model_radiomics: Dict[str, Any] = field(
    #     default_factory = lambda: {
    #         "hidden_dim": 128, 
    #     },
    # )


