# Training HITNET with custom dataset.
The paper link can be found [here](https://arxiv.org/pdf/2007.12140) and its supplemental [here] (https://openaccess.thecvf.com/content/CVPR2021/supplemental/Tankovich_HITNet_Hierarchical_Iterative_CVPR_2021_supplemental.pdf).

## Dataset
My dataset found in `dataset/processed_data`
refer to the code `dataset/depth_dataset.py` for the data loading of my dataset.
refer to the code '`dataset/kitti2012.py` for example dataloading of the origin project.

## Requirements
The training code can be found in `train.py`. Help me to modify the code for my dataset. The library used for this project is quiet old, so I need to update the code to fit the new version of PyTorch. The training code is based on the original code from the HITNET project, but I have made some modifications to adapt it to my dataset and the new PyTorch version.