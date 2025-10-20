# 3D Audio-Visual Segmentation
Artem Sokolov, Swapnil Bhosale, Xiatian Zhu  
### NeurIPS 2024 Workshop on Audio Imagination
[![Project page](https://img.shields.io/badge/3D_Audio--Visual_Segmentation-%F0%9F%8C%90Website-purple?style=flat)](https://x-up-lab.github.io/research/3d-audio-visual-segmentation/) [![arXiv](https://img.shields.io/badge/arXiv-2411.02236-b31b1b.svg)](https://arxiv.org/abs/2411.02236) [![Dataset](https://img.shields.io/badge/3DAVS--S34--O7_Dataset-8A2BE2.svg)](https://drive.google.com/drive/folders/17yhfD9kBJhoutOYZhrze4AkpSf1KwpfL?usp=sharing)

![teaser](figures/2d_avs_vs_3d_avs.png)

**This repository is the official implementation of "3D Audio-Visual Segmentation".** In this paper, we introduce a novel research problem, 3D Audio-Visual Segmentation, extending the existing AVS to the 3D output space. To facilitate this research, we create the very first simulation based benchmark, 3DAVS-S34-O7, providing photorealistic 3D scene environments with grounded spatial audio under single-instance and multi-instance settings, across 34 scenes and 7 object categories. Subsequently, we propose a new approach, EchoSegnet, characterized by integrating the ready-to-use knowledge from pretrained 2D audio-visual foundation models synergistically with 3D visual scene representation through spatial audio-aware mask alignment and refinement.

## Updates
- The 3DAVS-S34-O7 dataset and EchoSegnet code are now available !🎉 
- Data & Code coming soon!

## 3DAVS-S34-O7 dataset
The dataset can be downloaded from this [link](https://drive.google.com/drive/folders/17yhfD9kBJhoutOYZhrze4AkpSf1KwpfL?usp=sharing).
Please refer to the [description](./docs/dataset.md) for details on the included data and usage.

The dataset is an Adapted Material derived from and compiled using several external resources. You must comply with the non-commercial terms of the original licenses:
* Habitat-Matterport3D (HM3D): Restricted to Non-Commercial Research Use.
* BBC Sound Effects: Restricted to Non-Commercial/Educational Use.
* ESC-50 Dataset: Licensed under Creative Commons Attribution-NonCommercial 4.0 (CC-BY-NC 4.0).

By using the 3DAVS-S34-O7 dataset, you agree to the full terms of the [CC-BY-NC 4.0 Public License](https://creativecommons.org/licenses/by-nc/4.0/).

## Method: EchoSegnet
![teaser](figures/method.png)

## Installation
See the [Installation Guide](./docs/install.md) for environment setup and dependency installation to run the evaluation.

## Evaluation
Refer to the [Evaluation Guide](./docs/evaluation.md) for instructions on running evaluation of Echosegnet on the proposed 3DAVS-S34-O7 dataset.

## Citation
If you find our project useful, please use the following BibTeX entry:
```bibtex
@inproceedings{sokolov20243daudiovisualsegmentation,
    title     = {3D Audio-Visual Segmentation},
    author    = {Sokolov, Artem and Bhosale, Swapnil and Zhu, Xiatian},
    booktitle = {Audio Imagination: NeurIPS 2024 Workshop AI-Driven Speech, Music, and Sound Generation},
    year      = {2024}
}

```

## Contact
For feedback or questions please contact [Artem Sokolov](mailto:artemiojosesokolov@gmail.com)
