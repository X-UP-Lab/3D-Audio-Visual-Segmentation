# 3DAVS-S34-O7 dataset

Please see [paper](https://arxiv.org/pdf/2411.02236) for a description of the 3DAVS-S34-O7 dataset, in particular Appendix A.3.

The Echosegnet pipeline uses **train_cameras.json** and **test_cameras.json**, which are automatically generated for each scene by running audio_informed_gaussian_splatting/train.py with the **--eval** flag. In this process, every 8th frame is assigned to the test set. These splits are deterministic when the 3D Gaussian Splatting (3D-GS) training is performed using the provided COLMAP data.
To ensure strict reproducibility and prevent data leakage, **camera_splits.json** file is also provided for each scene. This file defines the same train/test partitions and can be used for convenience or by alternative (non–3D-GS-based) methods.

For the Echosegnet method, only the training frames and their corresponding audio clips are used for:

 - generating 2D masks for OWOD-BIND,
 - training the 3D Gaussian Splatting (GS) scene,
 - constructing the Audio Intensity Map, and
 - lifting 2D masks into 3D space.

Test frames are used exclusively for evaluation in the run_lifting_aisrm.py script.

***

### Dataset Structure

NOTE: The combination of **audio_clips_binaural**, **input**, **semantic_masks**, and **camera_splits.json** is independent of 3D-GS and can be used with other methods, preserving the same dataset splits and settings to ensure results are comparable with our Echosegnet approach.

- **audio_clips_binaural/** – Original binaural audio clips (1 s, 44.1 kHz) corresponding to each frame, for both training and testing subsets.  
- **audio_clips_mono/** – Supplementary mono clips used for the OWOD-BIND setup in the paper (not part of the main dataset). Each 1 s clip is extended to 2 s by appending 0.5 s from adjacent clips (for train frames adjacent to test, 1 s from neighboring train frames is appended, to avoid leakage from test to train).  
- **input/** – Original RGB images captured at 1008 × 1008 × 3 resolution (120 per scene).  
- **semantic_masks/** – Ground-truth 2D binary masks highlighting only the sound-emitting object or instance in each frame.  
- **images/** – Undistorted and registered COLMAP images aligned with reconstructed camera poses, used for 3D GS training. Some frames may be omitted relative to `input/` if COLMAP could not successfully register them.  
- **distorted/** – COLMAP outputs to be used for 3DGS training (distorted views).  
- **sparse/** – COLMAP outputs to be used for 3DGS training (sparse reconstruction data).  
- **stereo/** – COLMAP outputs to be used for 3DGS training (stereo image pairs).  
- **camera_splits.json** – **(Important)** Metadata file defining the train/test camera split for each observation.

```
3DAVS-S34-O7/
├── part_1 (single instance setup)
│   ├── <room_id_1>
│   │   ├── <idx_1>
│   │   │   ├── audio_clips_binaural/ 
│   │   │   ├── audio_clips_mono/ 
│   │   │   ├── semantic_masks/ 
│   │   │   ├── camera_splits.json
│   │   │   ├── images/ 
│   │   │   ├── input/
│   │   │   ├── distorted/ 
│   │   │   ├── sparse/ 
│   │   │   └── stereo/
│   │   └── <idx_2>
│   │       └── ...
│   └── <room_id_2>
│       └── ...
└── part_2 (multi instance setup)
    ├── <room_id_3>
    │   ├── <idx_3>
    │   │   ├── audio_clips_binaural/
    │   │   ├── audio_clips_mono/
    │   │   ├── semantic_masks/
    │   │   ├── camera_splits.json
    │   │   ├── images/
    │   │   ├── input/
    │   │   ├── distorted/ 
    │   │   ├── sparse/ 
    │   │   └── stereo/ 
    │   └── ...
    └── ...
```
