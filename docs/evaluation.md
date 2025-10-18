# EchosegNet Evaluation on 3DAVS-S34-O7

Download the dataset from [link](https://drive.google.com/file/d/1HG53HrcCnWCKBXF0lVVtEEkqebL-N9jT/view?usp=sharing) and place it in the `dataset` directory, so that the full path to the dataset is `dataset/3DAVS-S34-O7`.

***

## Step 1. Train Audio-Informed 3D Gaussian Splatting for each dataset scene

### Command

```bash
bash ./src/avs_3d/scripts/train_gaussian_splatting_dataset.sh ./dataset/3DAVS-S34-O7
```

The trained Audio-Informed 3D Gaussian Splatting models will be saved to `dataset/3DAVS-S34-O7/<part>/<room>/<idx>/3dgs_scene`.

***

## Step 2. Run OWOD-BIND to get 2D masks

### Command

```bash
python ./src/avs_3d/run_owodbind.py --root_dir ./dataset/3DAVS-S34-O7 --config ./src/avs_3d/echosegnet/configs/owod_bind_config.yaml
```

The 2D masks of sound-emitting objects will be saved in the scene directory `dataset/3DAVS-S34-O7/<part>/<room>/<idx>/owob_bind_train_frames`.

***

## Step 3. Run 2D-3D Lifting and Audio-Informed Spatial Refinement Module (AISRM)

### Command
The script below will run for each scene, both with and without AISRM. It also automatically computes F-Score and mIoU for each scene, however, the aggregation should be done as described in the next step.

```bash
bash ./src/avs_3d/scripts/run_lifting_aisrm_dataset.sh ./dataset/3DAVS-S34-O7
```

Predicted 3D segmented point clouds will be saved to `dataset/3DAVS-S34-O7/<part>/<room>/<idx>/3dgs_scene` and rendered 2D segmentation masks will be saved to `dataset/3DAVS-S34-O7/<part>/<room>/<idx>/echosegnet_output`.
***

## Step 4. Aggregate Final Metrics of Evaluation
Aggregate the final quantitative metrics for the dataset (e.g., **mIoU**, **F-Score**).

### Command

```bash
# Single-instance setting, without AISRM
python ./src/avs_3d/aggregate_metrics.py --root ./dataset/3DAVS-S34-O7 --part 1 --mode "no_aisrm"
```

```bash
# Single-instance setting, with AISRM
python ./src/avs_3d/aggregate_metrics.py --root ./dataset/3DAVS-S34-O7 --part 1 --mode "aisrm"
```

```bash
# Multi-instance setting, without AISRM
python ./src/avs_3d/aggregate_metrics.py --root ./dataset/3DAVS-S34-O7 --part 2 --mode "no_aisrm"
```

```bash
# Multi-instance setting, with AISRM
python ./src/avs_3d/aggregate_metrics.py --root ./dataset/3DAVS-S34-O7 --part 2 --mode "aisrm"
```
