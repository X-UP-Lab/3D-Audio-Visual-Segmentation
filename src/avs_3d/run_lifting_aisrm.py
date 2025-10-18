import os
import json
import logging
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as func
import cv2
from plyfile import PlyData, PlyElement
from sklearn.cluster import DBSCAN

from avs_3d.echosegnet.audio_informed_gaussian_splatting.scene.gaussian_model import GaussianModel
from avs_3d.echosegnet.audio_informed_gaussian_splatting.scene import Scene
from avs_3d.echosegnet.audio_informed_gaussian_splatting.gaussian_renderer import render
from avs_3d.echosegnet.audio_informed_gaussian_splatting.arguments import ModelParams, PipelineParams

from avs_3d.echosegnet.sagd_utils import (
    mask_inverse,
    ensemble,
    gaussian_decomp
)

from avs_3d.echosegnet.aisrm.audio_informed_refinement import (
    run_dbscan_clustering,
    get_distance,
    audio_informed_refiner
)

from avs_3d.metrics import mask_iou, Eval_Fmeasure


def setup_logging(quiet: bool = False):
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=level)


def get_combined_args(parser: ArgumentParser) -> Namespace:
    """Parse CLI args and merge with cfg_args file in the scene folder (CLI overrides file).
    """
    args_cmdline = parser.parse_args()
    model_path = args_cmdline.path

    cfgfile_string = "Namespace()"
    cfgfilepath = os.path.join(model_path, "cfg_args")
    if os.path.exists(cfgfilepath):
        logging.info("Config file loaded: %s", cfgfilepath)
        with open(cfgfilepath, "r") as f:
            cfgfile_string = f.read()
    else:
        logging.info("No config file found at %s, using defaults.", cfgfilepath)

    args_cfgfile = eval(cfgfile_string)
    merged = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v is not None:
            merged[k] = v
    return Namespace(**merged)


def save_gs(pc, indices_mask, save_path: str):
    xyz = pc._xyz.detach().cpu()[indices_mask].numpy()
    normals = np.zeros_like(xyz)
    f_dc = pc._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu()[indices_mask].numpy()
    f_rest = pc._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu()[indices_mask].numpy()
    opacities = pc._opacity.detach().cpu()[indices_mask].numpy()
    scale = pc._scaling.detach().cpu()[indices_mask].numpy()
    rotation = pc._rotation.detach().cpu()[indices_mask].numpy()
    audio_intensity = pc._audio_intensity.detach().cpu()[indices_mask].numpy()

    dtype_full = [(attribute, 'f4') for attribute in pc.construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, audio_intensity.reshape(-1, 1)), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    PlyData([el]).write(save_path)


def main():
    parser = ArgumentParser(description="3D Audio-Visual Segmentation: lifting + AISRM")
    model_params = ModelParams(parser, sentinel=True)
    _pipeline_params = PipelineParams(parser)

    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--iteration', type=int, default=30000)
    parser.add_argument('--gd_interval', type=int, default=4)
    parser.add_argument('--use_aisrm', action='store_true')
    parser.add_argument('--threshold_value', type=float, default=0.3)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--eps', type=float, default=0.04)
    parser.add_argument('--min_samples', type=int, default=6)
    parser.add_argument('--quiet', action='store_true')

    args = get_combined_args(parser)    
    scene_dir = os.path.dirname(args.path)
    
    setup_logging(args.quiet)
    logging.info("Starting segmentation pipeline for path %s", args.path)
    dataset = model_params.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    train_cameras = scene.getTrainCameras()
    test_cameras = scene.getTestCameras()

    dataset.white_background = False
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda:0')

    xyz = gaussians.get_xyz

    # load per-view masks and assemble multiview masks
    multiview_masks = []
    masks_to_lift = []
    for i, view in enumerate(train_cameras):
        image_name = view.image_name
        mask_path = os.path.join(scene_dir, 'owod_bind_train_frames', image_name + '.png')
        if not os.path.exists(mask_path):
            logging.warning('Mask not found for %s', mask_path)
            continue
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = torch.from_numpy(mask / 255.).to('cuda:0').long()
        masks_to_lift.append(mask)

        point_mask, indices_mask = mask_inverse(xyz, view, mask)
        multiview_masks.append(point_mask.unsqueeze(-1))

    # Run ensemble voting, modified approach from 
    # Xu et al., "SAGD: Boundary-Enhanced Segment Anything in 3D Gaussian via Gaussian Decomposition"
    _, final_mask = ensemble(multiview_masks, args.threshold_value)
    mapping_dict = {i: int(e) for i, e in enumerate(final_mask.cpu().numpy())}

    grid_points = gaussians._xyz.detach().cpu().numpy()
    target_object_grid_points = grid_points[final_mask]

    # Audio intensity processing
    audio_intensity_values = gaussians._audio_intensity.detach().cpu().numpy()
    audio_intensity = np.where(audio_intensity_values > 0.85, audio_intensity_values, 0)
    audio_intensity_mask = audio_intensity > 0
    normalized_audio_intensity = audio_intensity[audio_intensity_mask]
    if normalized_audio_intensity.size:
        normalized_audio_intensity = normalized_audio_intensity / np.sum(normalized_audio_intensity)
        audio_intensity_av = np.average(grid_points[audio_intensity_mask.squeeze(-1)], axis=0, weights=normalized_audio_intensity)
    else:
        audio_intensity_av = None

    # Run DBSCAN clustering and get large clusters
    large_cluster_masks = run_dbscan_clustering(target_object_grid_points, eps=args.eps, min_samples=args.min_samples)

    # Run AISRM refinement and get the large cluster closest to the Audio Intensity Centre
    chosen_mask = None
    if args.use_aisrm and large_cluster_masks:
        chosen_mask = audio_informed_refiner(large_cluster_masks, target_object_grid_points, audio_intensity_av)

    if chosen_mask is not None:
        target_cluster_mask = chosen_mask
        final_mask_after_filtration = torch.tensor([mapping_dict[i] for i, e in enumerate(target_cluster_mask) if e], dtype=torch.int64)
    else:
        final_mask_after_filtration = final_mask

    # Gaussian decomposition loop
    for i, view in enumerate(train_cameras):
        if args.gd_interval != -1 and i % args.gd_interval == 0:
            input_mask = masks_to_lift[i]
            gaussians = gaussian_decomp(gaussians, view, input_mask, final_mask_after_filtration.to('cuda:0'))

    # Save PLY
    save_gd_path = os.path.join(args.path, f'point_cloud/iteration_{args.iteration}/point_cloud_seg_gd_use_aisrm_{args.use_aisrm}.ply')
    save_gs(gaussians, final_mask_after_filtration, save_gd_path)

    # Render predicted segmentation using saved Gaussians
    seg_gaussians = GaussianModel(dataset.sh_degree)
    seg_gaussians.load_ply(save_gd_path)

    # train_renders_path = os.path.join(scene_dir, f'3d_segmentation_use_aisrm_{args.use_aisrm}', 'renders', 'train')
    test_renders_path = os.path.join(scene_dir, f'3d_segmentation_use_aisrm_{args.use_aisrm}', 'renders', 'test')
    # os.makedirs(train_renders_path, exist_ok=True)
    os.makedirs(test_renders_path, exist_ok=True)

    # train_renders_mask_path = os.path.join(scene_dir, f'3d_segmentation_use_aisrm_{args.use_aisrm}', 'masks', 'train')
    test_renders_mask_path = os.path.join(scene_dir, f'3d_segmentation_use_aisrm_{args.use_aisrm}', 'masks', 'test')
    # os.makedirs(train_renders_mask_path, exist_ok=True)
    os.makedirs(test_renders_mask_path, exist_ok=True)

    predicted_rendered_masks = []
    ground_truth_masks = []

    for idx, view in enumerate(test_cameras):
        image_name = view.image_name
        gt_mask_path = os.path.join(scene_dir, 'semantic_masks', image_name + '.png')
        if not os.path.exists(gt_mask_path):
            logging.warning('GT mask not found: %s', gt_mask_path)
            continue
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE) / 255.
        ground_truth_masks.append(gt_mask)

        render_pkg = render(view, seg_gaussians, _pipeline_params, background)
        render_image = (255 * np.clip(render_pkg['render'].permute(1, 2, 0).detach().cpu().numpy(), 0, 1)).astype('uint8')
        render_image = cv2.cvtColor(render_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(test_renders_path, f'{image_name}.png'), render_image)

        render_alpha = (render_pkg['alpha'].squeeze(0).detach().cpu().numpy() > 0.8).astype('uint8')
        predicted_rendered_masks.append(render_alpha)
        cv2.imwrite(os.path.join(test_renders_mask_path, f'{image_name}.png'), (255 * render_alpha).astype('uint8'))

    # Compute metrics
    predicted_rendered_masks = torch.from_numpy(np.array(predicted_rendered_masks)).to('cuda:0')
    ground_truth_masks = torch.from_numpy(np.array(ground_truth_masks)).to('cuda:0')

    metrics = {
        'mIoU': float(mask_iou(predicted_rendered_masks, ground_truth_masks).cpu().numpy()),
        'FScore': float(Eval_Fmeasure(predicted_rendered_masks, ground_truth_masks)),
        'use_AISRM': bool(args.use_aisrm)
    }

    logging.info('Metrics: %s', metrics)
    metrics_path = os.path.join(scene_dir, f'3d_segmentation_use_aisrm_{args.use_aisrm}/metrics_use_aisrm_{args.use_aisrm}.json')
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as fh:
        json.dump(metrics, fh, indent=4)


if __name__ == '__main__':
    main()
