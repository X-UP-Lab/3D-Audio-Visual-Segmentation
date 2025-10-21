# Implementation of OWOD-BIND
# Swapnil Bhosale, Haosen Yang, Diptesh Kanojia, and Xiatian Zhu.
# "Leveraging Foundation Models for Unsupervised Audio-Visual Segmentation."
# arXiv preprint arXiv:2309.06728, 2023.
# URL: https://api.semanticscholar.org/CorpusID:261705977
#
# Some parts of the implementation are inspired by:
# https://github.com/IDEA-Research/Grounded-Segment-Anything

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import argparse
import os
from imagebind import data
import cv2
import json
import torch
import numpy as np
import logging
import shutil
from tqdm import tqdm
from PIL import Image, ImageDraw
import yaml

from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from segment_anything import sam_model_registry, SamPredictor
from avs_3d.echosegnet.owod_bind.utils import (
    load_config, 
    segment_image, 
    get_indices_of_values_above_threshold, 
    load_and_transform_vision_data_from_pil_image
)

from torchvision import transforms
from avs_3d.echosegnet.owod_bind.submodules.caod.models.model import Model as CaodModel


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def run_caod_inference(model, image_paths, root_dir):
    """
    Run CAOD model inference
    """
    detections_map = {}
    for image_path in image_paths:
        rel_key = os.path.relpath(image_path, start=root_dir)
        detections = model.infer_image(image_path, caption=None)
        detections_map[rel_key] = detections
    return detections_map

def retriev_vision_and_audio(elements, audio_list, device, bind_model):
    """Retrieve joint vision-audio embeddings."""
    inputs = {
        ModalityType.VISION: load_and_transform_vision_data_from_pil_image(elements, device),
        ModalityType.AUDIO: data.load_and_transform_audio_data(audio_list, device),
    }
    with torch.no_grad():
        embeddings = bind_model(inputs)

    return torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run OWOD-Bind pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Path to the dataset root directory",
    )
    args = parser.parse_args()

    CONFIG = load_config(args.config)
    logging.info(f"Loaded configuration from {args.config}")

    root_dir = args.root_dir
    device = CONFIG["device"]

    # --------------------- Stage 1: Run CAOD ---------------------
    logging.info("=== Stage 1: Running CAOD inference ===")
    caod_cfg = CONFIG["caod"]
    caod_model = CaodModel(caod_cfg["model_name"], caod_cfg["checkpoint"]).get_model()

    # Find all 'input' directories to process
    input_dirs = [
        os.path.join(dirpath, "input")
        for dirpath, dirnames, _ in os.walk(root_dir)
        if "input" in dirnames
    ]

    all_caod_detections = {}
    for images_dir in tqdm(input_dirs, desc="CAOD Global Processing"):
        logging.info(f"Running CAOD on directory: {images_dir}")
        image_paths = [
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        scene_detections = run_caod_inference(caod_model, image_paths, root_dir)
        all_caod_detections.update(scene_detections)

    del caod_model # to clear VRAM
    torch.cuda.empty_cache()
    logging.info(f"Completed CAOD inference on {len(all_caod_detections)} total images.")

    # --------------------- Stage 2: Load SAM + ImageBind ---------------------
    logging.info("=== Stage 2: Loading SAM and ImageBind models ===")
    sam_cfg = CONFIG["sam"]
    sam_model = sam_model_registry[sam_cfg["model_name"]](checkpoint=sam_cfg["checkpoint"])
    sam_model.to(device=device)
    sam_predictor = SamPredictor(sam_model)

    bind_model = imagebind_model.imagebind_huge(pretrained=True)
    bind_model.eval()
    bind_model.to(device)

    # --------------------- Stage 3: Process Scenes ---------------------
    logging.info("=== Stage 3: Processing scenes ===")
    scenes = [
        os.path.dirname(d) for d in input_dirs 
    ]

    for scene_path in tqdm(scenes, desc="Processing Scenes"):
        logging.info(f"Processing scene: {scene_path}")
        
        if not os.path.exists(os.path.join(scene_path, "3dgs_scene")):
            logging.warning(f"Missing directory '3dgs_scene' for {scene_path}, skipping")
            continue
        
        with open(os.path.join(scene_path, "3dgs_scene", "train_cameras.json"), "r") as f:
            train_cameras = sorted([x["img_name"] for x in json.load(f)])

        owod_bind_path = os.path.join(scene_path, "owod_bind_train_frames")
        os.makedirs(owod_bind_path, exist_ok=True)

        for img_name in tqdm(train_cameras, desc=f"Frames in {os.path.basename(scene_path)}", leave=False):
            
            # 1. RETRIEVE CAOD DATA
            img_rel_path = os.path.relpath(
                os.path.join(scene_path, "input", f"{img_name}.png"), 
                start=root_dir
            )
            caod_bb = all_caod_detections.get(img_rel_path)
            if not caod_bb:
                logging.warning(f"No CAOD detection for {img_name}, skipping.")
                continue

            # 2. LOAD IMAGE
            img_path = os.path.join(scene_path, "input", f"{img_name}.png")
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.open(img_path)

            # 3. PROCESS BOUNDING BOXES
            bounding_boxes, scores = caod_bb
            img_height = img.shape[0]
            processed_boxes = []

            for bb, score in zip(bounding_boxes, scores):
                if score > CONFIG["caod"]["confidence_threshold"]:
                    if ( 
                        abs(bb[0] - bb[2]) < np.shape(img)[0] * 0.98 
                        and 
                        abs(bb[1] - bb[3]) < np.shape(img)[0] * 0.98 
                    ):
                        processed_boxes.append(bb)

                if len(processed_boxes) == 0:
                    processed_boxes.append(bounding_boxes[0])

            # 4. RUN SAM
            boxes_tensor = torch.tensor(processed_boxes, device=sam_predictor.device)
            sam_predictor.set_image(img)
            masks, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=sam_predictor.transform.apply_boxes_torch(boxes_tensor, img.shape[:2]),
                multimask_output=False,
            )
            masks_np = masks.cpu().numpy()

            # 5. CROP SEGMENTED OBJECTS
            cropped_objects = [
                segment_image(pil_img, mask[0]).crop(box)
                for mask, box in zip(masks_np, processed_boxes)
            ]

            # 6. RUN IMAGEBIND
            audio_clip_path = os.path.join(
                scene_path, "audio_clips_mono", f"frame_{int(img_name.split('_')[-1])}.wav"
            )
            vision_audio_scores = retriev_vision_and_audio(
                cropped_objects, [audio_clip_path], device, bind_model
            )
            
            indices_above_threshold = get_indices_of_values_above_threshold(
                vision_audio_scores, CONFIG["vision_audio_threshold"]
            )

            # 7. GENERATE AND SAVE FINAL MASK
            if indices_above_threshold: 
                selected_masks = masks_np[np.array(indices_above_threshold)]
                combined_mask = np.any(selected_masks.squeeze(axis=1), axis=0)
                final_mask = (combined_mask * 255).astype(np.uint8)
            else:
                final_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            
            Image.fromarray(final_mask).save(
                os.path.join(owod_bind_path, f"{img_name}.png"), format="PNG"
            )

    logging.info("=== Pipeline finished successfully! ===")
