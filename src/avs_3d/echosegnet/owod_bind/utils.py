from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import yaml
import torch

def segment_image(image, segmentation_mask):
    image_array = np.array(image)
    blurred_image_array = cv2.GaussianBlur(image_array, (21, 21), 0)
    segmented_image_array = np.where(segmentation_mask[..., None], image_array, blurred_image_array)
    segmented_image = Image.fromarray(segmented_image_array)
    return segmented_image

def get_indices_of_values_above_threshold(values, threshold):
    indices = []
    for index in range(len(values)):
        if values[index] > threshold:
            indices.append(index)
    return indices

def load_config(config_path: str):
    """Load experiment configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_and_transform_vision_data_from_pil_image(img_list, device):
    """Load and preprocess a list of PIL images for ImageBind."""
    if not img_list:
        return None

    transform_pipeline = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    image_outputs = [transform_pipeline(img).to(device) for img in img_list]
    return torch.stack(image_outputs, dim=0)



    