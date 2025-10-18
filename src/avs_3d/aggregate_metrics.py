from pathlib import Path
import json
import numpy as np
import argparse


def main(root_dir: Path, part: int, mode: str):
    if part == 1:
        part_dir = "part_1"
    elif part == 2:
        part_dir = "part_2"
    else:
        raise ValueError("Part must be 1 or 2.")

    base_root_dir = root_dir / part_dir

    metrics_aisrm = {'mIoU': [], 'FScore': []}
    metrics_no_aisrm = {'mIoU': [], 'FScore': []}
    results = []

    flats = ["5ZKStnWn8Zo", "q9vSo1VnCiC", "sT4fr6TAbpF", "wc2JMjhGNzB"]
    rooms = ["bathroom", "kitchen", "living_room"]
    indices = ["1", "2", "3"]

    for flat in flats:
        for room in rooms:
            for idx in indices:
                scene_path = base_root_dir / flat / room / idx
                target_output_path = scene_path

                if not target_output_path.exists():
                    continue

                # Define paths for both AISRM and non-AISRM metrics
                aisrm_subdir = "3d_segmentation_use_aisrm_True"
                no_aisrm_subdir = "3d_segmentation_use_aisrm_False"
                
                metrics_path_aisrm = target_output_path / aisrm_subdir / "metrics_use_aisrm_True.json"
                metrics_path_no_aisrm = target_output_path / no_aisrm_subdir / "metrics_use_aisrm_False.json"

                required_metrics_exist = (
                    (mode == 'aisrm' and metrics_path_aisrm.exists()) or
                    (mode == 'no_aisrm' and metrics_path_no_aisrm.exists()) or
                    (mode == 'compare' and metrics_path_aisrm.exists() and metrics_path_no_aisrm.exists())
                )

                if not required_metrics_exist:
                    print(f"[WARN] Missing metrics (Mode: {mode})")
                    continue
                
                metric_aisrm, metric_no_aisrm = None, None
                
                if mode in ['aisrm', 'compare']:
                    with open(metrics_path_aisrm) as f:
                        metric_aisrm = json.load(f)
                if mode in ['no_aisrm', 'compare']:
                    with open(metrics_path_no_aisrm) as f:
                        metric_no_aisrm = json.load(f)
                
                if metric_aisrm:
                    metrics_aisrm['mIoU'].append(metric_aisrm.get('mIoU'))
                    metrics_aisrm['FScore'].append(metric_aisrm.get('FScore'))
                
                if metric_no_aisrm:
                    metrics_no_aisrm['mIoU'].append(metric_no_aisrm.get('mIoU'))
                    metrics_no_aisrm['FScore'].append(metric_no_aisrm.get('FScore'))
    
    count_aisrm = len(metrics_aisrm['mIoU'])
    count_no_aisrm = len(metrics_no_aisrm['mIoU'])
    count = max(count_aisrm, count_no_aisrm)

    print(f"\n--- 3DAVS Evaluation (Part {part}) ---")
    print(f"Total valid scenes: {count}")

    if count > 0:
        if mode in ['aisrm', 'compare'] and count_aisrm > 0:
            print("\nWith AISRM:")
            print(f"  mIoU   = {np.mean(metrics_aisrm['mIoU'])}")
            print(f"  FScore = {np.mean(metrics_aisrm['FScore'])}")

        if mode in ['no_aisrm', 'compare'] and count_no_aisrm > 0:
            print("\nWithout AISRM:")
            print(f"  mIoU   = {np.mean(metrics_no_aisrm['mIoU'])}")
            print(f"  FScore = {np.mean(metrics_no_aisrm['FScore'])}")
        
        if mode == 'compare' and count_aisrm > 0 and count_no_aisrm > 0:
            print("\nDifference (AISRM - No AISRM):")
            miou_diff = np.mean(metrics_aisrm['mIoU']) - np.mean(metrics_no_aisrm['mIoU'])
            fscore_diff = np.mean(metrics_aisrm['FScore']) - np.mean(metrics_no_aisrm['FScore'])
            print(f"  mIoU Diff   = {miou_diff}")
            print(f"  FScore Diff = {fscore_diff}")
            
    else:
        print("No valid scenes found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 3DAVS AISRM metrics for Part 1 or Part 2.")
    parser.add_argument("--root", type=str, required=True, 
                        help="Path to the main 3DAVS dataset root (e.g., /path/to/generated_dataset/).")
    parser.add_argument("--part", type=int, choices=[1, 2], required=True, 
                        help="Which dataset part to evaluate (1 or 2).")
    parser.add_argument("--mode", type=str, choices=['compare', 'aisrm', 'no_aisrm'], default='compare',
                        help="Evaluation mode. 'compare' (default) shows both, 'aisrm' only shows with AISRM, 'no_aisrm' only shows without AISRM.")
    
    args = parser.parse_args()
    main(Path(args.root), args.part, args.mode)