# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Evaluate Fast-FoundationStereo on DrivingStereo dataset.

Usage:
  python scripts/eval_drivingstereo.py \
    --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
    --dataset_dir /path/to/DrivingStereo/train/2018-10-11-17-08-31 \
    --out_dir output/eval \
    --max_samples 10
"""

import os, sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

import argparse
import logging
import json
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

from core.utils.utils import InputPadder
from core.utils.frame_utils import readPFM
from Utils import AMP_DTYPE, set_logging_format, set_seed, vis_disparity


def compute_metrics(pred_disp, gt_disp, valid_mask=None):
    """Compute standard stereo metrics.

    Args:
        pred_disp: predicted disparity (H, W), float32
        gt_disp: ground truth disparity (H, W), float32
        valid_mask: optional bool mask (H, W). If None, uses gt_disp > 0.

    Returns:
        dict with EPE, bad_1, bad_3, D1_all, RMSE
    """
    if valid_mask is None:
        valid_mask = (gt_disp > 0) & np.isfinite(gt_disp)

    if valid_mask.sum() == 0:
        return None

    diff = np.abs(pred_disp - gt_disp)[valid_mask]
    gt_valid = gt_disp[valid_mask]

    epe = float(diff.mean())
    bad_1 = float((diff > 1.0).mean() * 100)
    bad_3 = float((diff > 3.0).mean() * 100)
    rmse = float(np.sqrt((diff ** 2).mean()))

    # D1-all: error > max(3px, 5% of GT)
    d1_thresh = np.maximum(3.0, 0.05 * gt_valid)
    d1_all = float((diff > d1_thresh).mean() * 100)

    return {
        'epe': epe,
        'bad_1': bad_1,
        'bad_3': bad_3,
        'd1_all': d1_all,
        'rmse': rmse,
        'valid_pixels': int(valid_mask.sum()),
        'total_pixels': int(valid_mask.size),
    }


def is_scene_dir(path):
    """Check if a directory is a DrivingStereo scene (has left/, right/, disparity/)."""
    p = Path(path)
    return (p / 'left').is_dir() and (p / 'right').is_dir() and (p / 'disparity').is_dir()


def find_pairs(dataset_dir):
    """Find all (left, right, disparity) triplets in one or more DrivingStereo scenes.

    Supports:
      - Single scene: --dataset_dir /path/to/scene/
      - Multiple scenes: --dataset_dir /path/to/train/ (contains many scene dirs)

    Returns:
        list of (left_path, right_path, disp_path) tuples sorted by filename.
    """
    root = Path(dataset_dir)

    # Determine scene directories
    if is_scene_dir(root):
        scene_dirs = [root]
    else:
        scene_dirs = sorted(d for d in root.iterdir() if d.is_dir() and is_scene_dir(d))
        if not scene_dirs:
            logging.warning(f"No DrivingStereo scene directories found under {dataset_dir}. "
                            f"Trying as single scene anyway...")
            scene_dirs = [root]

    pairs = []
    for scene_dir in scene_dirs:
        left_dir = scene_dir / 'left'
        right_dir = scene_dir / 'right'
        disp_dir = scene_dir / 'disparity'

        for left_path in sorted(left_dir.iterdir()):
            if left_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            img_name = left_path.name
            stem = left_path.stem

            # Right image: same filename
            right_path = right_dir / img_name
            if not right_path.exists():
                for ext in ('.jpg', '.jpeg', '.png'):
                    alt = right_dir / (stem + ext)
                    if alt.exists():
                        right_path = alt
                        break

            # Disparity: same stem with .pfm extension
            disp_path = disp_dir / (stem + '.pfm')
            if not disp_path.exists():
                continue

            if right_path.exists():
                pairs.append((str(left_path), str(right_path), str(disp_path)))

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to serialized .pth model')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to DrivingStereo scene directory (contains left/, right/, disparity/)')
    parser.add_argument('--out_dir', type=str, default='output/drivingstereo_eval',
                        help='Output directory')
    parser.add_argument('--valid_iters', type=int, default=8,
                        help='Number of refinement updates')
    parser.add_argument('--max_disp', type=int, default=192,
                        help='Maximum disparity')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Image scaling factor')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit number of samples (for quick test)')
    parser.add_argument('--save_vis', action='store_true',
                        help='Save disparity visualizations')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load model ---
    cfg_path = os.path.join(os.path.dirname(args.model_dir), 'cfg.yaml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    for k in args.__dict__:
        if args.__dict__[k] is not None:
            cfg[k] = args.__dict__[k]
    cfg = OmegaConf.create(cfg)
    logging.info(f"Config:\n{cfg}")

    model = torch.load(args.model_dir, map_location='cpu', weights_only=False)
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp
    model.cuda().eval()

    # --- Find samples ---
    pairs = find_pairs(args.dataset_dir)
    if not pairs:
        logging.error(f"No valid (left, right, disparity) triplets found in {args.dataset_dir}")
        sys.exit(1)
    logging.info(f"Found {len(pairs)} samples")

    if args.max_samples is not None:
        pairs = pairs[:args.max_samples]
        logging.info(f"Limited to {len(pairs)} samples")

    # --- Evaluate ---
    all_metrics = []
    scene_metrics = {}  # Per-scene aggregation

    for left_file, right_file, disp_file in tqdm(pairs, desc='Evaluating'):
        # Read images
        img0 = imageio.imread(left_file)
        img1 = imageio.imread(right_file)
        if len(img0.shape) == 2:
            img0 = np.tile(img0[..., None], (1, 1, 3))
            img1 = np.tile(img1[..., None], (1, 1, 3))
        img0 = img0[..., :3]
        img1 = img1[..., :3]
        H, W = img0.shape[:2]

        # Scale
        if args.scale != 1.0:
            img0 = cv2.resize(img0, fx=args.scale, fy=args.scale, dsize=None)
            img1 = cv2.resize(img1, dsize=(img0.shape[1], img0.shape[0]))

        # Inference
        img0_t = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
        img1_t = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(img0_t.shape, divis_by=32, force_square=False)
        img0_t, img1_t = padder.pad(img0_t, img1_t)

        with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
            disp = model.forward(img0_t, img1_t, iters=args.valid_iters, test_mode=True,
                                 optimize_build_volume='pytorch1')
        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(img0.shape[0], img0.shape[1])

        # Read GT disparity (at original resolution)
        gt_disp = readPFM(disp_file)

        # If we scaled the image, scale GT to match prediction resolution
        if args.scale != 1.0:
            gt_disp = cv2.resize(gt_disp, (disp.shape[1], disp.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

        # If GT resolution differs from prediction, resize prediction to match GT
        if gt_disp.shape != disp.shape:
            disp = cv2.resize(disp, (gt_disp.shape[1], gt_disp.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        # Metrics
        metrics = compute_metrics(disp, gt_disp)
        if metrics is not None:
            sample_name = os.path.splitext(os.path.basename(left_file))[0]
            scene_name = os.path.basename(os.path.dirname(os.path.dirname(left_file)))
            metrics['sample'] = sample_name
            metrics['scene'] = scene_name
            all_metrics.append(metrics)

            # Per-scene tracking
            if scene_name not in scene_metrics:
                scene_metrics[scene_name] = []
            scene_metrics[scene_name].append(metrics)

        # Optional: save disparity visualization
        if args.save_vis:
            vis_dir = os.path.join(args.out_dir, 'vis')
            os.makedirs(vis_dir, exist_ok=True)
            vis = vis_disparity(disp)
            out_name = os.path.splitext(os.path.basename(left_file))[0] + '_disp.png'
            imageio.imwrite(os.path.join(vis_dir, out_name), vis)

    # --- Aggregate results ---
    if not all_metrics:
        logging.error("No valid samples processed")
        sys.exit(1)

    # Overall averages
    total_valid = sum(m['valid_pixels'] for m in all_metrics)
    overall = {}
    for key in ('epe', 'bad_1', 'bad_3', 'd1_all', 'rmse'):
        # Weight by valid pixels
        overall[key] = sum(m[key] * m['valid_pixels'] for m in all_metrics) / total_valid

    # Per-scene summaries
    scene_summaries = {}
    for scene, ms in scene_metrics.items():
        sw = sum(m['valid_pixels'] for m in ms)
        scene_summaries[scene] = {
            key: sum(m[key] * m['valid_pixels'] for m in ms) / sw
            for key in ('epe', 'bad_1', 'bad_3', 'd1_all', 'rmse')
        }
        scene_summaries[scene]['num_samples'] = len(ms)

    # --- Print results ---
    logging.info("\n" + "=" * 60)
    logging.info("DrivingStereo Evaluation Results")
    logging.info("=" * 60)
    logging.info(f"{'Scene':<30} {'#Samples':>8} {'EPE':>8} {'bad-1%':>8} {'bad-3%':>8} {'D1-all%':>8} {'RMSE':>8}")
    logging.info("-" * 60)
    for scene, sm in scene_summaries.items():
        logging.info(f"{scene:<30} {sm['num_samples']:>8} "
                     f"{sm['epe']:>8.3f} {sm['bad_1']:>7.2f}% {sm['bad_3']:>7.2f}% "
                     f"{sm['d1_all']:>7.2f}% {sm['rmse']:>8.3f}")
    logging.info("-" * 60)
    logging.info(f"{'OVERALL':<30} {len(all_metrics):>8} "
                 f"{overall['epe']:>8.3f} {overall['bad_1']:>7.2f}% {overall['bad_3']:>7.2f}% "
                 f"{overall['d1_all']:>7.2f}% {overall['rmse']:>8.3f}")
    logging.info("=" * 60)

    # --- Save results ---
    results = {
        'overall': overall,
        'per_scene': {
            scene: {k: v for k, v in sm.items() if k != 'num_samples'}
            for scene, sm in scene_summaries.items()
        },
        'per_sample': all_metrics,
        'num_samples': len(all_metrics),
    }
    results_path = os.path.join(args.out_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
