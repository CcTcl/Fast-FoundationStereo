"""Visualize input/output across weather conditions — one sample per condition."""
import os, sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

import cv2, imageio, torch, yaml, numpy as np
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, set_seed, vis_disparity

proj_root = os.path.join(code_dir, '..')
out_dir = os.path.join(proj_root, 'output/full_eval/vis_conditions')
os.makedirs(out_dir, exist_ok=True)

MODEL = 'weights/20-30-48/model_best_bp2_serialize.pth'
CONDS = ['foggy', 'cloudy', 'sunny', 'rainy']
SAMPLE_IDX = 0  # first frame per condition

set_seed(0)
torch.autograd.set_grad_enabled(False)

# Load model
cfg_path = os.path.join(os.path.dirname(MODEL), 'cfg.yaml')
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
cfg['valid_iters'] = 4
cfg['max_disp'] = 192
cfg = OmegaConf.create(cfg)
model = torch.load(MODEL, map_location='cpu', weights_only=False)
model.args.valid_iters = 4
model.args.max_disp = 192
if 'normalize' not in model.args:
    model.args.normalize = True
model.cuda().eval()

rows = []

for cond in CONDS:
    # Find sample frame
    base = os.path.join(proj_root, 'data', cond, cond)
    left_dir = os.path.join(base, 'left-image-half-size')
    right_dir = os.path.join(base, 'right-image-half-size')
    disp_dir = os.path.join(base, 'disparity-map-half-size')

    left_files = sorted(os.listdir(left_dir))
    fname = left_files[SAMPLE_IDX]
    stem = os.path.splitext(fname)[0]

    left = imageio.imread(os.path.join(left_dir, fname))
    right = imageio.imread(os.path.join(right_dir, fname))

    # GT disparity
    gt = cv2.imread(os.path.join(disp_dir, stem + '.png'), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 256.0

    # Inference
    img0 = left[..., :3].copy()
    img1 = right[..., :3].copy()
    H, W = img0.shape[:2]

    img0_t = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    img1_t = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0_t.shape, divis_by=32, force_square=False)
    img0_t, img1_t = padder.pad(img0_t, img1_t)

    with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
        pred = model.forward(img0_t, img1_t, iters=4, test_mode=True, optimize_build_volume='pytorch1')
    pred = padder.unpad(pred.float())
    pred = pred.data.cpu().numpy().reshape(H, W)

    # Match resolution if needed
    if gt.shape != pred.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Error map
    valid = gt > 0
    err_map = np.zeros_like(gt)
    err_map[valid] = np.abs(pred[valid] - gt[valid])
    err_map = np.clip(err_map, 0, 5)  # cap at 5px for visualization

    # Visualize disparities
    gt_vis = vis_disparity(gt, invalid_thres=1e9)
    pred_vis = vis_disparity(pred)
    err_vis = cv2.applyColorMap((err_map / 5 * 255).astype(np.uint8), cv2.COLORMAP_HOT)[..., ::-1]
    err_vis[~valid] = 0

    # Concatenate row: [left | right | GT disp | Pred disp | Error map]
    left_rgb = cv2.resize(left[..., :3], (gt.shape[1], gt.shape[0]))
    right_rgb = cv2.resize(right[..., :3], (gt.shape[1], gt.shape[0]))

    row = np.concatenate([left_rgb, right_rgb, gt_vis, pred_vis, err_vis], axis=1)

    # Add condition label
    label_h = 28
    label = np.ones((label_h, row.shape[1], 3), dtype=np.uint8) * 40
    cv2.putText(label, f'{cond}  |  Left  |  Right  |  GT Disparity  |  Predicted  |  Error (0-5px)',
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    row = np.concatenate([label, row], axis=0)
    rows.append(row)

# Resize all rows to same width
max_w = max(r.shape[1] for r in rows)
rows = [cv2.resize(r, (max_w, r.shape[0])) if r.shape[1] != max_w else r for r in rows]

# Stack all conditions vertically
result = np.concatenate(rows, axis=0)

# Scale for display (max width 1800px)
scale = min(1800 / result.shape[1], 1.0)
if scale < 1:
    result = cv2.resize(result, None, fx=scale, fy=scale)

out_path = os.path.join(out_dir, 'all_conditions.png')
imageio.imwrite(out_path, result)
print(f'Saved: {out_path} ({result.shape[1]}x{result.shape[0]})')
