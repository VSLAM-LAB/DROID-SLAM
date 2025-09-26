import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import pkg_resources
import pandas as pd
import numpy as np
import argparse
import torch
import yaml
import time
import csv
import cv2
import sys
import os

sys.path.append('droid_slam')
from droid_slam.droid import Droid

timestamps = []

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def load_calibration(calibration_yaml: Path):
    fs = cv2.FileStorage(str(calibration_yaml), cv2.FILE_STORAGE_READ)
    def read_real(key: str) -> float:
        node = fs.getNode(key)
        return float(node.real()) if not node.empty() else 0.0

    fx, fy, cx, cy = map(read_real, ["Camera0.fx", "Camera0.fy", "Camera0.cx", "Camera0.cy"])
    k1, k2, p1, p2, k3 = map(read_real, ["Camera0.k1", "Camera0.k2", "Camera0.p1", "Camera0.p2", "Camera0.k3"])
    fs.release()

    K = np.array([[fx, 0,  cx],
                  [0,  fy, cy],
                  [0,  0,   1]], dtype=np.float32)
    dist = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    has_dist = not np.allclose(dist, 0.0)
    return K, dist, has_dist

def image_stream(sequence_path: Path, rgb_csv: Path, calibration_yaml: Path, target_pixels: int = 384*512):
    """ image generator """
    global timestamps
    K, dist, has_dist = load_calibration(calibration_yaml)

    # Load rgb images
    df = pd.read_csv(rgb_csv)       
    image_list = df['path_rgb0'].to_list()
    timestamps = df['ts_rgb0 (s)'].to_list()

    # Undistort and resize images
    for t, imrel in enumerate(image_list):
        impath = sequence_path / imrel
        image = cv2.imread(impath)
        if has_dist:
            image = cv2.undistort(image, K, dist)

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt(target_pixels / (h0 * w0)))
        w1 = int(w0 * np.sqrt(target_pixels / (h0 * w0)))

        image = cv2.resize(image, (w1, h1), interpolation=cv2.INTER_AREA)

        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.tensor([
            K[0,0] * (w1 / w0),
            K[1,1] * (h1 / h0),
            K[0,2] * (w1 / w0),
            K[1,2] * (h1 / h0)
        ], dtype=torch.float32)
        
        yield t, image[None], intrinsics.clone()

def main():  
    print("\nRunning vslamlab_droidslam_mono.py ...\n")  

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--sequence_path", type=Path, required=True)
    parser.add_argument("--calibration_yaml", type=Path, required=True)
    parser.add_argument("--rgb_csv", type=Path, required=True)
    parser.add_argument("--exp_folder", type=Path, required=True)
    parser.add_argument("--exp_it", type=str, default="0")
    parser.add_argument("--settings_yaml", type=Path, default=None)
    parser.add_argument("--verbose", type=str, help="verbose")
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--weights", type=Path, default=None)

    args, _ = parser.parse_known_args()

    verbose = bool(int(args.verbose))
    args.disable_vis = not bool(int(args.verbose))
    args.upsample = bool(int(args.upsample))

    settings_path = args.settings_yaml
    if not os.path.exists(settings_path):
        settings_path = pkg_resources.resource_filename(
            'droid_slam.configs', 'vslamlab_droidslam-dev_settings.yaml'
        )

    with open(settings_path, 'r') as f:
        settings = yaml.safe_load(f)
    S = settings.get('settings', {})

    # Attach required settings to args
    args.t0 = int(S.get('t0', 0))
    args.stride = int(S.get('stride', 1))
    args.buffer = int(S.get('buffer', 200))
    args.beta = float(S.get('beta', 0.3))
    args.filter_thresh = float(S.get('filter_thresh', 0.1))
    args.warmup = int(S.get('warmup', 0))
    args.keyframe_thresh = float(S.get('keyframe_thresh', 1.2))
    args.frontend_thresh = float(S.get('frontend_thresh', 12.0))
    args.frontend_window = int(S.get('frontend_window', 25))
    args.frontend_radius = int(S.get('frontend_radius', 2))
    args.frontend_nms = int(S.get('frontend_nms', 2))
    args.backend_thresh = float(S.get('backend_thresh', 24.0))
    args.backend_radius = int(S.get('backend_radius', 2))
    args.backend_nms = int(S.get('backend_nms', 2))
    args.stereo = False
    args.depth = False

    torch.multiprocessing.set_start_method('spawn')

    droid = None
    for (t, image, intrinsics) in tqdm(image_stream(args.sequence_path, args.rgb_csv, args.calibration_yaml)):
        if t < args.t0:
            continue

        if verbose:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]         
            droid = Droid(args)
            time.sleep(5)

        droid.track(t, image, intrinsics=intrinsics)

    traj_est = droid.terminate(image_stream(args.sequence_path, args.rgb_csv, args.calibration_yaml))
    
    keyframe_csv = args.exp_folder / f"{args.exp_it.zfill(5)}_KeyFrameTrajectory.csv"
    with open(keyframe_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        for i in range(len(timestamps)):
            ts = timestamps[i]
            tx, ty, tz, qx, qy, qz, qw = traj_est[i][:7]
            writer.writerow([ts, tx, ty, tz, qx, qy, qz, qw])

if __name__ == '__main__':
    main()