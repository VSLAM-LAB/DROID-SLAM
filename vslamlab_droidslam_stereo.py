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

    def read_int(key: str) -> int:
        node = fs.getNode(key)
        if node.empty():
           return 0.0
        return int(node.real())

    fx, fy, cx, cy = map(read_real, ["Camera0.fx", "Camera0.fy", "Camera0.cx", "Camera0.cy"])

    h0 = read_int("Camera0.h")
    w0 = read_int("Camera0.w")

    fs.release()

    K = np.array([[fx, 0,  cx],
                  [0,  fy, cy],
                  [0,  0,   1]], dtype=np.float32)
    return K, h0, w0


def image_stream(sequence_path: Path, rgb_csv: Path, calibration_yaml: Path, use_stereo: bool = False, image_size=[320, 512]):
    """ image generator """ 
    global timestamps 
    K, ht0, wd0 = load_calibration(calibration_yaml)

    # Load rgb images
    df = pd.read_csv(rgb_csv)       
    images_left = df['path_rgb0'].to_list()
    images_right = df['path_rgb1'].to_list()
    timestamps = df['ts_rgb0 (s)'].to_list()

    for t, (imgL, imgR) in enumerate(zip(images_left, images_right)):
        imgL = os.path.join(sequence_path, imgL)
        imgR = os.path.join(sequence_path, imgR)

        if use_stereo and not os.path.isfile(imgR):
            continue

        #images = [cv2.remap(cv2.imread(imgL), map_l[0], map_l[1], interpolation=cv2.INTER_LINEAR)]
        images = [cv2.imread(imgL)]
        if use_stereo:
            images += [cv2.imread(imgR)]
        
        images = torch.from_numpy(np.stack(images, 0))
        images = images.permute(0, 3, 1, 2).to("cuda:0", dtype=torch.float32)
        images = F.interpolate(images, image_size, mode="bilinear", align_corners=False)
        
        intrinsics_vec = [K[0,0], K[1,1], K[0,2], K[1,2]]
        intrinsics = torch.as_tensor(intrinsics_vec).cuda()
        intrinsics[0] *= image_size[1] / wd0
        intrinsics[1] *= image_size[0] / ht0
        intrinsics[2] *= image_size[1] / wd0
        intrinsics[3] *= image_size[0] / ht0

        yield t, images, intrinsics.clone()
    
def main():    
    print("\nRunning vslamlab_droidslam_stereo.py ...\n")  

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
    args.stereo = True
    args.depth = False

    torch.multiprocessing.set_start_method('spawn')

    droid = None
    for (t, image, intrinsics) in tqdm(image_stream(args.sequence_path, args.rgb_csv, args.calibration_yaml, use_stereo=True)):

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