
import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp

from lietorch import SE3
from .base import RGBDDataset
from .stream import RGBDStream

cur_path = osp.dirname(osp.abspath(__file__))
test_split = osp.join(cur_path, 'tartan_test.txt')
test_split = open(test_split).read().split()


class VSLAM_LAB(RGBDDataset):

    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0
    DEPTH_SCALE_save = 1.0

    def __init__(self, mode='training', **kwargs):
        self.mode = mode
        self.n_frames = 2
        super(VSLAM_LAB, self).__init__(name='vslamlab', **kwargs)

    @staticmethod 
    def is_test_scene(scene):
        # print(scene, any(x in scene for x in test_split))
        return any(x in scene for x in test_split)

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building VSLAM-LAB dataset")

        scene_info = {}

        #scenes = glob.glob(osp.join(self.root, '*/*/*/*'))
        scenes = glob.glob(osp.join(self.root, '*'))

        for scene in tqdm(sorted(scenes)):

            #images = sorted(glob.glob(osp.join(scene, 'image_left/*.png')))
            images = sorted(
                glob.glob(osp.join(scene, 'rgb_0_resized/*.png')) +
                glob.glob(osp.join(scene, 'rgb_0_resized/*.jpg'))
            )

            #depths = sorted(glob.glob(osp.join(scene, 'depth_left/*.npy')))
            depths = sorted(
                glob.glob(osp.join(scene, 'depth_0_npy/*.npy'))
            )

            #poses = np.loadtxt(osp.join(scene, 'pose_left.txt'), delimiter=' ')  
            #poses[:,:3] /= VSLAM_LAB.DEPTH_SCALE
            #poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]
            
            import pandas as pd
            poses_df = pd.read_csv(osp.join(scene, "groundtruth.csv"), engine='python', sep=None)
            cols = ['tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
            poses = poses_df[cols].to_numpy()
            poses[:, :3] /= VSLAM_LAB.DEPTH_SCALE # scale translation columns
            intrinsics = [VSLAM_LAB.calib_read()] * len(images)

            # graph of co-visible frames based on flow
            graph = self.build_frame_graph(poses, depths, intrinsics)
            scene = '/'.join(scene.split('/'))
            scene_info[scene] = {'images': images, 'depths': depths, 
                'poses': poses, 'intrinsics': intrinsics, 'graph': graph}

        print("Finished building VSLAM-LAB dataset")
        return scene_info

    @staticmethod
    def calib_read():
        return np.array([368.0, 367.94, 367.6933, 208.1934])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        # depth = np.load(depth_file) / VSLAM_LAB.DEPTH_SCALE
        # depth[depth==np.nan] = 1.0
        # depth[depth==np.inf] = 1.0
        # return depth
        depth = np.load(depth_file).astype(np.float32) #cv2.imread(depth_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth /= (VSLAM_LAB.DEPTH_SCALE_save * VSLAM_LAB.DEPTH_SCALE)
        depth = np.nan_to_num(depth, nan=1.0, posinf=1.0, neginf=1.0)
        depth[depth == 0] = 1.0
        return depth


class VSLAM_LAB_Stream(RGBDStream):
    def __init__(self, datapath, **kwargs):
        super(VSLAM_LAB_Stream, self).__init__(datapath=datapath, **kwargs)

    def _build_dataset_index(self):
        """ build list of images, poses, depths, and intrinsics """
        self.root = 'datasets/TartanAir'
        
        scene = osp.join(self.root, self.datapath)
        image_glob = osp.join(scene, 'image_left/*.png')
        images = sorted(glob.glob(image_glob))

        poses = np.loadtxt(osp.join(scene, 'pose_left.txt'), delimiter=' ')
        poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]

        poses = SE3(torch.as_tensor(poses))
        poses = poses[[0]].inv() * poses
        poses = poses.data.cpu().numpy()

        intrinsic = self.calib_read(self.datapath)
        intrinsics = np.tile(intrinsic[None], (len(images), 1))

        self.images = images[::int(self.frame_rate)]
        self.poses = poses[::int(self.frame_rate)]
        self.intrinsics = intrinsics[::int(self.frame_rate)]

    @staticmethod
    def calib_read(datapath):
        return np.array([368.0, 367.94, 367.6933, 208.1934])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)


class VSLAM_LAB_TestStream(RGBDStream):
    def __init__(self, datapath, **kwargs):
        super(VSLAM_LAB_TestStream, self).__init__(datapath=datapath, **kwargs)

    def _build_dataset_index(self):
        """ build list of images, poses, depths, and intrinsics """
        self.root = 'datasets/mono'
        image_glob = osp.join(self.root, self.datapath, '*.png')
        images = sorted(glob.glob(image_glob))

        poses = np.loadtxt(osp.join(self.root, 'mono_gt', self.datapath + '.txt'), delimiter=' ')
        poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]

        poses = SE3(torch.as_tensor(poses))
        poses = poses[[0]].inv() * poses
        poses = poses.data.cpu().numpy()

        intrinsic = self.calib_read(self.datapath)
        intrinsics = np.tile(intrinsic[None], (len(images), 1))

        self.images = images[::int(self.frame_rate)]
        self.poses = poses[::int(self.frame_rate)]
        self.intrinsics = intrinsics[::int(self.frame_rate)]

    @staticmethod
    def calib_read(datapath):
        return np.array([368.0, 367.94, 367.6933, 208.1934])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)
