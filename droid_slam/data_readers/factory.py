
import pickle
import os
import os.path as osp

# RGBD-Dataset
from .tartan import TartanAir
from .vslamlab import VSLAM_LAB

from .stream import ImageStream
from .stream import StereoStream
from .stream import RGBDStream

# streaming datasets for inference
from .tartan import TartanAirStream
from .tartan import TartanAirTestStream
from .vslamlab import VSLAM_LAB_Stream

def dataset_factory(dataset_list, **kwargs):
    """ create a combined dataset """

    from torch.utils.data import ConcatDataset

    if dataset_list[0] == 'tartan' in dataset_list:
        dataset_map = { 'tartan': (TartanAir, ) }
    else:
        dataset_map = { 'vslamlab': (VSLAM_LAB, ) }

    db_list = []
    for key in dataset_list:
        # cache datasets for faster future loading
        db = dataset_map[key][0](**kwargs)

        print("Dataset {} has {} images".format(key, len(db)))
        db_list.append(db)

    return ConcatDataset(db_list)
            

def create_datastream(dataset_path, **kwargs):
    """ create data_loader to stream images 1 by 1 """

    from torch.utils.data import DataLoader

    if 'TartanAir' in dataset_path:
        db = TartanAirStream(dataset_path, **kwargs)
    else:
        db = VSLAM_LAB_Stream(dataset_path, **kwargs)

    # if osp.isfile(osp.join(dataset_path, 'calibration.txt')):
    #     db = ETH3DStream(dataset_path, **kwargs)

    # elif osp.isdir(osp.join(dataset_path, 'image_left')):
    #     db = TartanAirStream(dataset_path, **kwargs)

    # elif osp.isfile(osp.join(dataset_path, 'rgb.txt')):
    #     db = TUMStream(dataset_path, **kwargs)

    # elif osp.isdir(osp.join(dataset_path, 'mav0')):
    #     db = EurocStream(dataset_path, **kwargs)

    # elif osp.isfile(osp.join(dataset_path, 'calib.txt')):
    #     db = KITTIStream(dataset_path, **kwargs)

    # else:
    #     # db = TartanAirStream(dataset_path, **kwargs)
    #     db = TartanAirTestStream(dataset_path, **kwargs)
    
    stream = DataLoader(db, shuffle=False, batch_size=1, num_workers=4)
    return stream


def create_imagestream(dataset_path, **kwargs):
    """ create data_loader to stream images 1 by 1 """
    from torch.utils.data import DataLoader

    db = ImageStream(dataset_path, **kwargs)
    return DataLoader(db, shuffle=False, batch_size=1, num_workers=4)

def create_stereostream(dataset_path, **kwargs):
    """ create data_loader to stream images 1 by 1 """
    from torch.utils.data import DataLoader

    db = StereoStream(dataset_path, **kwargs)
    return DataLoader(db, shuffle=False, batch_size=1, num_workers=4)

def create_rgbdstream(dataset_path, **kwargs):
    """ create data_loader to stream images 1 by 1 """
    from torch.utils.data import DataLoader

    db = RGBDStream(dataset_path, **kwargs)
    return DataLoader(db, shuffle=False, batch_size=1, num_workers=4)

