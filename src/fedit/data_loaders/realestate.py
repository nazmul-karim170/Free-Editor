import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import glob
import cv2
import pickle 

class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.intrinsics = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def unnormalize_intrinsics(intrinsics, h, w):
    intrinsics[0] *= w
    intrinsics[1] *= h
    return intrinsics


def parse_pose_file(file):
    f = open(file, "r")
    cam_params = {}
    for i, line in enumerate(f):
        if i == 0:
            continue
        entry = [float(x) for x in line.split()]
        id = int(entry[0])
        cam_params[id] = Camera(entry)
    return cam_params


# only for training
class RealEstateDataset(Dataset):
    def __init__(self, args, mode, **kwargs):
        self.folder_path = os.path.join(args.rootdir, "../../../data/realestate10k/")
        self.mode = mode                    ## train / test / validation
        self.num_source_views = args.num_source_views
        self.target_h, self.target_w = 450, 800
        assert mode in ["train", "test"]
        self.camera_path =  os.path.join(self.folder_path, "cameras")
        self.frame_path = os.path.join(self.folder_path, 'frames')
        train_scenes = [name for name in os.listdir(self.frame_path) if os.path.isdir(os.path.join(self.frame_path, name))]

        for scene in train_scenes:
            self.scene_path =  os.path.join(self.frame_path, scene)

            ## Load the Metadata containing 
            with open(self.scene_path.joinpath(f"{scene}_edited_metadata.pkl"), "rb") as file:
                self.scene_metadata = pickle.load(file)
            
            self.metadata.extend(self.scene_metadata)

    def __len__(self):
        return len(self.all_rgb_files)

    def __getitem__(self, idx):

        ## Load the starting and target view
        metadata = self.metadata[idx]
        starting_view  = imageio.imread(os.path.join(self.folder_path, metadata["starting_view_file"]))
        rgb    = imageio.imread(os.path.join(self.folder_path,metadata["target_view_file"]))
        # render_pose    = metadata["render_pose"] 
        camera  = metadata["target_camera_matrices"]
        start_camera   = metadata["starting_camera_matrices"]
        scene_name     = metadata["scene_name"]
        nearest_pose_ids = metadata["nearest_pose_ids"]          ## make sure to select at least (2*self.num_source_views) 
        nearest_pose_ids = np.random.choice(nearest_pose_ids, self.num_source_views, replace=False)

        scene_path =  os.path.join(self.folder_path, scene_name)
        rgb_files = [os.path.join(scene_path, f) for f in sorted(os.listdir(scene_path))]
        timestamps = [int(os.path.basename(rgb_file).split(".")[0]) for rgb_file in rgb_files]
        sorted_ids = np.argsort(timestamps)
        rgb_files = np.array(rgb_files)[sorted_ids]
        timestamps = np.array(timestamps)[sorted_ids]

        assert (timestamps == sorted(timestamps)).all()
        # num_frames = len(rgb_files)
        # window_size = 32
        # shift = np.random.randint(low=-1, high=2)
        # id_render = np.random.randint(low=4, high=num_frames - 4 - 1)

        ## resize the image to target size
        rgb = cv2.resize(rgb, dsize=(self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
        rgb = rgb.astype(np.float32) / 255.0

        camera_file = os.path.join(self.camera_path, scene_name) + ".txt"
        cam_params = parse_pose_file(camera_file)
        
        # get depth range
        depth_range = torch.tensor([1.0, 100.0])

        src_rgbs = []
        src_cameras = []

        ## Starting View 
        starting_view = cv2.resize(starting_view, dsize=(self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
        starting_view = starting_view.astype(np.float32) / 255.0
        src_rgbs.append(starting_view)
        src_cameras.append(start_camera)

        for id in id_feat:
            src_rgb = imageio.imread(rgb_files[id])
            
            ## Resize the image to target size
            src_rgb = cv2.resize(
                src_rgb, dsize=(self.target_w, self.target_h), interpolation=cv2.INTER_AREA
            )
            src_rgb = src_rgb.astype(np.float32) / 255.0
            src_rgbs.append(src_rgb)

            img_size = src_rgb.shape[:2]
            cam_param = cam_params[timestamps[id]]
            src_camera = np.concatenate(
                (
                    list(img_size),
                    unnormalize_intrinsics(
                        cam_param.intrinsics, self.target_h, self.target_w
                    ).flatten(),
                    cam_param.c2w_mat.flatten(),
                )
            ).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs)
        src_cameras = np.stack(src_cameras)

        return {
            "rgb": torch.from_numpy(rgb),
            "camera": torch.from_numpy(camera),
            "rgb_path": os.path.join(self.folder_path,metadata["target_view_file"]),
            "src_rgbs": torch.from_numpy(src_rgbs),      ## First one is the starting view
            "src_cameras": torch.from_numpy(src_cameras),
            "depth_range": depth_range,
        }
