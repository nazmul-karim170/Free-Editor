import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import glob
import sys
import pickle 

sys.path.append("../")
from .data_utils import deepvoxels_parse_intrinsics, get_nearest_pose_ids, rectify_inplane_rotation


class DeepVoxelsDataset(Dataset):
    def __init__(self, args, subset, scenes="vase", **kwargs):  # string or list

        self.folder_path = os.path.join(args.rootdir, "../../../data/deepvoxels/")
        self.rectify_inplane_rotation = args.rectify_inplane_rotation
        self.subset = subset  # train / test / validation
        self.num_source_views = args.num_source_views
        self.testskip = args.testskip
        if isinstance(scenes, str):
            scenes = [scenes]

        self.scenes = scenes
        self.scene_path_list_train = glob.glob(self.folder_path + "train") 
        self.scene_path_list_test  =  glob.glob(self.folder_path + "test") 
        self.scene_path_list_val   = glob.glob(self.folder_path + "validation")

        self.scene_path_list = []
        self.data_sets = []
        for scene in scenes:
            if str(scene) in self.scene_path_list_train:
                self.scene_path =  os.path.join(args.scene_path_list_train, scene)
                self.data_sets.append("train")
            elif str(scene) in self.scene_path_list_test:
                self.scene_path =  os.path.join(args.scene_path_list_test, scene)
                self.data_sets.append("test")
            else:
                self.scene_path =  os.path.join(args.scene_path_list_val, scene)   
                self.data_sets.append("val")             

            ## Load the Metadata containing 
            with open(self.scene_path.joinpath(f"{scene}_edited_metadata.pkl"), "rb") as file:
                self.scene_metadata = pickle.load(file)
            
            self.metadata.extend(self.scene_metadata)
            self.scene_path_list.append(self.scene_path)

    def __len__(self):
        return len(self.all_rgb_files)

    def __getitem__(self, idx):
        ## Load the starting and target view
        scene_path = self.scene_path_list[idx]
        data_set = self.data_sets[idx]
        metadata = self.metadata[idx]
        starting_view  = imageio.imread(os.path.join(self.folder_path, metadata["starting_view_file"])).astype(np.float32) / 255.0
        rgb    = imageio.imread(os.path.join(self.folder_path, metadata["target_view_file"])).astype(np.float32) / 255.0
        render_pose    = metadata["render_pose"] 
        camera  = metadata["target_camera_matrices"]
        start_camera   = metadata["starting_camera_matrices"]
        scene     = metadata["scene_name"]                       ## Scene Path
        nearest_pose_ids = metadata["nearest_pose_ids"]          ## make sure to select at least (2*self.num_source_views) 
        depth_range = metadata["depth_range"]
        nearest_pose_ids = np.random.choice(nearest_pose_ids, self.num_source_views, replace=False)

        intrinsics_file = os.path.join(scene_path, "intrinsics.txt")
        intrinsics = deepvoxels_parse_intrinsics(intrinsics_file, 512)[0]

        train_rgb_files = sorted(
            glob.glob(os.path.join(scene_path, "rgb", "*"))
        )
        train_poses_files = [
            f.replace("rgb", "pose").replace("png", "txt") for f in train_rgb_files
        ]
        train_poses = np.stack(
            [np.loadtxt(file).reshape(4, 4) for file in train_poses_files], axis=0
        )

        src_rgbs = []
        src_cameras = []
        src_rgbs.append(starting_view)
        src_cameras.append(start_camera)
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.0
            train_pose = train_poses[id]
            if self.rectify_inplane_rotation:
                src_pose, src_rgb = rectify_inplane_rotation(train_pose, render_pose, src_rgb)

            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate(
                (list(img_size), intrinsics.flatten(), train_pose.flatten())
            ).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)

        # origin_depth = np.linalg.inv(render_pose.reshape(4, 4))[2, 3]
        # rgb_file = os.path.join(self.folder_path, metadata["target_view_file"])

        # if "cube" in rgb_file:
        #     near_depth = origin_depth - 1.0
        #     far_depth = origin_depth + 1
        # else:
        #     near_depth = origin_depth - 0.8
        #     far_depth = origin_depth + 0.8

        # depth_range = torch.tensor([near_depth, far_depth])

        return {
            "rgb": torch.from_numpy(rgb[..., :3]),
            "camera": torch.from_numpy(camera),
            "rgb_path": os.path.join(self.folder_path, metadata["target_view_file"]),
            "src_rgbs": torch.from_numpy(src_rgbs[..., :3]),
            "src_cameras": torch.from_numpy(src_cameras),
            "depth_range": depth_range,
            "scene_path": self.scene_path,
        }
