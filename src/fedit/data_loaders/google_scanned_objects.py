import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import glob
import sys
import pickle 
sys.path.append("../")
from .data_utils import rectify_inplane_rotation, get_nearest_pose_ids


# only for training
class GoogleScannedDataset(Dataset):
    def __init__(self, args, mode, **kwargs):
        self.folder_path = os.path.join(args.rootdir, "../../../data/google_scanned_objects/")
        self.num_source_views = args.num_source_views
        self.rectify_inplane_rotation = args.rectify_inplane_rotation
        self.scene_path_list = glob.glob(os.path.join(self.folder_path, "*"))

        num_files = 250
        self.frame_path = self.folder_path

        train_scenes = [name for name in os.listdir(self.frame_path) if os.path.isdir(os.path.join(self.frame_path, name))]

        for scene in train_scenes:
            self.scene_path =  os.path.join(self.frame_path, scene)

            ## Load the Metadata containing 
            with open(self.scene_path.joinpath(f"{scene}_edited_metadata.pkl"), "rb") as file:
                self.scene_metadata = pickle.load(file)
            
            self.metadata.extend(self.scene_metadata)


        # for i, scene_path in enumerate(self.scene_path_list):

        #     rgb_files = [
        #         os.path.join(scene_path, "rgb", f)
        #         for f in sorted(os.listdir(os.path.join(scene_path, "rgb")))
        #     ]
        #     pose_files = [f.replace("rgb", "pose").replace("png", "txt") for f in rgb_files]
        #     intrinsics_files = [
        #         f.replace("rgb", "intrinsics").replace("png", "txt") for f in rgb_files
        #     ]

        #     if np.min([len(rgb_files), len(pose_files), len(intrinsics_files)]) < num_files:
        #         print(scene_path)
        #         continue

        #     all_rgb_files.append(rgb_files)
        #     all_pose_files.append(pose_files)
        #     all_intrinsics_files.append(intrinsics_files)

        # index = np.arange(len(all_rgb_files))
        # self.all_rgb_files = np.array(all_rgb_files)[index]
        # self.all_pose_files = np.array(all_pose_files)[index]
        # self.all_intrinsics_files = np.array(all_intrinsics_files)[index]

    def __len__(self):
        return len(self.all_rgb_files)

    def __getitem__(self, idx):

        ## Load the starting and target view
        metadata = self.metadata[idx]
        starting_view  = imageio.imread(os.path.join(self.folder_path,metadata["starting_view_file"])).astype(np.float32) / 255.0
        rgb    = imageio.imread(os.path.join(self.folder_path,metadata["target_view_file"])).astype(np.float32) / 255.0
        render_pose    = metadata["render_pose"] 
        camera  = metadata["target_camera_matrices"]
        start_camera   = metadata["starting_camera_matrices"]
        scene_name     = metadata["scene_name"]
        id_feat = metadata["nearest_pose_ids"]          ## make sure to select at least (2*self.num_source_views) 
        id_feat = np.random.choice(id_feat, self.num_source_views, replace=False)

        scene_path =  os.path.join(self.folder_path, scene_name)

        rgb_files = [
            os.path.join(scene_path, "rgb", f)
            for f in sorted(os.listdir(os.path.join(scene_path, "rgb")))
        ]
        pose_files = [f.replace("rgb", "pose").replace("png", "txt") for f in rgb_files]
        intrinsics_files = [
            f.replace("rgb", "intrinsics").replace("png", "txt") for f in rgb_files
        ]

        # get depth range
        min_ratio = 0.1
        origin_depth = np.linalg.inv(render_pose)[2, 3]
        max_radius = 0.5 * np.sqrt(2) * 1.1
        near_depth = max(origin_depth - max_radius, min_ratio * origin_depth)
        far_depth = origin_depth + max_radius
        depth_range = torch.tensor([near_depth, far_depth])

        src_rgbs = []
        src_cameras = []
        src_rgbs.append(starting_view)
        src_cameras.append(start_camera)
        for id in id_feat:
            src_rgb = imageio.imread(rgb_files[id]).astype(np.float32) / 255.0
            pose = np.loadtxt(pose_files[id])
            if self.rectify_inplane_rotation:
                pose, src_rgb = rectify_inplane_rotation(pose.reshape(4, 4), render_pose, src_rgb)

            src_rgbs.append(src_rgb)
            intrinsics = np.loadtxt(intrinsics_files[id])
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), intrinsics, pose.flatten())).astype(
                np.float32
            )
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs)
        src_cameras = np.stack(src_cameras)

        return {
            "rgb": torch.from_numpy(rgb),
            "camera": torch.from_numpy(camera),
            "rgb_path": os.path.join(self.folder_path,metadata["target_view_file"]),
            "src_rgbs": torch.from_numpy(src_rgbs),
            "src_cameras": torch.from_numpy(src_cameras),
            "depth_range": depth_range,
        }
