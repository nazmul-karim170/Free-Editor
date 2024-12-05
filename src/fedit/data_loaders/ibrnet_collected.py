import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import glob
import sys
import pickle 

sys.path.append("../")
from .data_utils import rectify_inplane_rotation, random_crop, random_flip, get_nearest_pose_ids
from .llff_data_utils import load_llff_data, batch_parse_llff_poses


class IBRNetCollectedDataset(Dataset):
    def __init__(self, args, mode, random_crop=True, **kwargs):
        self.folder_path1 = os.path.join(args.rootdir, "../../../data/ibrnet_collected_1/")
        self.folder_path2 = os.path.join(args.rootdir, "../../../data/ibrnet_collected_2/")
        self.rectify_inplane_rotation = args.rectify_inplane_rotation
        self.mode = mode  # train / test / validation
        self.num_source_views = args.num_source_views
        self.random_crop = random_crop
        self.args = args
        all_scenes = glob.glob(self.folder_path1 + "*") + glob.glob(self.folder_path2 + "*")

        for scene in all_scenes:
            self.scene_path =  scene

            ## Load the Metadata containing 
            with open(self.scene_path.joinpath(f"{scene}_edited_metadata.pkl"), "rb") as file:
                self.scene_metadata = pickle.load(file)
            
            self.metadata.extend(self.scene_metadata)

    def __len__(self):
        return len(self.render_rgb_files)

    def __getitem__(self, idx):
        ## Load the starting and target view
        metadata = self.metadata[idx]
        starting_view  = imageio.imread(metadata["starting_view_file"]).astype(np.float32) / 255.0
        rgb    = imageio.imread(metadata["target_view_file"]).astype(np.float32) / 255.0
        render_pose    = metadata["render_pose"] 
        camera  = metadata["target_camera_matrices"]
        start_camera   = metadata["starting_camera_matrices"]
        scene     = metadata["scene_name"]                       ## Scene Path
        nearest_pose_ids = metadata["nearest_pose_ids"]          ## make sure to select at least (2*self.num_source_views) 
        depth_range = metadata["depth_range"]
        nearest_pose_ids = np.random.choice(nearest_pose_ids, self.num_source_views, replace=False)

        if "ibrnet_collected_2" in scene:
            factor = 8
        else:
            factor = 2
        _, poses, bds, render_poses, i_test, rgb_files = load_llff_data(
            scene, load_imgs=False, factor=factor
        )
        near_depth = np.min(bds)
        far_depth = np.max(bds)
        intrinsics, c2w_mats = batch_parse_llff_poses(poses)
        if self.mode == "train":
            i_train = np.array(np.arange(int(poses.shape[0])))
            i_render = i_train
        else:
            i_test = np.arange(poses.shape[0])[:: self.args.llffhold]
            i_train = np.array(
                [
                    j
                    for j in np.arange(int(poses.shape[0]))
                    if (j not in i_test and j not in i_test)
                ]
            )
            i_render = i_test

        train_intrinsics = intrinsics[i_render]
        train_rgb_files = np.array(rgb_files)[i_render].tolist()
        train_poses = c2w_mats[i_render]

        src_rgbs = []
        src_cameras = []
        src_rgbs.append(starting_view)
        src_cameras.append(start_camera)
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.0
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]
            if self.rectify_inplane_rotation:
                train_pose, src_rgb = rectify_inplane_rotation(train_pose, render_pose, src_rgb)

            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate(
                (list(img_size), train_intrinsics_.flatten(), train_pose.flatten())
            ).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)

        if self.mode == "train" and self.random_crop:
            rgb, camera, src_rgbs, src_cameras = random_crop(rgb, camera, src_rgbs, src_cameras)

        if self.mode == "train" and np.random.choice([0, 1], p=[0.5, 0.5]):
            rgb, camera, src_rgbs, src_cameras = random_flip(rgb, camera, src_rgbs, src_cameras)

        # depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.5])

        return {
            "rgb": torch.from_numpy(rgb[..., :3]),
            "camera": torch.from_numpy(camera),
            "rgb_path": metadata["starting_view_file"],
            "src_rgbs": torch.from_numpy(src_rgbs[..., :3]),
            "src_cameras": torch.from_numpy(src_cameras),
            "depth_range": depth_range,
        }
