import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import sys
import pickle 

sys.path.append("../")
from .data_utils import random_crop, random_flip, get_nearest_pose_ids
from .llff_data_utils import load_llff_data, batch_parse_llff_poses


class LLFFDataset(Dataset):
    def __init__(self, args, mode, **kwargs):
        base_dir = os.path.join(args.rootdir, "data/real_iconic_noface/")
        self.dataset_dir = base_dir
        self.args = args
        self.mode = mode  ## train / test / validation
        self.num_source_views = args.num_source_views
        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []

        scenes = os.listdir(base_dir)
        self.scene_path_list = []
        for scene in scenes:
            self.scene_path =  os.path.join(base_dir, scene)

            ## Load the Metadata containing 
            with open(self.scene_path.joinpath(f"{scene}_edited_metadata.pkl"), "rb") as file:
                self.scene_metadata = pickle.load(file)
            
            self.metadata.extend(self.scene_metadata)
            self.scene_path_list.append(self.scene_path)

        for i, scene in enumerate(scenes):
            scene_path = os.path.join(base_dir, scene)
            _, poses, bds, render_poses, i_test, rgb_files = load_llff_data(
                scene_path, load_imgs=False, factor=4
            )
            near_depth = np.min(bds)
            far_depth = np.max(bds)
            intrinsics, c2w_mats = batch_parse_llff_poses(poses)

            if mode == "train":
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

            self.train_intrinsics.append(intrinsics[i_train])
            self.train_poses.append(c2w_mats[i_train])
            self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())
            num_render = len(i_render)
            self.render_rgb_files.extend(np.array(rgb_files)[i_render].tolist())
            self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics[i_render]])
            self.render_poses.extend([c2w_mat for c2w_mat in c2w_mats[i_render]])
            self.render_depth_range.extend([[near_depth, far_depth]] * num_render)
            self.render_train_set_ids.extend([i] * num_render)

    def __len__(self):
        return len(self.render_rgb_files)

    def __getitem__(self, idx):
        metadata = self.metadata[idx]
        starting_view  = imageio.imread(os.path.join(self.dataset_dir, metadata["starting_view_file"])).astype(np.float32) / 255.0
        rgb    = imageio.imread(os.path.join(self.dataset_dir,metadata["target_view_file"])).astype(np.float32) / 255.0
        render_pose    = metadata["render_pose"] 
        camera  = metadata["target_camera_matrices"]
        start_camera   = metadata["starting_camera_matrices"]
        scene     = metadata["scene_name"]                       ## Scene Path
        nearest_pose_ids = metadata["nearest_pose_ids"]          ## make sure to select at least (2*self.num_source_views) 
        depth_range = metadata["depth_range"]
        nearest_pose_ids = np.random.choice(nearest_pose_ids, self.num_source_views, replace=False)
        scene_path =  os.path.join(self.dataset_dir, scene)
        
        _, poses, bds, render_poses, i_test, rgb_files = load_llff_data(
            scene_path, load_imgs=False, factor=4
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


        mean_depth = np.mean(depth_range)
        world_center = (render_pose.dot(np.array([[0, 0, mean_depth, 1]]).T)).flatten()[:3]

        train_rgb_files = np.array(rgb_files)[i_render].tolist()
        train_poses = c2w_mats[i_render]
        train_intrinsics = intrinsics[i_render]

        src_rgbs = []
        src_cameras = []
        src_rgb.append(starting_view)
        src_cameras.append(start_camera)
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.0
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]
            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate(
                (list(img_size), train_intrinsics_.flatten(), train_pose.flatten())
            ).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)
        if self.mode == "train":
            crop_h = np.random.randint(low=250, high=750)
            crop_h = crop_h + 1 if crop_h % 2 == 1 else crop_h
            crop_w = int(400 * 600 / crop_h)
            crop_w = crop_w + 1 if crop_w % 2 == 1 else crop_w
            rgb, camera, src_rgbs, src_cameras = random_crop(
                rgb, camera, src_rgbs, src_cameras, (crop_h, crop_w)
            )

        if self.mode == "train" and np.random.choice([0, 1]):
            rgb, camera, src_rgbs, src_cameras = random_flip(rgb, camera, src_rgbs, src_cameras)

        depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.6])

        return {
            "rgb": torch.from_numpy(rgb[..., :3]),
            "camera": torch.from_numpy(camera),
            "rgb_path": os.path.join(self.dataset_dir,metadata["target_view_file"]),
            "src_rgbs": torch.from_numpy(src_rgbs[..., :3]),
            "src_cameras": torch.from_numpy(src_cameras),
            "depth_range": depth_range,
        }
