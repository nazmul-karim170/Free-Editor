import os
import numpy as np
import imageio
import torch
import sys

sys.path.append("../")
from torch.utils.data import Dataset
from .data_utils import random_crop, get_nearest_pose_ids
from .shiny_data_utils import load_llff_data, batch_parse_llff_poses
import pickle 

class ShinyDataset(Dataset):
    def __init__(self, args, mode, scenes=(), random_crop=True, **kwargs):
        self.folder_path = os.path.join(args.rootdir, "data/shiny/")
        self.args = args
        self.mode = mode  # train / test / validation
        self.num_source_views = args.num_source_views
        self.random_crop = random_crop
        # self.render_rgb_files = []
        # self.render_intrinsics = []
        # self.render_poses = []
        # self.render_train_set_ids = []
        # self.render_depth_range = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []

        all_scenes = os.listdir(self.folder_path)
        if len(scenes) > 0:
            if isinstance(scenes, str):
                scenes = [scenes]
        else:
            scenes = all_scenes

        print("loading {} for {}".format(scenes, mode))
        for i, scene in enumerate(scenes):
            self.scene_path =  os.path.join(self.frame_path, scene)

            ## Load the Metadata containing 
            with open(self.scene_path.joinpath(f"{scene}_edited_metadata.pkl"), "rb") as file:
                self.scene_metadata = pickle.load(file)         
            self.metadata.extend(self.scene_metadata)

    def __len__(self):
        return (
            len(self.render_rgb_files) * 100000
            if self.mode == "train"
            else len(self.render_rgb_files)
        )

    def __getitem__(self, idx):

        ## Load the starting and target view
        metadata = self.metadata[idx]
        starting_view  = imageio.imread(os.path.join(self.folder_path, metadata["starting_view_file"]))/255
        rgb    = imageio.imread(os.path.join(self.folder_path,metadata["target_view_file"]))/255
        # render_pose    = metadata["render_pose"] 
        camera  = metadata["target_camera_matrices"]
        start_camera   = metadata["starting_camera_matrices"]
        scene_name     = metadata["scene_name"]
        nearest_pose_ids = metadata["nearest_pose_ids"]          ## make sure to select at least (2*self.num_source_views) 
        depth_range = metadata["depth_range"]
        nearest_pose_ids = np.random.choice(nearest_pose_ids, self.num_source_views, replace=False)

        scene_path = os.path.join(self.folder_path, scene_name)
        _, poses, bds, render_poses, intrinsic, i_test = load_llff_data(
            scene_path, load_imgs=False, factor=8, render_style="", split_train_val=0
        )
        if len(intrinsic) == 3:
            H, W, f = intrinsic
            cx = W / 2.0
            cy = H / 2.0
            fx = f
            fy = f
        else:
            H, W, fx, fy, cx, cy = intrinsic
        H = H / 8
        W = W / 8
        fx = fx / 8
        fy = fy / 8
        cx = cx / 8
        cy = cy / 8
        image_dir = os.path.join(scene_path, "images_8")
        rgb_files = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))]
        _, c2w_mats = batch_parse_llff_poses(poses)

        intrinsics_data = [[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        
        ## Flatten any numpy arrays to scalars
        intrinsics_data_cleaned = [
            [float(x) if isinstance(x, np.ndarray) else x for x in row]
            for row in intrinsics_data
        ]
        
        ## Convert to a numpy array
        intrinsics = np.array(intrinsics_data_cleaned, dtype=np.float32)

        intrinsics = intrinsics[None, :, :].repeat(len(c2w_mats), axis=0)
        near_depth = np.min(bds)
        far_depth = np.max(bds)
        i_test = np.arange(poses.shape[0])[:: self.args.llffhold]
        i_train = np.array(
            [
                j
                for j in np.arange(int(poses.shape[0]))
                if (j not in i_test and j not in i_test)
            ]
        )

        if self.mode == "train":
            i_render = i_train
        else:
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

            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate(
                (list(img_size), train_intrinsics_.flatten(), train_pose.flatten())
            ).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)
        if self.mode == "train" and self.random_crop:
            crop_h = np.random.randint(low=250, high=750)
            crop_h = crop_h + 1 if crop_h % 2 == 1 else crop_h
            crop_w = int(400 * 600 / crop_h)
            crop_w = crop_w + 1 if crop_w % 2 == 1 else crop_w
            rgb, camera, src_rgbs, src_cameras = random_crop(
                rgb, camera, src_rgbs, src_cameras, (crop_h, crop_w)
            )

        # depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.6])
        return {
            "rgb": torch.from_numpy(rgb[..., :3]),
            "camera": torch.from_numpy(camera),
            "rgb_path":  metadata["starting_view_file"],
            "src_rgbs": torch.from_numpy(src_rgbs[..., :3]),
            "src_cameras": torch.from_numpy(src_cameras),
            "depth_range": depth_range,
        }
