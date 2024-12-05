import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import sys
import json
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

sys.path.append("../")
from .data_utils import rectify_inplane_rotation, get_nearest_pose_ids


def read_cameras(pose_file):
    basedir = os.path.dirname(pose_file)
    with open(pose_file, "r") as fp:
        meta = json.load(fp)

    camera_angle_x = float(meta["camera_angle_x"])
    rgb_files = []
    c2w_mats = []

    img = imageio.imread(os.path.join(basedir, meta["frames"][0]["file_path"] + ".png"))
    H, W = img.shape[:2]
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
    intrinsics = get_intrinsics_from_hwf(H, W, focal)

    for i, frame in enumerate(meta["frames"]):
        rgb_file = os.path.join(basedir, meta["frames"][i]["file_path"][2:] + ".png")
        rgb_files.append(rgb_file)
        c2w = np.array(frame["transform_matrix"])
        w2c_blender = np.linalg.inv(c2w)
        w2c_opencv = w2c_blender
        w2c_opencv[1:3] *= -1
        c2w_opencv = np.linalg.inv(w2c_opencv)
        c2w_mats.append(c2w_opencv)
    c2w_mats = np.array(c2w_mats)
    return rgb_files, np.array([intrinsics] * len(meta["frames"])), c2w_mats

def get_intrinsics_from_hwf(h, w, focal):
    return np.array(
        [[focal, 0, 1.0 * w / 2, 0], [0, focal, 1.0 * h / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )

class NerfSynthGenerationDataset(Dataset):
    def __init__(
        self,
        args,
        mode,
        # scenes=('chair', 'drum', 'lego', 'hotdog', 'materials', 'mic', 'ship'),
        scenes=(),
        **kwargs
    ):
        self.folder_path = os.path.join(args.rootdir, "../../../data/nerf_synthetic/")  ## change this according to your data path
        self.rectify_inplane_rotation = args.rectify_inplane_rotation
        if mode == "validation":
            mode = "val"
        assert mode in ["train", "val", "test"]
        self.mode = mode            ## train / test / val
        self.num_source_views = args.num_source_views
        self.testskip = args.testskip

        all_scenes = ("chair", "drums", "lego", "hotdog", "materials", "mic", "ship")
        if len(scenes) > 0:
            if isinstance(scenes, str):
                scenes = [scenes]
        else:
            scenes = all_scenes

        print("loading {} for {}".format(scenes, mode))
        self.render_rgb_files = []
        self.render_poses = []
        self.render_intrinsics = []

        for scene in scenes:
            self.scene_path = os.path.join(self.folder_path, scene)
            pose_file = os.path.join(self.scene_path, "transforms_{}.json".format(mode))
            rgb_files, intrinsics, poses = read_cameras(pose_file)
            if self.mode != "train":
                rgb_files = rgb_files[:: self.testskip]
                intrinsics = intrinsics[:: self.testskip]
                poses = poses[:: self.testskip]
            self.render_rgb_files.extend(rgb_files)
            self.render_poses.extend(poses)
            self.render_intrinsics.extend(intrinsics)

    def __len__(self):
        return len(self.render_rgb_files)

    def __getitem__(self, idx):
        rgb_file = self.render_rgb_files[idx]
        render_pose = self.render_poses[idx]  ## target view
        render_intrinsics = self.render_intrinsics[idx]
        train_pose_file = os.path.join("/".join(rgb_file.split("/")[:-2]), "transforms_train.json")
        train_rgb_files, train_intrinsics, train_poses = read_cameras(train_pose_file)

        if self.mode == "train":
            id_render = int(os.path.basename(rgb_file)[:-4].split("_")[1])
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.3, 0.5, 0.2])
        else:
            id_render = -1
            subsample_factor = 1

        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.0
        rgb = rgb[..., [-1]] * rgb[..., :3] + 1 - rgb[..., [-1]]         
        img_size = rgb.shape[:2]
        camera = np.concatenate(
            (list(img_size), render_intrinsics.flatten(), render_pose.flatten())
        ).astype(np.float32)

        max_dif =  len(train_rgb_files) - self.num_source_views

        # if self.dataset_generation:
        nearest_pose_ids = get_nearest_pose_ids(
            render_pose,
            train_poses,
            int(self.num_source_views * 3 + 1),  #+int(max_dif/3)
            tar_id=id_render,
            angular_dist_method="vector",
        )

        ## Nearest Pose Ids, we select 10 more poses out of which we will choose the starting view
        nearest_pose_ids = np.random.choice(nearest_pose_ids, self.num_source_views+1, replace=False)

        assert id_render not in nearest_pose_ids

        ## Occasionally include input image (Why!)
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == "train":
            nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

        src_rgbs    = []
        src_cameras = []
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.0
            src_rgb = src_rgb[..., [-1]] * src_rgb[..., :3] + 1 - src_rgb[..., [-1]]
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

        # id_starting = int(len(nearest_pose_ids)/2)

        ## Select the Farthest from target view 
        # if self.dataset_generation:
        # starting_pose_id = np.random.choice(np.arange(self.num_source_views, self.num_source_views+int(max_dif/3)), 1, replace=False)
        # starting_pose_id = np.random.choice(self.num_source_views, 1, replace=False)

        ## Output Arguments         
        starting_rgb = src_rgbs[0]
        starting_camera = src_cameras[0]
        src_rgbs     = np.stack(src_rgbs[1:], axis=0)
        src_cameras  = np.stack(src_cameras[1:], axis=0)
        nearest_pose_ids = nearest_pose_ids[1:]

        ## Depth Range
        near_depth = 2.0
        far_depth = 6.0
        depth_range = torch.tensor([near_depth, far_depth])

        ## To generate the caption
        caption_rgb = imageio.imread(train_rgb_files[0]).astype(np.float32) / 255.0
        caption_rgb = caption_rgb[..., [-1]] * caption_rgb[..., :3] + 1 - caption_rgb[..., [-1]]  ## Alpha blending based on the opacity (it is a must)

        return {
            "caption_rgb": torch.from_numpy(caption_rgb[..., :3]),
            "traget_rgb": rgb[..., :3],
            "target_camera_matrices": camera,
            "starting_view": starting_rgb[..., :3],
            "starting_camera_matrices": starting_camera,
            "nearest_pose_ids": nearest_pose_ids,
            # "target_rgb_path": rgb_file,
            "num_images_in_scene": len(train_rgb_files),
            "train_pose_file": train_pose_file,
            "render_pose": render_pose,
            # "src_rgbs": src_rgbs[..., :3],
            # "src_cameras": src_cameras,
            "depth_range": depth_range,
        }