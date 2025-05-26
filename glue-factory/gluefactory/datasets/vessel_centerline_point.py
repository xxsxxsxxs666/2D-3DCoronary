"""
Simply load images from a folder or nested folders (does not have any split),
and apply homographic adaptations to it. Yields an image pair without border
artifacts.
"""

import argparse
import logging
import random

from pathlib import Path


import torch
from omegaconf import OmegaConf
import numpy as np
import json

from ..utils.tools import fork_rng
from .augmentations import IdentityAugmentation
from .base_dataset import BaseDataset
from ..utils.image import read_image


logger = logging.getLogger(__name__)

class VesselDataset(BaseDataset):
    default_conf = {
        # image search
        "data_dir": "homographies/",
        "train_path": "jpg/",  # the subdirectory with the images
        "val_path": "jpg/",
        "glob": "*.npz",
        # splits
        "train_size": 50, # 100,
        "val_size": 10,
        "shuffle_seed": 0,  # or None to skip
        # image loading
        "reseed": False,
        "save_list": True,
        "using_existing_list": True,
        "existing_list_path": None,
        "using_static_size": True,
    }

    def _init(self, conf):
        data_dir = Path(conf.data_dir)

        train_dir = data_dir / conf.train_path
        val_dir = data_dir / conf.val_path
        existing_list_path = Path(conf.existing_list_path) if conf.existing_list_path is not None else data_dir

        if ((existing_list_path / "train_list.json").exists() and (existing_list_path / "val_list.json").exists()
                and conf.using_existing_list):
            with open(existing_list_path / "train_list.json", "r") as f:
                train_images = json.load(f)
                train_images = [Path(i) for i in train_images]
            with open(existing_list_path / "val_list.json", "r") as f:
                val_images = json.load(f)
                if conf.using_static_size:
                    val_images = [Path(i) for i in random.sample(val_images, conf.val_size)]
                else:
                    val_images = [Path(i) for i in val_images]
            self.images = {"train": sorted(train_images), "val": sorted(val_images)}
            logger.info(f"Found {len(train_images)} images in train list, "
                        f"Found {len(val_images)} images in train list")
            return
        train_images = []

        glob = [conf.glob] if isinstance(conf.glob, str) else conf.glob
        # save the train_list in the data_dir using json

        for g in glob:
            train_images += list(train_dir.glob("*/*/*/*/" + g))

        val_images = []

        glob = [conf.glob] if isinstance(conf.glob, str) else conf.glob
        for g in glob:
            val_images += list(val_dir.glob("*/*/*/*/" + g))

        if len(train_images) == 0 or len(val_images) == 0:
            raise ValueError(f"Cannot find any image in train of val folder: {data_dir}.")

        train_images = [i.relative_to(data_dir).as_posix() for i in train_images]
        train_images = sorted(train_images)  # for deterministic behavior

        val_images = [i.relative_to(data_dir).as_posix() for i in val_images]
        val_images = sorted(val_images)  # for deterministic behavior

        if conf.save_list:
            with open(existing_list_path / "train_list.json", "w") as f:
                train_images_save = [str(i) for i in train_images]
                json.dump(train_images_save, f)
            with open(existing_list_path / "val_list.json", "w") as f:
                val_images_save = [str(i) for i in val_images]
                json.dump(val_images_save, f)

        logger.info(f"Found {len(train_images)} images in train folder, "
                    f"Found {len(val_images)} images in train folder")

        if conf.shuffle_seed is not None:
            np.random.RandomState(conf.shuffle_seed).shuffle(train_images)

        self.images = {"train": train_images, "val": val_images}

    def get_dataset(self, split):
        return _Dataset(self.conf, self.images[split], split)



class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, image_names, split):
        self.conf = conf
        self.image_names = np.array(image_names)
        self.image_dir = Path(conf.data_dir)

        self.img_to_tensor = IdentityAugmentation()


    def __getitem__(self, idx):
        if self.conf.reseed:
            with fork_rng(self.conf.seed + idx, False):
                return self.getitem(idx)
        else:
            return self.getitem(idx)

    def getitem(self, idx):
        name = self.image_names[idx]
        gt = np.load(str(self.image_dir / name))
        image0_path = self.image_dir / str(name).replace("gt.npz", "image0.png")
        image1_path = self.image_dir / str(name).replace("gt.npz", "image1.png")
        image0 = read_image(path=image0_path, grayscale=False)
        image1 = read_image(path=image1_path, grayscale=False)
        if image0 is None or image1 is None:
            logging.error("Image %s could not be read.", name)

        # image0 = image0.astype(np.float32) / 255.0 # normalize, but our data are binary data
        image0_size = np.array(image0.shape[:2][::-1], dtype=np.float32)
        image1_size = np.array(image1.shape[:2][::-1], dtype=np.float32)
        # 'cl_keypoints0', 'cl_keypoints1', 'origin_lm0', 'origin_lm1', 'update_lm', 'gt_assignment',
        # 'gt_matches0', 'gt_matches1', 'gt_matching_scores0', 'gt_matching_scores1'

        data = {
            "keypoints0": torch.tensor(gt["cl_keypoints0"], dtype=torch.float32),
            "keypoints1": torch.tensor(gt["cl_keypoints1"], dtype=torch.float32),
            "origin_lm0": torch.tensor(gt["origin_lm0"], dtype=torch.float32),
            "origin_lm1": torch.tensor(gt["origin_lm1"], dtype=torch.float32),
            "update_lm": torch.tensor(gt["update_lm"], dtype=torch.float32),
            "gt_assignment": torch.tensor(gt["gt_assignment"].squeeze(0), dtype=torch.float32),
            "gt_matches0": torch.tensor(gt["gt_matches0"].squeeze(0), dtype=torch.float32),
            "gt_matches1": torch.tensor(gt["gt_matches1"].squeeze(0), dtype=torch.float32),
            "gt_matching_scores0": torch.tensor(gt["gt_matching_scores0"].squeeze(0), dtype=torch.float32),
            "gt_matching_scores1": torch.tensor(gt["gt_matching_scores1"].squeeze(0), dtype=torch.float32),
            "view0": {"image": self.img_to_tensor(image0, return_tensor=True), "image_size": image0_size},
            "view1": {"image": self.img_to_tensor(image1, return_tensor=True), "image_size": image1_size},
        }

        return data

    def __len__(self):
        return len(self.image_names)


def visualize(args):
    conf = {
        "batch_size": 1,
        "num_workers": 1,
        "prefetch_factor": 1,
        "data_dir": "/mnt/maui/CTA_Coronary/project/xiongxs/Data/",
        "train_path": "51_train_lightglue_L/",
        "val_path": "51_val_lightglue_L/",
    }
    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
    dataset = HomographyDataset(conf)
    loader = dataset.get_data_loader("train")
    logger.info("The dataset has %d elements.", len(loader))

    with fork_rng(seed=dataset.conf.seed):
        for _, data in zip(range(args.num_items), loader):
            print(data)



if __name__ == "__main__":
    from .. import logger  # overwrite the logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_items", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--dotlist", nargs="*")
    args = parser.parse_intermixed_args()
    visualize(args)


