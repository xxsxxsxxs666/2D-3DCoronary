"""
Simply load images from a folder or nested folders (does not have any split),
and apply homographic adaptations to it. Yields an image pair without border
artifacts.
"""

import argparse
import logging
import random

from pathlib import Path

import numpy
import torch
from omegaconf import OmegaConf
import numpy as np
import json

from ..utils.tools import fork_rng
from .augmentations import IdentityAugmentation
from .base_dataset import BaseDataset
from ..utils.image import read_image
from  ..utils.vessel_tool import sample_points2, VesselTree, generate_2d_image_numpy, generate_2d_image_parreral, binary_image_2_points
import copy



logger = logging.getLogger(__name__)

class VesselDataset(BaseDataset):
    default_conf = {
        # image search
        "data_dir": "homographies/",
        "train_path": "jpg/",  # the subdirectory with the images
        "val_path": "jpg/",
        "glob": "*.npz",
        "image_size": (224, 224),
        "solid_tree_path": "gluefactory/utils/vessel_structure/coronary_artery_tree.json",
        "ambiguous_tree_path": "gluefactory/utils/vessel_structure/ambiguous.json",
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
        "assignment_mode": "single0_lax1",
        "dsa_p_cut_p": 0.8,
        "dsa_percentage_cut_num": 2,
        "dsa_percentage_threshold": (0.0, 0.4),
        "dsa_h_cut_p": 0.8,
        "dsa_head_cut_num": 2,
        "dsa_head_cutting_threshold": (0.1, 0.4),
        "dsa_root_cut_num": 1,
        "dsa_r_cut_p": 0.8,
        "dsa_root_cutting_threshold": (0, 0.5),
        "cta_p_cut_p": 0.8,
        "cta_percentage_cut_num": 1,
        "cta_percentage_threshold": (0.0, 0.2),
        "cta_h_cut_p": 0.8,
        "cta_head_cut_num": 1,
        "cta_head_cutting_threshold": (0.1, 0.2),
        "cta_root_cut_num": 1,
        "cta_r_cut_p": 0.8,
        "cta_root_cutting_threshold": (0, 0.5),
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
            existing_list_path.mkdir(parents=True, exist_ok=True)
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
        self.vessel_tree = VesselTree(solid_tree_path=conf.solid_tree_path,
                                      ambiguous_tree_path=conf.ambiguous_tree_path)
        self.img_to_tensor = IdentityAugmentation()
    def __getitem__(self, idx):
        if self.conf.reseed:
            with fork_rng(self.conf.seed + idx, False):
                return self.getitem(idx)
        else:
            return self.getitem(idx)

    def random_cut(self, label, case_tree, branch_start_end_index: dict, p_cut_p=0.5,
                   percentage_cut_num=3, percentage_threshold=(0.0, 0.5), h_cut_p=0.5,
                   head_cut_num=3, head_cutting_threshold=(0, 0.5),
                   root_cut_num=1, r_cut_p=0.5, root_cutting_threshold=(0, 0.5)):
        cut_tree = copy.deepcopy(case_tree)
        cut_index = np.zeros(len(label), dtype=bool)
        if percentage_cut_num > 0 and random.random() < p_cut_p:
            cutting_percentage = self.vessel_tree.get_cutting_percentage_for_each_point(case_tree=case_tree,
                                                                                        label=label)
            remove_list, remove_percentage = \
                self.vessel_tree.random_generate_cutting_id_and_percentage(cutting_percentage=cutting_percentage,
                                                                           label=label,
                                                                           case_tree=case_tree,
                                                                           cutting_branch_num=percentage_cut_num,
                                                                           percentage_threshold=percentage_threshold)
            cut_index, cut_tree = self.vessel_tree.cut_branch_percentage(case_tree=case_tree,
                                                                         branch_start_end_index=branch_start_end_index,
                                                                         label=label,
                                                                         cut_branch_list=remove_list,
                                                                         cut_percentage_list=remove_percentage)

        if head_cut_num > 0 and random.random() < h_cut_p:
            cutting_percentage = self.vessel_tree.get_cutting_percentage_for_each_point(case_tree=cut_tree,
                                                                                        label=label)
            remove_list, remove_percentage = \
                self.vessel_tree.random_generate_cutting_id_for_head_blocking(cutting_percentage=cutting_percentage,
                                                                              label=label,
                                                                              case_tree=cut_tree,
                                                                              cutting_branch_num=head_cut_num,
                                                                              percentage_threshold=head_cutting_threshold)

            cut_index_, cut_tree = self.vessel_tree.cut_branch_percentage(case_tree=cut_tree,
                                                                          branch_start_end_index=branch_start_end_index,
                                                                          label=label,
                                                                          cut_branch_list=remove_list,
                                                                          cut_percentage_list=remove_percentage)
            cut_index = cut_index | cut_index_

        if root_cut_num > 0 and random.random() < r_cut_p:
            cut_index_, cut_tree = self.vessel_tree.random_root_cutting(label=label, case_tree=cut_tree,
                                                                        percentage_threshold=root_cutting_threshold)
            cut_index = cut_index | cut_index_

        return cut_index


    def create_2D3DSample_matching(self, d, mode="single0", sample_num_0=1024, sample_num_1=1024,
                                   lax_distance_threshold=0.2):
        """assert: loaded matching is [1, 2, 3, 4....]"""
        """
        :param d:
        :param mode: single_0, single_1, bi-directional, bi-directional-lax
        :param sample_num:
        :return:
        """

        cl_keypoints0_o = d["cl_keypoints0"]
        cl_keypoints1_o = d["cl_keypoints1"]
        label_0 = d["cl_labels0"]
        label_1 = d["cl_labels1"]
        radius_0 = d["cl_radius0"]
        radius_1 = d["cl_radius1"]

        case_tree, branch_start_end_index = self.vessel_tree.generate_case_tree(label=label_0, keypoints=cl_keypoints0_o)
        cut_index0 = self.random_cut(label=label_0, case_tree=case_tree, branch_start_end_index=branch_start_end_index,
                                     p_cut_p=self.conf.dsa_p_cut_p,
                                     percentage_cut_num=self.conf.dsa_percentage_cut_num,
                                     percentage_threshold=self.conf.dsa_percentage_threshold,
                                     h_cut_p=self.conf.dsa_h_cut_p,
                                     head_cut_num=self.conf.dsa_head_cut_num,
                                     head_cutting_threshold=self.conf.dsa_head_cutting_threshold,
                                     r_cut_p=self.conf.dsa_r_cut_p,
                                     root_cutting_threshold=self.conf.dsa_root_cutting_threshold)
        cut_index1 = self.random_cut(label=label_0, case_tree=case_tree, branch_start_end_index=branch_start_end_index,
                                     p_cut_p=self.conf.cta_p_cut_p,
                                     percentage_cut_num=self.conf.cta_percentage_cut_num,
                                     percentage_threshold=self.conf.cta_percentage_threshold,
                                     h_cut_p=self.conf.cta_h_cut_p,
                                     head_cut_num=self.conf.cta_head_cut_num,
                                     head_cutting_threshold=self.conf.cta_head_cutting_threshold,
                                     r_cut_p=self.conf.cta_r_cut_p,
                                     root_cutting_threshold=self.conf.cta_root_cutting_threshold)

        image0 = generate_2d_image_parreral(cl_keypoints0_o[~cut_index0], radius_0[~cut_index0], self.conf.image_size)
        image1 = generate_2d_image_parreral(cl_keypoints1_o[~cut_index1], radius_1[~cut_index1], self.conf.image_size)

        in_points_index_0 = (cl_keypoints0_o[:, 0] > 0) & (cl_keypoints0_o[:, 0] < image0.shape[1]) & \
                            (cl_keypoints0_o[:, 1] > 0) & (cl_keypoints0_o[:, 1] < image0.shape[0])
        in_points_index_1 = (cl_keypoints1_o[:, 0] > 0) & (cl_keypoints1_o[:, 0] < image1.shape[1]) & \
                            (cl_keypoints1_o[:, 1] > 0) & (cl_keypoints1_o[:, 1] < image1.shape[0])

        # from IPython import embed; embed(colors="Linux")
        cut_index = cut_index0 | cut_index1 | ~in_points_index_0 | ~in_points_index_1
        cut_index1_addout = cut_index1 | ~in_points_index_1


        cl2d_keypoints0 = binary_image_2_points(img=image0, sample_num=sample_num_0)
        cl_keypoints1 = sample_points2(cl_keypoints1_o[~cut_index1_addout], sample_num_1)

        # 现在3D投影的也需要采样
        gt_matches1_o = np.arange(cl_keypoints1_o.shape[0])
        gt_matches1_o[cut_index] = -1
        gt_matches0_o = np.arange(cl_keypoints0_o.shape[0])
        gt_matches0_o[cut_index] = -1

        nearest_point_index = np.linalg.norm(cl2d_keypoints0[:, None, :] - cl_keypoints0_o[None, :, :], axis=-1).argmin(
            axis=1)
        nearest_point_index_2 = np.linalg.norm(cl_keypoints1_o[:, None, :] - cl_keypoints1[None, :, :], axis=-1).argmin(
            axis=1)
        nearest_point_index_3 = np.linalg.norm(cl_keypoints1[:, None, :] - cl_keypoints1_o[None, :, :], axis=-1).argmin(
            axis=1)
        gt_matches0_2d_invalid = gt_matches0_o[nearest_point_index] < 0
        gt_matches0_2d = nearest_point_index_2[gt_matches0_o[nearest_point_index]]
        gt_matches0_2d[gt_matches0_2d_invalid] = -1

        # m_cl2d_keypoints0 = cl2d_keypoints0[matching_index]
        # m_cl_keypoints0 = cl_keypoints1[gt_matches0_2d[matching_index]]
        gt_matches1_2d_invalid = gt_matches1_o[nearest_point_index_3] < 0
        gt_matches1_2d = np.zeros(cl_keypoints1.shape[0])
        gt_matches1_2d[gt_matches1_2d_invalid] = -1
        if mode == "single0":
            gt2d_assignment = torch.zeros(size=(cl2d_keypoints0.shape[0], cl_keypoints1.shape[0]), dtype=torch.bool)
            src = torch.ones(size=(cl2d_keypoints0.shape[0], 1), dtype=torch.bool)
            src[gt_matches0_2d_invalid] = False
            gt_matches0_2d_ = gt_matches0_2d.copy()
            gt_matches0_2d_[gt_matches0_2d_invalid] = 0
            gt2d_assignment.scatter_(-1, torch.tensor(gt_matches0_2d_).unsqueeze(-1), src)

        elif mode == "single0_lax0":
            distance_constraint = np.linalg.norm(cl2d_keypoints0[:, None, :] -
                                                 cl2d_keypoints0[None, :, :], axis=-1) < lax_distance_threshold
            x, y = np.where(distance_constraint)
            z = nearest_point_index[y]
            gt2d_assignment = torch.zeros(size=(cl2d_keypoints0.shape[0], cl_keypoints1.shape[0]), dtype=torch.bool)
            gt2d_assignment[torch.tensor(x), torch.tensor(z)] = True

        elif mode == "single0_lax01":
            distance_constraint0 = np.linalg.norm(cl2d_keypoints0[:, None, :] -
                                                  cl2d_keypoints0[None, :, :], axis=-1) < lax_distance_threshold
            x, y = np.where(distance_constraint0)
            z = nearest_point_index[y]
            nearest_point_after_lax = cl_keypoints1[y]
            distance_constraint1 = np.linalg.norm(nearest_point_after_lax[:, None, :] -
                                                  nearest_point_after_lax[None, :, :], axis=-1) < lax_distance_threshold
            m, n = np.where(distance_constraint1)
            gt2d_assignment = torch.zeros(size=(cl2d_keypoints0.shape[0], cl_keypoints1.shape[0]), dtype=torch.bool)
            gt2d_assignment[torch.tensor(x[m]), torch.tensor(z[n])] = True

        elif mode == "single0_lax1":
            nearest_point = cl_keypoints1[gt_matches0_2d]
            distance_constraint1 = np.linalg.norm(nearest_point[:, None, :] -
                                                  nearest_point[None, :, :], axis=-1) < lax_distance_threshold
            m, n = np.where(distance_constraint1)
            gt2d_assignment = torch.zeros(size=(cl2d_keypoints0.shape[0], cl_keypoints1.shape[0]), dtype=torch.bool)
            filter_index = gt_matches0_2d_invalid[m]
            gt2d_assignment[torch.tensor(m[~filter_index]), torch.tensor(gt_matches0_2d[n[~filter_index]])] = True
        else:
            raise ValueError(f"mode {mode} not supported")

        image0_size = np.array(image0.shape[:2][::-1], dtype=np.float32)
        image1_size = np.array(image1.shape[:2][::-1], dtype=np.float32)


        data = {
            "view0": {"image": self.img_to_tensor(image0, return_tensor=True), "image_size": image0_size},
            "view1": {"image": self.img_to_tensor(image1, return_tensor=True), "image_size": image1_size},
            "keypoints0": torch.tensor(cl2d_keypoints0, dtype=torch.float32),
            "keypoints1": torch.tensor(cl_keypoints1, dtype=torch.float32),
            "origin_lm0": torch.tensor(d["origin_lm0"], dtype=torch.float32),
            "origin_lm1": torch.tensor(d["origin_lm1"], dtype=torch.float32),
            "update_lm": torch.tensor(d["update_lm"], dtype=torch.float32),
            "gt_assignment": gt2d_assignment,
            "gt_matches0": torch.tensor(gt_matches0_2d, dtype=torch.float32),
            "gt_matches1": torch.tensor(gt_matches1_2d, dtype=torch.float32),
        }
        return data

    def getitem(self, idx):
        name = self.image_names[idx]
        gt = np.load(str(self.image_dir / name))


        return self.create_2D3DSample_matching(d=gt, mode=self.conf.assignment_mode)

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


