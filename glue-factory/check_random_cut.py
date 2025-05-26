#  In real picture we only have png, so we need to sample the point when loading the data
import numpy as np
import pandas as pd
import os
import json
from skimage.morphology import skeletonize
import cv2
from pathlib import Path
import torch
from gluefactory.models.matchers import lightglue
from omegaconf import OmegaConf
from gluefactory.datasets import get_dataset
from gluefactory.models import get_model
from gluefactory.datasets.augmentations import IdentityAugmentation
from gluefactory.utils.tensor import batch_to_device
from gluefactory.visualization import viz2d
from gluefactory.utils.vessel_visualization import plot_vessel_tree, plot_vessel_percentage
from gluefactory.utils.vessel_tool import generate_2d_image_parreral, VesselTree
import matplotlib.pyplot as plt
to_ctr = OmegaConf.to_container  # convert DictConfig to dict


def convert_keypoints_to_list_form(array_form):
    general_list = []
    print("input array shape : ", array_form.shape)
    for i in range(array_form.shape[0]):
        general_list.append(array_form[i,:].tolist())
    return general_list


def sample_points2(points, num):
    interval = (len(points) - 1) / (num - 1)
    idx = [int(np.floor(interval * i)) for i in range(num)]
    idx[-1] = min(idx[-1], len(points) - 1)
    return np.array(points[idx])


def binary_image_2_points(img: np.array, sample_num: int = None):
    sk = skeletonize((img/255).sum(axis=-1) > 0)
    y, x = np.where(sk > 0)  # h, w -> y, x
    # change iy to num x 2
    points_coords = np.concatenate([x[:, None], y[:, None]], axis=1)
    return sample_points2(points_coords, num=sample_num) if sample_num is not None else points_coords


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    """Read an image from path as RGB or grayscale"""
    if not Path(path).exists():
        raise FileNotFoundError(f"No image at path {path}.")
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }


def convert_dict_to_name(d):
    # using key and item, join them with "_" if the items is None, them pass it
    return "_".join([str(k) + "_" + str(v) if v is not None else str(k) for k, v in d.items()])


if __name__ == "__main__":
    conf_path = Path(
        "../../../glue-factory/gluefactory/configs/superpoint+lightglue_homography_vessel_offline_cut_45.yaml")
    conf = OmegaConf.load(conf_path)
    dataset = get_dataset(conf.data.name)(conf.data)
    val_dataset = dataset.get_dataset("train")
    random_index = np.random.choice(len(val_dataset), 50)
    # use val_dataset to get the data
    for i in random_index:

        data = val_dataset[i]
        f1, f2 = data["f1"], data["f2"]
        if True:
            image0, image1 = data["view0"]["image"].numpy().squeeze(0), data["view1"]["image"].numpy().squeeze(0)
            keypoints0, keypoints1 = data["keypoints0"].numpy(), data["keypoints1"].numpy()
            keypoints0_o, keypoints1_o = data["keypoints0_o"].numpy(), data["keypoints1_o"].numpy()
            label0, label1 = data["label0"].numpy(), data["label1"].numpy()
            radius0, radius1 = data["radius0"].numpy(), data["radius1"].numpy()
            cut_index = data["cut_index"].numpy()
            cut_index0, cut_index1 = data["cut_index0"].numpy(), data["cut_index1"].numpy()
            cut_index0_0, cut_index0_1, cut_index0_2 = data["cut_index0_0"].numpy(), data["cut_index0_1"].numpy(), data[
                "cut_index0_2"].numpy()
            cut_index1_0, cut_index1_1, cut_index1_2 = data["cut_index1_0"].numpy(), data["cut_index1_1"].numpy(), data[
                "cut_index1_2"].numpy()
            rm0_0, rmp0_0, rm0_1, rmp0_1 = data["rm0_0"], data["rmp0_0"], data["rm0_1"], data["rmp0_1"]
            rm1_0, rmp1_0, rm1_1, rmp1_1 = data["rm1_0"], data["rmp1_0"], data["rm1_1"], data["rmp1_1"]
            origin_lm0, origin_lm1, update_lm = data["origin_lm0"].numpy(), data["origin_lm1"].numpy(), data[
                "update_lm"].numpy()
            gt_assignment = data["gt_assignment"].numpy()
            gt_matches0, gt_matches1 = data["gt_matches0"].numpy().astype(np.int16), data["gt_matches1"].numpy().astype(
                np.int16)

            # visualize the image

            # show before cutting
            image0_o = generate_2d_image_parreral(keypoints0_o, radius=radius0, image_size=image0.shape)
            image1_o = generate_2d_image_parreral(keypoints1_o, radius=radius1, image_size=image1.shape)
            viz2d.plot_images([image0_o, image1_o])
            plot_vessel_tree(keypoints_list=[keypoints0_o, keypoints1_o],
                             label_list=[label0, label1], show_legend=True)
            plt.show()

            # show cut_percentage
            vessel_tree = VesselTree()
            case_tree, branch_start_end_index = vessel_tree.generate_case_tree(keypoints0_o, label0)
            cut_percentage = vessel_tree.get_cutting_percentage_for_each_point(case_tree, label0)
            # show the cutting percentage
            viz2d.plot_images([image0_o, image1_o])
            plot_vessel_percentage([keypoints0_o, keypoints1_o], c=cut_percentage, cmap="Reds")
            plt.show()

            # # show after cutting
            viz2d.plot_images([image0, image1])
            plot_vessel_tree(keypoints_list=[keypoints0_o[~cut_index0], keypoints1_o[~cut_index1]],
                             label_list=[label0[~cut_index0], label1[~cut_index1]])
            plot_vessel_tree(keypoints_list=[keypoints0_o[cut_index0_0], keypoints1_o[cut_index1_0]],
                             label_list=[label0[cut_index0_0], label1[cut_index1_0]], one_color="red",
                             show_legend=True, legend_name="P Cut",
                             linewidth=4, alpha=0.5)
            plot_vessel_tree(keypoints_list=[keypoints0_o[cut_index0_1], keypoints1_o[cut_index1_1]],
                             label_list=[label0[cut_index0_1], label1[cut_index1_1]], one_color="y",
                             show_legend=True, legend_name="H Cut",
                             linewidth=4, alpha=0.5)
            plot_vessel_tree(keypoints_list=[keypoints0_o[cut_index0_2], keypoints1_o[cut_index1_2]],
                             label_list=[label0[cut_index0_2], label1[cut_index1_2]], one_color="c",
                             show_legend=True, legend_name="R Cut",
                             linewidth=4, alpha=0.5)
            plt.show()
            #
            #
            # # show after cutting
            viz2d.plot_images([image0, image1])
            viz2d.plot_keypoints([keypoints0, keypoints1], a=1.0, colors="blue")
            valid_index = gt_matches0 > -1
            kpt1 = keypoints0[valid_index]
            kpt2 = keypoints1[gt_matches0[valid_index]]
            viz2d.plot_matches(kpt1, kpt2, a=0.2, color="lime")
            plt.show()
            print(f"error in {i}: {val_dataset.image_names[i]}")
            print("hello")
            # from IPython import embed; embed(colors="linux")
        print(i, len(val_dataset))
        pass


