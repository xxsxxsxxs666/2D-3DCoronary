from skimage.morphology import skeletonize
import cv2
import numpy as np
from pathlib import Path
import torch
from treelib import Tree
import copy
import json

def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }


def sample_points2(points, num, return_idx=False):
    interval = (len(points) - 1) / (num - 1)
    idx = [int(np.floor(interval * i)) for i in range(num)]
    idx[-1] = min(idx[-1], len(points) - 1)
    if return_idx:
        return np.array(points[idx]), idx
    return np.array(points[idx])


def binary_image_2_points(img: np.array, sample_num: int = None):
    shape = img.shape
    if len(shape) == 3:
        sk = skeletonize((img / 255).sum(axis=-1) > 0)
    else:
        sk = skeletonize(img > 0)
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


def generate_2d_image_numpy(points_position: np.array, radius: np.array, image_size: tuple):
    """
    :param points_position: m x 2
    :param radius: m x 1
    :param image_size: 2 x 1; case: (H, W)
    :return: image array
    """
    x = np.arange(image_size[0])
    y = np.arange(image_size[1])
    xx, yy = np.meshgrid(x, y)  # xx: H x W; yy: H x W

    image = (np.sqrt(((xx[:, :, None] - points_position[:, 0]) ** 2 +
                     (yy[:, :, None] - points_position[:, 1]) ** 2)) - radius[None, None, :]).min(-1) < 0
    return image


def caculate_circle_index(x, y, radius, half_box_size=None):
    """
    :param x: float
    :param y: float
    :param radius: float
    :return: circle index
    """
    if half_box_size is None:
        half_box_size = np.ceil(radius).astype(int)
    x_center = x - np.floor(x)
    y_center = y - np.floor(y)
    xx = np.arange(-half_box_size, half_box_size + 1)
    yy = np.arange(-half_box_size, half_box_size + 1)
    xx, yy = np.meshgrid(xx, yy)
    x_index, y_index = np.where(((xx - x_center) ** 2 + (yy-y_center) ** 2) < radius ** 2)
    x_index_final = xx[x_index, y_index] + np.floor(x)
    y_index_final = yy[x_index, y_index] + np.floor(y)
    return x_index_final.astype(int), y_index_final.astype(int)


def caculate_circle_parrallel(x, y, radius, half_box_size=None):
    """
    :param x: float
    :param y: float
    :param radius: float
    :return: circle index
    """
    if half_box_size is None:
        half_box_size = np.ceil(radius).astype(int)
    x_center = x - np.floor(x)
    y_center = y - np.floor(y)
    xx = np.arange(-half_box_size, half_box_size + 1)
    yy = np.arange(-half_box_size, half_box_size + 1)
    xx, yy = np.meshgrid(xx, yy, indexing="ij")
    x_index, y_index, l_index = (
        np.where(((xx[:, :, None] - x_center) ** 2 + (yy[:, :, None]-y_center) ** 2) < radius ** 2))  # H x W x m
    x_index_final = xx[x_index, y_index] + np.around(x[l_index])
    y_index_final = yy[x_index, y_index] + np.around(y[l_index])
    return x_index_final.astype(int), y_index_final.astype(int)


def generate_2d_image_loop(points_position: np.array, radius: np.array, image_size: np.array):
    """
    :param points_position: m x 2
    :param radius: m x 1
    :param image_size: 2 x 1; case: (H, W)
    :return: image array
    """
    image = np.zeros(image_size)
    for i, (x, y) in enumerate(points_position):
        x_index, y_index = caculate_circle_index(x, y, radius[i])
        image[y_index, x_index] = 1
    return image


def generate_2d_image_parreral(points_position: np.array, radius: np.array, image_size: np.array):
    """
    :param points_position: m x 2 (x->w, y->h)
    :param radius: m x 1
    :param image_size: 2 x 1; case: (H, W)
    :return: image array
    """
    image = np.zeros(image_size)
    x_index, y_index = caculate_circle_parrallel(points_position[:, 1], points_position[:, 0], radius=radius,
                                                 half_box_size=np.ceil(max(radius)))
    filter = (x_index >= 0) & (x_index < image_size[0]) & (y_index >= 0) & (y_index < image_size[1])
    image[x_index[filter], y_index[filter]] = 1
    return image

def caculate_3d_circle_parrallel(x, y, z, radius, half_box_size=None):
    """
    :param x: float
    :param y: float
    :param radius: float
    :return: circle index
    """
    if half_box_size is None:
        half_box_size = np.ceil(radius).astype(int)
    x_center = x - np.floor(x)
    y_center = y - np.floor(y)
    z_center = z - np.floor(z)
    xx = np.arange(-half_box_size, half_box_size + 1)
    yy = np.arange(-half_box_size, half_box_size + 1)
    zz = np.arange(-half_box_size, half_box_size + 1)
    xx, yy, zz = np.meshgrid(xx, yy, zz, indexing="ij")
    pass
    x_index, y_index, z_index, l_index = (
        np.where(((xx[:, :, :, None] - x_center) ** 2 + (yy[:, :, :, None]-y_center) ** 2) + (zz[:, :, :, None]-z_center) ** 2
                 < radius ** 2))  # H x W x m
    x_index_final = xx[x_index, y_index, z_index] + np.floor(x[l_index])
    y_index_final = yy[x_index, y_index, z_index] + np.floor(y[l_index])
    z_index_final = zz[x_index, y_index, z_index] + np.floor(z[l_index])
    return x_index_final.astype(int), y_index_final.astype(int), z_index_final.astype(int)


def generate_3d_image_parreral(points_position: np.array, radius: np.array, image_size: np.array):
    """
    :param points_position: image coordinate
    :param radius: m x 1
    :param image_size: 2 x 1; case: (H, W)
    :return: image array
    """
    image = np.zeros(image_size)
    x_index, y_index, z_index = caculate_3d_circle_parrallel(points_position[:, 0], points_position[:, 1], points_position[:, 2],
                                                             radius=radius,
                                                             half_box_size=np.ceil(max(radius)))
    filter = (x_index >= 0) & (x_index < image_size[0]) & (y_index >= 0) & (y_index < image_size[1]) & \
             (z_index >= 0) & (z_index < image_size[2])

    image[x_index[filter], y_index[filter], z_index[filter], ] = 1
    return image


def caculate_3d_circle_parrallel_torch(x, y, z, radius, device, half_box_size=None):
    """
    :param x: float
    :param y: float
    :param radius: float
    :return: circle index
    """
    if half_box_size is None:
        half_box_size = np.ceil(radius).astype(int)
    x_center = x - torch.floor(x)
    y_center = y - torch.floor(y)
    z_center = z - torch.floor(z)
    xx = torch.arange(-half_box_size, half_box_size + 1)
    yy = torch.arange(-half_box_size, half_box_size + 1)
    zz = torch.arange(-half_box_size, half_box_size + 1)
    xx, yy, zz = torch.meshgrid(xx, yy, zz, indexing="ij")
    xx = xx.to(device)
    yy = yy.to(device)
    zz = zz.to(device)
    x_index, y_index, z_index, l_index = (
        torch.where(((xx[:, :, :, None] - x_center) ** 2 + (yy[:, :, :, None]-y_center) ** 2) + (zz[:, :, :, None]-z_center) ** 2
                    < radius ** 2))  # H x W x m
    x_index_final = xx[x_index, y_index, z_index] + torch.round(x[l_index])
    y_index_final = yy[x_index, y_index, z_index] + torch.round(y[l_index])
    z_index_final = zz[x_index, y_index, z_index] + torch.round(z[l_index])
    return x_index_final.long(), y_index_final.long(), z_index_final.long()


def generate_3d_image_parreral_torch(points_position: torch.Tensor, radius: torch.Tensor, image_size: tuple):
    """
    :param points_position: image coordinate
    :param radius: m x 1
    :param image_size: 2 x 1; case: (H, W)
    :return: image array
    """
    device = points_position.device
    image = torch.zeros(image_size, device=device, dtype=torch.long)
    x_index, y_index, z_index = caculate_3d_circle_parrallel_torch(points_position[:, 0], points_position[:, 1], points_position[:, 2],
                                                                   radius=radius,
                                                                   half_box_size=torch.ceil(radius.max()),
                                                                   device=device)
    filter = (x_index >= 0) & (x_index < image_size[0]) & (y_index >= 0) & (y_index < image_size[1]) & \
             (z_index >= 0) & (z_index < image_size[2])

    image[x_index[filter], y_index[filter], z_index[filter], ] = 1
    return image


def find_overlapping_area(points_position: np.array, label: np.array, radius: np.array, image_size: np.array, around_threshold):
    """
    :param points_position: m x 2
    :param radius: m x 1
    :param image_size: 2 x 1; case: (H, W)
    :return: image array
    """
    image = np.zeros(image_size)
    label_list, start_index = np.unique(label, return_index=True)
    _, end_index = np.unique(label[::-1], return_index=True)
    end_index = len(label) - end_index
    # for each label using function generate_2d_image_parallel
    for i, (start, end) in enumerate(zip(start_index, end_index)):
        image += generate_2d_image_parreral(points_position[start:end], radius[start:end], image_size)

    # cut the area around start_point
    select_point_index = []
    for start_point_index, end_point_index in zip(start_index, end_index):
        extend_index = np.arange(start_point_index, min(start_point_index + around_threshold, end_point_index))
        select_point_index.extend(extend_index)

    # using generate_2d_image_parallel to generate the image around the start_point
    image_around = generate_2d_image_parreral(points_position[select_point_index], radius[select_point_index], image_size)

    return image, image_around


def get_landmark_idx(cta_tree_label):
    if 5 in cta_tree_label:  # LM in label of current CTA tree
        ctl_indices = np.where(cta_tree_label == 5)[0]
    else:  # Current CTA tree is of right coronary, use RCA centerline
        ctl_indices = np.where(cta_tree_label == 1)[0]
        ctl_indices = ctl_indices[:10]
    ctl_mid_idx = ctl_indices[len(ctl_indices) // 2]
    return ctl_mid_idx


def generate_2d_image_torch(points_position: torch.tensor, radius: torch.tensor, image_size: tuple):
    """
    :param points_position: m x 2
    :param radius: m x 1
    :param image_size: 2 x 1; case: (H, W)
    :return: image array
    """
    x = torch.arange(image_size[0])
    y = torch.arange(image_size[1])
    xx, yy = torch.meshgrid(x, y)  # xx: H x W; yy: H x W

    image = (np.sqrt(((xx[:, :, None] - points_position[:, 0]) ** 2 +
                     (yy[:, :, None] - points_position[:, 1]) ** 2)) - radius[None, None, :]).min(-1) < 0
    return image

def convert_data_to_int(data):
    data_ = {}
    for k, v in data.items():
        if isinstance(k, str) and k.isdigit():
            k = int(k)
        if isinstance(v, dict):
            data_[k] = convert_data_to_int(v)
        else:
            data_[k] = v
    return data_


def vessel_bound(points, h, w, side=2):
    min_h, max_h, min_w, max_w = (np.floor(min(0, points[:, 1].min() - side)).astype(np.int16),
                                  np.ceil(max(h, points[:, 1].max()) + side).astype(np.int16),
                                  np.floor(min(0, points[:, 0].min()) - side).astype(np.int16),
                                  np.ceil(max(w, points[:, 0].max()) + side).astype(np.int16))
    return min_h, max_h, min_w, max_w


def get_landmark_idx(cta_tree_label):
    if 5 in cta_tree_label: # LM in label of current CTA tree
        ctl_indices = np.where(cta_tree_label == 5)[0]
    else:  # Current CTA tree is of right coronary, use RCA centerline
        ctl_indices = np.where(cta_tree_label == 1)[0]
        ctl_indices = ctl_indices[:10]
    ctl_mid_idx = ctl_indices[len(ctl_indices) // 2]
    return ctl_mid_idx


def get_gt_matching(cl_keypoints0_o, cl_keypoints1_o, cl_keypoints1, cl2d_keypoints0,
                    cut_index0, cut_index1, cut_index, mode="single0_lax1", lax_distance_threshold=2):

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
    # get gt_assignment
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

    data = {
        "gt_assignment": gt2d_assignment,
        "gt_matches0": torch.tensor(gt_matches0_2d, dtype=torch.float32),
        "gt_matches1": torch.tensor(gt_matches1_2d, dtype=torch.float32),
    }
    return data



class VesselTree:
    def __init__(self, solid_tree_path="gluefactory/utils/vessel_structure/coronary_artery_tree.json",
                 ambiguous_tree_path="gluefactory/utils/vessel_structure/ambiguous.json"):
        """
        :param solid_tree_path: solid branch relationship
        :param ambiguous_tree_path: for S_Side, D_Side ...
        """
        self.solid_tree = self.load_tree_from_json(solid_tree_path)
        with open(ambiguous_tree_path, "r") as f:
            self.ambiguous_tree = json.load(f)
        # get all tag in solid_tree
        self.solid_list = [node.identifier for node in self.solid_tree.all_nodes()]
        self.ambiguous_list = list(self.ambiguous_tree.keys())
        if not isinstance(self.ambiguous_list[0], int):
            self.ambiguous_tree = convert_data_to_int(self.ambiguous_tree)
            self.ambiguous_list = list(self.ambiguous_tree.keys())

    def check_is_ambiguous(self, label):
        """
        :param label: m x 1
        :return: bool m x 1
        """
        return np.isin(label, self.ambiguous_list).sum() > 0

    def generate_case_tree(self, keypoints: np.array, label: np.array):
        """
        :param label: m x 1
        :param keypoints: m x 2 or m x 3
        :return:
        """
        case_branch_id_list, branch_start_index = np.unique(label, return_index=True)
        branch_end_index = np.unique(label[::-1], return_index=True)[1]
        branch_end_index = len(label) - branch_end_index
        branch_start_end_index = {branch_id: (start, end) for branch_id, start, end in
                                  zip(case_branch_id_list, branch_start_index, branch_end_index)}
        branch_start = keypoints[branch_start_index]
        id2index = {branch_id: i for i, branch_id in enumerate(case_branch_id_list)}

        # get intersection between case_branch_id_list and solid_list
        case_branch_id_list_solid = list(set(case_branch_id_list) & set(self.solid_list))
        cut_list = list(set(self.solid_list) - set(case_branch_id_list_solid))
        # get intersection between case_branch_id_list and ambiguous_list
        case_branch_id_list_ambiguous = list(set(case_branch_id_list) & set(self.ambiguous_list))

        # copy the solid_tree and only remain the node in case_branch_id_list_solid using while
        case_tree = copy.deepcopy(self.solid_tree)

        for cut_id in cut_list:
            cut_node = case_tree.get_node(cut_id)
            if cut_node is not None:
                # print(cut_id, f"It's children {case_tree.is_branch(cut_node.identifier)} will be removed.")
                case_tree.remove_node(cut_node.identifier)

        ambiguous_index = np.isin(case_branch_id_list, case_branch_id_list_ambiguous)
        root_index = case_branch_id_list == self.solid_tree.root


        # calculate the distance between branch_start and other keypoints labeled in ambiguous_tree['branch_id']
        distance_map = np.linalg.norm(np.array(branch_start)[:, None, :] - keypoints[None, :, :], axis=-1)
        mask = np.ones_like(distance_map, dtype=bool)
        for i, branch_id in enumerate(case_branch_id_list):
            if branch_id in case_branch_id_list_ambiguous:
                for parent_id in self.ambiguous_tree[branch_id]["parents"]:
                    if parent_id in case_branch_id_list:
                        start, end = branch_start_end_index[parent_id]
                        mask[i, start:end] = False
            elif self.solid_tree.parent(branch_id) is not None:
                # using self.solid_tree to get parent id
                start, end = branch_start_end_index[self.solid_tree.parent(branch_id).identifier]
                mask[i, start:end] = False
        distance_map[mask] = np.inf
        # get the nearest point, if the nearest distance is inf ,than return -1
        mount_index = distance_map.argmin(axis=1)
        mount_index[root_index] = -1
        ambiguous_point_parent_id = {}
        for id in case_branch_id_list_ambiguous:
            ambiguous_point_parent_id[id] = label[mount_index[id2index[id]]]

        # add child node to the parent node
        for i, branch_id in enumerate(sorted(case_branch_id_list_ambiguous)):
            parent_node = case_tree.get_node(ambiguous_point_parent_id[branch_id])
            if parent_node is None:
                print(branch_id)
            case_tree.create_node(tag=branch_id, identifier=branch_id, parent=parent_node.identifier, data={})

        # add children information in data when loop the solid tree,
        # the content is that children start is in which percentage of parent
        for node in case_tree.all_nodes():
            start, end = branch_start_end_index[node.identifier][0].item(), branch_start_end_index[node.identifier][1].item()
            branch_length = (end - start)
            node.data["start_end_length"] = (start,
                                             end,
                                             branch_length)
            if case_tree.is_branch(node.identifier):
                children = case_tree.children(node.identifier)
                node.data["children_mount"] = {}
                node.data["children_mount_index"] = {}
                for child in children:
                    mount_percentage = (mount_index[id2index[child.identifier]] -
                                        branch_start_end_index[node.identifier][0] + 1) / branch_length
                    if mount_percentage > 1:
                        print(f"Branch {child.identifier} is not in the branch {node.identifier}.")
                    assert 0 <= mount_percentage <= 1, f"Mount percentage should be in [0, 1], but got {mount_percentage}."
                    node.data["children_mount"][child.identifier] = mount_percentage.item()
                    node.data["children_mount_index"][child.identifier] = mount_index[id2index[child.identifier]]

            if node.is_leaf():
                node.data["children_mount"] = None

        return case_tree, branch_start_end_index

    @staticmethod
    def random_generate_cutting_id_and_percentage(cutting_percentage: np.array,
                                                  label: np.array,
                                                  case_tree: Tree,
                                                  cutting_branch_num: int = 5,
                                                  percentage_threshold: tuple = (0.0, 0.5)):
        """
        :param percentage_threshold: float, the threshold of the cutting percentage
        :param cutting_percentage: float, the percentage of the cutting branch, which means when cutting the branch, the percentage of the vessel will be cut
        :param label: np.array, m x 1
        :param case_tree: treelib.Tree each node contains data {"children_mount":{child_id: percentage, ...}}, from self.generate_case_tree
        :param cutting_branch_num: int, the number of the cutting branch
        :return: cut_branch_list, cut_percentage_list
        """
        # randomly select N branch to cut and make sure the sum of the cutting percentage is less than cutting_percentage
        cut_branch_list = []
        cut_percentage_list = []
        scope_index = (cutting_percentage < percentage_threshold[1]) & (cutting_percentage > percentage_threshold[0])
        current_cutting_percentage = 0
        select_id = np.arange(len(label))
        while current_cutting_percentage < percentage_threshold[1] and len(cut_branch_list) < cutting_branch_num:
            if np.sum(scope_index) == 0:
                break
            select_index = np.random.choice(select_id[scope_index], size=1)
            branch_id = label[select_index][0]
            percentage = cutting_percentage[select_index]
            if branch_id in cut_branch_list:
                continue
            cut_branch_list.append(branch_id)
            start, _, length = case_tree.get_node(branch_id).data["start_end_length"]
            branch_percentage = (select_index - start + 1) / length
            # move children branch if branch_percentage is larger than the mount_percentage
            if case_tree.is_branch(branch_id):
                children = case_tree.children(branch_id)
                for child in children:
                    child_percentage = case_tree.get_node(branch_id).data["children_mount"][child.identifier]
                    if branch_percentage < child_percentage:
                        # TODO: need to be accelerated using start_end_index
                        index = label == child.identifier
                        scope_index *= ~index

            cut_percentage_list.append((1 - branch_percentage).item())
            current_cutting_percentage += percentage

        return cut_branch_list, cut_percentage_list

    @staticmethod
    def random_generate_cutting_id_for_head_blocking(cutting_percentage: np.array,
                                                     label: np.array,
                                                     case_tree: Tree,
                                                     cutting_branch_num: int = 5,
                                                     percentage_threshold: tuple = (0.0, 0.5), ):
        """
        :param percentage_threshold: float, the threshold of the cutting percentage
        :param cutting_percentage: float, the percentage of the cutting branch, which means when cutting the branch, the percentage of the vessel will be cut
        :param case_tree: treelib.Tree each node contains data {"children_mount":{child_id: percentage, ...}}, from self.generate_case_tree
        :param label: np.array, m x 1
        :param cutting_branch_num: int, the number of the cutting branch
        :return: cut_branch_list, cut_percentage_list
        """
        # randomly select N branch to cut and make sure the sum of the cutting percentage is less than cutting_percentage
        cut_branch_list = []
        node_list = [node for node in case_tree.all_nodes()]
        start_point_index = np.array([node.data["start_end_length"][0] for node in node_list])
        filter_index = (cutting_percentage[start_point_index] < percentage_threshold[1]) & \
                       (cutting_percentage[start_point_index] > percentage_threshold[0])
        out_label_list = label[start_point_index[filter_index]]
        current_cutting_percentage = 0
        potential_list = []
        for node in case_tree.all_nodes():
            if not node.is_root():
                parent_node = case_tree.parent(node.identifier)
                mount_percentage = parent_node.data["children_mount"][node.identifier]
                if mount_percentage < 0.8 and mount_percentage > 0.2 and node.identifier not in out_label_list:
                    potential_list.append(node.identifier)

        while current_cutting_percentage < percentage_threshold[1] and len(cut_branch_list) < cutting_branch_num:
            if len(potential_list) == 0:
                break
            branch_id = np.random.choice(potential_list, size=1).item()
            if branch_id in cut_branch_list:
                continue
            cut_branch_list.append(branch_id)
            # move branch_id from potential_list
            potential_list.remove(branch_id)
            if case_tree.is_branch(branch_id):
                # move children branch in potential_list if exist
                potential_list = list(set(potential_list) - set(case_tree.children(branch_id)))

        return cut_branch_list, [1 for i in cut_branch_list]

    @staticmethod
    def random_root_cutting(
            label: np.array,
            case_tree: Tree,
            percentage_threshold: tuple = (0.0, 0.5), ):
        """
        :param percentage_threshold: float, the threshold of the cutting percentage
        :param cutting_percentage: float, the percentage of the cutting branch, which means when cutting the branch, the percentage of the vessel will be cut
        :param case_tree: treelib.Tree each node contains data {"children_mount":{child_id: percentage, ...}}, from self.generate_case_tree
        :param label: np.array, m x 1
        :param cutting_branch_num: int, the number of the cutting branch
        :return: cut_branch_list, cut_percentage_list
        """
        flag = False
        cut_index = np.zeros_like(label, dtype=bool)
        cut_tree = copy.deepcopy(case_tree)
        root_id = cut_tree.root
        root_node = cut_tree.get_node(root_id)
        start, end, length = root_node.data["start_end_length"]
        if len(list(root_node.data["children_mount"].values())) == 0:
            print("Root node do not have children.")
            flag = True
            # from IPython import embed;embed(colors="linux")
            return cut_index, cut_tree, flag
        children_mount_min = min(root_node.data["children_mount"].values())
        percentage_threshold = (percentage_threshold[0], min(percentage_threshold[1], children_mount_min))
        cut_percentage = np.random.uniform(*percentage_threshold)
        cut_index[start: int(start + length * cut_percentage)] = True
        # modify the start_end_length of the root node, also modify the children_mount
        root_node.data["start_end_length"] = (start, int(start + length * cut_percentage), length * cut_percentage)
        # because it is the root node, so it must have children
        assert len(root_node.data["children_mount"].keys()) > 0, "Root node do not have children."
        for child_id, percentage in root_node.data["children_mount"].items():
            cut_length = int(length * cut_percentage)
            if cut_length > 0:
                root_node.data["children_mount"][child_id] = (root_node.data["children_mount"][child_id] * length
                                                             - cut_length) / cut_length

        return cut_index, cut_tree

    @staticmethod
    def cut_branch_percentage(case_tree, branch_start_end_index, label, cut_branch_list, cut_percentage_list):
        """
        :param case_tree: treelib.Tree each node contains data {"children_mount":{child_id: percentage, ...}}, from self.generate_case_tree
        :param branch_start_end_index: dict, m x 2 , from self.generate_case_tree
        :param label: np.array, m x 1
        :param cut_branch_list: k x 1, contains the branch_id that need to be cut
        :param cut_percentage: k x 1, contains the percentage that need to be cut
        :return: cut_tree, cut_index (bool) m x 1
        """
        # TODO: the data (mount_percentage) in cut_tree is not correct, need to be fixed
        flag = False
        cut_tree = copy.deepcopy(case_tree)
        cut_index = np.zeros_like(label, dtype=bool)
        existing_id_list = [node.identifier for node in case_tree.all_nodes()]
        assert np.isin(cut_branch_list, existing_id_list).all(), \
            "Some branch_id in cut_branch_list not found in case_tree."
        # sort the cut_branch_list by the level of the branch
        sorted_indices = sorted(range(len(cut_branch_list)), key=lambda index: case_tree.level(cut_branch_list[index]),
                                reverse=True)
        cut_branch_list = [cut_branch_list[i] for i in sorted_indices]
        cut_percentage_list = [cut_percentage_list[i] for i in sorted_indices]
        while len(cut_branch_list) > 0:
            branch_id = cut_branch_list.pop()
            cut_percentage = cut_percentage_list.pop()
            cut_node = cut_tree.get_node(branch_id)
            if cut_node is None:
                print(f"Branch {branch_id} not found in case_tree.")
                flag = True
                # from IPython import embed; embed(colors="linux")
                continue
            if cut_percentage == 0:
                continue

            start, end = (cut_tree.get_node(branch_id).data["start_end_length"][0],
                          cut_tree.get_node(branch_id).data["start_end_length"][1])
            cut_index[int(end - (end - start) * cut_percentage):end] = True
            cut_node.data["start_end_length"] = (
                cut_node.data["start_end_length"][0],
                int(end - (end - start) * cut_percentage),
                int(end - (end - start) * cut_percentage) - cut_node.data["start_end_length"][0]
            )
            if len(cut_tree.is_branch(branch_id)) > 0:
                # if the cut_percentage is larger than node.data["children_mount"][child_id], than cut the child node
                pop_list = []
                for child_id, percentage in cut_node.data["children_mount"].items():
                    if cut_percentage < 1:
                        cut_node.data["children_mount"][child_id] = percentage / (1 - cut_percentage)
                    if (1 - percentage) <= cut_percentage:
                        # TODO: Move all the children of the child_id to the cut_branch_list. Already Done!
                        subtree = cut_tree.remove_subtree(child_id)
                        pop_list.append(child_id)
                        cut_node.data["children_mount_index"].pop(child_id)
                        for child_child in subtree.all_nodes():
                            child_child_id = child_child.identifier
                            start, end = (child_child.data["start_end_length"][0],
                                          child_child.data["start_end_length"][1])
                            cut_index[start: end] = True
                            if child_child_id in cut_branch_list:
                                cut_percentage_list.pop(cut_branch_list.index(child_child_id))
                                cut_branch_list.remove(child_child_id)
                                # move corresponding cut_percentage
                for child_id in pop_list:
                    cut_node.data["children_mount"].pop(child_id)

            if cut_percentage == 1:
                cut_tree.remove_node(branch_id)
        return cut_index, cut_tree

    @staticmethod
    def get_cutting_percentage_for_each_point(case_tree, label):
        """
        :param case_tree: treelib.Tree each node contains data {"children_mount":{child_id: percentage, ...}}, from self.generate_case_tree
        :param label: np.array, m x 1
        :param branch_start_end_index: dict, m x 2 , from self.generate_case_tree
        :return: cut_tree, cut_index (bool) m x 1
        """
        cutting_percentage = np.zeros_like(label, dtype=np.float32)
        node_list = [node for node in case_tree.all_nodes()]
        # sort by level
        node_list = sorted(node_list, key=lambda x: case_tree.level(x.identifier), reverse=True)
        for node in node_list:
            start, end = node.data["start_end_length"][0], node.data["start_end_length"][1]
            branch_length = node.data["start_end_length"][2]
            cutting_percentage[start:end] = np.arange(branch_length, 0, -1)
            node.data["mount_length"] = branch_length
            if case_tree.is_branch(node.identifier):
                children = case_tree.children(node.identifier)
                children_id_list = [child.identifier for child in children]
                children_id_list = sorted(children_id_list, key=lambda x: (1 - node.data["children_mount"][x]))
                for child_id in children_id_list:
                    child = case_tree.get_node(child_id)
                    mount_index = node.data["children_mount_index"][child_id]
                    cutting_percentage[start: (mount_index + 1)] += child.data["mount_length"]
                    node.data["mount_length"] += child.data["mount_length"]
        cutting_percentage /= cutting_percentage.max()

        return cutting_percentage

    def random_cut(self, label, case_tree, branch_start_end_index: dict, p_cut_p=0.5,
                   percentage_cut_num=3, percentage_threshold=(0.0, 0.5), h_cut_p=0.5,
                   head_cut_num=3, head_cutting_threshold=(0, 0.5),
                   root_cut_num=1, r_cut_p=0.5, root_cutting_threshold=(0, 0.5)):
        cut_tree = copy.deepcopy(case_tree)
        cut_index = np.zeros(len(label), dtype=bool)
        if percentage_cut_num > 0 and np.random.random() < p_cut_p:
            cutting_percentage = self.get_cutting_percentage_for_each_point(case_tree=case_tree,
                                                                            label=label)
            remove_list, remove_percentage = \
                self.random_generate_cutting_id_and_percentage(cutting_percentage=cutting_percentage,
                                                               label=label,
                                                               case_tree=case_tree,
                                                               cutting_branch_num=percentage_cut_num,
                                                               percentage_threshold=percentage_threshold)
            cut_index, cut_tree = self.cut_branch_percentage(case_tree=case_tree,
                                                             branch_start_end_index=branch_start_end_index,
                                                             label=label,
                                                             cut_branch_list=remove_list,
                                                             cut_percentage_list=remove_percentage)

        if head_cut_num > 0 and np.random.random() < h_cut_p:
            cutting_percentage = self.get_cutting_percentage_for_each_point(case_tree=cut_tree,
                                                                                        label=label)
            remove_list, remove_percentage = \
                self.random_generate_cutting_id_for_head_blocking(cutting_percentage=cutting_percentage,
                                                                  label=label,
                                                                  case_tree=cut_tree,
                                                                  cutting_branch_num=head_cut_num,
                                                                  percentage_threshold=head_cutting_threshold)

            cut_index_, cut_tree = self.cut_branch_percentage(case_tree=cut_tree,
                                                              branch_start_end_index=branch_start_end_index,
                                                              label=label,
                                                              cut_branch_list=remove_list,
                                                              cut_percentage_list=remove_percentage)
            cut_index = cut_index | cut_index_

        if root_cut_num > 0 and np.random.random() < r_cut_p:
            cut_index_, cut_tree = self.random_root_cutting(label=label, case_tree=cut_tree,
                                                            percentage_threshold=root_cutting_threshold)
            cut_index = cut_index | cut_index_

        return cut_index

    @staticmethod
    def get_longest_id(branch_start_end_index):
        max_id = -1
        max_length = 0
        for id, info in branch_start_end_index.items():
            length = info[1] - info[0]
            if length > max_length:
                max_id = id
                max_length = length
        return max_id

    @staticmethod
    def load_tree_from_json(json_file):
        with open(json_file, 'r') as f:
            tree_data = json.load(f)

        tree = Tree()

        def add_nodes(parent_id, node_data):
            for children in node_data:
                children_id = list(children.keys())[0]
                children_info = children[children_id]
                tree.create_node(tag=int(children_id), identifier=int(children_id), parent=int(parent_id),
                                 data=convert_data_to_int(children_info.get('data', {})))
                if 'children' in children_info:
                    add_nodes(children_id, children_info['children'])

        root_id = list(tree_data.keys())[0]
        root_info = tree_data[root_id]
        root_label = root_id

        tree.create_node(tag=int(root_label), identifier=int(root_id),
                         data=convert_data_to_int(root_info.get('data', {})))
        if 'children' in root_info:
            add_nodes(root_id, root_info['children'])

        return tree






