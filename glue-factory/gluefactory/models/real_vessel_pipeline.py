"""
A two-view sparse feature matching pipeline.

This model contains sub-models for each step:
    feature extraction, feature matching, outlier filtering, pose estimation.
Each step is optional, and the features or matches can be provided as input.
Default: SuperPoint with nearest neighbor matching.

Convention for the matches: m0[i] is the index of the keypoint in image 1
that corresponds to the keypoint i in image 0. m0[i] = -1 if i is unmatched.
"""

from omegaconf import OmegaConf

from . import get_model
from .base_model import BaseModel
import torch

to_ctr = OmegaConf.to_container  # convert DictConfig to dict

def sample_descriptors_fix_sampling(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = keypoints / (keypoints.new_tensor([w, h]) * s)
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=False
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors

class VesselPipeline(BaseModel):
    default_conf = {
        "extractor": {
            "name": None,
            "trainable": False,
        },
        "matcher": {"name": None},
        "filter": {"name": None},
        "solver": {"name": None},
        "ground_truth": {"name": None},
        "allow_no_extract": False,
        "run_gt_in_forward": False,
    }
    required_data_keys = ["view0", "view1"]
    strict_conf = False  # need to pass new confs to children models
    components = [
        "extractor",
        "matcher",
        "filter",
        "solver",
        "ground_truth",
    ]

    def _init(self, conf):
        if conf.extractor.name:
            self.extractor = get_model(conf.extractor.name)(to_ctr(conf.extractor))

        if conf.matcher.name:
            self.matcher = get_model(conf.matcher.name)(to_ctr(conf.matcher))

        if conf.filter.name:
            self.filter = get_model(conf.filter.name)(to_ctr(conf.filter))

        if conf.solver.name:
            self.solver = get_model(conf.solver.name)(to_ctr(conf.solver))

        if conf.ground_truth.name:
            self.ground_truth = get_model(conf.ground_truth.name)(
                to_ctr(conf.ground_truth)
            )

    def extract_view(self, data, i):
        data_i = data[f"view{i}"]
        pred_i = data_i.get("cache", {})
        # skip_extract = len(pred_i) > 0 and self.conf.allow_no_extract
        result_dict = self.extractor(data_i)
        # B x num_keypoints x dim_num
        pred_i[f'descriptors{i}'] = (
            sample_descriptors_fix_sampling(data[f'keypoints{i}'], result_dict['descriptors'], 8)).transpose(-1, -2)
        return pred_i

    def _forward(self, data):
        pred0 = self.extract_view(data, "0")
        pred1 = self.extract_view(data, "1")
        pred = {
            **pred0,
            **pred1,
        }
        if self.conf.matcher.name:
            pred = {**pred, **self.matcher({**data, **pred})}
        return pred

    def loss(self, pred, data):
        losses = {}
        metrics = {}
        total = 0

        # get labels
        if self.conf.ground_truth.name and not self.conf.run_gt_in_forward:
            gt_pred = self.ground_truth({**data, **pred})
            pred.update({f"gt_{k}": v for k, v in gt_pred.items()})

        for k in self.components:
            apply = True
            if "apply_loss" in self.conf[k].keys():
                apply = self.conf[k].apply_loss
            if self.conf[k].name and apply:
                try:
                    losses_, metrics_ = getattr(self, k).loss(pred, {**pred, **data})
                except NotImplementedError:
                    continue
                losses = {**losses, **losses_}
                metrics = {**metrics, **metrics_}
                total = losses_["total"] + total
        return {**losses, "total": total}, metrics
