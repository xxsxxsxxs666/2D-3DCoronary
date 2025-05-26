# If we are on colab: this clones the repo and installs the dependencies
from pathlib import Path

# if Path.cwd().name != "LightGlue":
#     !git clone --quiet https://github.com/cvg/LightGlue/
#     %cd LightGlue
#     !pip install --progress-bar off --quiet -e .

from lightglue import LightGlue, SuperPoint, DISK, SIFT
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)
images = Path("assets")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

extractor_type = "superpoint"
if extractor_type == "superpoint":
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
elif extractor_type == "sift":
    extractor = SIFT(max_num_keypoints=2048).eval().to(device)  # load the extractor
elif extractor_type == "disk":
    extractor = DISK(max_num_keypoints=2048).eval().to(device)
else:
    raise ValueError("Unknown extractor type")

matcher = LightGlue(features=extractor_type).eval().to(device)

# paired images are in H:\CTA2DSA\LightGlue\paired_data\CCTA_segmentation_projection and H:\CTA2DSA\LightGlue\paired_data\DSA_alg_mask_png
# 这两个文件夹中每个文件前9位是一样的，所以可以直接对应
# 迭代读取两个文件夹中的文件，然后进行匹配

CCTA_projection_path = Path("H:/CTA2DSA/LightGlue/new_paired_data/cta_image")
DSA_mask_path = Path("H:/CTA2DSA/LightGlue/new_paired_data/dsa_image")
save_path = Path(f"H:/CTA2DSA/LightGlue/output/new_vessel_{extractor_type}")
save_path.mkdir(parents=True, exist_ok=True)

for file in CCTA_projection_path.iterdir():
    if file.suffix == ".png":
        img1 = load_image(file)
        img0 = load_image(DSA_mask_path / file.name)
        feats0 = extractor.extract(img0.to(device))
        feats1 = extractor.extract(img1.to(device))
        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]  # remove batch dimension

        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        axes = viz2d.plot_images([img0, img1])
        viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
        viz2d.add_text(0, f'Stop after {matches01["stop"]} layers; \n patient_id: {file.name}', fs=20)
        # save the matched image
        plt.show()
        plt.savefig(save_path / file.name)

        kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
        viz2d.plot_images([img0, img1])
        viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
        plt.show()


# image0 = load_image(images / "DSC_0411.JPG")
# image1 = load_image(images / "DSC_0410.JPG")
#
# feats0 = extractor.extract(image0.to(device))
# feats1 = extractor.extract(image1.to(device))
# matches01 = matcher({"image0": feats0, "image1": feats1})
# feats0, feats1, matches01 = [
#     rbd(x) for x in [feats0, feats1, matches01]
# ]  # remove batch dimension
#
# kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
# m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
#
# axes = viz2d.plot_images([image0, image1])
# viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
# viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
#
# kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
# viz2d.plot_images([image0, image1])
# viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
# plt.show()

