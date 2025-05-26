# 导入必要的库
from pathlib import Path
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd, resize_image
from lightglue import viz2d
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np

# 设置设备
torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义辅助函数
def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


def frame_from_video(frame: np.ndarray, resize: int = None, **kwargs) -> torch.Tensor:
    """Process a video frame and convert it to a torch tensor."""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if resize is not None:
        image, _ = resize_image(image, resize, **kwargs)
    return numpy_image_to_torch(image)


# 加载特征提取器和匹配器
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features="superpoint").eval().to(device)

# 加载图像
type = "chicken"
images = Path(f"images/{type}")
image_path = images / f"{type}_fix.JPG"
video_path = images / f"{type}_video.mp4"
output_dir = Path(f"output/{type}")
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "video").mkdir(parents=True, exist_ok=True)


image0 = load_image(image_path)
image0_feats = extractor.extract(image0.to(device))

# 打开视频文件
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 获取视频的帧率和总帧数
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
total_duration = total_frames / fps

# 设置采样时间间隔（秒）
sample_interval_seconds = 0.05  # 每秒采样一帧

# 计算采样帧间隔
frame_interval = int(fps * sample_interval_seconds)
frame_count = 0

# 处理视频帧
image_files = []
# 处理视频帧
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 只处理每隔 frame_interval 帧
    if frame_count % frame_interval == 0:
        # 使用自定义的 frame_from_video 函数处理视频帧
        image1 = frame_from_video(frame).to(device)
        image1_feats = extractor.extract(image1)

        # 匹配特征
        matches01 = matcher({"image0": image0_feats, "image1": image1_feats})
        image0_feats_move_batch, image1_feats_move_batch, matches01_move_batch = [rbd(x) for x in [image0_feats, image1_feats, matches01]]

        kpts0, kpts1, matches = image0_feats_move_batch["keypoints"], image1_feats_move_batch["keypoints"], matches01_move_batch["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        # 可视化
        axes = viz2d.plot_images([image0, image1])
        viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
        viz2d.add_text(0, f'Stop after {matches01_move_batch["stop"]} layers', fs=20)

        output_file = output_dir / f"frame_{frame_count:04d}.png"
        plt.savefig(output_file)
        image_files.append(str(output_file))

    frame_count += 1

cap.release()


# 创建视频
output_video_path = str(output_dir / "output_video.mp4")
frame = cv2.imread(image_files[0])
height, width, layers = frame.shape

video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

for image_file in image_files:
    video.write(cv2.imread(image_file))

video.release()