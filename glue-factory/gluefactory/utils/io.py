import json
import pandas as pd
import cv2
from pathlib import Path
import numpy as np
import SimpleITK as sitk

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

def read_json(file_path:str):
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data:dict, file_path:str, log=False):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    if log:
        print(f"成功保存至{file_path}")


def read_nii(file_path: Path) -> np.ndarray:
    """Read a .nii file from path"""
    image = sitk.ReadImage(str(file_path))
    return sitk.GetArrayFromImage(image)


def save_nii(data: np.ndarray, file_path: Path, reference_path: Path = None, reference_itk_image=None) -> None:
    """Save a .nii file to path"""
    image = sitk.GetImageFromArray(data)
    if reference_itk_image is not None:
        image.CopyInformation(reference_itk_image)
    elif reference_path is not None:
        reference = sitk.ReadImage(str, reference_path)
        image.CopyInformation(reference)
    sitk.WriteImage(image, str(file_path))


# here is a class, 可以方便地将数据存成excel, 每个例子都会又名字，且会得到mertic1:xxx, metric2:xxx, metric的名字不定，数量也不定，但是不同例子之间是相同的


class MetricRecorder:
    def __init__(self, columns:list):
        self.data = pd.DataFrame(columns=columns)

    def add(self, name:str, **metrics):
        for metric in metrics.keys():
            if metric not in self.data.columns:
                self.data[metric] = None

        new_row = {'name': name}
        new_row.update(metrics)
        self.data = pd.concat([self.data, pd.DataFrame([new_row], index=[0])], ignore_index=True)

    def to_excel(self, file_path: str):
        self.data.to_excel(file_path, index=False)

    def to_json(self, file_path: str, log=False):
        result = self.data.to_dict(orient='records')
        with open(file_path, 'w') as f:
            json.dump(result, f, indent=4)
        if log:
            print(f"成功保存至{file_path}")

    def load_excel(self, file_path: str):
        if Path(file_path).exists():
            # delete it
            Path(file_path).unlink()
        self.data = pd.read_excel(file_path)

    def calculate_mean(self):
        # 会计算所有的metric的平均值, 注意第一列和第一行是名字，不是metric

        for metric in self.data.columns[1:]:
            self.data.loc['mean', metric] = self.data[metric].mean()

    def show_data(self):
        print(self.data)



if __name__ == "__main__":
    recorder = MetricRecorder(['Name', 'mse', 'angle'])
    recorder.add('example1', mse=0.1, angle=0.2)
    recorder.add('example2', mse=0.2, angle=0.3)
    recorder.to_excel('/home/xiaosong_xiong/Intern_project/CTA2DSA/script/results/example.xlsx')
