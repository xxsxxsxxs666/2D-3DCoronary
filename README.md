# Vascular-topology-aware Deep Structure Matching for 2D DSA and 3D CTA Rigid Registration
This repository implements a deep learning-based method for rigid DSA-CTA registration in cardiac navigation. The approach integrates a vascular-topology-aware framework with transformer-based techniques to improve vascular registration.
## Vascular-topology-aware Deep Structure Matching Framework
![Project Diagram](IPMI_pipeline.png) 
### Key Features:
- **Vascular Structure Matching**: Novel framework for 2D DSA and 3D CTA rigid registration, addressing vessel overlap and missing branches.
- **Realistic Data Generation**: Simulation strategy to generate datasets that replicate real-world challenges, such as branch missing and vessel overlap.
- **Comprehensive Datasets**: Utilizes both large clinical and simulated datasets for training and evaluation.

The method demonstrates effectiveness and robustness on both simulated and real-world datasets.

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![GitHub Stars](https://img.shields.io/github/stars/xxsxxsxxs666/2D-3DCoronary?style=social)
## TODO List
### 1. Inference  
- [x] Open-source inference model weights  
- [ ] Release example inference script and test data
### 2. Data  
- [ ] Open-source partial simulation data  
- [ ] Open-source partial real-world data (considering data privacy)  
### 3. Simulation Module  
- [x] Organize and release simulation code  
- [ ] Provide example simulation data  
- [ ] Write usage documentation for the simulation module  
### 4. Training  
- [x] Open-source training code  

> **News**: Our paper **"Vascular-topology-aware Deep Structure Matching for 2D DSA and 3D CTA Rigid Registration"** has been accepted to the **2025 International Conference on Information Processing in Medical Imaging (IPMI)**! 


## Citation & Acknowledgment

If you find this work helpful, please consider giving this repository a star ⭐ and citing our paper:
### Citation
```
Xiong, X., et al. (2025). Vascular-topology-aware Deep Structure Matching for 2D DSA and 3D CTA Rigid Registration. 
*Proceedings of the 2025 International Conference on Information Processing in Medical Imaging (IPMI)*.

```

### Acknowledgment
This work was conducted at **United Imaging Intelligence (联影智能)** and **ShanghaiTech University (上海科技大学)**.  
We sincerely appreciate their support.

<p align="center">
  <img src="assets/logo_uii.png" alt="United Imaging Intelligence Logo" height="80">
  <img src="assets/shanghaitech.png" alt="ShanghaiTech University Logo" height="80">
</p>