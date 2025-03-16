# Topology and Occlusion-Aware Rigid Registration for 2D/3D Coronary Arteries Alignment: Overcoming Branch Missing and Vessel Overlap
This repository implements a deep learning-based method for rigid DSA-CTA registration in cardiac navigation. The approach integrates a vascular-topology-aware framework with transformer-based techniques to improve vascular registration.
## Visuals
![Project Diagram](IPMI_pipeline.png) 
### Key Features:
- **Vascular Structure Matching**: Novel framework for 2D DSA and 3D CTA rigid registration, addressing vessel overlap and missing branches.
- **Realistic Data Generation**: Simulation strategy to generate datasets that replicate real-world challenges, such as branch missing and vessel overlap.
- **Comprehensive Datasets**: Utilizes both large clinical and simulated datasets for training and evaluation.

The method demonstrates effectiveness and robustness on both simulated and real-world datasets.

## TODO List
### 1. Inference  
- [ ] Open-source inference model weights  
- [ ] Release example inference script and test data
### 2. Data  
- [ ] Open-source partial simulation data  
- [ ] Open-source partial real-world data (considering data privacy)  
### 3. Simulation Module  
- [ ] Organize and release simulation code  
- [ ] Provide example simulation data  
- [ ] Write usage documentation for the simulation module  
### 4. Training  
- [ ] Open-source training code  
