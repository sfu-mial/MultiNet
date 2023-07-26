# Multitask Deep Learning Reconstruction and Localization of Lesions in Limited Angle Diffuse Optical Tomography
This is the software repository "MultiNet" for [our](https://ieeexplore.ieee.org/document/9557304) [paper](#cite) solving problem of breast cancer lesion reconstruction and localization in novel deep spatial attention based paradigms.

## Motivation
Diffuse optical tomography (DOT) leverages near-infrared light propagation through tissue to assess its optical properties and identify abnormalities. DOT image reconstruction is an ill-posed problem due to the highly scattered photons in the medium and the smaller number of measurements compared to the number of unknowns. Limited-angle DOT reduces probe complexity at the cost of increased reconstruction complexity.
Reconstructions are thus commonly marred by artifacts and, as a result, it is difficult to obtain an accurate reconstruction of target objects, e.g., malignant lesions.Reconstruction does not always ensure good localization of small lesions. Furthermore, conventional optimization-based reconstruction methods are computationally expensive, rendering them too slow for real-time imaging applications.

Our goal is to develop a fast and accurate image reconstruction method using deep learning, where multitask learning ensures accurate lesion localization in addition to improved reconstruction.
We apply spatial-wise attention and a distance transform based loss function in a novel multitask learning formulation to improve localization and reconstruction compared to single-task optimized methods.
Given the scarcity of real-world sensor-image pairs required for training supervised deep learning models,
we leverage physics-based simulation to generate synthetic datasets and use a transfer learning module to align the sensor domain distribution between in silico and real-world data, while taking advantage of cross-domain learning.
Applying our method, we find that we can reconstruct and localize lesions faithfully while allowing real-time reconstruction. We also demonstrate that the present algorithm can reconstruct multiple cancer lesions.
The results demonstrate that multitask learning provides sharper and more accurate reconstruction.
Both quantitative and qualitative results on phantom and real data indicate the superiority of our multitask method in the reconstruction and localization of lesions in tissue compared to state-of-the-art methods.
## Keywords
Diffuse optical tomography, image reconstruction, deep learning, multi-frequency, tissue estimation, lesion classification, diagnosis, multitask learning, transfer learning, handheld probe.
## Citation
<a name="Cite"></a>
```bibtext
@article{ben2022orthogonal,
  title={Orthogonal Multi-frequency Fusion Based Image Reconstruction and Diagnosis in Diffuse Optical Tomography},
  author={Ben Yedder, Hanene and Cardoen, Ben and Shokoufi, Majid and Golnaraghi, Farid and Hamarneh, Ghassan},
  year={2022},
  publisher={TechRxiv}
}
```
