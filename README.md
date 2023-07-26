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
Diffuse optical tomography, image reconstruction, deep learning, tissue estimation, lesion localisation, multitask learning, transfer learning, handheld probe.
## Citation
<a name="Cite"></a>
```bibtext
@ARTICLE{9557304,
  author={Ben Yedder, Hanene and Cardoen, Ben and Shokoufi, Majid and Golnaraghi, Farid and Hamarneh, Ghassan},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Multitask Deep Learning Reconstruction and Localization of Lesions in Limited Angle Diffuse Optical Tomography}, 
  year={2022},
  volume={41},
  number={3},
  pages={515-530},
  doi={10.1109/TMI.2021.3117276}}
```
## Table of contents
1. [Contribution](#contribution)
2. [Installation](#install)
3. [Usage](#usage)
4. [Questions?](#faq)

### Contribution
<a name="contribution"></a>
- We investigate the benefit of multitask deep learning on the quality of DOT reconstruction and breast lesion localization.
- We leverage a deep spatial-wise attention network to adaptively re-weight features and attend to the most important ones.
- We introduce a distance transform loss to improve lesion localization.
- We present the first work that tests DL-based DOT reconstruction generalization on real patient data.
- We conduct experiments to assess the trade-offs between network characteristics, scanning probe design, and performance.
### Installation
<a name="install"></a>

```bash
git clone https://github.com/haneneby/MultiNet.git  
cd MultiNet
conda env create -f requirement.yml --name MultiNet
conda activate MultiNet
```
### Usage
<a name="Usage"></a>
For quick hints about commands:
```bash
cd MultiNet
python3 MultiNet.py -h
```
#### Training
<a name="Training"></a>
```bash
export CUDA_VISIBLE_DEVICES=0 #or change to your GPU config
mkdir myoutput
cd MultiNet
python3 MultiNet.py --epochs 100 --outputfolder ../myoutput.      


```
This will show something like:
```bash
[FuseNet++.py:100 -          initializer() ] Writing output in /dev/shm/MultiNet/MultiNet/../myoutput
[FuseNet++.py:101 -          initializer() ] Logging directory /dev/shm/MultiNet/MultiNet/../myoutput
[FuseNet++.py:104 -          initializer() ] CONF::		 epochs -> 100
[FuseNet++.py:104 -          initializer() ] CONF::		 lr -> 0.0001
[FuseNet++.py:104 -          initializer() ] CONF::		 seed -> 2
[FuseNet++.py:104 -          initializer() ] CONF::		 device -> gpu
[FuseNet++.py:104 -          initializer() ] CONF::		 orth -> True
[FuseNet++.py:104 -          initializer() ] CONF::		 batchsize -> 16
[FuseNet++.py:104 -          initializer() ] CONF::		 beta -> 0.2
[FuseNet++.py:104 -          initializer() ] CONF::		 gamma -> 0.5
[FuseNet++.py:104 -          initializer() ] CONF::		 checkpoint -> None
[FuseNet++.py:104 -          initializer() ] CONF::		 datasetdirectory -> ./data/data_samples/
[FuseNet++.py:104 -          initializer() ] CONF::		 outputfolder -> ../myoutput
[FuseNet++.py:104 -          initializer() ] CONF::		 checkpointdirectory -> .
[FuseNet++.py:104 -          initializer() ] CONF::		 mode -> train
[FuseNet++.py:104 -          initializer() ] CONF::		 outputdirectory -> /dev/shm/MultiNet/MultiNet/../myoutput
[FuseNet++.py:104 -          initializer() ] CONF::		 logdir -> /dev/shm/MultiNet/MultiNet/../myoutput
...
Epoch 1/100
  16/1939 [..............................] - ETA: 34:23 - loss: 19.4238
```

This will train the network and save output in `myoutput`.
Examples of outputs are presented in [Figures](MultiNet/Figures) 
<!--![images/reconst](MultiNet/Images/test_generated_image-19.png?=100x100)-->
#### Evaluation
For evaluation, put all your test measurments in a folder and set it path as an argument.Examples are available under [data_samples](MultiNet/data). Then run the following command:

<a name="Evaluation"></a>
```bash
mkdir myoutput
cd MultiNet
python3 MultiNet.py  --input testdatadir --outputfolder ../myoutput  --mode test
```

The results will be saved output in `myoutput`. 
### Questions?
<a name="faq"></a>
Please create a [new issue](https://github.com/haneneby/MultiNet/issues/new/choose) detailing concisely, yet complete what issue you encountered, in a reproducible way.
