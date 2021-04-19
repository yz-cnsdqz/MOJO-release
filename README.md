# We are **MO**re than our **JO**ints: Predicting How 3D Bodies Move


This repo contains the official implementation of our paper:
```
@inproceedings{Zhang:CVPR:2021,
  title = {We are More than Our Joints: Predicting how {3D} Bodies Move},
  author = {Zhang, Yan and Black, Michael J. and Tang, Siyu},
  booktitle = {Proceedings IEEE/CVF Conf.~on Computer Vision and Pattern Recognition (CVPR)},
  month = jun,
  year = {2021},
  month_numeric = {6}
}
```

## License 

todo

## Environment
* **Tested OS:** Linux Ubuntu 18.04
* **Packages:**
    * Python >= 3.6
    * [PyTorch](https://pytorch.org) >= 1.2
    * Tensorboard
* **Note**: All scripts should be run from the root of this repo to avoid path issues. 
Also, please fix some path configs in the code, otherwise errors will occur.


## Datasets
In MOJO, we have used [AMASS](https://amass.is.tue.mpg.de/), [Human3.6M](http://vision.imar.ro/human3.6m/description.php), and [HumanEva](http://humaneva.is.tue.mpg.de/).

For **Human3.6M** and **HumanEva**, we follow the same pre-processing step as in [DLow](), [VideoPose3D](https://github.com/facebookresearch/VideoPose3D), and others. Please
refer to their pages, e.g. [this one](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md), for details.

For **AMASS**, we perform canonicalization of motion sequences with our own procedures. The details are in `experiments/utils_canonicalize_amass.py`.
We find this sequence canonicalization procedure is important.



## Models
For human body modeling, we employ the [SMPL-X](https://smpl-x.is.tue.mpg.de/) parametric body model. You need to follow their license and download.
Based on SMPL-X, we can use the body joints or a sparse set of body mesh vertices (the body markers) to represent the body. 
* **CMU** It has 41 markers, the corresponding SMPL-X mesh vertex ID can be downloaded [here]().
* **SSM2** It has 64 markers, the corresponding SMPL-X mesh vertex ID can be downloaded [here]().
* **Joints** We used 22 joints. No need to download, but just obtain them from the SMPL-X body model. See details in the code.


Our CVAE model configurations are in `experiments/cfg`. The pre-trained checkpoints can be downloaded [here]()




























_______





This repo contains the official implementation of our paper:

DLow: Diversifying Latent Flows for Diverse Human Motion Prediction
Ye Yuan, Kris Kitani
**ECCV 2020**
[[website](https://www.ye-yuan.com/dlow)] [[paper](https://arxiv.org/pdf/2003.08386.pdf)] [[talk](https://youtu.be/c45ss6Tcb2A)] [[summary](https://youtu.be/nVYGHnRB1_M)] [[demo](https://youtu.be/64OEdSadb00)]

# Installation
### Datasets
* Please follow the data preprocessing steps ([DATASETS.md](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md)) inside the [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) repo. Place the prepocessed data ``data_3d_h36m.npz`` (Human3.6M) and ``data_3d_humaneva15.npz`` (HumanEva-I) under the ``data`` folder.
### Environment
* **Tested OS:** MacOS, Linux
* **Packages:**
    * Python >= 3.6
    * [PyTorch](https://pytorch.org) >= 0.4
    * Tensorboard
* **Note**: All scripts should be run from the root of this repo to avoid path issues.

### Pretrained Models
* Download our pretrained models from [Google Drive](https://drive.google.com/file/d/1k5uDeUXrvtwZPN-lJNPSO8tPvHH6Gj55/view?usp=sharing) (or [BaiduYun](https://pan.baidu.com/s/1Ye6bHXcX6lNVMLaXJyzyWg), password: y9ph) and place the unzipped ``results`` folder inside the root of this repo.

# Train
### Configs
We have provided 4 example YAML configs inside ``experiments/cfg``:
* `h36m_nsamp10.yml` and `h36m_nsamp50.yml` for Human3.6M for number of samples 10 and 50 respectively.
* `humaneva_nsamp10.yml` and `humaneva_nsamp50.yml` for HumanEva-I for number of samples 10 and 50 respectively.
* These configs also have corresponding pretrained models inside ``results``.

### Train VAE
```
python experiments/exp_vae.py --cfg h36m_nsamp10
```

### Train DLow (After VAE is trained)
```
python experiments/exp_dlow.py --cfg h36m_nsamp10
```

# Test
### Visualize Motion Samples
```
python experiments/eval.py --cfg h36m_nsamp10 --mode vis
```
Useful keyboard shortcuts for the visualization GUI:
| Key           | Functionality |
| ------------- | ------------- |
| d             | test next motion data
| c             | save current animation as `out/video.mp4` |
| space         | stop/resume animation |
| 1             | show DLow motion samples |
| 2             | show VAE motion samples |


### Compute Metrics
```
python experiments/eval.py --cfg h36m_nsamp50 --mode stats
```


# Citation
If you find our work useful in your research, please consider citing our paper [DLow](https://www.ye-yuan.com/dlow):
```
@inproceedings{yuan2020dlow,
  title={Dlow: Diversifying latent flows for diverse human motion prediction},
  author={Yuan, Ye and Kitani, Kris},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

# Acknowledgement
Part of the code is borrowed from the [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) repo.

# License

The software in this repo is freely available for free non-commercial use. Please see the [license](LICENSE) for further details.
