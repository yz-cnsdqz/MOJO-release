# We are **MO**re than our **JO**ints: Predicting How 3D Bodies Move


## Citation
This repo contains the official implementation of our paper MOJO:
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
We employ [__CC BY-NC-SA 4.0__](LICENSE) for the MOJO code, which covers
```
models/fittingop.py
experiments/utils/batch_gen_amass.py
experiments/utils/utils_canonicalize_amass.py
experiments/utils/utils_fitting_jts2mesh.py
experiments/utils/vislib.py
experiments/vis_*_amass.py
```

The rest part are developed based on [DLow](https://github.com/Khrylx/DLow). According to their license, the implementation
follows its [CMU license](https://github.com/Khrylx/DLow/blob/master/LICENSE).



## Environment & code structure
* **Tested OS:** Linux Ubuntu 18.04
* **Packages:**
    * Python >= 3.6
    * [PyTorch](https://pytorch.org) >= 1.2
    * Tensorboard
    * [smplx](https://pypi.org/project/smplx/)
    * [vposer](https://github.com/nghorbani/human_body_prior)
    * others
* **Note**: All scripts should be run from the root of this repo to avoid path issues.
Also, please fix some path configs in the code, otherwise errors will occur.

### Training
The training is split to two steps. Provided we have a config file in `experiments/cfg/amass_mojo_f9_nsamp50.yml`, we can do
* `python experiments/train_MOJO_vae.py --cfg amass_mojo_f9_nsamp50` to train the MOJO
* `python experiments/train_MOJO_dlow.py --cfg amass_mojo_f9_nsamp50` to train the DLow


### Evaluation
These `experiments/eval_*.py` files are for evaluation.
For `eval_*_pred.py`, they can be used either to evaluate the results while predicting, or to save results to a file for further evaluation and visualization. An example is `python experiments/eval_kps_pred.py --cfg amass_mojo_f9_nsamp50 --mode vis`, which is to save files to the folder `results/amass_mojo_f9_nsamp50`.


### Generation
In MOJO, the recursive projection scheme is to get 3D bodies from markers and keep the body valid. The relevant implementation is mainly in `models/fittingop.py` and `experiments/test_recursive_proj.py`. An example to run is

```
python experiments/test_recursive_proj.py --cfg amass_mojo_f9_nsamp50 --testdata ACCAD --gpu_index 0
```



## Datasets
In MOJO, we have used [AMASS](https://amass.is.tue.mpg.de/), [Human3.6M](http://vision.imar.ro/human3.6m/description.php), and [HumanEva](http://humaneva.is.tue.mpg.de/).

For **Human3.6M** and **HumanEva**, we follow the same pre-processing step as in [DLow](https://github.com/Khrylx/DLow), [VideoPose3D](https://github.com/facebookresearch/VideoPose3D), and others. Please
refer to their pages, e.g. [this one](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md), for details.

For **AMASS**, we perform canonicalization of motion sequences with our own procedures. The details are in `experiments/utils_canonicalize_amass.py`.
We find this sequence canonicalization procedure is important. The canonicalized AMASS used in our work can be downloaded [here](https://drive.google.com/file/d/14WTJRZvvmmVs9AlPtSGYMf1VI5haaj9q/view?usp=sharing), which includes the
random sample names used in our experiments.



## Models
For human body modeling, we employ the [SMPL-X](https://smpl-x.is.tue.mpg.de/) parametric body model. You need to follow their license and download.
Based on SMPL-X, we can use the body joints or a sparse set of body mesh vertices (the body markers) to represent the body.
* **CMU** It has 41 markers, the corresponding SMPL-X mesh vertex ID can be downloaded [here](https://drive.google.com/file/d/1CcNBZCXA7_Naa0SGlYKCxk_ecnzftbSj/view?usp=sharing).
* **SSM2** It has 64 markers, the corresponding SMPL-X mesh vertex ID can be downloaded [here](https://drive.google.com/file/d/1ozQuVjXoDLiZ3YGV-7RpauJlunPfcx_d/view?usp=sharing).
* **Joints** We used 22 joints. No need to download, but just obtain them from the SMPL-X body model. See details in the code.

Our CVAE model configurations are in `experiments/cfg`. The pre-trained checkpoints can be downloaded [here](https://drive.google.com/drive/folders/1Zger3DVlcBilosYpuMM-Q6tVCCuHDa_h?usp=sharing).






# Acknowledgement & Disclaimer
We thank Nima Ghorbani for the advice on the body marker setting and the {\bf AMASS} dataset.
We thank Yinghao Huang, Cornelia K\"{o}hler, Victoria Fern\'{a}ndez Abrevaya, and Qianli Ma for proofreading.
We thank Xinchen Yan and Ye Yuan for discussions on baseline methods.
We thank Shaofei Wang and Siwei Zhang for their help with the user study and the presentation, respectively.


MJB has received research gift funds from Adobe, Intel, Nvidia, Facebook, and Amazon. While MJB is a part-time employee of Amazon, his research was performed solely at, and funded solely by, Max Planck. MJB has financial interests in Amazon Datagen Technologies, and Meshcapade GmbH.
