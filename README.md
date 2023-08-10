# GazeNeRF: 3D-Aware Gaze Redirection with Neural Radiance Fields

This repository is the official PyTorch implementation for CVPR 2023 paper\
[“GazeNeRF: 3D-Aware Gaze Redirection with Neural Radiance Fields”](https://arxiv.org/abs/2212.04823)

- Authors: [Alessandro Ruzzi*](https://alessandroruzzi.github.io), [Xiangwei Shi*](), [Xi Wang](https://ait.ethz.ch/people/xiwang), [Gengyan Li](https://ait.ethz.ch/people/lig), [Shalini De Mello](https://research.nvidia.com/person/shalini-de-mello), [Hyung Jin Chang](https://www.birmingham.ac.uk/staff/profiles/computer-science/academic-staff/chang-jin-hyung.aspx), [Xucong Zhang](https://cvlab-tudelft.github.io/people/xucong-zhang/), [Otmar Hilliges](https://ait.ethz.ch/people/hilliges)
- [Project Page](https://x-shi.github.io/GazeNeRF.github.io/)

https://github.com/AlessandroRuzzi/GazeNeRF/assets/76208014/7607b8fc-2aa5-45fc-8f0f-542d9dab9597

## Requirements
The models in our paper are trained with Python 3.8.8, PyTorch 1.12.0, CUDA 11.3, and CentOS 7.9.2009.

To install the required packages, run:\
`pip install -r requirements.txt`

To install the required libraries for data preprocessing, please refer to [this repository](https://github.com/CrisHY1995/headnerf/tree/main).

## Dataset
Our models are trained with ETH-XGaze dataset, and evaluated with ETH-XGaze, Columbia, MPIIFaceGaze, and GazeCapture datasets. The preprocessing code is mainly based on the repository of [data normalization](https://github.com/xucong-zhang/data-preprocessing-gaze), the repository of [HeadNeRF](https://github.com/CrisHY1995/headnerf/tree/main) and [this repository](https://github.com/switchablenorms/CelebAMask-HQ).

To preprocess the datasets, run:\
`python dataset_pre_precessing.py --dataset_dir=/path/to/your/dataset --dataset_name=eth_xgaze --output_dir=/path/to/your/output/directory`

- We do not provide the dataset download links.

## Training
To train the GazeNeRF model, run\
`python train.py --batch_size=2 --log=true --learning_rate=0.0001 --img_dir='/path/to/your/ETH-XGaze/training/dataset'`

Our model was trained on a single NVIDIA A40 GPU.

## Evaluation
To evaluate the trained model, run\
`python evaluate_metrics.py --log=true --num_epochs=75 --model_path=checkpoints/your_checkpoints.json`

To generate the interpolation demos, run\
`python evaluate.py --model_path=checkpoints/your_checkpoints.json --img_dir='/path/to/your/ETH-XGaze/test/dataset'`

## Pre-trained model
You can download the pre-trained GazeNeRF [here](https://drive.google.com/file/d/100ksmOoWc5kFB0V4eT0RZecI9N1Hr2vu/view?usp=sharing) and the gaze estimator [here](https://drive.google.com/file/d/1YFQjLYx187XyhGj6SGEONmgV3lBieJsn/view?usp=share_link).

## Citation

```
@InProceedings{ruzzi2023gazenerf,
    author    = {Ruzzi, Alessandro and Shi, Xiangwei and Wang, Xi and Li, Gengyan and De Mello, Shalini and Chang, Hyung Jin and Zhang, Xucong and Hilliges, Otmar},
    title     = {GazeNeRF: 3D-Aware Gaze Redirection with Neural Radiance Fields},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2023},
    pages     = {9676--9685}
}
```

