VPVQA - Viewport-Patch Extraction Enhanced 360 Video Quality Assessment

## Installation

We recommend to run the code with virtualenv. The code is developed with Python3.

Please install other prerequisites with the following command after invoking a virtual env.

```
pip install -r requirements.txt
```
All packages are required to run the code.

## Dataset

Please prepare a dataset if you want to evaluate in batch or train the code from scratch on your own GPUs. The dataset should be in json format, e.g. your\_dataset.json

```
{
    "test": {
        "dis": ["dis_1.yuv", "dis_2.yuv"],
        "ref": ["ref_1.yuv", "ref_2.yuv"],
        "fps": [30, 24],
        "mos": [94.2, 55.8],
        "height": [1080, 720],
        "width": [1920, 1280]
    },
    "train": {
        "dis": ["dis_3.yuv", "dis_4.yuv"],
        "ref": ["ref_3.yuv", "ref_4.yuv"],
        "fps": [50, 24],
        "mos": [85.2, 51.8],
        "height": [320, 720],
        "width": [640, 1280]
    }
}
```
For the time being, only YUV is supported. We will update modules to read bitstream.
```

## Train from scratch

Prepare dataset as above and simply run:

```
python train.py --multi_gpu --video_dir /dir/to/yuv --score_file_path /path/to/your_dataset.json --save_model ./save/your_new_trained.pt
```
Please check train.sh and opts.py if you would like to tweak other hyper-parameters.

##
Pre-trained Models

Due to GitHub's file size limitations, the pre-trained model weights (.pt files) are not included in this repository.

You can download the trained models from the following link:

Download Link: https://pan.quark.cn/s/4cad40085253?pwd=i3kD

We will try to answer above questions. Stay tuned.

