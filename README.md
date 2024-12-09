# Source Separation


## About

This repository contains an implementation of two DL models for
Audio-Visual Source Separation: TASnet and RTFS-Net.


## Installation


0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

## How To Use

### Data preparation:

1. First of all you need to prepare your dataset with this structure:

```bash
NameOfTheDirectoryWithUtterances
├── audio
│   ├── mix
│   │   ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│   │   ├── FirstSpeakerID2_SecondSpeakerID2.wav
│   │   .
│   │   .
│   │   .
│   │   └── FirstSpeakerIDn_SecondSpeakerIDn.wav
│   ├── s1 # ground truth for the speaker s1, may not be given
│   │   ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│   │   ├── FirstSpeakerID2_SecondSpeakerID2.wav
│   │   .
│   │   .
│   │   .
│   │   └── FirstSpeakerIDn_SecondSpeakerIDn.wav
│   └── s2 # ground truth for the speaker s2, may not be given
│       ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│       ├── FirstSpeakerID2_SecondSpeakerID2.wav
│       .
│       .
│       .
│       └── FirstSpeakerIDn_SecondSpeakerIDn.wav
└── mouths # contains video information for all speakers
    ├── FirstOrSecondSpeakerID1.npz # npz mouth-crop
    ├── FirstOrSecondSpeakerID2.npz
    .
    .
    .
    └── FirstOrSecondSpeakerIDn.npz
```

2. Generation video embedding

### Embeddings

We used open-source project for generation video embeddings: [repo](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/tree/master?tab=readme-ov-file).
But original repo contained some problems so we forked repo and fixed them: [forked repo](https://github.com/dikirillov/Lipreading_using_Temporal_Convolutional_Networks/).

### Guide for embedding extraction:

Preparation:
```
git clone https://github.com/dikirillov/Lipreading_using_Temporal_Convolutional_Networks/
pip install -r Lipreading_using_Temporal_Convolutional_Networks/requirements.txt
```

Exctraction:

Embedding extraction: [model url ](https://drive.google.com/file/d/1TGFG0dW5M3rBErgU8i0N7M1ys9YMIvgm/view).

```Shell
python3 Lipreading_using_Temporal_Convolutional_Networks/main.py --modality video \
        --extract-feats \
        --config-path 'Lipreading_using_Temporal_Convolutional_Networks/configs/lrw_resnet18_dctcn_boundary.json' \
        --model-path <PATH-TO-DOWNLOADED-MODEL> \
        --mouth-patch-path <MOUTH-PATCH-PATH>
```

### Training

If you want to retrain model you can use train script with your config:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

For example if you want to retrain RTFS model with your dataset you should
change src/configs/rtfs.yaml(set correct dataset dir) and then run command:

```bash
python3 train.py dataset
```

### Inference


You can use inference.py to generate separated audio from your dataset:

```bash
python3 inference.py
```
Before that you should configure run in 'configs/inference.yaml':
0) You should set path to your dataset in config 'configs/datasets/rtfs_eval.yaml'.
1) You should set path to checpoint file in from_pretrained field.
2) You may change save_path.
