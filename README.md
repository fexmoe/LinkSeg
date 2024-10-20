# LinkSeg: Using Pairwise Link Prediction and Graph Attention Networks for Music Structure Analysis
This repository contains a [PyTorch](http://pytorch.org/) implementation of the paper [Using Pairwise Link Prediction and Graph Attention Networks for Music Structure Analysis](https://hal.science/hal-04665063/) 
presented at ISMIR 2024.

## Table of Contents
0. [Requirements](#requirements)
0. [Dataset](#dataset)
0. [Training](#training)
0. [Inference](#inference)
0. [Citing](#citing)
0. [Contact](#contact)

## Requirements
```
conda env create -f environment.yml
```

## Dataset
The dataset structure follows that of the [MSAF](https://ismir2015.ismir.net/LBD/LBD30.pdf) package:
```
dataset/
├── audio                   # audio files (.mp3, .wav, .aiff)
├── audio_npy               # audio files (.npy)
├── features                # feature files (.json)
└── references              # references files (.jams) (if available, necessary for training)
```

To preprocess some dataset, run:
```
python preprocess_data.py --data_path {dataset_path}
```
This will handle the /audio_npy and /features files. 

## Training
To train a new LinkSeg model, run:
```
python train.py --data_path {dataset_path}
```

## Inference
To make predictions using a trained model, run:
```
python predict.py --test_data_path {dataset_path} --model_name {path_to_model}
```

To use the 7-class pre-trained model, run:
```
python predict.py --test_data_path {dataset_path} --model_name ../data/model_7_classes.pt
```

## Citing
```
@inproceedings{buisson2022learning,
  title={Using Pairwise Link Prediction and Graph Attention Networks for Music Structure Analysis},
  author={Buisson, Morgan and McFee, Brian and Essid, Slim},
  booktitle={International Society for Music Information Retrieval (ISMIR)},
  year={2024}
}
```

## Contact
morgan.buisson@telecom-paris.fr
