<div  align="center">
  
# LinkSeg: Using Pairwise Link Prediction and Graph Attention Networks for Music Structure Analysis
  
[Morgan Buisson](https://morgan76.github.io/)<sup>1</sup>,
[Brian McFee](https://brianmcfee.net/)<sup>2,3</sup>,
[Slim Essid](https://slimessid.github.io/research/)<sup>1</sup> <br>
<sup>1</sup>  LTCI, Télécom Paris, Institut Polytechnique de Paris, France <br> <sup>2</sup>  Music and Audio Research Laboratory, New York University, USA <br> <sup>3</sup> Center for Data Science, New York University, USA
  
<p align="center">
<img src="linkseg.png" width="500">
</p align="center">
</div>

This repository contains the official [PyTorch](http://pytorch.org/) implementation of the paper [Using Pairwise Link Prediction and Graph Attention Networks for Music Structure Analysis](https://hal.science/hal-04665063/) 
presented at ISMIR 2024. 

We introduce **LinkSeg**, a novel approach to music structure analysis based on pairwise link prediction. This method consists in predicting whether any pair of time instants within a track belongs to the same structural element (segment or section). This problem comes down to classifying each individual component (i.e. link) of the track's self-similarity matrix into one of the three categories: "same-segment", "same-section" or "different section". The link features calculated from this task are then combined with frame-wise features through a graph attention module to predict segment boundaries and musical section labels. 

This project is focused on the segmentation of popular music genres, therefore, predicted section labels follow a 7-class taxonomy containing: "Intro", "Verse", "Chorus", "Bridge", "Instrumental", "Outro" and "Silence". 

This repository provides code for training the system from scratch along with some pre-trained checkpoints for predicting the structure of new tracks. 

## Table of Contents
0. [Requirements](#requirements)
0. [Dataset](#dataset)
0. [Training](#training)
0. [Inference](#inference)
0. [Citing](#citing)
0. [Contact](#contact)

## Requirements
Install FFmpeg for ubuntu:

```shell
sudo apt install ffmpeg
```

For macOS:

```shell
brew install ffmpeg
```

Create new environment and install dependencies:
```
conda create -n YOUR_ENV_NAME python=3.9
conda activate YOUR_ENV_NAME
pip install git+https://github.com/CPJKU/madmom # install madmom from github
pip install -r requirements.txt
cd src/
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
python preprocess_data.py --data_path <dataset_path>
```
This will handle the creation of [`dataset/audio_npy/`](dataset/audio_npy/) (beat estimation) and [`dataset/features/`](dataset/features/) files (conversion to numpy files). 

## Training
To train a new LinkSeg model, run:
```
python train.py --data_path <dataset_path>
```

The default label taxonomy contains 7 section labels: "Intro", "Verse", "Chorus", "Bridge", "Instrumental", "Outro" and "Silence". A second taxonomy containing "Pre-chorus" and "Post-chorus" labels can be used:
```
python train.py --data_path <dataset_path> --nb_section_labels 9
``` 

## Inference
To make predictions using a trained model, first make sure that the test dataset is processed: 
```
python preprocess_data.py --data_path <test_dataset_path>
```

Then run:
```
python predict.py --test_data_path <test_dataset_path> --model_name <path_to_model>
```

To use the 7-class pre-trained model (on a 75% split of Harmonix), run:
```
python predict.py --test_data_path <dataset_path> --model_name ../data/model_7_classes.pt
```

To use the 9-class pre-trained model, run:
```
python predict.py --test_data_path <dataset_path> --model_name ../data/model_9_classes.pt
```

By default, segmentation predictions will be saved in [JAMS](https://jams.readthedocs.io/en/stable/quickstart.html) file format under the [`dataset/predictions/`](dataset/predictions/) directory. 

Keep in mind that boundary predictions are calculated from the features of two consecutive time frames $x\prime \prime_{i}$, $x\prime \prime_{i+1}$ and the features $e\prime_{i,i+1}$ of the link connecting them. Therefore, boundary predictions fall **in-between** consecutive estimated beat locations. 


## Segmentation Example

<div  align="center">
<p align="center">
<img src="segmentation_example_midnight_city.png" width="1000">
</p align="center">
</div>

Segmentation results for the track ***M83 - Midnight City*** from the Harmonix dataset. The top two rows display the class activation and boundary curves over time. The bottom rows show the final predictions and the reference annotations. Black and red dotted lines indicate predicted and annotated segment boundaries.

<div  align="center">
<p align="center">
<img src="ssm_example_midnight_city.png" width="800">
</p align="center">
</div>

The self-similarity matrix of the track ***M83 - Midnight City*** obtained from the output embeddings of the graph attention module. Red dashed lines indicate annotated (left) and predicted (right) segment boundaries.

## Citing
```bib
@inproceedings{buisson:hal-04665063,
  title={Using Pairwise Link Prediction and Graph Attention Networks for Music Structure Analysis},
  author={Buisson, Morgan and McFee, Brian and Essid, Slim},
  booktitle={Proceedings of the 25th International Society for Music Information Retrieval Conference (ISMIR)},
  year={2024}
}
```

## Contact
morgan.buisson@telecom-paris.fr
