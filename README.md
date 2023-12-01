# hubness-reduction-improves-sbert-semantic-spaces
Code for the article: [Hubness Reduction Improves Sentence-BERT Semantic Spaces](https://arxiv.org/abs/2311.18364) by Betrix M. G. Nielsen and Lars Kai Hansen. Article accepted for publication at NLDL 2024.

Trained models, embeddings and result files can be found at: [DTU Data](https://doi.org/10.11583/DTU.c.6165561.v1)

## Description
We explore the high-dimensional problem of hubness in sentence-BERT embeddings. Hubness results in asymmetric neighborhood relations, such that some texts (the hubs) are neighbours of many other texts while most texts (so-called anti-hubs), are neighbours of few or no other texts.
This code includes implementation of training, embedding generation and performance evaluation of sentence-BERT models with and without hubness reduction. Evaluation is with respect to hubness measures and K-Nearest Neighbours classification. 
Code for generating the figures in the article is in src/make_figures/article.py. Note that result files are needed to generate the figures.  

## Installation guide
The outline of the installation is the following:

**1. Create and activate conda environment**

**2. Install scikit hubness package**

**1. Create and activate conda environment**

If you are installing on a linux machine with GPU, use the linux_gpu.yml file provided and the commands:
```
conda create -f linux_gpu.yml
conda activate s_bert_hub
```
If you are on a windows machine, use the commands: 
```
conda create -f windows.yml
conda activate s_bert_hub
```
and either
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
for gpu, or 
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
for cpu. 

**2. Install scikit hubness package** 

Run the command:
```
python -m pip install git+https://github.com/VarIr/scikit-hubness.git
```

## Datasets

Due to the size, the Yahoo Answers dataset is not included in this repo. It can be downloaded from https://www.kaggle.com/datasets/yacharki/yahoo-answers-10-categories-for-nlp-csv. 
See more details on datasets in the readme and source files found in their folders. 


## Usage

To see all options, use:
```
python run.py -h
```
To train models use:
```
python run.py train --output-folder=<file> --training-parameters=<file> [options]
```
training_parameters.json contains all the training parameters used in the article.
For example, to train sentence-BERT models with settings from training_parameters_small_example.json on the stsbenchmark dataset using the gpu and saving the output in the "output" folder, run the command
```
python run.py train --output-folder=output --training-parameters=training_parameters_small_example.json --cuda
```
To get embeddings from a single trained model, use:
```
python run.py embeddings --output-folder=<file> --model-name=<string> --dataset=<string> --embeddings-save-to=<file> [options]
```
For example, to get embeddings for the train split of the 20 Newsgroup dataset with your model "sts_bert_microsoft-mpnet-base_cos_ORTHOGONAL_z_False_n_False_c_False_seed0" from the folder "output/models" and save the embeddings in "emb_test.h5" use
```
python run.py embeddings --output-folder=output --model-name=sts_bert_microsoft-mpnet-base_cos_ORTHOGONAL_z_False_n_False_c_False_seed0 --dataset=newsgroups.train --embeddings-save-to=emb_test.h5 --cuda
```
To get embeddings from all models in a folder, use:
```
python run.py embeddings --output-folder=<file> --model-name=<string> --dataset=<string> --model-names-path=<file> [options]
```
For example, to get embeddings for the train split of the 20 Newsgroup dataset with all models with the prefix "sts_bert" from the folder "output/models" and save the embeddings in "output/embeddings" use
```
python run.py embeddings --output-folder=output --model-name=sts_bert --dataset=newsgroups.train --model-names-path=models --cuda
```
To get the performance of the embeddings use:
```
python run.py performance --output-folder=<file> --dataset=<string> --model-names-path=<file> [options]
```
For example, to get the knn performance of the embeddings from the "output/embeddings" folder on the test split of 20 Newsgroups with both no hubness reduction and f-norm use:
```
python run.py performance --output-folder=output --dataset=newsgroups.test --model-names-path=embeddings --force-dist=none,normal_all --result-type=knn
```
To get the hubness performance of the embeddings from the "output/embeddings" folder on the test split of 20 Newsgroups with both no hubness reduction and f-norm use:
```
python run.py performance --output-folder=output --dataset=newsgroups.test --model-names-path=embeddings --force-dist=none,normal_all --result-type=hubness
```

## Authors and acknowledgment
Authors of article: Beatrix M. G. Nielsen and Lars Kai Hansen

This work was supported by the Danish Pioneer Centre for AI, DNRF grant number P1. 

## License
Apache License 2.0. See LICENSE.

