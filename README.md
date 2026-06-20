# Supplementary material

This repository contains the source code and trained models for the paper 'Automating Color Image Steganography Research from First Principles', published at ESORICS 2026.

## Setup

```bash
# setup the virtual environment
python3.12 -m venv .venv
source .venv/bin/activate
# install dependencies
pip install .
```

## Structure

This archive is structured as follows:

| Folder               | Contents                                                                     |
|----------------------|------------------------------------------------------------------------------|
| data                 | Training-test split.                                                         |
| detector_predictions | Logits produced by our trained detectors.                                    |
| example_images       | Example images from all methods in the study.                                |
| models               | Weights, logs, and metadata from the training of the proposed model.         |
| source_code          | Source code for generating the dataset, training, embedding, and evaluation. |


The source code lets you reproduce the following steps:


| Script                               | Behavior                                                                         |
|--------------------------------------|----------------------------------------------------------------------------------|
| detector_predictions/calculate_pe.py | Recalculates the PE from all the detector logits in the folder.                  |
| source_code/measure_speed.py         | Measures the embedding speed of HILL, WOW and the proposed method.               |
| source_code/embed_proposed.py        | Simulates steganographic embedding into a cover image using the proposed method. |
| source_code/prepare_dataset.py       | Downloads and prepares the ALASKA dataset.                                       |
| source_code/train_gan.py             | Performs the adversarial training. (computationally expensive)                   |
| source_code/visualize_probability.py | Performs the adversarial training. (computationally expensive)                   |


## Dataset

We carried out our experiments on the ALASKA dataset, which is available at https://alaska.utt.fr/.
The ALASKA dataset is released under the Creative Commons BY-NC-ND licence.
We have the explicit permission from the dataset's authors to publish the example images.


## Baseline method

For the end-to-end methods, we use the following implementations:

| Method         | Source                                                                                                                      | License     | Note                                          |
|----------------|-----------------------------------------------------------------------------------------------------------------------------|-------------|-----------------------------------------------|
| HILL, WOW      | https://github.com/UIBK-uncover/conseal                                                                                     | MPL 2.0     |                                               |
| Gina           |                                                                                                                             |             | kindly provided by Weixiang Li                |
| CPV            |                                                                                                                             |             | kindly provided by Bin Li                     |
| PTS            | https://github.com/Sanakkk3/Color-Image-Steganography-Using-Generative-Adversarial-Networks-with-a-Phased-Training-Strategy |             | kindly provided by Saixing Zhou and Weiqi Luo |

We provide example images and the prediction vectors over the test set for these methods.


## Publishing

This material is released under a MPL licence.
