# Rhetorical-Roles
Semantic Segmentation of Legal Documents via Rhetorical Roles

The repository contains the full codebase of experiments and results of the paper "Semantic Segmentation of Legal Documents via Rhetorical Roles". 

You can get RR dataset in the Dataset folder.

Our contributions can be summarized as below:
* We create a new corpus of legal documents annotated with rhetorical role labels. To the best of our knowledge, this is the largest RR corpus. We release the corpus and model implementations and experiments code.
* We propose new multi-task learning (MTL) based deep learning model with document level rhetorical role shift as an auxiliary task for segmenting the document into rhetorical role units. We experiment with various text classification models and show that the proposed model performs better than the existing models. We further show that our method is robust against domain transfer.
* Given that annotating legal documents with RR is a tedious process, we perform model distillation experiments with the proposed MTL model and attempt to leverage unlabeled data to enhance the performance.
* To the best of our knowledge, we are first to show an application of RR for the task of judgment prediction.

## License

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

The RR corpus and software follows [CC-BY-NC](CC-BY-NC) license. Thus, users can share and adapt our dataset if they give credit to us and do not use our dataset for any commercial purposes.
