# Rhetorical-Roles
Semantic Segmentation of Legal Documents via Rhetorical Roles

The repository contains the full codebase of experiments and results of the paper "Semantic Segmentation of Legal Documents via Rhetorical Roles". 

You can get RR dataset in the Data folder.

Our contributions can be summarized as below:
* We create a new corpus of legal documents annotated with rhetorical role labels. To the best of our knowledge, this is the largest RR corpus. We release the corpus and model implementations and experiments code.
* We propose new multi-task learning (MTL) based deep learning model with document level rhetorical role shift as an auxiliary task for segmenting the document into rhetorical role units. We experiment with various text classification models and show that the proposed model performs better than the existing models. We further show that our method is robust against domain transfer.
* Given that annotating legal documents with RR is a tedious process, we perform model distillation experiments with the proposed MTL model and attempt to leverage unlabeled data to enhance the performance.


## License

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

The RR corpus and software follows [CC-BY-NC](CC-BY-NC) license. Thus, users can share and adapt our dataset if they give credit to us and do not use our dataset for any commercial purposes.

## Citation

```
@inproceedings{malik-etal-2022-semantic-segmentation-rr,
    title = "Semantic Segmentation of Legal Documents via Rhetorical Roles",
    author = "Malik, Vijit and 
              Sanjay, Rishabh and 
              Guha, Shouvik and 
              Hazarika, Angshuman and 
              Nigam, Shubham Kumar and
              Bhattacharya, Arnab and 
              Modi, Ashutosh",
    booktitle = "Proceedings of the Natural Legal Language Processing Workshop 2022",
    month = December,
    year = "2022",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    abstract = "Legal documents are unstructured, use legal jargon, and have considerable length, making them difficult to process automatically via conventional text processing techniques. A legal document processing system would benefit substantially if the documents could be segmented into coherent information units. This resource paper proposes a new corpus of legal documents annotated (with the help of legal experts) with a set of 13 semantically coherent units labels (referred to as Rhetorical Roles), e.g., facts, arguments, statute, issue, precedent, ruling, and ratio. We perform a thorough analysis of the corpus and the annotations. For automatically segmenting the legal documents, we experiment with the task of rhetorical role prediction: given a document, predict the text segments corresponding to various roles. Using the created corpus, we experiment extensively with various deep learning-based baseline models for the task. Further, we develop a multitask learning (MTL) based deep model with document rhetorical role label shift as an auxiliary
task for segmenting a legal document. The proposed model shows superior performance over the existing models. We also experiment with model performance in the case of domain transfer and model distillation techniques to see the model performance in limited data conditions.",
}
```

## Contact

In case of any queries, please contact <ashutoshm.iitk@gmail.com> and <vijitvm21@gmail.com> and <rishabh.lfs@gmail.com>.
