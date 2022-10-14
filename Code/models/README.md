## Details about models

The main model which was proposed in the paper is the **mtl-model.ipynb** file. 

### Training the model :

 - The first step is to use the **generate-bert-embeddings.ipynb** file in the feature_generation folder to get BERT embeddings for your data. 
**Input format** : Please refer to the IT and CL json files to get a better idea.  Please refer to the Data directory.
**Output format** : Each document with seperate text file with one sentence on each line with format : ``emb1 <SPACE> emb2....<SPACE> emb768 <TAB> label``
- Then you need to use to **BERTSC.ipynb** to get the shift embeddings. Suppose there are n sentences in a document then this returns **n-1** 768 dimension embeddings. The embeddings correspond to the consecutive pairs of sentences so the **i<sup>th</sup>** embedding is for sentence pair s<sub>i</sub> and s<sub>i+1</sub>. This file before generating the embeddings trains a model on the data so if you are using our data then we have a pretrained model [here](https://drive.google.com/drive/folders/1GetfwQwHkZIc1aAwH_m5t7bruW0cvpt5?usp=sharing). 
- Now we need to use these embeddings to train the model using the **mtl-model.ipynb** file. The expected input of this though is a folder with 4 sub-folders, train, test, train_binary and test_binary. The train and test are the same as the output of the first step. For the train_binary and test_binary we need to concat the embeddings from first and second step for each sentence in document and also update the labels to binary labels i.e 0 and 1. So suppose for a sentence **s<sub>i</sub>** in a document the new embeddings will be the  ``shift emb of si-1 and si <concat> si emb from first step <concat> shift emb of si and si+1`` which is of 2304 dimension. A null vector can be used for first and last pair. The updated label for a sentence **s<sub>i</sub>** will be 0 if **s<sub>i</sub> label** and **s<sub>i+1</sub> label** are same otherwise 1. For the first sentence of each document 0 can be used. The format of the documents in the train_binary and test_binary will be same as train and test folders. 

### Inference :

We are also providing a pre-trained model which can be directly used for inference purposes without training the model. The model can be found [here](https://drive.google.com/drive/folders/1vVS-lk8yAt_DonhaSVYYfAQ1ci1J8uxX?usp=sharing). Also, for inference those who want to infer on a custom data, the process of embedding generation is same only the labels can be ignored in that case.

