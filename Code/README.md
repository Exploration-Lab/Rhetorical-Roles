## Code 

Here we provide all the different codes which we have used in this project. Please refer to the `data` folder to get access to the data which we have used here.

**This folder contains 4 sub-folders:**
- Models
- Feature_generation
- Judgement_Prediction
- Utilities

`Models` contains the code for all the different models we used in this paper:
- Single Sentence Classification
- Baseline(BiLSTM-CRF)
- Proposed Models(MTL)

`Feature_generation` contains all the code which we used to get the representations of sentences:
- Pretrained BERT embeddings
- Handcrafted Features
- Shift Embeddings(Using SBERT and BERT-SC)

The code to generate sent2vec features is not provided as it is just a 2-3 line code. Please refer to this [link](https://github.com/epfml/sent2vec). The pretrained sent2vec model on 50,000 Supreme Court cases can be accessed [here](https://iitk-my.sharepoint.com/:u:/g/personal/ashutoshm_iitk_ac_in/EeKvkSBir0FBk9eJdy5pLI8BDDoRRZPDKucTJSYj-LxZEg?e=dTsgHH)

Also in the `Models` folder there is no explicit file for LSP model as it is the BiLSTM-CRF model with embeddings as BERT+Shift combined. The details can be found in the paper. 

`Judgement_Prediction` folder contains the code which was used to generate the data for judgement prediction using RPC, ROD roles. The generated data can then be used with a pre-trained BERT [model](https://drive.google.com/drive/folders/17nddWo9e4Z-rljF83jIq1aEb3w71DouZ?usp=sharing) fine-tuned on the judgement prediction task. The code for this can be accessed [here](https://github.com/Exploration-Lab/CJPE/blob/main/Models/transformers/trained_on_multi/BERT_training_notebook.ipynb).

The `Utilities` folder contains all other codes which were used in this project, like calculating `flies_kappa` and generating the plots.

