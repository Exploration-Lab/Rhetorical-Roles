# RR: Rhetorical Roles

This folder contains the official code pipeline for generating the Rhetorical Roles.

# Requirements

1. Libraries:

   1.1 Install torch: `pip install torch`

   1.2 Install transformers: `pip install transformers`

   1.3 Install keras: `pip install keras`

   1.4 Install keras.preprocessing: `pip install Keras-Preprocessing`

   1.5 Install nltk: `pip install nltk`

2. Folder Structure

   Root directory consists of the following:

   2.1 `Input_Data` folder

   2.2 `Output_Data` folder

   2.3 `SiameseBERT_7labels_full` (This folder contains the
   BERT model.)

   2.4 `all_models` folder

   2.3 The python script named `rr_pipeline.py`

   2.4 This ReadMe file.

The following folders will be created inside the Output_Data folder by the code.

1. Emb_Output_Data

   1.1 gen_emb

   1.2 bertsc_emb

2. File_RR

3. RR_Output_Data

Note:

1. Emb_Output_Data folder can be deleted after the processing as the data stored in this folder consumes space.
2. Final RR output for each file will be store in the File_RR folder.
3. RR_Ouput_Data contains the combined json for all the files.

# Models and Data
1. The models for `SiameseBERT_7labels_full` and `all_models` can be downloaded from here. Please download these folders and replace the corresponding folders in the root directory with these folders.

2. The data folder present in the same link contains the data for training the above models. But the data is not required to run this pipeline as models in the above link are already trained.
# File structures inside

# 1. Input_Data

INPUT_DATA folder contains ".sav" files which contain the sentences for all the files.

    Example: ilpcr_data.sav

The internal structure for this file is as follows:

    {
        'candidate_data':{
            'file_id1':[sentence1, sentence2, sentence3, ...],
            'file_id2':[sentence1, sentence2, sentence3, ...],
            ...
        }
    }

# 2. Output_Data/File_RR

A folder with the name of the sav file in the Input_Data folder will be created inside the Output_Data/File_RR folder. This folder will contain the RR output for each file in the form of json.

    Example: ilpcr_data

Total files in this ilpcr_data folder will be equal to the number of file_ids in the ilpcr_data.sav file.

Example: file_id1.json, file_id2.json, ...

The internal structure of these json files is as follows:
For file_id1.json:

    {
        'file_id1':[
            [
                "sentence1","rr_label1"
            ],
            [
                "sentence2","rr_label2"
            ],
            ...
        ]
    }

# 3. Output_Data/RR_Output_Data

This folder contains the combined json for all the files in the form of a single json file.

    Example: ilpcr_dataset_rr_results.json

The internal structure of this json file is as follows:

    {
        'file_id1':[
            [
                "sentence1","rr_label1"
            ],
            [
                "sentence2","rr_label2"
            ],
            ...
        ],
        'file_id2':[
            [
                "sentence1","rr_label1"
            ],
            [
                "sentence2","rr_label2"
            ],
            ...
        ],
        ...
    }
