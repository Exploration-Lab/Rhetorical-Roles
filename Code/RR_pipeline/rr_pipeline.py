from torch import nn
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from keras_preprocessing.sequence import pad_sequences
import torch
#!pip install transformers
from transformers import BertTokenizer, BertModel
import os
import time
import numpy as np
import pickle as pkl
import pandas as pd
import json
import nltk
from nltk import sent_tokenize
from tqdm import tqdm
nltk.download('punkt')

# bertsc imports

# rr model imports

raw_input_path = "./Input_Data/"
output_path = "./Output_Data/"
emb_output_path = "./Output_Data/Emb_Output_Data/"
rr_output_path = "./Output_Data/RR_Output_Data/"
file_rr_output_path = "./Output_Data/File_RR/"

os.makedirs(emb_output_path, exist_ok=True)
print("Created emb_output_path ---->", emb_output_path)
os.makedirs(rr_output_path, exist_ok=True)
print("Created rr_output_path ---->", rr_output_path)
os.makedirs(file_rr_output_path, exist_ok=True)
print("Created file_rr_output_path ---->", file_rr_output_path)

# Load BERTSC Model for bertsc embeddings

bertsc_model_path = "./SiameseBERT_7labels_full/"
print("Loading bertsc model from path::::: ", bertsc_model_path)
model_bertsc = BertForSequenceClassification.from_pretrained(
    bertsc_model_path, output_hidden_states=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_bertsc.to(device)
tokenizer_bertsc = BertTokenizer.from_pretrained(bertsc_model_path)
print("::::::::::::: BertSC model loading success ::::::::::")
# exit()
'''
    Data Conversion for BERTSC Model
'''


def json_to_df(data):
    sentences_1 = []
    sentences_2 = []
    for doc in data.keys():
        length_sentences = len(data[doc])
        for i, sentence in enumerate(data[doc]):
            if(i == length_sentences-1):
                break
            sentences_1.append(data[doc][i])
            sentences_2.append(data[doc][i+1])
    df = pd.DataFrame(list(zip(sentences_1, sentences_2)),
                      columns=['Sentence 1', 'Sentence 2'])
    return df


'''
    Function to get imput ids for each sentences using the tokenizer
'''


def input_id_maker(dataf, tokenizer):
    input_ids = []
    lengths = []
    token_type_ids = []
    for i in tqdm(range(len(dataf['Sentence 1']))):
        sen1 = dataf['Sentence 1'].iloc[i]
        sen1_t = tokenizer.tokenize(sen1)
        sen2 = dataf['Sentence 2'].iloc[i]
        sen2_t = tokenizer.tokenize(sen2)
        if(len(sen1_t) > 253):
            sen1_t = sen1_t[:253]
        if(len(sen2_t) > 253):
            sen2_t = sen2_t[:253]
        CLS = tokenizer.cls_token
        SEP = tokenizer.sep_token
        sen_full = [CLS] + sen1_t + [SEP] + sen2_t + [SEP]
        tok_type_ids_0 = [0 for i in range(len(sen1_t)+2)]
        tok_type_ids_1 = [1 for i in range(512-len(sen1_t)-2)]
        tok_type_ids = tok_type_ids_0 + tok_type_ids_1
        token_type_ids.append(tok_type_ids)
        encoded_sent = tokenizer.convert_tokens_to_ids(sen_full)
        input_ids.append(encoded_sent)
        lengths.append(len(encoded_sent))
    input_ids = pad_sequences(
        input_ids, maxlen=256, value=0, dtype="long", truncating="pre", padding="post")
    tok_type_ids = []
    return input_ids, lengths, token_type_ids


'''
    This functions returns the attention mask for given input id
'''


def att_masking(input_ids):
    attention_masks = []
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    return attention_masks


'''
    This function returns the [CLS] embedding for a given input_id and attention mask
'''


def get_output_for_one_vec(input_id, att_mask):
    input_ids = torch.tensor(input_id)
    att_masks = torch.tensor(att_mask)
    input_ids = input_ids.unsqueeze(0)
    att_masks = att_masks.unsqueeze(0)
    model_bertsc.eval()
    input_ids = input_ids.to(device)
    att_masks = att_masks.to(device)
    with torch.no_grad():
        output = model_bertsc(input_ids=input_ids,
                              token_type_ids=None, attention_mask=att_masks)
    vec = output["hidden_states"][12][0][0]
    vec = vec.detach().cpu().numpy()
    return vec


'''
    Function to Concatenate Embeddings
'''


def get_s_emb(path_gen, path_bert):
    with open(path_gen, 'rb') as f:
        gen_emb = json.load(f)
    # load npy
    # print("Loading....",path_bert)
    bert_emb = np.load(path_bert)
    gen_emb_np = {}
    for key, val in gen_emb.items():
        val_np = []
        for v in val:
            val_np.append(np.float32(v))
        gen_emb_np[key] = np.array(val_np)
    gen_emb = gen_emb_np
    s_emb = list()
    for i, key in enumerate(list(gen_emb.keys())):
        gei = gen_emb[key]
        if len(bert_emb) < 1:
            bei_1 = [0] * 768
            bei = [0]*768
        else:
            if i == 0:
                # print(i)
                bei_1 = [0] * 768
                bei = bert_emb[0]
            elif i == (len(gen_emb)-1):
                # print(i)
                bei_1 = bert_emb[len(gen_emb)-2]
                bei = [0]*768
            else:
                bei_1 = bert_emb[i-1]
                bei = bert_emb[i]
        si = np.concatenate((bei_1, gei, bei), axis=0)
        s_emb.append(si)
    s_emb = np.array(s_emb)
    return s_emb


# RR model for RR results
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
'''
Shift Module:
    A Bi-LSTM is used to generate feature vectors for each sentence from the sentence embeddings. 
    The feature vectors are actually context-aware sentence embeddings. 
    These are then fed to a feed-forward network to obtain emission scores for each class at each sentence.
'''


class LSTM_Emitter_Binary(nn.Module):
    def __init__(self, n_tags, emb_dim, hidden_dim, drop=0.5, device='cuda'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(emb_dim, hidden_dim // 2,
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(drop)
        self.hidden2tag = nn.Linear(hidden_dim, n_tags)
        self.hidden = None
        self.device = device

    def init_hidden(self, batch_size):
        return (torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device), torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device))

    def forward(self, sequences):
        ## sequences: tensor[batch_size, max_seq_len, emb_dim]
        # initialize hidden state
        self.hidden = self.init_hidden(sequences.shape[0])
        # generate context-aware sentence embeddings (feature vectors)
        # tensor[batch_size, max_seq_len, emb_dim] --> tensor[batch_size, max_seq_len, hidden_dim]
        x, self.hidden = self.lstm(sequences, self.hidden)
        x_new = self.dropout(x)
        # generate emission scores for each class at each sentence
        # tensor[batch_size, max_seq_len, hidden_dim] --> tensor[batch_size, max_seq_len, n_tags]
        x_new = self.hidden2tag(x_new)
        return x_new, x


'''
RR Module:
    A Bi-LSTM is used to generate feature vectors for each sentence from the sentence embeddings. 
    The feature vectors are actually context-aware sentence embeddings. 
    These are then fed to a feed-forward network to obtain emission scores for each class at each sentence.
'''


class LSTM_Emitter(nn.Module):
    def __init__(self, n_tags, emb_dim, hidden_dim, drop=0.5, device='cuda'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(emb_dim, hidden_dim // 2,
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(drop)
        self.hidden2tag = nn.Linear(2*hidden_dim, n_tags)
        self.hidden = None
        self.device = device

    def init_hidden(self, batch_size):
        return (torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device), torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device))

    def forward(self, sequences, hidden_binary):
        ## sequences: tensor[batch_size, max_seq_len, emb_dim]
        # initialize hidden state
        self.hidden = self.init_hidden(sequences.shape[0])
        # generate context-aware sentence embeddings (feature vectors)
        # tensor[batch_size, max_seq_len, emb_dim] --> tensor[batch_size, max_seq_len, hidden_dim]
        x, self.hidden = self.lstm(sequences, self.hidden)
        final = torch.zeros(
            (x.shape[0], x.shape[1], 2*x.shape[2])).to(self.device)
        # Concat the hidden states of both Shift and RR Module LSTM's and then pass through a linear layer to get emission scores for RR Module
        for batch_name, doc in enumerate(x):
            for i, sent in enumerate(doc):
                final[batch_name][i] = torch.cat(
                    (x[batch_name][i], hidden_binary[batch_name][i]), 0)
        final = self.dropout(final)
        # generate emission scores for each class at each sentence
        # tensor[batch_size, max_seq_len, hidden_dim] --> tensor[batch_size, max_seq_len, n_tags]
        final = self.hidden2tag(final)
        return final


'''
    A linear-chain CRF is fed with the emission scores at each sentence, 
    and it finds out the optimal sequence of tags by learning the transition scores.
'''


class CRF(nn.Module):
    def __init__(self, n_tags, sos_tag_idx, eos_tag_idx, pad_tag_idx=None):
        super().__init__()
        self.n_tags = n_tags
        self.SOS_TAG_IDX = sos_tag_idx
        self.EOS_TAG_IDX = eos_tag_idx
        self.PAD_TAG_IDX = pad_tag_idx
        self.transitions = nn.Parameter(torch.empty(self.n_tags, self.n_tags))
        self.init_weights()

    def init_weights(self):
        # initialize transitions from random uniform distribution between -0.1 and 0.1
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        # enforce constraints (rows = from, cols = to) with a big negative number.
        # exp(-1000000) ~ 0
        # no transitions to SOS
        self.transitions.data[:, self.SOS_TAG_IDX] = -1000000.0
        # no transition from EOS
        self.transitions.data[self.EOS_TAG_IDX, :] = -1000000.0
        if self.PAD_TAG_IDX is not None:
            # no transitions from pad except to pad
            self.transitions.data[self.PAD_TAG_IDX, :] = -1000000.0
            self.transitions.data[:, self.PAD_TAG_IDX] = -1000000.0
            # transitions allowed from end and pad to pad
            self.transitions.data[self.PAD_TAG_IDX, self.EOS_TAG_IDX] = 0.0
            self.transitions.data[self.PAD_TAG_IDX, self.PAD_TAG_IDX] = 0.0

    def forward(self, emissions, tags, mask=None):
        ## emissions: tensor[batch_size, seq_len, n_tags]
        ## tags: tensor[batch_size, seq_len]
        # mask: tensor[batch_size, seq_len], indicates valid positions (0 for pad)
        return -self.log_likelihood(emissions, tags, mask=mask)

    def log_likelihood(self, emissions, tags, mask=None):
        if mask is None:
            mask = torch.ones(emissions.shape[:2])
        scores = self._compute_scores(emissions, tags, mask=mask)
        partition = self._compute_log_partition(emissions, mask=mask)
        return torch.sum(scores - partition)

    # find out the optimal tag sequence using Viterbi Decoding Algorithm
    def decode(self, emissions, mask=None):
        if mask is None:
            mask = torch.ones(emissions.shape[:2])
        scores, sequences = self._viterbi_decode(emissions, mask)
        return scores, sequences

    def _compute_scores(self, emissions, tags, mask):
        batch_size, seq_len = tags.shape
        if(torch.cuda.is_available()):
            scores = torch.zeros(batch_size).cuda()
        else:
            scores = torch.zeros(batch_size)
        # save first and last tags for later
        first_tags = tags[:, 0]
        last_valid_idx = mask.int().sum(1) - 1
        last_tags = tags.gather(1, last_valid_idx.unsqueeze(1)).squeeze()
        # add transition from SOS to first tags for each sample in batch
        t_scores = self.transitions[self.SOS_TAG_IDX, first_tags]
        # add emission scores of the first tag for each sample in batch
        e_scores = emissions[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze()
        scores += e_scores + t_scores
        # repeat for every remaining word
        for i in range(1, seq_len):
            is_valid = mask[:, i]
            prev_tags = tags[:, i - 1]
            curr_tags = tags[:, i]
            e_scores = emissions[:, i].gather(
                1, curr_tags.unsqueeze(1)).squeeze()
            t_scores = self.transitions[prev_tags, curr_tags]
            # apply the mask
            e_scores = e_scores * is_valid
            t_scores = t_scores * is_valid
            scores += e_scores + t_scores
        # add transition from last tag to EOS for each sample in batch
        scores += self.transitions[last_tags, self.EOS_TAG_IDX]
        return scores

    # compute the partition function in log-space using forward algorithm
    def _compute_log_partition(self, emissions, mask):
        batch_size, seq_len, n_tags = emissions.shape
        # in the first step, SOS has all the scores
        alphas = self.transitions[self.SOS_TAG_IDX,
                                  :].unsqueeze(0) + emissions[:, 0]
        for i in range(1, seq_len):
            # tensor[batch_size, n_tags] -> tensor[batch_size, 1, n_tags]
            e_scores = emissions[:, i].unsqueeze(1)
            # tensor[n_tags, n_tags] -> tensor[batch_size, n_tags, n_tags]
            t_scores = self.transitions.unsqueeze(0)
            # tensor[batch_size, n_tags] -> tensor[batch_size, n_tags, 1]
            a_scores = alphas.unsqueeze(2)
            scores = e_scores + t_scores + a_scores
            new_alphas = torch.logsumexp(scores, dim=1)
            # set alphas if the mask is valid, else keep current values
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * new_alphas + (1 - is_valid) * alphas
        # add scores for final transition
        last_transition = self.transitions[:, self.EOS_TAG_IDX]
        end_scores = alphas + last_transition.unsqueeze(0)
        # return log_sum_exp
        return torch.logsumexp(end_scores, dim=1)

    # return a list of optimal tag sequence for each example in the batch
    def _viterbi_decode(self, emissions, mask):
        batch_size, seq_len, n_tags = emissions.shape
        # in the first iteration, SOS will have all the scores and then, the max
        alphas = self.transitions[self.SOS_TAG_IDX,
                                  :].unsqueeze(0) + emissions[:, 0]
        backpointers = []
        for i in range(1, seq_len):
            # tensor[batch_size, n_tags] -> tensor[batch_size, 1, n_tags]
            e_scores = emissions[:, i].unsqueeze(1)
            # tensor[n_tags, n_tags] -> tensor[batch_size, n_tags, n_tags]
            t_scores = self.transitions.unsqueeze(0)
            # tensor[batch_size, n_tags] -> tensor[batch_size, n_tags, 1]
            a_scores = alphas.unsqueeze(2)
            scores = e_scores + t_scores + a_scores
            # find the highest score and tag, instead of log_sum_exp
            max_scores, max_score_tags = torch.max(scores, dim=1)
            # set alphas if the mask is valid, otherwise keep the current values
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * max_scores + (1 - is_valid) * alphas
            backpointers.append(max_score_tags.t())
        # add scores for final transition
        last_transition = self.transitions[:, self.EOS_TAG_IDX]
        end_scores = alphas + last_transition.unsqueeze(0)
        # get the final most probable score and the final most probable tag
        max_final_scores, max_final_tags = torch.max(end_scores, dim=1)
        # find the best sequence of labels for each sample in the batch
        best_sequences = []
        emission_lengths = mask.int().sum(dim=1)
        for i in range(batch_size):
            # recover the original sentence length for the i-th sample in the batch
            sample_length = emission_lengths[i].item()
            # recover the max tag for the last timestep
            sample_final_tag = max_final_tags[i].item()
            # limit the backpointers until the last but one
            # since the last corresponds to the sample_final_tag
            sample_backpointers = backpointers[: sample_length - 1]
            # follow the backpointers to build the sequence of labels
            sample_path = self._find_best_path(
                i, sample_final_tag, sample_backpointers)
            # add this path to the list of best sequences
            best_sequences.append(sample_path)
        return max_final_scores, best_sequences

    # auxiliary function to find the best path sequence for a specific example
    def _find_best_path(self, sample_id, best_tag, backpointers):
        # backpointers: list[tensor[seq_len_i - 1, n_tags, batch_size]], seq_len_i is the length of the i-th sample of the batch
        # add the final best_tag to our best path
        best_path = [best_tag]

        # traverse the backpointers in backwards
        for backpointers_t in reversed(backpointers):

            # recover the best_tag at this timestep
            best_tag = backpointers_t[best_tag][sample_id].item()

            # append to the beginning of the list so we don't need to reverse it later
            best_path.insert(0, best_tag)

        return best_path


'''
    MTL Model to classify. Our Architecture which used the RR component and 
    Shift component parallely to get the emission scores and then they are 
    fed into the CRF to get the appropriate probabilities for each label.
'''


class Hier_LSTM_CRF_Classifier(nn.Module):
    def __init__(self, n_tags, sent_emb_dim, sos_tag_idx, eos_tag_idx, pad_tag_idx, vocab_size=0, pad_word_idx=0, pretrained=False, device='cuda'):
        super().__init__()

        self.emb_dim = sent_emb_dim
        self.pretrained = pretrained
        self.device = device
        self.pad_tag_idx = pad_tag_idx
        self.pad_word_idx = pad_word_idx

        # RR Modele
        self.emitter = LSTM_Emitter(
            n_tags, 3*sent_emb_dim, sent_emb_dim, 0.5, self.device).to(self.device)
        self.crf = CRF(n_tags, sos_tag_idx, eos_tag_idx,
                       pad_tag_idx).to(self.device)

        # Shift or Binary Module
        self.emitter_binary = LSTM_Emitter_Binary(
            5, 3*sent_emb_dim, sent_emb_dim, 0.5, self.device).to(self.device)
        self.crf_binary = CRF(5, sos_tag_idx, eos_tag_idx,
                              pad_tag_idx).to(self.device)

    def forward(self, x, x_binary):
        batch_size = len(x)
        seq_lengths = [len(doc) for doc in x]
        max_seq_len = max(seq_lengths)

        ## x: list[batch_size, sents_per_doc, sent_emb_dim]
        tensor_x = [torch.tensor(
            doc, dtype=torch.float, requires_grad=True) for doc in x]
        tensor_x_binary = [torch.tensor(
            doc, dtype=torch.float, requires_grad=True) for doc in x_binary]

        # list[batch_size, sents_per_doc, sent_emb_dim] --> tensor[batch_size, max_seq_len, sent_emb_dim]
        tensor_x = nn.utils.rnn.pad_sequence(
            tensor_x, batch_first=True).to(self.device)
        tensor_x_binary = nn.utils.rnn.pad_sequence(
            tensor_x_binary, batch_first=True).to(self.device)

        self.mask = torch.zeros(batch_size, max_seq_len).to(self.device)
        for i, sl in enumerate(seq_lengths):
            self.mask[i, :sl] = 1

        # Get hidden states of Shift Module and pass them to the RR Module for emission score calculation for RR Module
        self.emissions_binary, self.hidden_binary = self.emitter_binary(
            tensor_x_binary)
        self.emissions = self.emitter(tensor_x, self.hidden_binary)

        # Passing the emission scores to the CRF to get the final sequence of tags
        _, path = self.crf.decode(self.emissions, mask=self.mask)
        _, path_binary = self.crf_binary.decode(
            self.emissions_binary, mask=self.mask)
        return path, path_binary

    def _loss(self, y):
        # list[batch_size, sents_per_doc] --> tensor[batch_size, max_seq_len]
        tensor_y = [torch.tensor(doc, dtype=torch.long) for doc in y]
        tensor_y = nn.utils.rnn.pad_sequence(
            tensor_y, batch_first=True, padding_value=self.pad_tag_idx).to(self.device)

        nll = self.crf(self.emissions, tensor_y, mask=self.mask)
        return nll

    def _loss_binary(self, y_binary):
        # list[batch_size, sents_per_doc] --> tensor[batch_size, max_seq_len]
        tensor_y_binary = [torch.tensor(doc, dtype=torch.long)
                           for doc in y_binary]
        tensor_y_binary = nn.utils.rnn.pad_sequence(
            tensor_y_binary, batch_first=True, padding_value=self.pad_tag_idx).to(self.device)

        nll_binary = self.crf_binary(
            self.emissions_binary, tensor_y_binary, mask=self.mask)
        return nll_binary


'''
    Top-level module which uses a Hierarchical-LSTM-CRF to classify.
    Sentence embeddings are then passed to LSTM_Emitter to generate emission scores, 
    and finally CRF is used to obtain optimal tag sequence. 
    Emission scores are fed to the CRF to generate optimal tag sequence.
'''
# class Hier_LSTM_CRF_Classifier(nn.Module):
#     def __init__(self, n_tags, sent_emb_dim, sos_tag_idx, eos_tag_idx, pad_tag_idx, vocab_size = 0, word_emb_dim = 0, pad_word_idx = 0, pretrained = False, device = 'cuda'):
#         super().__init__()

#         self.emb_dim = sent_emb_dim
#         self.pretrained = pretrained
#         self.device = device
#         self.pad_tag_idx = pad_tag_idx
#         self.pad_word_idx = pad_word_idx

#         self.emitter = LSTM_Emitter(n_tags, sent_emb_dim, sent_emb_dim, 0.5, self.device).to(self.device)
#         self.crf = CRF(n_tags, sos_tag_idx, eos_tag_idx, pad_tag_idx).to(self.device)


#     def forward(self, x):
#         batch_size = len(x)
#         seq_lengths = [len(doc) for doc in x]
#         max_seq_len = max(seq_lengths)

#         ## x: list[batch_size, sents_per_doc, sent_emb_dim]
#         tensor_x = [torch.tensor(doc, dtype = torch.float, requires_grad = True) for doc in x]

#         ## list[batch_size, sents_per_doc, sent_emb_dim] --> tensor[batch_size, max_seq_len, sent_emb_dim]
#         tensor_x = nn.utils.rnn.pad_sequence(tensor_x, batch_first = True).to(self.device)

#         self.mask = torch.zeros(batch_size, max_seq_len).to(self.device)
#         for i, sl in enumerate(seq_lengths):
#             self.mask[i, :sl] = 1

#         self.emissions = self.emitter(tensor_x)
#         _, path = self.crf.decode(self.emissions, mask = self.mask)
#         return path

#     def _loss(self, y):
#         ##  list[batch_size, sents_per_doc] --> tensor[batch_size, max_seq_len]
#         tensor_y = [torch.tensor(doc, dtype = torch.long) for doc in y]
#         tensor_y = nn.utils.rnn.pad_sequence(tensor_y, batch_first = True, padding_value = self.pad_tag_idx).to(self.device)

#         nll = self.crf(self.emissions, tensor_y, mask = self.mask)
#         return nll

print("Loading.... tag2idx")
tag2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "Fact": 3, "RulingByLowerCourt": 4, "Argument": 5,
           "Statute": 6, "RatioOfTheDecision": 7, "RulingByPresentCourt": 8, "Precedent": 9, "Dissent": 10}
idx2tag = {v: k for k, v in tag2idx.items()}

print("Loading....rr model")
save_path = "all_models/saved_new/"
emb_dim = 768
device_type = 'cuda'
model_best = Hier_LSTM_CRF_Classifier(len(
    tag2idx), emb_dim, tag2idx['<start>'], tag2idx['<end>'], tag2idx['<pad>'], vocab_size=2, pretrained=True, device=device_type).to(device_type)
model_state = torch.load(save_path + 'model_state.tar')
# print("Printing keys in model_state..........")
# print(model_state.keys())
model_best.load_state_dict(model_state['state_dict'])
print(" RR Model Load Successful")

# generate_embedding section (saves embeddings in json files)
print("Inside Generate_embedding..........")
tokenizer_gen_emb = BertTokenizer.from_pretrained("bert-base-uncased")
model_gen_emb = BertModel.from_pretrained("bert-base-uncased")
model_gen_emb.eval()  # Setting model to evaluation mode
print('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_gen_emb.to(device)
print("Gen emb model loading sucessful......")
# print("Exiting the code.....")
# exit()
print("***************************Starting the actual processing**************************")
file_list = os.listdir(raw_input_path)  # list of all sav files
print("Number of sav files in the location ",
      raw_input_path, " is ", len(file_list))


for file_name in tqdm(file_list, desc="sav files:"):

    file_path = raw_input_path + file_name
    print("Loading file:", file_name)
    with open(file_path, "rb") as f:
        train_it = pkl.load(f)

    it_train_files = list(train_it['candidate_data'].keys())
    print("Number of files in ", file_name, " is: ", len(it_train_files))

    out_gen = emb_output_path+"gen_emb/"+file_name.split(".")[0]
    print("Creating gen embbeding output path: ", out_gen)
    os.makedirs(out_gen, exist_ok=True)
    print("Created gen emb output path: ", out_gen)
    out_bertsc = emb_output_path+"bertsc_emb/"+file_name.split(".")[0]+"/"
    print("Creating gen embbeding output path: ", out_bertsc)
    os.makedirs(out_bertsc, exist_ok=True)
    print("Created gen emb output path: ", out_bertsc)
    # uncomment from here
    # Getting the embeddings for case in train_it, similarly follow for other files
    # Here we use the embedding corresponding to the [CLS] token as the sentences representation
    for case in tqdm(it_train_files, desc="Generate Emb:"):
        sentences = train_it['candidate_data'][case]
        all_text = {}
        start_time = time.time()
        for idx in range(len(sentences)):
            text = sentences[idx]
            marked_text = "[CLS] " + text + " [SEP]"
            tokenized_text = tokenizer_gen_emb.tokenize(marked_text)

            if(len(tokenized_text) > 510):
                tokenized_text = tokenized_text[:510] + ['[SEP]']

            indexed_tokens = tokenizer_gen_emb.convert_tokens_to_ids(
                tokenized_text)

            segments_ids = [1] * len(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens]).to(device)
            segments_tensors = torch.tensor([segments_ids]).to(device)

            with torch.no_grad():
                outputs = model_gen_emb(tokens_tensor, segments_tensors)

            emb = outputs[0].squeeze()[0].flatten().tolist()

            emb = [str(round(i, 5)) for i in emb]
            all_text[idx] = emb

        json_object = json.dumps(all_text, indent=4)
        with open(os.path.join(out_gen, str(case)+".json"), "w") as f:
            f.write(json_object)

    print("***********Finished Generating embeddings for docs in ",
          file_name, " ************")

    # Creating BERTSC embeddings

    print("***********Starting BERTSC embeddings for docs in ",
          file_name, " ************")
    data_tr_CL = train_it['candidate_data']
    train_df_CL = json_to_df(data_tr_CL)
    train_input_ids, train_lengths, train_token_type_ids = input_id_maker(
        train_df_CL, tokenizer_bertsc)
    train_attention_masks = att_masking(train_input_ids)
    # Getting embeddings for sentences
    clsembs_train = []
    for i in tqdm(range(len(train_input_ids)), desc="BertSC Embeddings"):
        ii = train_input_ids[i]
        clsembs_train.append(get_output_for_one_vec(
            ii, train_attention_masks[i]))

    # Saving the embeddings
    i = 0
    for key in tqdm(it_train_files, desc="Saving BertSC Embs:"):
        limit = len(data_tr_CL[key])
        sp = clsembs_train[i:i+limit-1]
        np.save(out_bertsc + str(key), np.array(sp))
        i = i+limit-1

    print("************Finished BERTSC embeddings for docs in ",
          file_name, " *************")

    # Rhetorical Roles Results - RRR
    rr_results = dict()
    print(":::::::::::::::::Starting Rhetorical Roles Results for ",
          file_name, "::::::::::::::::")
    file_rr_dir = file_rr_output_path+file_name.split(".")[0]+"/"
    os.makedirs(file_rr_dir, exist_ok=True)
    print("len of it_train_files:", len(it_train_files))
    for i in tqdm(range(len(it_train_files)), desc="Rhetorical Roles: "):
        case_id = it_train_files[i]
        sentences = train_it['candidate_data'][case_id]
        file_gen_emb = out_gen+"/"+str(case_id)+".json"
        file_bertsc_emb = out_bertsc+str(case_id)+".npy"
        batch_x_binary = get_s_emb(
            file_gen_emb, file_bertsc_emb)  # 3*768 size emb
        batch_x_binary = batch_x_binary.reshape(
            (1, batch_x_binary.shape[0], batch_x_binary.shape[1]))
        #print("For ",file_name," batch_x_binary shape: ",batch_x_binary.shape)
        #print("Predicting labels for ",file_name,"........")
        pred, pred_binary = model_best(batch_x_binary, batch_x_binary)
        # #print("Prediction Successful")
        # Convert tag_ids to tags
        tag_lst = [idx2tag[idx] for idx in pred[0]]

        # create list of pairs for sentence and its RR
        # save list corresponding to doc id as key-value pair in dict
        rr_results[case_id] = list(zip(sentences, tag_lst))
        with open(file_rr_dir+str(case_id)+".json", "w") as f:
            f.write(json.dumps(
                {case_id: list(zip(sentences, tag_lst))}, indent=4))

    json_object = json.dumps(rr_results, indent=4)
    with open(rr_output_path+file_name.split(".")[0]+"_rr_results"+".json", "w") as f:
        f.write(json_object)
    print("************Finished Rhetorical Roles for docs in ",
          file_name, " *************")

print("Total Processing Finished... check ",
      rr_output_path, " for the results")
print("ciao!!")
