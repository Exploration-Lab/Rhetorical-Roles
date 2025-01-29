### Dataset: ### 
The dataset and leaderboard are available on [Hugging Face](https://huggingface.co/spaces/Exploration-Lab/IL-TUR-Leaderboard) 

Please note that data is free to use for academic research and commercial usage of data is not allowed.

**Corpus Statistics:**

| Domain  | Number of Documents | Number of Sentences   | Average Number of Sentences per Document |
| ------- |:-------------------:| :-------------------: | :--------------------------------------: |
| CL      | 50                  | 13328                 | 266                                      |
| IT      | 50                  | 7856                  | 157                                      |
| Total   | 100                 | 21184                 | 212                                      |

**Label Descriptions:**  There are eight main rhetorical roles plus one ‘none’ label. During the annotation, the documents were annotated with thirteen fine-grained labels since some could be sub-divided into more fine-grained classes.

1. **Fact (FAC):** These are the facts specific to the case based on which the arguments have been made and judgment has been issued. In addition to Fact, we also have the fine-grained label:
    1. **Issues (ISS)**. The issues which have been framed/accepted by the present court for adjudication.

2. **Argument (ARG):** The arguments in the case were divided in two more fine-grained sub-labels: 
    1. **Argument Petitioner (ARG-P):** Arguments which have been put forward by the petitioner/appellant in the case before the present court and by the same party in lower courts (where it may have been petitioner/respondent).
    2. **Argument Respondent (ARG-R):** Arguments which have been put forward by the respondent in the case before the present court and by the same party in lower
courts (where it may have been petitioner/respondent).

3. **Statute (STA):** The laws referred in the case.

4. **Dissent (DIS):** Any dissenting opinion expressed by a judge in the present judgment/decision.

5. **Precedent (PRE):** The precedents in the documents were divided into 3 finer labels, 
    1. **Precedent Relied Upon (PRE-R):** The precedents which have been relied upon by the present court for adjudication. These may or may not have been raised by the advocates of the parties
and amicus curiae. 
   2. **Precedent Not Relied Upon (PRE-NR):** The precedents which have not been relied upon by the present court for adjudication. These may have been raised by the advocates of the parties and amicus curiae. 
    3. **Precedent Overruled (PRE-O):** Any precedents (past cases) on the same issue which have been overruled through the current judgment.

6. **Ruling By Lower Court (RLC):** Decisions of the lower courts which dealt with the same case.

7. **Ratio Of The Decision (ROD):** The principle which has been established by the current judgment/decision which can be used in future cases. Does not include the obiter dicta which is based on observations applicable to the specific case only.

8. **Ruling By Present Court (RPC):** The decision of the court on the issues which have been framed/accepted by the present court for adjudication.

9. **None (NON):** any other matter in the judgment which does not fall in any of the above-mentioned categories.

A single sentence can sometimes represent multiple rhetorical roles. Each expert could also assign secondary and tertiary rhetorical roles to a single sentence to handle such scenarios.

As an example, suppose a sentence is a ‘Fact’ but could also be an ‘Argument’ according to the legal expert. In that case, the expert could assign the rhetorical roles ‘Primary Fact’ and ‘Secondary Argument’ to that sentence. We extended it to the tertiary level as well to handle rare cases.

Each case contains the following keys in the JSON: 
```
'sentences': Preprocessed judgments.

'user_1_primary', 'user_2_primary' and 'user_3_primary': Primary rhetorical role assigned by the users (law experts) to sentences.

'user_1_secondary', 'user_2_secondary' and 'user_3_secondary': Secondary rhetorical role assigned by the users (law experts) to sentences, if any.

'user_1_tertiary', 'user_2_tertiary' and 'user_3_tertiary': Tertiary rhetorical role assigned by the users (law experts) to sentences, if any.

'user_1_overall', 'user_2_overall' and 'user_3_overall': Highest hierarchy label 

'complete': Majority voted label across the three users
```
