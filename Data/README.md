You can find the dataset from [here](), which contains 6 JSON files (Train, Dev, and Test) for CL and IT domains.

Corpus Statistics:



A single sentence can sometimes represent multiple rhetorical roles. Each expert could also assign secondary and tertiary rhetorical roles to a single sentence to handle such scenarios. 

As an example, suppose a sentence is a ‘Fact’ but could also be an ‘Argument’ according to the legal expert. In that case, the expert could assign the rhetorical roles ‘Primary Fact’ and ‘Secondary Argument’ to that sentence. We extended it to the tertiary level as well to handle rare cases.

Each case contains the following keys in the JSON: 

'sentences': Preprocessed judgments.

'user_1_primary', 'user_2_primary' and 'user_3_primary': Primary rhetorical role assigned by the users (law experts) to sentences.

'user_1_secondary', 'user_2_secondary' and 'user_3_secondary': Secondary rhetorical role assigned by the users (law experts) to sentences, if any.

'user_1_tertiary', 'user_2_tertiary' and 'user_3_tertiary': Tertiary rhetorical role assigned by the users (law experts) to sentences, if any.

'user_1_overall', 'user_2_overall' and 'user_3_overall':

'complete':
