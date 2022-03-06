# BMI 203 Project 7: Neural Network


# Importing Dependencies
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike


# Defining preprocessing functions
def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one hot encoding of a list of nucleic acid sequences
    for use as input into a fully connected neural net.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence
            length due to the one hot encoding scheme for nucleic acids.

            For example, if we encode 
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    """
    encoding = {
        "A": [1, 0, 0, 0],
        "T": [0, 1, 0, 0],
        "C": [0, 0, 1, 0],
        "G": [0, 0, 0, 1]
    }
    encoded_seq_arr = np.zeros((len(seq_arr), len(seq_arr[0])*4))
    for seq_idx, seq in enumerate(seq_arr):
        for base_idx, base in enumerate(seq):
            encoded_seq_arr[seq_idx, base_idx*4:base_idx*4+4] += encoding[base]

    return encoded_seq_arr


def sample_seqs(seqs: List[str],
                labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample your sequences to account for class imbalance. 
    Consider this as a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    pos_seqs = np.array(seqs, dtype=object)[labels]
    neg_seqs = np.array(seqs, dtype=object)[[not label for label in labels]]

    if len(pos_seqs) > len(neg_seqs):
        boosted = list(neg_seqs[np.random.randint(len(neg_seqs), size=len(pos_seqs))])
        balanced_seqs = list(pos_seqs) + boosted
        balanced_labels = [True] * len(pos_seqs) + [False] * len(boosted)
    elif len(pos_seqs) < len(neg_seqs):
        boosted = list(pos_seqs[np.random.randint(len(pos_seqs), size=len(neg_seqs))])
        balanced_seqs = list(neg_seqs) + boosted
        balanced_labels = [False] * len(neg_seqs) + [True] * len(boosted)
    else:
        balanced_seqs = seqs
        balanced_labels = labels

    return balanced_seqs, balanced_labels
