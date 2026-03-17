# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
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
    if len(seqs) != len(labels):
        raise ValueError("seqs and labels must have the same length")

    pos_seqs = [seq for seq, lab in zip(seqs, labels) if lab]
    neg_seqs = [seq for seq, lab in zip(seqs, labels) if not lab]

    if len(pos_seqs) == 0 or len(neg_seqs) == 0:
        raise ValueError("Both positive and negative examples are required")

    target_n = max(len(pos_seqs), len(neg_seqs))

    pos_idx = np.random.choice(len(pos_seqs), size=target_n, replace=True)
    neg_idx = np.random.choice(len(neg_seqs), size=target_n, replace=True)

    sampled_pos = [pos_seqs[i] for i in pos_idx]
    sampled_neg = [neg_seqs[i] for i in neg_idx]

    sampled_seqs = sampled_pos + sampled_neg
    sampled_labels = [True] * target_n + [False] * target_n

    shuffle_idx = np.random.permutation(len(sampled_seqs))
    sampled_seqs = [sampled_seqs[i] for i in shuffle_idx]
    sampled_labels = [sampled_labels[i] for i in shuffle_idx]

    return sampled_seqs, sampled_labels


def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    if len(seq_arr) == 0:
        return np.array([])

    seq_len = len(seq_arr[0])
    for seq in seq_arr:
        if len(seq) != seq_len:
            raise ValueError("All sequences must have the same length")

    base_to_vec = {
        "A": [1, 0, 0, 0],
        "T": [0, 1, 0, 0],
        "C": [0, 0, 1, 0],
        "G": [0, 0, 0, 1],
    }

    encodings = []
    for seq in seq_arr:
        seq = seq.upper()
        encoding = []
        for base in seq:
            if base not in base_to_vec:
                raise ValueError(f"Invalid DNA base: {base}")
            encoding.extend(base_to_vec[base])
        encodings.append(encoding)

    return np.array(encodings, dtype=np.float64)