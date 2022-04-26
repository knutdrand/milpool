import numpy as np
import pytest
from milpool.conditional_kmer_encoding import ConditionalKmerEncoding, ConditionalKmerCounter


@pytest.fixture
def sequence():
    return "acgtg".upper()

@pytest.fixture
def one_counts():
    return {"AG": {"C": 1},
            "CT": {"G": 1},
            "GG": {"T": 1}}


def test_conditional_kmer_encoding(sequence):
    seq = np.array([ord(c) for c in sequence], dtype=np.uint8)
    encoder = ConditionalKmerEncoding(3, 1)
    encoded = encoder.from_bytes(seq)
    assert np.all(encoder.to_bytes(encoded) == seq), (encoder.to_bytes(encoded), seq)


def test_conditional_kmer_counting(sequence, one_counts):
    seq = np.array([ord(c) for c in sequence], dtype=np.uint8)
    counter = ConditionalKmerCounter(3, 1)
    counter.count(seq)
    d = counter.to_dict()
    assert d == one_counts
