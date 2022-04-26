import numpy as np
from bionumpy.encodings import ACTGEncoding
from bionumpy.kmers import TwoBitHash
from collections import defaultdict


class ConditionalKmerEncoding:
    def __init__(self, k, i, alphabet_size=4):
        self._k = k
        self._i = i
        self._alphabet_size = alphabet_size
        self._convolution = self._get_convolution()
        
    def _get_convolution(self):
        i = self._i
        conv = self._alphabet_size**np.arange(1, self._k+1)
        conv[i+1:] = conv[i:-1]
        conv[i] = 0
        return conv

    def from_bytes(self, byte_array):
        actg_encoded = ACTGEncoding.from_bytes(byte_array)
        actg_encoded = np.lib.stride_tricks.sliding_window_view(actg_encoded, self._k)
        print(self._convolution.shape, actg_encoded.shape)
        return (self._convolution*actg_encoded).sum(axis=-1) + actg_encoded[..., self._i]

    def to_bytes(self, encoded):
        i = self._i
        b = (encoded[:, None] // self._convolution) % self._alphabet_size
        b[:, i] = encoded % self._alphabet_size
        b = np.concatenate([b[:, 0], b[-1, 1:]])
        return ACTGEncoding.to_bytes(b)

    def to_string(self, encoded):
        i = self._i
        bs = (encoded // (self._alphabet_size**np.arange(self._k))) % self._alphabet_size
        bs = ACTGEncoding.to_bytes(bs)
        return bytes(bs).decode()


class ConditionalKmerCounter:
    def __init__(self, k, i, alphabet_size=4):
        self._k = k
        self._i = i
        self._alphabet_size = alphabet_size
        self.shape = (self._alphabet_size**(k-1), alphabet_size)
        self.count_matrix = np.zeros(self.shape, dtype="int")
        self._encoding = ConditionalKmerEncoding(k, i, alphabet_size)

    def count(self, sequence):
        """ Sequence needs to be 1-alphabet_size encoded"""
        encoded = self._encoding.from_bytes(sequence)
        self.count_matrix += np.bincount(encoded, minlength=self.count_matrix.size).reshape(self.shape)

    def to_dict(self):
        d = defaultdict(dict)
        encodings = (self._encoding.to_string(c) for c in range(self.count_matrix.size))
        for i, encoding in enumerate(encodings):
            if self.count_matrix.ravel()[i] > 0:
                print(i, encoding)
                d[encoding[1:]][encoding[0]] = self.count_matrix.ravel()[i]
        return d
