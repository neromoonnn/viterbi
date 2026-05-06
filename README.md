# Viterbi Decoder

A Python implementation of convolutional encoding and Viterbi decoding.

## Overview

This project implements:

- **Trellis diagram generation** from generator polynomials
- **Convolutional encoding** of binary input sequences
- **Viterbi decoding** to recover the original message from encoded output using Hamming distance metrics

## Usage

```python
import numpy as np
from viterbi import convolutional_encode, convolutional_decode

bits = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1])
constraint_length = np.int32(7)
generator = np.array([0b1_110111, 0b1_101001, 0b001_011])

encoded = convolutional_encode(bits, constraint_length, generator)
decoded = convolutional_decode(encoded, constraint_length, generator)

print(np.array_equal(bits, decoded))  # True
```

Or run directly:

```bash
python viterbi.py
```

## Dependencies

- [NumPy](https://numpy.org/)

## API

### `polynomial_to_trellis(constraint_length, generator)`

Builds the trellis (next-state and output tables) from the constraint length and generator polynomials.

### `convolutional_encode(input, constraint_length, generator)`

Encodes a binary input array using the specified convolutional code.

### `convolutional_decode(input, constraint_length, generator)`

Decodes a received convolutionally-encoded sequence back to the original bits using the Viterbi algorithm with Hamming distance as the branch metric.
