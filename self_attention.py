import numpy as np


def self_attention(X):
    batch_size, seq_len, d_model = X.shape
    d_k = d_model

    W_q = np.random.randn(d_model, d_k)
    W_k = np.random.randn(d_model, d_k)
    W_v = np.random.randn(d_model, d_k)

    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)

    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)

    output = attention_weights @ V

    return output, attention_weights
