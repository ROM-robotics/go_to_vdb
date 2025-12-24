import torch
import torch.nn as nn
import math

word_vocab = {
    "I": 0,
    "am": 1,
    "hero": 2
}

vocab_size = len(word_vocab)
d_model = 4  


word_embedding = nn.Embedding(vocab_size, d_model) # random embedding matrix

sentence = "I am hero"
word_ids = torch.tensor([word_vocab[w] for w in sentence.split()])

print("Words:", sentence.split())
print("Word IDs:", word_ids.tolist())

word_vectors = word_embedding(word_ids)

print("\nWord embeddings:")
print(word_vectors)
print("Shape:", word_vectors.shape)


def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)

    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pe[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))

    return pe


seq_len = len(word_ids)
pos_encoding = positional_encoding(seq_len, d_model)

print("\nPositional Encoding:")
print(pos_encoding)
print("Shape:", pos_encoding.shape)


final_input = word_vectors + pos_encoding  # element-wise addition (+)

print("\nFinal input to Transformer (Embedding + Position):")
print(final_input)
print("Shape:", final_input.shape)
