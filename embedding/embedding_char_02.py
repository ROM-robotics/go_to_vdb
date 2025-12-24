import torch
import torch.nn as nn

char_vocab = {'h': 0, 'i': 1, ' ': 2, '!': 3}
vocab_size = len(char_vocab)
d_model = 3

char_embedding = nn.Embedding(vocab_size, d_model)

text = "hi!"
char_ids = torch.tensor([char_vocab[c] for c in text])

print("Characters:", list(text))
print("Char IDs:", char_ids.tolist())

char_vectors = char_embedding(char_ids)

print("Char embeddings:")
print(char_vectors)