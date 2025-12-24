## GPT Style tokenization
## pip install sentencepiece ( for multiligual models )

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "Im hero and used to save the world sisnce 2111."
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print("Tokens:", tokens)
print("Token IDs:", token_ids)


# Output:
#Tokens: ['I', 'Ġlove', 'ĠN', 'LP']
#Token IDs: [40, 1842, 399, 19930]


decoded = tokenizer.decode(token_ids)
print(decoded)


# GPT-style Tokenization (Byte Pair Encoding - BPE)


