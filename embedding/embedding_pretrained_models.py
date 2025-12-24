from transformers import AutoTokenizer, AutoModel
import torch

# ----------------------------
# 1. Load pretrained BERT
# ----------------------------
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model.eval()  # disable dropout randomness

# ----------------------------
# 2. Input sentence
# ----------------------------
sentence = ["with", "several", "words"]
inputs = tokenizer(sentence, return_tensors="pt")

input_ids = inputs["input_ids"]
token_type_ids = inputs.get(
    "token_type_ids",
    torch.zeros_like(input_ids)
)

print("Input IDs:")
print(input_ids)
print("Shape:", input_ids.shape)

# ----------------------------
# 3. Word Embeddings (token only)
# ----------------------------
word_embeddings = model.embeddings.word_embeddings(input_ids)

print("\n[1] Word embeddings (token only)")
print(word_embeddings)
print("Shape:", word_embeddings.shape)

# ----------------------------
# 4. Position Embeddings
# ----------------------------
seq_len = input_ids.size(1)
position_ids = torch.arange(seq_len).unsqueeze(0)
print("\nPosition IDs:")
print(position_ids)
position_embeddings = model.embeddings.position_embeddings(position_ids)

print("\n[2] Position embeddings")
print(position_embeddings)
print("Shape:", position_embeddings.shape)

# ----------------------------
# 5. Token Type (Segment) Embeddings
# ----------------------------
token_type_embeddings = model.embeddings.token_type_embeddings(token_type_ids)

print("\n[3] Token type (segment) embeddings")
print(token_type_embeddings)
print("Shape:", token_type_embeddings.shape)

# ----------------------------
# 6. Embedding WITHOUT position encoding
#    (word + token_type)
# ----------------------------
embedding_no_position = word_embeddings + token_type_embeddings

print("\n[4] Embedding WITHOUT position encoding")
print(embedding_no_position)
print("Shape:", embedding_no_position.shape)

# ----------------------------
# 7. Embedding WITH position encoding
#    (before LayerNorm)
# ----------------------------
embedding_with_position = embedding_no_position + position_embeddings

print("\n[5] Embedding WITH position encoding (before LayerNorm)")
print(embedding_with_position)
print("Shape:", embedding_with_position.shape)

# ----------------------------
# 8. Final BERT embedding
#    (LayerNorm + Dropout)
# ----------------------------
final_embeddings = model.embeddings.LayerNorm(embedding_with_position)
final_embeddings = model.embeddings.dropout(final_embeddings)

print("\n[6] Final BERT embedding (used by encoder)")
print(final_embeddings)
print("Shape:", final_embeddings.shape)

# ----------------------------
# 9. Verification 
# ----------------------------
hf_embeddings = model.embeddings(input_ids)

print("\nVerification with HuggingFace embeddings:")
print("Same as model.embeddings(...)? ->",
      torch.allclose(final_embeddings, hf_embeddings))
