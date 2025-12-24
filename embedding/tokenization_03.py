text = "I am hero."

# Encode string to UTF-8 bytes

# ဒါကို Tokenization မလုပ်ခင်မှာ initial token IDs ထုတ်တယ်လို့ပြောလို့ရပါတယ်။
byte_sequence = text.encode("utf-8")
print("Byte sequence:", list(byte_sequence))

decoded_text = bytes(byte_sequence).decode("utf-8")
print(decoded_text)

vocab = {i: chr(i) for i in range(256)}
token_ids = list(text.encode("utf-8"))
tokens = [chr(i) for i in token_ids]
print("vocab: ", vocab)
print("-----------------------------------------")
print("Tokens:", tokens)
print("Token IDs:", token_ids)


# string → byte → byte IDs → BPE merge step
# Example: frequent pair ('l','l') → 'll' token
merge_map = {(108, 108): 256}  # assign new token id 256
new_token_ids = []

i = 0
while i < len(token_ids):
    if i < len(token_ids)-1 and (token_ids[i], token_ids[i+1]) in merge_map:
        new_token_ids.append(merge_map[(token_ids[i], token_ids[i+1])])
        i += 2
    else:
        new_token_ids.append(token_ids[i])
        i += 1

print("After merge:", new_token_ids)
