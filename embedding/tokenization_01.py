text = "I love you more than NLP"

tokens = text.split(" ")
print(tokens)


#❌ Problem:
#Vocabulary ကြီး
#Unknown words မကိုင်နိုင်


vocab = {
    "I": 0,
    "love": 1,
    "NLP": 2,
    "you": 3,
    "more": 4,
    "than": 5,
}

text = "I love you more than NLP"
tokens = text.split()

token_ids = [vocab[token] for token in tokens]
print(token_ids)
