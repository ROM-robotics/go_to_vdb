#### Embedding ကဘာလုပ်သလဲ?
```
Embedding lookup ဆိုတာ
token ID ကို index အဖြစ်သုံးပြီး
embedding matrix ထဲက row (တည်နေရာ) ကို
တန်းထုတ်ယူတာပဲ ဖြစ်ပါတယ်။
```
#### Positional Encoding ရဲ့ အဓိကရည်ရွယ်ချက် ?

```
Token ရဲ့ meaning (embedding) +
Token ရဲ့ order (position)
ကို vector တစ်ခုထဲမှာ ပေါင်းပေးဖို့
```


* Frequency Based -> TF-IDF


* Prediction Based -> 

* Word to vector (WordToVec) - developed by google 2013
    - Two main architectures
        1. CBOW - Continuous Bags of Words
            Predict target word from surrounding words
        2. SKIP-GRAM - 
            Predict surrounding words from target word
        
* GLOVE - Global Vectors for world representations (Stanford 2014)

* Contexual Based Embeddeding
    - ELMo (2018)

```
Tokenization အပြီး embedding လုပ်တဲ့ အချိန်မှာ static (Word2Vec / GloVe) or contextual (ELMo / BERT / GPT) လုပ်နိုင်တယ်

Frequency-based (TF-IDF) က classical NLP မှာ သုံး

Transformer-based embedding = modern NLP standard
```


## 1️⃣ Tokenization ပြီးရင် ဘာကျန်နေသလဲ?

Tokenization ပြီးရင် model လက်ထဲမှာ ရှိတာက —

```
"I am hero"
→ Tokens → ["I", "am", "hero"]
→ Token IDs → [73, 256, 512]
```

ဒီ **Token ID တွေက number သက်သက်ပဲ**
အဓိပ္ပါယ် (meaning) မပါသေး

### ပြဿနာ

* 73 နဲ့ 256 ဆိုတာ **ဘာဆိုင်လဲ** model မသိ
* “hero” နဲ့ “warrior” ဆင်တူတာ မမြင်နိုင်


## Word Embedding ဆိုတာဘာလဲ?

> **Token ID → dense vector** ပြောင်းပေးတာ

```
73     → [ 0.12, -0.34, 0.91, ... ]
256    → [ 0.10, -0.30, 0.88, ... ]
512    → [ 0.56,  0.22, -0.11, ... ]
```

* Dimension = 100 / 300 / 768 / 4096 …
* Vector တစ်ခုက **word ရဲ့ meaning** ကို ကိုယ်စားပြု

---

## Embedding မလုပ်ရင် ဘာဖြစ်မလဲ?

### ❌ One-hot encoding သုံးလို့ရ၊ သူ့ကို သုံးမယ်ဆိုရင်?

```
"hero" = [0, 0, 0, 1, 0, ...]
"warrior" = [0, 0, 0, 0, 1, ...]
```

* Similarity = 0
* Memory အရမ်းစား
* Generalization မရ

Embedding က **dense + semantic**

---

## Embedding က model ကို ဘာပေးလဲ?

### (1) Meaning (Semantic similarity)

```
cosine(hero, warrior) ≈ 0.8
cosine(hero, table)   ≈ 0.1
```

Model က “hero ~ warrior” နားလည်လာ

---

### (2) Generalization

* “hero” ကို မမြင်ဖူးသေးပေမဲ့
* embedding space ထဲမှာ နီးရင်
* behavior ဆင်တူစေ

---

## Transformer context မှာ Embedding ရဲ့နေရာ

```
Token IDs
  ↓
Token Embedding
  ↓
+ Positional Embedding
  ↓
Transformer Layers
```

---




