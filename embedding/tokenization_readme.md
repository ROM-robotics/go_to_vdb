**အားလုံးက BPE တစ်မျိုးတည်းပဲ သုံးတာ မဟုတ်ပါဘူး**။
LLM / NLP model တွေမှာ **tokenizer algorithm မျိုးစုံ** သုံးကြပြီး
model design + language target ပေါ်မူတည်ပါတယ်။

---

## 1 Tokenizer Algorithm အမျိုးအစား အကြမ်းဖျဉ်း

| Tokenizer  | Idea                       |
| ---------- | -------------------------- |
| BPE        | Frequent pair merge        |
| WordPiece  | Likelihood-based merge     |
| Unigram LM | Probabilistic segmentation |
| Character  | Char-level                 |
| Byte-level | Raw bytes                  |

---

## 2 Model Family အလိုက် ဘာသုံးလဲ?

---

### 🔹 GPT Series (GPT-2 / GPT-3 / GPT-4*)

**Tokenizer:** Byte-level BPE

* Byte (0–255) ကနေစ
* OOV မရှိ
* English-centric
* Space-aware (`Ġword`)

👉 **GPT = BPE (byte-level)**

---

### 🔹 LLaMA / LLaMA-2 / LLaMA-3

**Tokenizer:** SentencePiece (BPE variant)

* Subword BPE
* No explicit space token
* Unicode-based
* Multilingual ပိုကောင်း

👉 **LLaMA = BPE (SentencePiece)**

---

### 🔹 BERT / RoBERTa

**Tokenizer:** WordPiece

* Likelihood-based
* `[CLS]`, `[SEP]`
* Masked LM friendly

👉 **BERT ≠ BPE (but similar idea)**

---

### 🔹 T5

**Tokenizer:** SentencePiece (Unigram LM)

* Probabilistic segmentation
* Multilingual friendly
* No whitespace dependency

👉 **T5 = Unigram LM**

---

### 🔹 XLM-R

**Tokenizer:** SentencePiece (Unigram)

* 100+ languages
* Unicode normalization

---

### 🔹 BLOOM

**Tokenizer:** Byte-level BPE

* Similar to GPT
* Multilingual tuned

---

### 🔹 mBERT

**Tokenizer:** WordPiece

---

### 🔹 Whisper (Speech → Text)

**Tokenizer:** Byte-level BPE

* Robust to noisy transcripts

---

## 3 BPE vs WordPiece vs Unigram (Conceptual)

| Feature      | BPE           | WordPiece           | Unigram       |
| ------------ | ------------- | ------------------- | ------------- |
| Training     | Greedy merge  | Likelihood maximize | LM-based      |
| Segmentation | Deterministic | Deterministic       | Probabilistic |
| Multilingual | Medium        | Medium              | High          |
| OOV          | ❌             | ❌                   | ❌             |

---

## 4 ဘာကြောင့် Model တစ်ခုချင်း မတူတာလဲ?

### 🔹 Language property

* English → BPE OK
* Burmese / Thai → SentencePiece ပိုကောင်း

### 🔹 Training objective

* Causal LM → BPE
* Masked LM → WordPiece

### 🔹 Engineering

* Byte-level → robust input
* Unicode-level → compact vocab

---

## 5 Myanmar / Low-resource Language Perspective 

| Choice        | Result                       |
| ------------- | ---------------------------- |
| Byte BPE      | Robust but token length ကြီး |
| SentencePiece | Better segmentation          |
| WordPiece     | Needs whitespace             |

👉 **Myanmar NLP → SentencePiece (Unigram/BPE)** သင့်တော်

---

### BPE သို့မဟုတ် Sentence Piece သို့မဟုတ် word pieces / ပြီးသွားတာနဲ့ embedding လုပ်ဖို့လိုအပ်လာပါပြီ။ Words embeddeding လုပ်တဲ့အချိန်မှာ သုံးလို့ရတဲ့ algorithms တွေကတော့ 
