### RAG မှာ LEANN + FAISS နှစ်ခု hybrid သုံးပါမယ်။

FAISS ဆိုတာက facebook AI Similarity Search ပါ။​ Meta က ထုတ်ပေးထားတာပါ။ သူ့ Algorithm က Approximate Nearest Neighbor ဖြစ်ပါတယ်။ Vector similarity search အတွက် အသုံးပြုတဲ့ library တစ်ခုဖြစ်ပါတယ်။ ကျွန်တော်တို့ Attention Model ကို Query အနေနဲ့ပေးချင်တဲ့ Sentence ကို embedding matrix (768-dim vector) ပြောင်း ပြီး Vector သန်းချီထဲက Attention က လိုချင်တဲ့ sentence နဲ့ အနီးဆုံး 5 ခု ကို ရှာပါ စသဖြင့်ပြုလုပ်ပေးတာပါ။ 

FAISS မပါရင်
vector 1 သန်း × cosine similarity = အရမ်းနှေး ပြီး

FAISS ပါရင်
milliseconds အတွင်းပြီး ပါတယ်။


LEANN (Learned Approximate Nearest Neighbor)ကတော့ FAISS ရဲ့ backend algorithm တစ်ခုဖြစ်တဲ့ Approximate Nearest Neighbor အသုံးပြုပြီး ပြန်လည်တည်ဆောက်ထားတဲ့ ကောင်ပါ။ 


FAISS က မှာ ပါဝင်တဲ့ index အမျိုးအစားတွေကတော့ 

Flat (Exact) - သူကတော့ Attention Model က Vector Database ထဲက data တွေကို ဘယ်လောက်တိတိကျကျလိုချင်သလဲ တိုင်းတာပေးတာပါ။ ထပ်တူညီအောင်ဆွဲထုတ်ချင်တယ်ဆိုရင် နှေးမယ် ၊ ကြာမယ်ပေါ့ဗျာ။ 

IVF (Inverted File Index) - Cluster လုပ်ပြီးရှာတာဖြစ်ပါတယ်။ 

HNSW - Graph-based - Accuracy + speed balance

PQ / OPQ - Vector compression နဲ့ RAM စားသက်သာစေချင်လို့ quantize လုပ်ပြီးရှာတာပါ။ 


ဒါဆိုရင် FAISS က embedding vector db ထဲမှာ ANN -Approximate Nearest Neighbor Algorithm နဲ့ searching ပေးတာဖြစ်ပြီးတော့ LEANN ကတော့ search strategy တစ်ခုအနေနဲ့အလုပ်လုပ်ပေးမှာပါ။ 

Ok , LEANN မှာဘာတွေအားသာချက်ရှိလဲ ရှိုးလိုက်ရအောင်၊​ 

Vector search (nearest neighbor search) ကို heuristic မဟုတ်ဘဲ
ML/NN model နဲ့ ရှာမှာဖြစ်ပါတယ်။ ပုံမှန် RAG pipeline မှာ FAISS, HNSW, IVF လို graph / tree / clustering အခြေခံ ANN တွေကိုသုံးကြပါတယ်။ LEANN ကတော့ search policy ကို neural network နဲ့ Train ထားပေးတာ ဖြစ်ပါတယ်။

```
Query → Embedding → ANN index → Top-k chunks → LLM
```

```
Query → Embedding
      → Learned policy (NN)
      → Which nodes to visit?
      → Fewer comparisons
      → Top-k chunks
      → LLM
```

LEANN မှာ Nearest neighbor search ကို Graph traversal, Node visiting order 




### Setup leann on some hosts.

https://github.com/yichuan-w/LEANN
