```python
from sentence_transformers import SentenceTransformer

# ၁။ Model ကို Load လုပ်မယ် (သေးငယ်ပြီး မြန်ဆန်တဲ့ Model တစ်ခုခုကို ရွေးပါ)
model = SentenceTransformer('all-MiniLM-L6-v2')

# ၂။ ပြောင်းချင်တဲ့ စာသားများ
sentences = ["ပန်းသီးသည် အရသာရှိသည်", "ငှက်ပျောသီးသည် အဝါရောင်ရှိသည်"]

# ၃။ Vector (Embeddings) အဖြစ် ပြောင်းလဲခြင်း
embeddings = model.encode(sentences)

print(embeddings.shape) # Output: (2, 384) - ဆိုလိုတာက စာကြောင်း ၂ ကြောင်း၊ တစ်ကြောင်းချင်းစီမှာ နံပါတ် ၃၈၄ လုံးပါပါတယ်။
print(embeddings[0])    # ပထမစာကြောင်းရဲ့ Vector ကို ကြည့်ခြင်း
```