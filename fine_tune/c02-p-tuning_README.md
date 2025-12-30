# 📚 c02-p-tuning.ipynb - အသေးစိတ် သင်ကြားချက် (မြန်မာဘာသာ)

## 🎯 ဒီ Notebook က ဘာလုပ်တာလဲ?

ဒီ notebook က **P-Tuning v2** နည်းလမ်းကို သုံးပြီး ROS 2 command generation လုပ်ပါတယ်။ P-Tuning က Soft Prompt Tuning ထက် ပိုခေတ်မီပြီး MLP (Multi-Layer Perceptron) network နဲ့ prompt embeddings တွေကို ပိုကောင်းအောင် လုပ်ပါတယ်။

---

## 📖 Cell 1: Libraries Import

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
```

### ရှင်းလင်းချက်:
c01 နဲ့ အတူတူပဲ ဖြစ်ပါတယ်။ PyTorch နဲ့ Transformers library တွေကို import လုပ်တာပါ။

---

## ⚙️ Cell 2: Model Configuration

```python
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

### ရှင်းလင်းချက်:
- အတူတူပဲ Qwen model ကို သုံးမယ်
- GPU ရှိရင် automatic သုံးမယ်

---

## 🔧 Cell 3: Training Hyperparameters

```python
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
N_PROMPT_TOKENS = 20
LR = 1e-4  
EPOCHS = 50
MAX_NEW_TOKENS = 64
```

### ရှင်းလင်းချက်:
- **DTYPE**: Data type သတ်မှတ်ခြင်း
  - **bfloat16**: GPU မှာ memory သက်သာပြီး မြန်တယ် (16-bit floating point)
  - **float32**: CPU မှာ သုံးတယ် (32-bit - ပိုတိကျတယ်)
- **EPOCHS**: 50 - c01 ထက် နည်းတယ်။ P-Tuning က ပိုထိရောက်လို့ epochs နည်းသုံးလို့ရတယ်

**သင်ခန်းစာ**: Mixed precision (bfloat16) က modern training မှာ အရေးကြီးတယ်။ Speed နဲ့ memory သက်သာတယ်၊ accuracy လည်း မကျဘူး။

---

## 📊 Cell 4: Training Data

```python
train_data = [
    ("Move forward 2 meters", "ros2 topic pub ..."),
    ("Turn left 90 degrees", "ros2 service call ..."),
    ("Navigate to waypoint A", "ros2 action send_goal ...")
]
```

### ရှင်းလင်းချက်:
c01 နဲ့ အတူတူပဲ။ Input-output pairs ၃ ခု။

---

## 🤖 Cell 5: Model & Tokenizer Setup

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, device_map="auto")
for p in model.parameters():
    p.requires_grad = False
model.eval()
```

### ရှင်းလင်းချက်:
- **torch_dtype=DTYPE**: Model ကို bfloat16/float32 နဲ့ load လုပ်တယ်
- **device_map="auto"**: GPU/CPU ကို အလိုအလျောက် ခွဲဝေ စီမံပေးတယ်
- Model parameters freeze - c01 နဲ့ အတူတူ

**သင်ခန်းစာ**: device_map="auto" က multi-GPU သို့ large model တွေအတွက် အသုံးဝင်တယ်။ Memory မလုံလောက်ရင် model ကို GPU/CPU ကြား ခွဲဝေပေးတယ်။

---

## 🎨 Cell 6: P-Tuning v2 Prompt Module (အဓိက ခြားနားချက်!)

```python
class PTuningV2Prompt(nn.Module):
    def __init__(self, n_tokens, hidden_size, dtype):
        super().__init__()
        self.virtual_tokens = torch.arange(n_tokens)
        self.embedding = nn.Embedding(n_tokens, hidden_size, dtype=dtype)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size, dtype=dtype)
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
```

### ရှင်းလင်းချက်:

**အဓိက ခြားနားချက် c01 နဲ့**:

### 1️⃣ Virtual Tokens
```python
self.virtual_tokens = torch.arange(n_tokens)  # [0, 1, 2, ..., 19]
```
- Token IDs တွေကို သီးခြား ဖန်တီးတယ်
- 0-19 အထိ numbers တွေ

### 2️⃣ Dedicated Embedding Layer
```python
self.embedding = nn.Embedding(n_tokens, hidden_size)
```
- Model ရဲ့ embedding တွေကို မကူးဘူး
- **အသစ် ဖန်တီးတယ်** - ဒါက ပိုလွတ်လပ်တယ်

### 3️⃣ MLP Network (အဓိက နည်းပညာ!)
```python
self.mlp = nn.Sequential(
    nn.Linear(hidden_size, hidden_size),  # Layer 1
    nn.Tanh(),                            # Activation
    nn.Linear(hidden_size, hidden_size)   # Layer 2
)
```

**MLP က ဘာလုပ်တာလဲ?**
- Embedding တွေကို **transform** လုပ်တယ်
- **Non-linear transformation** ဖြစ်တယ် (Tanh activation)
- Prompt embeddings တွေကို ပိုရှုပ်ထွေးတဲ့ patterns သင်ယူစေတယ်

**Tanh activation**:
- Values တွေကို -1 နဲ့ 1 ကြားမှာ ထားတယ်
- Smooth gradients ရတယ်

### 4️⃣ Weight Initialization
```python
nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
```
- Embeddings တွေကို random values နဲ့ စတင်တယ်
- Normal distribution: mean=0, standard deviation=0.02
- ဒါက training ကို stable ဖြစ်စေတယ်

### 5️⃣ Forward Method
```python
def forward(self, batch_size, device):
    tokens = self.virtual_tokens.to(device)
    x = self.embedding(tokens)     # [20, hidden_size]
    x = self.mlp(x)                # Transform
    return x.unsqueeze(0).expand(batch_size, -1, -1)
```

**Flow**:
1. Virtual tokens → Device (GPU/CPU)
2. Embedding lookup
3. **MLP transformation** ⭐
4. Expand to batch size

**သင်ခန်းစာ**: P-Tuning v2 က MLP network နဲ့ prompt embeddings တွေကို **transform** လုပ်တယ်။ ဒါက simple lookup (c01) ထက် ပို expressive ဖြစ်တယ်။ Model က ပိုရှုပ်ထွေးတဲ့ patterns တွေ သင်ယူနိုင်တယ်။

---

## 🔧 Cell 7: Initialize Prompt Encoder

```python
prompt_encoder = PTuningV2Prompt(
    N_PROMPT_TOKENS,
    model.config.hidden_size,
    DTYPE
).to(model.device)
```

### ရှင်းလင်းချက်:
- PTuningV2Prompt object ဖန်တီးတယ်
- **model.config.hidden_size**: Model ရဲ့ hidden dimension ကို အသုံးပြုတယ် (ပုံမှန် 1536 သို့ 2048)
- Model နဲ့ အတူတူ device ပေါ် ထားတယ်

**သင်ခန်းစာ**: Prompt encoder နဲ့ base model က တူညီတဲ့ hidden dimension ရှိရမယ်။ မဟုတ်ရင် tensor dimensions မကိုက်ဘူး။

---

## 📉 Cell 8: Loss Function

```python
def compute_loss(input_text, target_text):
    # Tokenization
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    target_ids = tokenizer(target_text, return_tensors="pt").input_ids.to(model.device)
    
    # Concatenate
    full_ids = torch.cat([input_ids, target_ids], dim=1)
    
    # Token embeddings
    token_embeds = model.get_input_embeddings()(full_ids).to(DTYPE)
    
    # Prompt embeddings from encoder
    prompt_embeds = prompt_encoder(batch_size, model.device)
    
    # Concat prompt + tokens
    full_embeds = torch.cat([prompt_embeds, token_embeds], dim=1)
    
    # Labels
    labels = torch.cat([
        torch.full((batch_size, N_PROMPT_TOKENS + input_ids.size(1)), -100, ...),
        target_ids
    ], dim=1)
    
    # Forward pass
    outputs = model(inputs_embeds=full_embeds, attention_mask=attention_mask, labels=labels)
    return outputs.loss
```

### ရှင်းလင်းချက်:

**c01 နဲ့ အဓိက ခြားနားချက်**:
```python
# c01:
prompt_embeds = soft_prompt(batch_size)

# c02 (P-Tuning):
prompt_embeds = prompt_encoder(batch_size, model.device)
```

P-Tuning မှာ prompt encoder က:
1. Virtual tokens → Embedding
2. **MLP transformation** 🔥
3. Output prompt embeddings

**သင်ခန်းစာ**: Loss function က c01 နဲ့ တူပေမယ့် prompt generation process က ပိုရှုပ်ထွေးတယ်။ MLP က additional learning capacity ပေးတယ်။

---

## 🏋️ Cell 9: Training Loop

```python
optimizer = torch.optim.AdamW(prompt_encoder.parameters(), lr=LR)

for epoch in range(EPOCHS):
    total_loss = 0.0
    
    for inp, out in train_data:
        loss = compute_loss(inp, out)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_encoder.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1:03d} | Loss: {total_loss:.4f}")
```

### ရှင်းလင်းချက်:
- c01 နဲ့ တူတယ်
- **prompt_encoder.parameters()**: Embedding + MLP parameters အားလုံး optimize လုပ်တယ်
- 50 epochs သာ - P-Tuning က ပို efficient

**Trainable Parameters**:
- Embedding weights: `n_tokens × hidden_size`
- MLP Layer 1: `hidden_size × hidden_size + bias`
- MLP Layer 2: `hidden_size × hidden_size + bias`

**သင်ခန်းစာ**: P-Tuning မှာ parameters အနည်းငယ် ပိုများပေမယ့် (MLP ကြောင့်) ပိုထိရောက်တယ်။ Epochs နည်းသုံးလို့ရတယ်။

---

## 💾 Cell 10: Save Prompt Encoder

```python
torch.save(prompt_encoder.state_dict(), "p_tuning_ros2.pt")
```

### ရှင်းလင်းချက်:
- Embedding + MLP အားလုံး သိမ်းတယ်
- File size က c01 ထက် အနည်းငယ် ပိုကြီးတယ် (MLP parameters ကြောင့်)

---

## 🔮 Cell 11: Inference Function

```python
def infer_ros2_command(human_input):
    input_ids = tokenizer(human_input, return_tensors="pt").input_ids.to(model.device)
    input_embeds = model.get_input_embeddings()(input_ids).to(DTYPE)
    
    prompt_embeds = prompt_encoder(1, model.device)
    full_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)
    
    with torch.no_grad():
        output_ids = model.generate(
            inputs_embeds=full_embeds,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

### ရှင်းလင်းချက်:
- c01 နဲ့ တူပေမယ့် **prompt_encoder** သုံးတယ်
- MLP transformation အလိုအလျောက် ဖြစ်တယ်
- **with torch.no_grad()**: Gradients မတွက်ဘူး (inference mode)

---

## 🧪 Cell 12: Testing

```python
tests = [
    "Move forward 2 meters",
    "Turn left 90 degrees",
    "Navigate to waypoint A"
]

for t in tests:
    print("Input :", t)
    print("Output:", infer_ros2_command(t))
    print("-" * 80)
```

### ရှင်းလင်းချက်:
- Training data မှာရှိတဲ့ examples တွေကို test လုပ်တယ်
- Model က မှတ်မိလား စစ်တယ်

---

## 🔍 P-Tuning v2 vs Soft Prompt Tuning နှိုင်းယှဉ်ချက်

| Feature | Soft Prompt (c01) | P-Tuning v2 (c02) |
|---------|------------------|-------------------|
| **Prompt Generation** | Direct embedding lookup | Embedding → MLP → Output |
| **Learnable Components** | Embedding only | Embedding + MLP |
| **Parameters** | နည်းတယ် | အနည်းငယ် များတယ် |
| **Expressiveness** | Simple | ပိုရှုပ်ထွေးတယ် |
| **Training Speed** | မြန်တယ် | နည်းနည်း နှေးတယ် |
| **Performance** | ကောင်းတယ် | ပိုကောင်းတယ် (ပုံမှန်) |
| **Initialization** | Model embeddings ကပုံ | Random + normal distribution |
| **Best For** | Simple tasks | Complex tasks |

---

## 🎓 အဓိက သင်ခန်းစာများ

### ✅ P-Tuning v2 ရဲ့ အားသာချက်များ:

1. **MLP Network**: Non-linear transformations ကြောင့် ပိုရှုပ်ထွေးတဲ့ patterns သင်ယူနိုင်တယ်
2. **Better Generalization**: Unseen inputs တွေအတွက် ပိုကောင်းတယ်
3. **Independent Embeddings**: Model embeddings ပေါ် မမှီခိုဘူး
4. **Fewer Epochs**: ပို efficient learning

### 📌 Technical Insights:

**MLP က ဘာကြောင့် အရေးကြီးသလဲ?**
- Simple embedding lookup က **linear transformation** ပဲ
- MLP က **non-linear transformation** ပေးတယ်
- Input space ကနေ ပိုကောင်းတဲ့ representation space ကို map လုပ်နိုင်တယ်

**Tanh Activation**:
- Smooth gradients → stable training
- Bounded output [-1, 1] → prevents explosion
- Symmetric around zero → balanced learning

### 💡 Best Practices:

1. **Hidden Size**: Model နဲ့ match လုပ်ဖို့ အရေးကြီးတယ်
2. **MLP Depth**: 2-layer က အများဆုံး သုံးတယ်။ ပိုများရင် overfitting ဖြစ်နိုင်တယ်
3. **Initialization**: Normal distribution (std=0.02) က stable training ပေးတယ်
4. **Learning Rate**: 1e-4 က good starting point
5. **Epochs**: 50-100 က ပုံမှန် လုံလောက်တယ်

### 🔧 Troubleshooting:

**Loss မကျဘူး?**
- Learning rate လျှော့ကြည့်ပါ (1e-5)
- MLP initialization ပြန်စစ်ပါ
- Data quality စစ်ပါ

**Memory error?**
- bfloat16 သုံးပါ
- Batch size လျှော့ပါ
- Prompt tokens အရေအတွက် လျှော့ပါ

**Inference slow?**
- Max tokens လျှော့ပါ
- do_sample=False သုံးပါ (deterministic)

---

## 🤔 စဉ်းစားစရာ မေးခွန်းများ:

1. MLP layers ထပ်ထည့်ရင် performance ပိုကောင်းမလား?
2. Tanh အစား ReLU သုံးရင် ဘာဖြစ်မလဲ?
3. Virtual tokens အရေအတွက် ဘယ်လောက် သင့်တော်သလဲ?
4. Production မှာ inference speed ပိုမြန်အောင် ဘယ်လို optimize လုပ်မလဲ?

**စမ်းသပ်ကြည့်ပါ!** 🚀 P-Tuning v2 က research paper မှာ အထောက်အထားပြထားတဲ့ ထိရောက်တဲ့ နည်းလမ်းဖြစ်ပါတယ်။
