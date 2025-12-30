# 📚 c01-prompt-tuning.ipynb - အသေးစိတ် သင်ကြားချက် (မြန်မာဘာသာ)

## 🎯 ဒီ Notebook က ဘာလုပ်တာလဲ?

ဒီ notebook က **Soft Prompt Tuning** နည်းလမ်းကို သုံးပြီး ROS 2 (Robot Operating System) အတွက် command generation လုပ်နိုင်အောင် AI model ကို သင်ကြားပေးတာပါ။ 

---

## 📖 Cell 1: Library များ Import လုပ်ခြင်း

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
```

### ရှင်းလင်းချက်:
- **torch**: PyTorch library ပါ။ Deep Learning အတွက် အဓိက tool ပါ
- **torch.nn**: Neural Network layers တွေ တည်ဆောက်ဖို့ module ပါ
- **AutoModelForCausalLM**: Pre-trained language model ကို အလိုအလျောက် load လုပ်ပေးပီးတော့ text generation လုပ်ပေးတဲ့ factory class ပါ
- **AutoTokenizer**: Text ကို tokens (numbers) အဖြစ် ပြောင်းပေးတဲ့ tool ပါ

**သင်ခန်းစာ**: AI model တွေ အလုပ်လုပ်ဖို့ အရင်ဆုံး လိုအပ်တဲ့ tools တွေကို import လုပ်ရပါတယ်။

---

## ⚙️ Cell 2: Configuration (ပြင်ဆင်သတ်မှတ်ချက်များ)

```python
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_PROMPT_TOKENS = 20
LR = 1e-4
EPOCHS = 150
MAX_NEW_TOKENS = 64
```

### ရှင်းလင်းချက်:
- **MODEL_NAME**: သုံးမယ့် AI model ရဲ့ နာမည်။ Qwen2.5-Coder က code generation အတွက် အထူးပြုလုပ်ထားတဲ့ model ပါ
- **DEVICE**: GPU ရှိရင် "cuda" သုံးမယ်၊ မရှိရင် CPU သုံးမယ်။ GPU က training မြန်ပါတယ်
- **N_PROMPT_TOKENS**: Soft prompt အတွက် token ၂၀ လုံး သုံးမယ်။ ဒါက learnable parameters တွေပါ
- **LR** (Learning Rate): ၀.၀၀၀၁ - model ဘယ်လောက် မြန်မြန် သင်ယူမလဲ ဆိုတာ ထိန်းချုပ်တာပါ
- **EPOCHS**: ၁၅၀ ကြိမ် data ကို ထပ်ခါထပ်ခါ သင်ကြားမယ်
- **MAX_NEW_TOKENS**: Output မှာ အများဆုံး ၆၄ token ထုတ်ပေးမယ်

**သင်ခန်းစာ**: Training မစခင် settings တွေကို သေချာ သတ်မှတ်ရပါတယ်။ Learning rate က အရေးကြီးဆုံးပါ - များလွန်းရင် model ပျက်စီးနိုင်၊ နည်းလွန်းရင် သင်ယူဖို့ အချိန်ကြာပါတယ်။

---

## 📊 Cell 3: Training Data (သင်ကြားစရာ အချက်အလက်များ)

```python
train_data = [
    ("Move forward 2 meters", "ros2 topic pub /cmd_vel ..."),
    ("Turn left 90 degrees", "ros2 service call /rotate_robot ..."),
    ("Navigate to waypoint A", "ros2 action send_goal ...")
]
```

### ရှင်းလင်းချက်:
ဒါက **input-output pairs** ပါ။ Model ကို သင်ကြားဖို့ ဥပမာတွေပါ:
- **Input**: လူသုံး ဘာသာစကား (e.g., "Move forward 2 meters")
- **Output**: ROS 2 command (technical format)

**သင်ခန်းစာ**: AI က ဥပမာတွေကို ကြည့်ပြီး pattern ရှာတယ်။ Data များလေ၊ ပိုကောင်းလေပါ။ ဒီမှာ ၃ ခုပဲ ရှိတာက demonstration အတွက်ပါ။

---

## 🤖 Cell 4: Model & Tokenizer Load လုပ်ခြင်း

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(...)
for p in model.parameters():
    p.requires_grad = False
model.eval()
```

### ရှင်းလင်းချက်:
1. **Tokenizer load**: Text ကို numbers အဖြစ် ပြောင်းဖို့
2. **Model load**: Pre-trained model ကို internet ကနေ download လုပ်တယ်
3. **Freeze parameters**: Model ရဲ့ original weights တွေကို **မပြောင်းစေဘူး**
4. **eval() mode**: Training mode မဟုတ်ဘဲ evaluation mode ထားတယ်

**သင်ခန်းစာ**: Soft Prompt Tuning က model တစ်ခုလုံးကို ပြန်မသင်ဘူး။ Prompt embeddings (သေးသေးလေးတွေ) ကိုပဲ သင်တယ်။ ဒါက memory သက်သာစေပြီး မြန်ပါတယ်။

---

## 🎨 Cell 5: Soft Prompt Module (အဓိက အစိတ်အပိုင်း)

```python
class SoftPrompt(nn.Module):
    def __init__(self, n_tokens, embedding_layer):
        super().__init__()
        init_prompt = embedding_layer.weight[:n_tokens].detach().clone()
        self.prompt_embeddings = nn.Parameter(init_prompt)
```

### ရှင်းလင်းချက်:
- **SoftPrompt**: Custom neural network class တစ်ခု
- **init_prompt**: Model ရဲ့ embedding table ကနေ ပထမဆုံး ၂၀ tokens တွေကို ကူးယူတယ်
- **nn.Parameter**: ဒါက learnable parameters ပါ - training အတွင်း ပြောင်းလဲမှာပါ
- **forward()**: Batch size လိုက် prompt embeddings ကို repeat လုပ်ပေးတယ်

**သင်ခန်းစာ**: Soft prompt က "virtual words" လိုမျိုးပါ။ စစ်မှန်တဲ့ words မဟုတ်ဘဲ number vectors တွေပါ။ Model က ဒီ vectors တွေကို သင်ယူပြီး task ကို ပိုကောင်းအောင် လုပ်ပါတယ်။

---

## 🔧 Cell 6: Soft Prompt Initialize လုပ်ခြင်း

```python
embedding_layer = model.get_input_embeddings()
soft_prompt = SoftPrompt(N_PROMPT_TOKENS, embedding_layer).to(...)
```

### ရှင်းလင်းချက်:
- Model ရဲ့ embedding layer ကို ယူတယ်
- SoftPrompt object တစ်ခု ဖန်တီးတယ်
- GPU/CPU ပေါ် ရောက်အောင် ပြောင်းတယ် (.to())

**သင်ခန်းစာ**: Object ဖန်တီးပြီးရင် သုံးလို့ ရပြီ။ ဒါက OOP (Object-Oriented Programming) style ပါ။

---

## 📉 Cell 7: Loss Function (အမှားတွက်ချက်ခြင်း)

```python
def compute_loss(input_text, target_text):
    # 1. Tokenize input & target
    # 2. Get embeddings
    # 3. Concat soft prompt + tokens
    # 4. Create attention mask & labels
    # 5. Forward pass → compute loss
```

### ရှင်းလင်းချက်:
**အဆင့် ၁**: Text တွေကို token IDs အဖြစ် ပြောင်းတယ်
**အဆင့် ၂**: Token IDs တွေကို embeddings (vector representations) အဖြစ် ပြောင်းတယ်
**အဆင့် ၃**: Soft prompt + input + target ကို ဆက်တယ်
**အဆင့် ၄**: Labels တွေ ပြင်ဆင်တယ် - prompt နဲ့ input ပိုင်းက -100 (ignore), target ပိုင်းကိုပဲ loss တွက်တယ်
**အဆင့် ၅**: Model ကို run ပြီး loss value ပြန်ရတယ်

**သင်ခန်းစာ**: Loss က model ဘယ်လောက် မှားသလဲ ဆိုတာ ပြတဲ့ number ပါ။ Training ရဲ့ ပန်းတိုင်က loss ကို လျှော့ချဖို့ပါ။

---

## 🏋️ Cell 8: Training Loop (သင်ကြားခြင်း လုပ်ငန်းစဉ်)

```python
optimizer = torch.optim.AdamW(soft_prompt.parameters(), lr=LR)
for epoch in range(EPOCHS):
    for inp, out in train_data:
        loss = compute_loss(inp, out)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(...)
        optimizer.step()
```

### ရှင်းလင်းချက်:
**Optimizer**: Parameters တွေကို update လုပ်တဲ့ algorithm (AdamW က လူကြိုက်များတဲ့ optimizer)

**Training လုပ်ငန်းစဉ်**:
1. **Forward pass**: Loss တွက်တယ်
2. **zero_grad()**: ယခင် gradients တွေ ရှင်းတယ်
3. **backward()**: Gradients တွေ တွက်တယ် (backpropagation)
4. **clip_grad_norm_**: Gradient explosion မဖြစ်အောင် ကန့်သတ်တယ်
5. **optimizer.step()**: Parameters တွေကို update လုပ်တယ်

**သင်ခန်းစာ**: Training က cycle ဖြစ်တယ် - loss တွက် → gradients တွက် → parameters ပြင် → ထပ်စပါ။ ၁၅၀ epochs ဆိုတာ ဒီ cycle ကို ၁၅၀ ကြိမ် ထပ်လုပ်တာပါ။

---

## 💾 Cell 9: Save Model (မော်ဒယ် သိမ်းဆည်းခြင်း)

```python
torch.save(soft_prompt.state_dict(), "soft_prompt_ros2.pt")
```

### ရှင်းလင်းချက်:
- **state_dict()**: Learnable parameters တွေ အားလုံးကို dictionary ပုံစံနဲ့ ယူတယ်
- **torch.save()**: File အဖြစ် သိမ်းတယ်
- **.pt extension**: PyTorch model file ကို ညွှန်းတယ်

**သင်ခန်းစာ**: Training ပြီးရင် results တွေ သိမ်းဖို့ မမေ့ပါနဲ့။ နောက်မှ load ပြန်လုပ်လို့ ရပါတယ်။

---

## 🔮 Cell 10: Inference Function (အသုံးပြုခြင်း)

```python
def infer_ros2_command(human_input):
    # 1. Tokenize input
    # 2. Get embeddings
    # 3. Add soft prompt
    # 4. Generate output
```

### ရှင်းလင်းချက်:
1. လူ့ဘာသာစကား input ကို tokens အဖြစ် ပြောင်းတယ်
2. Soft prompt embeddings ကို ရှေ့မှာ ထည့်တယ်
3. **model.generate()**: Auto-regressive generation - token တစ်ခုစီ ထုတ်တယ်
4. Tokens တွေကို text ပြန်ပြောင်းတယ်

**သင်ခန်းစာ**: Inference က training ထက် ရိုးရှင်းတယ်။ Gradients မလိုဘူး၊ output ထုတ်ဖို့ပဲ လိုတယ်။

---

## 🧪 Cell 11: Testing (စမ်းသပ်ခြင်း)

```python
tests = ["Move forward 9 meters", "Turn left 90 degrees", ...]
for t in tests:
    print(infer_ros2_command(t))
```

### ရှင်းလင်းချက်:
- Test cases ၃ ခု run တယ်
- တစ်ခုချင်းစီအတွက် ROS 2 command ထုတ်ပေးတယ်
- Results တွေ နှိုင်းယှဉ်ကြည့်တယ်

**သင်ခန်းစာ**: Testing က အရေးကြီးပါတယ်။ Model က unseen inputs တွေအတွက် ဘယ်လို အလုပ်လုပ်သလဲ ကြည့်ရတယ်။

---

## 🎓 အဓိက သင်ခန်းစာများ

### ✅ Soft Prompt Tuning ရဲ့ အားသာချက်များ:
1. **Parameter efficient**: Model တစ်ခုလုံးကို မပြင်ဘူး
2. **Fast training**: Learnable parameters နည်းတယ်
3. **Memory efficient**: GPU memory သက်သာတယ်
4. **Multiple tasks**: Prompt တစ်ခုချင်းစီကို task တစ်ခုချင်းစီအတွက် သိမ်းလို့ရတယ်

### 📌 အရေးကြီးတဲ့ Concepts:
- **Embeddings**: Words/tokens တွေကို dense vectors အဖြစ် represent လုပ်ခြင်း
- **Freezing**: Model parameters တွေကို မပြောင်းလဲစေခြင်း
- **Backpropagation**: Gradients တွေ တွက်ပြီး parameters update လုပ်ခြင်း
- **Loss function**: Model performance ကို measure လုပ်ခြင်း

### 💡 Tips & Tricks:
- Learning rate ကို သေးသေး စတင်ပါ (1e-4 to 1e-5)
- Prompt tokens အရေအတွက် စမ်းကြည့်ပါ (10-100 range)
- Training loss ကို monitor လုပ်ပါ - မကျဘူးဆိုရင် learning rate လျှော့ပါ
- GPU memory မလုံလောက်ရင် batch size လျှော့ပါ

---

## 🤔 စဉ်းစားစရာ မေးခွန်းများ:

1. Soft prompt tokens အရေအတွက် ပိုများရင် performance ပိုကောင်းမလား?
2. ဘာကြောင့် model parameters တွေကို freeze လုပ်ရတာလဲ?
3. Loss value ဘယ်လောက်ထိ ကျသင့်သလဲ?
4. Production မှာ သုံးဖို့ ဘာထပ်လုပ်ရမလဲ?

**အဖြေများ**: Notebook ကို လက်တွေ့ run ကြည့်ပြီး စမ်းသပ်ကြည့်ပါ! 🚀
