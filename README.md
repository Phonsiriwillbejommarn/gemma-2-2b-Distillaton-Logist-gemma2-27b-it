# Knowledge Distillation: gemma-2-27b-it → gemma-2-2b

ถ่ายทอดความรู้จาก **gemma-2-27b-it** (Teacher) สู่ **gemma-2-2b** (Student) ผ่าน 2 ขั้นตอน

```
Phase 1: SFT (Full Fine-Tuning)          Phase 2: Logit Distillation
┌─────────────────────────┐          ┌──────────────────────────────┐
│ gemma-2-2b (Base)       │          │ Teacher: gemma-2-27b-it (4-bit)│
│ + Opus Reasoning 3K     │  ──────► │ Student: SFT checkpoint      │
│ → sft_output/           │          │ + MATH 12.5K                 │
│ ~30-45 min (H100)       │          │ → distill_output/            │
└─────────────────────────┘          │ ~3-4 hrs (H100)              │
                                     └──────────────────────────────┘
```

---

## � ติดตั้ง

```bash
git clone https://github.com/Phonsiriwillbejommarn/Distillation.git
cd Distillation
pip install -r requirements.txt
```
pip uninstall -y wandb
pip install wandb --upgrade

---

## 🔑 ตั้งค่า API Keys

แก้ไขใน **ทั้ง `sft_gemma.py` และ `distill_gemma.py`**:

```python
MY_WANDB_KEY = "ใส่_wandb_key_จริง"   # https://wandb.ai/authorize
MY_HF_TOKEN  = "ใส่_hf_token_จริง"    # https://huggingface.co/settings/tokens (Write access)
```

---

## 🚀 Phase 1: SFT (Supervised Fine-Tuning)

สอน gemma-2-2b base ให้ทำ reasoning ด้วย [Opus Reasoning dataset](https://huggingface.co/datasets/nohurry/Opus-4.6-Reasoning-3000x-filtered)

```bash
python sft_gemma.py --config distill_config.yaml
```

| รายละเอียด | ค่า |
|-----------|-----|
| โมเดล | `gemma/gemma-2-2b` (Base, Full Fine-Tuning) |
| Dataset | `nohurry/Opus-4.6-Reasoning-3000x-filtered` (3K ข้อ) |
| Max tokens | 8192 |
| Batch size | 4 × 4 = 16 (effective) |
| เวลา (H100) | ~30-45 นาที |
| Output | `./sft_output/` + HF: `Phonsiri/gemma-2-2b-SFT-Reasoning` |

---

## 🧠 Phase 2: Knowledge Distillation

ถ่ายทอดจาก Teacher 32B สู่ Student (SFT checkpoint) ด้วย [MATH dataset](https://huggingface.co/datasets/rasbt/math_full_minus_math500)

```bash
python distill_gemma.py \
    --student_model ./sft_output \
    --config distill_config.yaml
```
# โหลด SFT Model ไว้ที่ folder ./sft_output
python -c "from huggingface_hub import snapshot_download; snapshot_download('Phonsiri/gemma-2-2b-Distilled', local_dir='./sft_output')"

# โหลด Checkpoint การ Distill ล่าสุด ไว้ที่ folder ./distill_output
python -c "from huggingface_hub import snapshot_download; snapshot_download('Phonsiri/gemma-2-2b-Math-Distilled', local_dir='./distill_output')"
# รันต่อจากเช็คพ้อยเดิม
python distill_gemma.py \
    --student_model ./sft_output \
    --config distill_config.yaml \
    --resume_from_checkpoint auto

| รายละเอียด | ค่า |
|-----------|-----|
| Teacher | `gemma/gemma-2-27b-it` (4-bit quantized, frozen) |
| Student | `./sft_output` (Full Fine-Tuning) |
| Dataset | `rasbt/math_full_minus_math500` (12.5K ข้อ) |
| Loss | `α × KL(teacher ∥ student) × T² + (1-α) × CE` |
| Alpha | 0.5, Temperature: 2.0 |
| Checkpoint | เซฟทุก 100 steps → push ไป HF Hub |
| เวลา (H100) | ~3-4 ชั่วโมง |
| Output | `./distill_output/` + HF: `Phonsiri/gemma-2-2b-Math-Distilled` |

---

## ⏸️ Resume จาก Checkpoint

ถ้า GPU หลุดกลางคัน หรือต้องการรันต่อจากเมื่อวาน:

**วิธีที่ 1: ดึงจากโฟลเดอร์รันล่าสุด (ง่ายที่สุด)**
```bash
python distill_gemma.py \
    --student_model ./sft_output \
    --config distill_config.yaml \
    --resume_from_checkpoint auto
```
ระบบจะเข้าไปหาโฟลเดอร์ล่าสุดใน `./distill_output` และทำต่อให้อัตโนมัติ

**วิธีที่ 2: ระบุโฟลเดอร์เอง (กรณีโหลดมาจาก HuggingFace)**
ถ้าย้ายเครื่อง แนะนำให้โหลดโฟลเดอร์ checkpoint มาไว้ในเครื่อง แล้วระบุ path ตรงๆ:
```bash
python distill_gemma.py \
    --student_model ./sft_output \
    --config distill_config.yaml \
    --resume_from_checkpoint ./distill_output/last-checkpoint
```

🚨 *Checkpoint ทุกอันจะทยอยถูก Push ขึ้น Hugging Face Model Hub ของคุณอัตโนมัติหากตั้ง `push_to_hub: true`*

---

## 🔍 ใช้งานโมเดลหลังเทรน

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Phonsiri/gemma-2-2b-Math-Distilled")
tokenizer = AutoTokenizer.from_pretrained("Phonsiri/gemma-2-2b-Math-Distilled")

messages = [{"role": "user", "content": "What is the sum of 1+2+3+...+100?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

# กำหนดจุดป้ายบอกทางให้หยุด (สำคัญมากสำหรับ Base model ที่เปลี่ยนมาใช้ ChatML)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|im_end|>")
]

outputs = model.generate(
    **inputs, 
    max_new_tokens=2048,
    eos_token_id=terminators
)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```

โมเดลจะตอบในรูปแบบ:
```
<|im_start|>assistant
<reasoning>
[กระบวนการคิด reasoning]
</reasoning>

[คำตอบสุดท้าย]
<|im_end|>
```

---

## ⚙️ CLI Overrides

ทุกค่าใน `distill_config.yaml` สามารถ override ผ่าน CLI:

```bash
python distill_gemma.py --alpha 0.7 --temperature 3.0 --learning_rate 1e-5
python sft_gemma.py --max_seq_length 4096 --num_train_epochs 1
```

---

## 📁 โครงสร้างไฟล์

```
├── sft_gemma.py               # Phase 1: SFT
├── distill_gemma.py           # Phase 2: Logit Distillation
├── distill_config.yaml        # Configuration ทั้งหมด
├── generate_teacher_data.py   # (Optional) สร้าง teacher responses
├── requirements.txt           # Dependencies
└── README.md
```

---

## 💻 GPU Requirements

| GPU | SFT | Distillation | VRAM ใช้ |
|-----|-----|-------------|---------|
| **H100 85GB** | ~30 min | ~3-4 hrs | ~60 GB |
| A100 80GB | ~1.5 hrs | ~8 hrs | ~55 GB |
| A100 40GB | ~2 hrs | ~12 hrs | ~38 GB |

> VRAM ขั้นต่ำ: ~38 GB (Teacher 32B 4-bit + Student 3B Full)

---
