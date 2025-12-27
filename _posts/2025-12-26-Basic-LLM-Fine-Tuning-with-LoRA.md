--- 
title: "LoRA를 이용한 LLM Fine Tunning"
description: LLM을 이용한 간단하게 파인 튜닝을 해보자.(PEFT) 
author: cylanokim
date: 2025-12-26 12:00:00 +0800
categories: [LLM]
tags: [LLM, LoRA, FineTuning]
pin: true
math: true
mermaid: true
---


## Step 0. 패키지 설치
```python
!pip install transformers==4.42.4
!pip install datasets
!pip install git+https://github.com/huggingface/peft.git@679bcd8777fxxxxxxxxx
```

```python
import datasets, torch
```

- transformers : Hugging Face에서 만든 라이브러리. 
- datasets : 대규모 데이터 셋을 빠르고, 메모리 효율적으로 다루기 위한 표준 도구
- huggingface/peft.git@6... : Huggingface/peft의 특정 커밋 버전으로 직접 설치
- @67xxx : 깃 커밋 해시 → 재현성 확보 

## Step 1. Load AutoModel (OPT-2.7B)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "facebook/opt-2.7b"
# Model Load
model_opt = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto", cache_dir="/home/ms/hf_cache/hub")
# Tokenizer Load
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/home/ms/hf_cache")
```

- 모델 정보 : https://huggingface.co/facebook/opt-2.7b
- OPT : Open Pretrained Transformer. Meta가 공개한 모델. 27억개. Decoder only
- 모델 클래스 비교

| 모델   | 클래스                     | 특징              |
| ---- | ----------------------- | --------------- |
| BERT | `AutoModelForMaskedLM`  | 양방향, [MASK] 예측  |
| T5   | `AutoModelForSeq2SeqLM` | Encoder-Decoder |
| GPT  | `AutoModelForCausalLM`  | 단방향, 생성         |

- device_map :모델의 각 파라미터를 어떤 디바이스(GPU/CPU)에 올릴지 정하는 배치 지도 
    - device_map="auto" : GPU VRAM 용량과 모델 파라미터 크기를 고려해 레이어 단위로 최적 배치를 진행한다. 
    - device_map="cuda" : GPU가 넉넉할때

```text
embed_tokens        → cuda:0
decoder.layers.0-15 → cuda:0
decoder.layers.16-23→ cpu
lm_head             → cuda:0
```

- cache_dir : Hugging Face가 다운로드한 모델 / 토크나이저 파일을 저장하는 로컬 경로

### 참고 함수
```python
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
```

## Step 2. LoRA Implementation

```python
from peft import LoraConfig, get_peft_model
# Setting for LoRA PEFT (fine-tuning QKV projection weight)
config = LoraConfig(
    r=4, # LoRA rank [2,4,8,16,64,...]
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"], # target modules ["fc1", "fc2", "q_proj", "k_proj", "v_proj"]
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to OPT Pre-Trained Model
model_opt.gradient_checkpointing_enable()
model_opt = get_peft_model(model_opt, config)
print_trainable_parameters(model_opt)
```

```text
trainable params: 1966080 || all params: 2653562880 || trainable%: 0.07409208256636451
```

## Step 3. Pre-Process & Tokenize Instruction Dataset

```python
!pip uninstall -y datasets
!pip install "datasets<4.0.0"
```


```python
from datasets import load_dataset

data = load_dataset("piqa", split="train[:10%]").select(range(100))
print(data.column_names)
```

```text
['goal', 'sol1', 'sol2', 'label']
```

- PIQA : Physical Interation Question Answering 데이터 셋

| column  | 의미                  |
| ------- | ------------------- |
| `goal`  | 사람이 하고 싶은 행동 설명     |
| `sol1`  | 첫 번째 가능한 해결 방법      |
| `sol2`  | 두 번째 가능한 해결 방법      |
| `label` | 정답 인덱스 (`0` 또는 `1`) |

```text
label = 0  → sol1 이 정답
label = 1  → sol2 이 정답
```

- 데이터 구조

<div style="text-align: center;">
  <img src="/assets/img/piqa_dataset.PNG" alt="piqa" width="500">
</div>


### Pre-Processing (concat goal and solution)

```python
def add_sol_with_label(example):
  sentence = example[column_names[0]] + " "
  answer = example[column_names[1]] if example["label"] == 0 else example[column_names[2]]

  example["sentence"] = sentence + answer
  return example
```

```python
# Pre-Processing PIQA train dataset
updated_data = data.map(add_sol_with_label)
updated_data = updated_data.remove_columns("goal")
updated_data = updated_data.remove_columns("label")
updated_data = updated_data.rename_column("sentence", "goal")
data = updated_data

# Tokenize
data = data.map(lambda samples:tokenizer(samples["goal"]), batched=True)
```

<div style="text-align: center;">
  <img src="/assets/img/piqa_dataset2.PNG" alt="piqa" width="500">
</div>

## Step 4. Text Generation Before Fine-Tuning

```python
text = "What is SOH in semiconductor manufacturing process?"
# Set max sequence length
max_token_number = 50

# Tokenize input sequence
inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

# Text Generation (model inference)
with torch.no_grad():
    outputs_opt = model_opt.generate(**inputs, max_tokens=max_token_number)
print(tokenizer.decode(outputs_opt[0], skip_special_tokens=True))
```

```text
What is SOH in semiconductor manufacturing process?
SOH is a term used to describe the amount of oxygen in the atmosphere. It is measured in parts per million (ppm).
SOH is a measure of the amount of oxygen in the atmosphere. It is measured in parts
```

파인 튜닝을 위한 간단한 데이터셋을 만들어보자. 아래 구조는 **instruction tuning**에서 전형적으로 사용되는 데이터셋 형태이다. 여기서 instruction은  모델에게 어떤 역할을 수행하라고 지시하는 상위 명령을 의미한다. 일종의 system prompt 역할을 하는 것으로 모델이 어떤 종류의 질문을 어떻게 답해야 하는지를 학습하게 한다. 이때 Decoder only 모델의 경우 `instruction + input + output`은 하나의 연속된 토큰 시퀀스로 모델에 들어간다. 

```python
data_etch = {
    "instruction": [
        "Tell me about the films used in the semiconductor process",
        "Tell me about the films used in the semiconductor process",
    ],
    "input": [
        "What is SION in semiconductor manufacturing process?",
        "What is SOH in semiconductor manufacturing process?",
    ],
    "output": [
        "In semiconductor manufacturing, SiON refers to silicon oxynitride.",
        "In semiconductor manufacturing, SOH(Spin On Hardmask) is a high carbon-containing polymer material.",
    ],
}
from datasets import Dataset
# Hugging Face Dataset으로 변환
dataset = Dataset.from_dict(data_etch)
```

```python
def preprocess(examples):
    texts = [
        f"Instruction: {inst}\nInput: {inp}\nAnswer: {out}"
        for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"])
    ]
    model_inputs = tokenizer(texts, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
```

## Step 5. HuggingFace Trainer Setting for LLM Fine-Tuning 

```python
import transformers

tokenizer.pad_token = tokenizer.eos_token

# Fine-Tuning Setting data -> tokenized
trainer = transformers.Trainer(
    model=model_opt,
    train_dataset=tokenized,
     args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=100,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=10,
        output_dir="outputs",
        report_to='none'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model_opt.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# Let's Fine-Tuning 🔥🔥🔥
trainer.train()
```
- `tokenizer.pad_token = tokenizer.eos_token`: GPT/LLaMA 계열은 pad_token이 없음. 
- `mlm=False` : GPT 계열, `mlm=True` : BERT 계열(Masked LLM) 

## Step 6. Text Generation after Fine-Tuning 📚📚

```python
text = "What is SOH in semiconductor manufacturing process?"
# Set max sequence length
max_token_number = 50

# Tokenize input sequence
inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

# Text Generation (model inference)
with torch.no_grad():
    outputs_opt = model_opt.generate(**inputs, max_tokens=max_token_number)
print(tokenizer.decode(outputs_opt[0], skip_special_tokens=True))
```

```text
What is SOH in semiconductor manufacturing process?
Spin-On Hardmask. In semiconductor manufacturing, it refers to silicon oxynitride.
```

