--- 
title: "Decoder Only Transformer 찍먹 해보기!"
description: 애국가 가사를 이용한, 정말 정말 간단한 Decoder only Model 찍먹해보기! 
author: cylanokim
date: 2025-09-21 19:00:00 +0800
categories: [Transformer, From Scratch]
tags: [transformer, decoder-only, scratch]
pin: true
math: true
mermaid: true
---

자연어 처리를 공부하다 보면 트렌스포머를 직접 구현하고, 학습까지 하는 과정을 기술한 블로그들을 많이 참고하게 됩니다. 그런 블로그 글을 보고 직접 구현하며, 트렌스포머의 구조와 데이터 처리 과정에 대한 이해를 많이 높일 수 있었지만, 항상 아쉬웠던 것은 많은 블로그들이 도대체 어떻게 생긴 데이터를 학습하는거지에 대한 설명이 부족했다는 것입니다. 보통 시작하면 갑자기 `Hugging Face`에서 제공하는 `datasets` 에서 뭔가를 다운 받고, 데이터가 어떻게 생긴지도 모르는 상황에서 `map` 함수로 전처리 딱 하고 `DataLoader`로 변환하더니 `Trainer` 에서 넣어 학습하더니만, 테스트 결과 몇 % 입니다~! 라고 하는 경우가 많다는 것입니다. 

```python 
from datasets import load_dataset
dataset = load_dataset("imdb")
...
```

그래서 오늘 포스팅에서는 대한민국 사람이라면 누구나 아는! 애국가를 데이터로 하여 Decoder-Only 모델을 간단하게 구현 해보겠습니다. 목적은 간단합니다. 애국가 가사의 한 소절을 입력하면, 아래와 같이 다음 소절을 예측하는 모델입니다. 애국가를 모르시진 않을테니....특별히 테스트 데이터를 만들 필요도 없습니다 :D 

| 입력 | 출력 |
|:-----:|:-----:|
| 동해물과 | 백두산이 |
| 무궁화 삼천리 | 화려강산 |
| 길이 보전하세 | 가을 하늘 공활한데 |


## 1. 데이터 준비
```python
text = """
동해물과 백두산이 마르고 닳도록 하느님이 보우하사 우리나라 만세
무궁화 삼천리 화려강산 대한사람 대한으로 길이 보전하세 남산 위에 저 소나무 철갑을 두른듯
바람 서리 불변함은 우리 기상 일세
무궁화 삼천리 화려강산 대한사람 대한으로 길이 보전하세
가을 하늘 공활한데 높고 구름없이 
밝은달은 우리가슴 일편단심일세
무궁화 삼천리 화려강산 대한사람 대한으로 길이 보전하세
이 기상과 이 맘으로 충성을 다하여
괴로우나 즐거우나 나라사랑하세
무궁화 삼천리 화려강산 대한사람 대한으로 길이 보전하세
"""
```
## 2. 토크나이징 
이번 포스팅에서는 아~~주 간단한 Decoder only 모델이기에, 토크나이징도 위 데이터 셋의 띄어 쓰기를 기준으로 나눠서 토큰으로 만들겠습니다.  
```python
# 1.데이터 토크나이징
vocab = text.split()
# vocab = ['동해물과', '백두산이', '마르고', '닳토록', '하나님이.....

# 2.토크나이징의 encoder(=stoi), decoder(=itos) 를 딕셔너리 형태로 구현
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}
vocab_size = len(vocab)
```

## 3. Dataset 만들기
Pytorch로 딥러닝 공부하면 항상 하는 `Dataset` → `Dataloader` 만들기! 아래와 같이 구성하였습니다. 
```python 
class AnthemDataset(Dataset):
    def __init__(self, vocab, seq_len):
        self.seq_len = seq_len
        self.data = []
        for i in range(len(vocab) - seq_len):
            x = vocab[i:i+seq_len]
            y = vocab[i+1:i+seq_len+1]
            self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x_ids = torch.tensor([stoi[ch] for ch in x], dtype=torch.long)
        y_ids = torch.tensor([stoi[ch] for ch in y], dtype=torch.long)
        return x_ids, y_ids

dataset = AnthemDataset(seq_len=4)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
```

간단하게 dataloader가 어떻게 생겼는지 확인하면....
```python
for xb, yb in dataloader:
    print("x : ", xb)
    print("y : ", yb)
    break

# x :  tensor([[40, 41, 58, 59],
#         [17, 18, 19, 20]])
# y :  tensor([[41, 58, 59, 60],
#         [18, 19, 20, 21]])
```
배치 크기 2로 x와 y가 각각 두 개씩 들어가 있습니다. 보시면 x는 [40, 41, 58, 59], y는 [41, 58, 59, 60]로 각 숫자는 토큰화 된 애국가 데이터의 토큰 번호입니다.

## 4. 모델 구현하기 
이제 디코더로만 이루어진 GPT 스타일 모델을 구현하도록 하겠습니다. 트렌스포머 공부하면서 가장 어려웠던 것은 텐서들이 어떻게 진행되고 변환되는지를 머리 속에서 그려지기까지 상당한 시간이 걸린다는 것이었습니다. 다음에 꼭 구체적으로 텐서의 크기가 어떻게 변환되는지 구체적으로 정리하는 포스팅도 계획중입니다. 아, 그리고 아래 코드에서 어텐션 부분은 간단한 모델을 위해 `torch`에서 제공하는 `nn.TransformerDecoderLayer`를 사용하였습니다. 추후에 어텐션 영역도 구체적으로 살피는 포스팅도 해보겠습니다.

```python
class DecoderOnly(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, max_len=20):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, use_mask=True): #=False
        B, T = x.shape
        token_embeddings = self.token_emb(x)  # (B, T, d_model)
        pos_embeddings = self.pos_emb(torch.arange(T, device=x.device))  # (T, d_model)
        h = token_embeddings + pos_embeddings.unsqueeze(0)

        mask = None
        if use_mask:
            mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)

        out = self.transformer(h, h, tgt_mask=mask)
        logits = self.lm_head(out)  # (B, T, vocab_size)
        return logits
```

그리고 모델을 사용하여 다음 가사를 예측하는 함수를 정의하였습니다. 즉 `["동해물과"]` 를 입력하면 모델이 생각하는 다음 가사를 반환하는 함수입니다. 그럼 아직 훈련이 되지 않은 모델은 `["동해물과"]` 다음 가사를 뭐라고 생각하는지 볼까요?

```python
def predict_next(model, start_seq, max_new_tokens=20):
    model.eval()
    x = torch.tensor([stoi[ch] for ch in start_seq], dtype=torch.long).unsqueeze(0)
    for _ in range(max_new_tokens):
        logits = model(x)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        x = torch.cat([x, torch.tensor([[next_id]])], dim=1)
    return ' '.join([itos[i] for i in x[0].tolist()])

print(predict_next(model, ["동해물과"], max_new_tokens=5))
# 동해물과 하늘 마르고 위에 불변함은 화려강산
```
`동해물과 하늘 마르고 위에.....` 이상한 소리를 하고 있네요 -.- 몇 번씩 실행해보면 동해물과 다음 가사가 계속 바뀌지만, 말도  안되는 소리를 하고 있습니다. 그런데 저는 이 부분이 중요한 것 같습니다. 즉, 입력한 데이터가 잘 텐서로 변환되어 모델을 잘 통과한다는 것이고, 이를 통해 모델의 훈련이 준비되었다는 것입니다.

## 5. 모델 학습하기
학습 ㄱㄱ
```python
for epoch in range(200):
    for xb, yb in dataloader:
        logits = model(xb)

        loss = loss_fn(logits.view(-1, vocab_size), yb.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 50 == 0:
        print(f"epoch {epoch} loss {loss.item():.4f}")

# epoch 0 loss 3.4258
# epoch 50 loss 0.0017
# epoch 100 loss 0.0005
# epoch 150 loss 0.0002
```
뭔가 loss가 줄긴했는데, 이것 만으로 언어 모델을 평가할 수는 없죠. `predict_next` 함수를 사용하여 아까와 동일하게 실험을 해보겠습니다.

```python
print(predict_next(model, ["동해물과"], max_new_tokens=5)) 
# -> 동해물과 백두산이 마르고 닳도록 하느님이 가을
print(predict_next(model, ["길이", "보전하세"], max_new_tokens=5))
# -> 길이 보전하세 남산 위에 저 남산 위에
# -> 길이 보전하세 가을 하늘 공활한데 가을 하늘
```
오~! 이 녀석 드디어 대한민국 귀화 조건 중 하나를 충족했네요 ㅎ 아주 기초적인 모델과 데이터 셋이지만, 적어도 애국가의 가사는 아는 녀석입니다. 

## 마치며 
정말 쉬운 데이터 셋인 애국가를 이용하여 GPT와 같은 Decoder Only 모델을 구현해보았습니다. 무엇보다 데이터 셋이 너무 익숙하고 토크나이징도 아~~주 간단히 해서, Decoder Only 모델이 어떻게 구현되고 학습화는지 간단하게 찍먹해 볼 수 있는 포스팅이었습니다. 앞으로도 AI 공부하면서 왠지 궁굼했던 내용을 좀 얇게, 때론 좀 깊게 파고드는 블로그를 만들도록 노력하겠습니다. 

감사합니다. 