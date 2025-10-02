--- 
title: "EmbeddingBag 간단히 살펴보기"
description: Embedding은 알겠는데 EmbeddingBag은 뭘까?
author: cylanokim
date: 2025-10-02 12:00:00 +0800
categories: [pytorch]
tags: [pytorch, DLRM]
pin: true
math: true
mermaid: true
---

pytorch를 이용하여 코드를 짜다보면 대충 tensor가 이렇게 진행될 것 같은데? 하고 넘어가지만, 자세하게 보면 모르는 것들이 많습니다. 그래서 가끔씩 작은 주제를 자세하게 분석해보는 시간을 갖어 보려고 하는데, 오늘은 `EmbeddingBag`이 무엇인지 정리해보겠습니다.

## 1. 기본 개념
- `nn.Embedding` : 정수 인덱스를 받아서 임베딩 벡터를 반환 (단일 토큰 → 벡터)
- `nn.EmbeddingBag` : 여러 인덱스를 받아 그에 해당하는 임베딩의 통계값 (합, 평균, 최대값)으로 `pooling`하여 벡터를 반환 

## 2. 주요 파라미터 
```python
nn.EmbeddingBag(
    num_embeddings, # 임베딩 사전의 크기, 총 토큰의 개수
    embedding_dim,  # 임베딩 차원의 크기 
    mode="mean"     # {'sum', 'mean', 'max'} 가능 (기본: 'mean')
)
```

## 3. 예시 코드
```python
import torch
import torch.nn as nn 

embedding_bag = nn.EmbeddingBag(num_embeddings=10, embedding_dim=3, mode='mean')

# 두 개의 문장을 표현한다고 가정
# 첫 번째 문장은 [1, 2, 4], 두 번째 문장은 [4, 3]
input = torch.tensor([1, 2, 4, 4, 3])  # 모든 토큰 인덱스 붙여서 나열
offsets = torch.tensor([0, 3])         # 각 문장의 시작 위치

# forward
output = embedding_bag(input, offsets)

print("출력 벡터:", output)
```
**출력** 
```bash
출력 벡터: tensor([
   [0.12, -0.43, 0.51],   # 문장1 ([1,2,4] 평균)
   [-0.33, 0.77, -0.12]   # 문장2 ([4,3] 평균)
])
```
**그림으로 보는 nn.Embeddingbag**

기본적으로 `nn.Embedding`은 **행 뽑기!** 과정입니다. 다만 `nn.EmbeddingBag`은 여러 행을 중복을 허용하여 뽑고 이를 배치 단위로 나눈 후 하나의 벡터로 만들어주는 과정입니다. 아래 그림에서 input 벡터는 임배딩 행렬에서 어떤 행(벡터)를 뽑을지 지정하는 것이고, offset은 마치 **포인터(pointer)** 처럼 어디서 각 배치, 묶음이 시작하는지 알려줍니다. 그리고 이 묶음(bag) 단위로 벡터 연산이 진행되는데 총 합, 평균, 최대값 연산이 가능합니다. 

<p align="left">
  <img src="/assets/img/EmbeddingBag.PNG" alt="EmbeddingBag" width="250">
</p>

## 4. EmbeddingBag이 사용되는 상황
1. 추천 시스템 (DLRM)
    - 관심사나 태그처럼 **multi-hot feature**, 하나의 feature에 여러 값이 동시에 들어가는 경우 하나의 bag으로 처리합니다.
    - `nn.EmbeddingBag`은 이런 multi-hot feature의 embedding을 하나의 벡터로 만들어 줍니다.
    - `DLRM`에서는 이후 dense fature와 concat 하여 클릭 확률을 예측하기도 합니다.
2. 텍스 분류
    - 문장 전체의 단어(토큰)을 sum/mean pooling 해서 하나의 문장 벡터로 만들 수 있습니다.
3. 정보 검색 / Ranking 모델
    - 문서/쿼리 안에 등장하는 단어들의 embedding을 bag 단위로 pooling하여 벡터로 만들기
    - 이후 쿼리 벡터와 문서 벡터의 유사도를 계산하여 정보 검색에 사용하기
