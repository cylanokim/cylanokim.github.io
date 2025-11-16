--- 
title: "AdamW로 가는 길 (2) : 그럼 AdmaW는 뭘까?"
description: Transformer 기반 모델을 학습 하는 프로젝트를 진행 하면서 AdamW에 대하여 자세한 공부가 필요하였다. 본 포스팅은 AdamW에 대하여 자세하게 이해하기 위한 포스팅이며, 총 2화로 진행될 예정이다. 
author: cylanokim
date: 2025-11-15 12:00:00 +0800
categories: [Principle]
tags: [ Adam, AdamW]
pin: true
math: true
mermaid: true
---

지난 포스팅에서 이어 Adma가 왜 weight decay 관점에서 문제가 있는지, 그리고 도대체 AdamW는 무엇인지 이어서 정리해보겠습니다. 딥러닝 처음 배울 때 부터 대부분의 강의에서 "optimizer는 Adam 그냥 쓰세요" 라는 말을 들을 만큼 강력한 툴이 었습니다. 그러나 L2 regularization이 적용된 Adam optimizer는 파라미터마다 크기다 다른 1차/2차 모멘트(m,v)를 사용해 가각 다른 크기의 학습률을 적용하기 때문에 파라미터 마다 감소 비율이 달라집니다. AdamW는 이 모든 사단의 원인이 바로 gradient에 regularization이 들어가서 임을 지적하며 고안됬습니다. 그리고 gradient는 본래의 역할 그대로의 역할만 하도록 설계되었습니다.

## 1. Gradinet 계산(No regularization!)

$$ 
g_t = \nabla_\theta \mathcal{L}_{\text{data}} 
$$

## 2. Decoupled Weight Decay 진행

$$
\theta_t \leftarrow \theta_t - \eta \lambda \theta_t
$$

## 3. Adam과 동일한 방법으로 Optimizer 진행

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t},
\qquad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

$$
\theta_{t+1}
= \theta_t
- \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

AdamW 옵티마이저는 Adam에서 gradient 계산시 진행하던 L2 regularization을 분리(decouple) 하고 따로 계산하는 방식으로 weight가 크면 클수록 더 감쇠하는 weight decay의 본질적인 목적을 위하여 고안된 방식입니다. AdamW는 여러 실험에서 일반화 성능이 더 좋고, 훈련 안전성이 크게 증가하였는데 Transformer 기반 모델에서 그 효과가 크다고 합니다. 실제로 현재 개발중인 Transformer 모델에서 테스트 해본 결과 test loss, val loss 모두 AdamW를 사용한 결과 낮았으며 훨씬 더 빠른 epoch에서 최적의 모델이 수렴한 것도 확인하였습니다.

사실 AdamW 공부 시작하면서 복잡한 수식에 많이 당황했었는데, 하나 하나 공부해 가면서 optimizer의 과정을 더 깊게 공부할 수 있는 기회였던 것 같습니다. 무엇보다 당연하게 생각하던 optimizer 하나를 천착하며 그 문제점을 파고든 사람들이 있음에 많이 놀랐고, 의외로 간단한 해결책으로 문제점을 해결한다는 것에도 놀랐던 시간이었네요. 


참고 자료
- https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
- https://hiddenbeginner.github.io/deeplearning/paperreview/2019/12/29/paper_review_AdamW.html
- I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” arXiv preprint arXiv:1711.05101, 2017.
https://arxiv.org/abs/1711.05101