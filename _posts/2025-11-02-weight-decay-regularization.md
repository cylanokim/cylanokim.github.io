--- 
title: "AdamW로 가는 길 (1) : Regularization이랑 Weight Decay, 같은거야?"
description: Transformer 기반 모델을 학습 하는 프로젝트를 진행 하면서 AdamW에 대하여 자세한 공부가 필요하였다. 본 포스팅은 AdamW에 대하여 자세하게 이해하기 위한 포스팅이며, 총 2화로 진행될 예정이다. 
author: cylanokim
date: 2025-11-08 12:00:00 +0800
categories: [Principle]
tags: [ Adam, AdamW]
pin: true
math: true
mermaid: true
---

Deep Learning 공부하면서 가장 조심해야하는 것이라고 모든 책과 강의에서 강조하는 것이 있죠. **오버피팅!!** ...

제가 배운 강의에서는 오버피팅이란, 데이터의 복잡도 대비 모델의 복잡도가 커서 모델이 학습 데이터의 노이즈까지 외우는 상태를 오버 피팅이라 정의하였습니다. 이를 해결하기 위해 정~~~말 많은 초식들이 개발되어 왔는데, 대표적인 방법이 Weight Decay와 Regularization입니다. 

## 1. Regularization과 Weight Decay 를 간단하게?

| 개념                       | 목적                                              |
|:------------------------ |:-----------------------------------------------:| 
| **Regularization** | weight의 크기에 비례하는 **패널티**를 손실 함수에 포함하는 것         | 
| **Weight Decay**         | 매 스탭 마다 weight를 일정 비율로 조금씩 줄이는 방식 |

## 2. L2 Regularization 수식

**기본 Loss (MSE라 가정)** :

<!-- $$
\mathcal{L}_{\text{data}}(\theta) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 \qquad (1)
$$ -->

**Regularization이 적용된 Loss** :

<!-- $$
\mathcal{L}(\theta) = \mathcal{L}_{\text{data}}(\theta) + \lambda \|\theta\|_2^2  \qquad\text{where}\;  \|\theta\|_2^2 = \sum_{i} \theta_i^2 \qquad (2)
$$ -->

<!-- 여기서 

$$
\|\theta\|_2^2 = \sum_{i} \theta_i^2
$$ -->

**학습을 위하여 Gradient가 진행되면?**

<!-- $$
\nabla_\theta \mathcal{L} = \nabla_\theta \mathcal{L}_{\text{data}} + 2\lambda \theta \qquad (3)
$$ -->

**파라미터 업데이트** : 

<!-- $$
\theta_{t+1}
= \theta_t - \eta\left(\nabla_\theta\mathcal{L}_{\text{data}} + 2\lambda\theta_t\right) \qquad (4)
$$ -->

이를 예쁘게 정리하면

<!-- $$
\theta_{t+1}
= (1 - 2\eta\lambda)\theta_t - \eta\nabla_\theta\mathcal{L}_{\text{data}} \qquad (5)
$$ -->

## 3. Weight Decay 수식 

Weight Decay는 아예 파라미터 업데이트 할때 마다 weight를 일정한 비율로 줄이는 것입니다. 즉 파라미터 업데이트 시 이전 파라미터를 일정한 비율로 살짝 줄인 후 Gradient Descent를 더하는 것입니다.  

<!-- $$
\theta_{t+1}
= (1 - \eta\lambda)\theta_t - \eta\nabla_\theta\mathcal{L}_{\text{data}} \qquad (6)
$$ -->

그런데 자세히 살펴 보면 이전 파라미터 앞에 상수를 곱한 체, Gradient Descent가 더한다는 관점에서, 위 두 수식은 사실상 같습니다. 즉 손실 함수에 weight가 커지는 것을 방지하는 패널티를 넣었더니 결국 Weight Decay와 똑같다는 것입니다. 그러면 Regularization이랑 Weight Decay는 똑같은 것일까요?

## 5. Regularization = Weight Decay?

사실 이거는 맞기도 하고 아니기도 합니다. 이를 자세하게 이해하려면 (3)에서 (4)로 당연한 듯 넘어갔던 파라미터 업데이트 과정을 고민해야 합니다. 업데이트 할 때 학습률 x Gradient를 기존 파라미터에서 빼주는 것은 매우 기본적인 딥러닝 과정입니다. 그러나 학습 과정에서 가장 많이 사용하는 Adam Optimizer는 그렇지 않습니다. Adam optimizer에서 파라미터가 업데이트 되는 과정을 수식으로 보면 다음과 같습니다. 


<!-- $$
m_t = \beta_1 m_{t-1} + (1 - \beta_1)\, (\nabla_\theta \mathcal{L}_{\text{data}} + 2\lambda \theta)
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, (\nabla_\theta \mathcal{L}_{\text{data}} + 2\lambda \theta)^2
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, 
\qquad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

$$
\theta_{t+1} 
= \theta_t 
- \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} \qquad (7)
$$ -->

Adam에서 gradient는 바로 학습률에 직접 곱해져 계산되지 않고 스케일링 과정이 진행되어 일종의 `effective 학습률`이 파라미터 업데이트에 적용됩니다. 즉 L2 Regularization 적용 시, 수식 (5), (6)과는 달리 Adam의 (7)에서는 모든 파라미터 θt​ 가 일정한 비율로 감소하지 않는 것입니다. 이는 오버 피팅을 막기 위해 파라미터가 값이 너무 커지면 그 파라미터의 비율만큼 줄이겠다고 고안된 Regularization의 본래 목적에서 벗어나는 것입니다. 이러한 문제로 고안된 것이 바로! AdamW 이며, 이에 대한 내용은 다음 포스팅에서 글을 이어 가겠습니다.  