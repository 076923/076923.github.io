---
layout: post
title: "Artificial Intelligence Theory : 가중치 초기화(Weight Initialization)"
tagline: "Batch Normalization"
image: /assets/images/ai.jpg
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['AI']
keywords: Artificial Intelligence, Weight Initialization, Scalar Initialization, Random Initialization, Uniform Initialization, Normal Initialization, Distribution Initialization, Xavier Initialization, Glorot initialization, He Initialization, Kaiming Initialization, Truncated Normal Distribution Initialization, Orthogonal Normal Distribution Initialization
ref: Theory-AI
category: Theory
permalink: /posts/AI-8/
comments: true
toc: true
plotly: true
---

## 가중치 초기화(Weight Initialization)

`가중치 초기화(Weight Initialization)`란 가중치의 초깃값을 설정하는 방법을 의미합니다.

`초기 가중치(Initial Weight)`에 따라 `기울기 소실(Vanishing Gradient)` 문제가 최소화되거나 `극솟값(Local Minimum)`에서 수렴되는 현상을 완화할 수 있습니다.

[Python Pytorch 강좌 : 제 5강 - 최적화(Optimization)][Pytorch-5강]의 그래프를 예시로 든다면 더 빠르게 `최솟값(Global Minimum)`에 수렴하고 안정적으로 수렴할 수 있습니다.

즉, 가중치 초기화에 따라 **더 적은 학습 횟수로 동일한 성능의 모델 결과를 얻을 수 있으며, 모델의 성능을 향상시킬 수 있습니다.**

<br>
<br>

## 상수 값으로 초기화

$$ W = a $$

<br>

상수 값으로 초기화하는 방법은 초기 가중치의 값을 **모두 같은 값**으로 초기화 하는 방법입니다.

예를 들어, **0 이나 1 또는 100 등 특정한 상수 값으로 모든 가중치를 동일하게 할당하는 방식입니다.**

대표적으로 0, 1, 특정값(Constant), 단위 행렬(Unit Matrix), 디렉 델타 함수(Dirac delta function)값 등이 있습니다.

모든 값을 같은 값으로 초기화하게 되면 스칼라(Scalar)가 아닌 **배열(Array) 구조의 가중치에서는 문제가 발생합니다.**

[Artificial Intelligence Theory : 순전파(Forward Propagation) & 역전파(Back Propagation)][AI-5강]에서 초기 가중치 값을 모두 0으로 할당한다면, **역전파 과정에서 모든 가중치가 동일하게 갱신됩니다.**

그러므로, `역전파(Back Propagation)`를 통해 가중치가 제대로 갱신되지 않게되어 학습이 정상적으로 진행되지 않습니다.

다만, `편향(Bias)`을 초기화하는 경우에는 0 이나 0.01 등의 형태로 초기화합니다.

<br>
<br>

## 무작위 값으로 초기화

$$ W = \mathcal{U}(a,\ b) $$

$$ W = \mathcal{N}(mean,\ std^2) $$

<br>

무작위 값으로 초기화하는 방법은 초기 가중치의 값을 **무작위 값이나 특정 분포** 형태로 초기화하는 방법입니다.

예를 들어, **정규 분포의 형태로 가중치의 값을 할당하는 방식입니다.**

대표적으로 무작위(Random), 균등 분포(Uniform Distribution), 정규 분포(Normal Distribution) 등이 있습니다.

무작위 값으로 가중치를 초기화하게 되면, `계층(Layer)`이 적거나 하나만 있는 경우에는 학습에 큰 영향을 미치진 않습니다.

다만 계층이 많아지고 깊어질 수록 활성화 값이 양 끝단에 치우치게 되어 `기울기 소실(Vanishing Gradient)` 현상이 발생하거나 상수 값으로 초기화한 형태와 동일한 문제가 발생하게 됩니다.

<br>
<br>

## 제이비어 초기화 & 글로럿 초기화

$$ W = \mathcal{U}(-a,\ a) $$

$$ a = gain \times \sqrt{\frac{6}{fan_{in} + fan_{out}}} $$

<br>

$$ W = \mathcal{N}(0,\ std^2) $$

$$ std = gain \times \sqrt{\frac{2}{fan_{in} + fan_{out}}} $$

<br>

`제이비어 초기화(Xavier Initialization)` 또는 `글로럿 초기화(Glorot initialization)` 방법은 `균등 분포(Uniform Distribution)`나 `정규 분포(Normal Distribution)`를 사용해 초기화합니다.

기존 확률 분포 초기화 방법과 차이점은 동일한 표준 편차를 사용하지 않고 `은닉층(Hidden Layer)`의 노드 수에 따라 다른 표준 편차를 할당합니다.

즉, 이전 계층의 노드 수(fan in)와 다음 계층의 노드 수(fan out)에 따라 표준 편차가 계산됩니다.

제이비어 초기화를 사용하게 되는 경우, **입력 데이터의 분산이 출력 데이터에서 유지되도록 가중치를 초기화**하므로 비선형 활성화(Sigmoid, tanh) 함수 등에서 더 강건(Robust)합니다.

- Tip : $$ gain $$은 스케일 값을 의미합니다.
- Tip : `torch.nn.init._calculate_fan_in_and_fan_out(weight)`를 통해 노드 수를 확인할 수 있습니다.

<br>
<br>

## 허 초기화 & 카이밍 초기화

$$ W = \mathcal{U}(-a,\ a) $$

$$ a = gain \times \sqrt{\frac{3}{fan_{in}}} $$

<br>

$$ W = \mathcal{N}(0,\ std^2) $$

$$ std = \frac{gain}{\sqrt{fan_{in}}} $$

<br>

`허 초기화(He Initialization)` 또는 `카이밍 초기화(Kaiming Initialization)` 방법은 `제이비어 초기화(Xavier Initialization)` 방법과 마찬가지로 `균등 분포(Uniform Distribution)`나 `정규 분포(Normal Distribution)`를 사용해 초기화합니다.

`제이비어 초기화(Xavier Initialization)` 방법은 렐루(ReLU) 함수에서 좋은 결과를 보이지 않아 적용할 수 없으므로 문제점을 보완한 방법입니다.

`허 초기화(He Initialization)` 방법은 다음 계층의 노드 크기를 고려하지 않고 이전 계층의 노드만 고려하여 계산합니다.

또한, **렐루(ReLU) 함수는 0 이하의 값을 모두 0으로 변경하기 때문에 입력값이 음수일 때 분산을 더 크게 주어 분산을 유지해 렐루(ReLU) 함수 등에 더 적합한 형태로 초기화합니다.**

<br>
<br>

## 잘린 정규 분포 초기화

$$ W = \mathcal{N}(mean,\ std^2) \ [a, b] $$

<br>

`잘린 정규 분포 초기화(Truncated Normal Distribution Initialization)`는 `정규 분포(Normal Distribution)`에서 최솟값($$ a $$)보다 작거나 최댓값($$ b $$)보다 큰 값을 제거한 확률 분포 형태입니다.

렐루(ReLU) 함수의 경우 입력값이 0 이하의 값을 모두 0으로 변경하기 때문에 정규 분포의 구조로 초기화할 때 **이러한 영향을 미치는 부분을 제거해서 할당할 수 있습니다.**

또한, 렐루(ReLU) 함수를 사용하는 경우 전체적으로 고른 분포보단 약간 양의 분포로 편향되는 것이 모델 성능에 더 좋은 효과를 낼 수 있습니다.

<br>
<br>

## 직교 초기화

`직교 초기화(Orthogonal Initialization)`는 `특잇값 분해(Singular Value Decomposition, SVD)`를 활용해 자기 자신을 제외한 나머지 모든 열, 행 벡터들과 직교이면서 동시에 단위 벡터인 행렬을 만드는 방법입니다.

`직교 행렬(Orthogonal Matrix)`의 **고윳값의 절댓값은 1이기 때문에 행렬 곱을 여러 번 수행하더라도** `기울기 폭주(Exploding Gradient)`나 `기울기 소실(Vanishing Gradient)`**이 발생하지 않습니다.**

직교 초기화는 주로 `순환 신경망(Recurrent Neural Network, RNN)`에서 주로 사용됩니다.

[AI-5강]: https://076923.github.io/posts/AI-5/
[Pytorch-5강]: https://076923.github.io/posts/Python-pytorch-5/

<br>
<br>

## 희소 정규 분포 초기화

$$ W = \mathcal{N}(0,\ 0.01) $$

<br>

`희소 정규 분포 초기화(Sparse Normal Distribution Initialization)`는 초기 가중치 값을 희소한(Sparse) 형태로 초기화하는 방법입니다.

단, 0이 아닌 값은 정규 분포의 형태로 할당되며, 희소 비율(sparsity)을 설정해 `희소 행렬(Sparse Matrix)`의 형태로 생성합니다.

<br>
<br>

* Writer by : 윤대희
