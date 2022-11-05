---
layout: post
title: "Python Pytorch 강좌 : 제 6강 - 단순 선형 회귀(Simple Linear Regression)"
tagline: "Python PyTorch Optimization"
image: /assets/images/pytorch.webp
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['PyTorch']
keywords: Python, Python Pytorch, Pytorch Simple Linear Regression, Pytorch HyperParameter, Pytorch HyperParameter Tuning, Pytorch Epoch, Pytorch Forward Propagation, Pytorch Back Propagation, Pytorch Underfitting, Pytorch Overfitting, Pytorch learning_rate, Pytorch Stochastic Gradient Descent, Pytorch SGD, Pytorch Mini-batch, Pytorch optimizer.zero_grad(), Pytorch cost.backward(), Pytorch optimizer.step()
ref: Python-PyTorch
category: Python
permalink: /posts/Python-pytorch-6/
comments: true
toc: true
---

## 단순 선형 회귀(Simple Linear Regression)

`단순 선형 회귀(Simple Linear Regression)`는 **하나의 독립 변수(Independent Variable)와 하나 이상의 종속 변수(Dependent Variable)** 사이의 선형 상관 관계를 분석하는 방법입니다.

하나의 종속 변수와의 관계를 분석하는 경우에는 `단변량 단일 선형 회귀(Univariate Simple Linear Regression)`가 되며, 두 개 이상의 종속 변수와의 관계를 분석하는 경우에는 `다변량 단일 선형 회귀(Multivariate Simple Linear Regression)`라 합니다.

즉, 종속 변수가 `스칼라(Scalar)` 형태($$ Y $$)를 가지면 `단변량(Univariate)`이 되며, `벡터(Vector)`의 형태($$ [Y_{1}, Y_{2}, ...] $$)를 가지면 `다변량(Multivariate)`이 됩니다.

선형 회귀에 관한 간략한 내용은 [Artificial Intelligence Theory : 지도 학습][2강]에서도 확인해 보실 수 있습니다.

<br>
<br>

## NumPy : 경사 하강법

### 메인 코드

{% highlight Python %}

import numpy as np


x = np.array(
    [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
    [11], [12], [13], [14], [15], [16], [17], [18], [19], [20],
    [21], [22], [23], [24], [25], [26], [27], [28], [29], [30]]
)
y = np.array(
    [[0.94], [1.98], [2.88], [3.92], [3.96], [4.55], [5.64], [6.3], [7.44], [9.1],
    [8.46], [9.5], [10.67], [11.16], [14], [11.83], [14.4], [14.25], [16.2], [16.32],
    [17.46], [19.8], [18], [21.34], [22], [22.5], [24.57], [26.04], [21.6], [28.8]]
)

weight = 0.0
bias = 0.0
learning_rate = 0.005

for epoch in range(10000):
    y_hat = weight * x + bias

    cost = ((y - y_hat) ** 2).mean()

    weight = weight - learning_rate * ((y_hat - y) * x).mean()
    bias = bias - learning_rate * (y_hat - y).mean()

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch : {epoch+1:4d}, Weight : {weight:.3f}, Bias : {bias:.3f}, Cost : {cost:.3f}")


{% endhighlight %}
**결과**
:    
Epoch : 1000, Weight : 0.872, Bias : -0.290, Cost : 1.377<br>
Epoch : 2000, Weight : 0.877, Bias : -0.391, Cost : 1.373<br>
Epoch : 3000, Weight : 0.878, Bias : -0.422, Cost : 1.372<br>
Epoch : 4000, Weight : 0.879, Bias : -0.432, Cost : 1.372<br>
Epoch : 5000, Weight : 0.879, Bias : -0.435, Cost : 1.372<br>
Epoch : 6000, Weight : 0.879, Bias : -0.436, Cost : 1.372<br>
Epoch : 7000, Weight : 0.879, Bias : -0.436, Cost : 1.372<br>
Epoch : 8000, Weight : 0.879, Bias : -0.436, Cost : 1.372<br>
Epoch : 9000, Weight : 0.879, Bias : -0.436, Cost : 1.372<br>
Epoch : 10000, Weight : 0.879, Bias : -0.436, Cost : 1.372<br>

<br>

#### 세부 코드

{% highlight Python %}

x = np.array(
    [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
    [11], [12], [13], [14], [15], [16], [17], [18], [19], [20],
    [21], [22], [23], [24], [25], [26], [27], [28], [29], [30]]
)
y = np.array(
    [[0.94], [1.98], [2.88], [3.92], [3.96], [4.55], [5.64], [6.3], [7.44], [9.1],
    [8.46], [9.5], [10.67], [11.16], [14], [11.83], [14.4], [14.25], [16.2], [16.32],
    [17.46], [19.8], [18], [21.34], [22], [22.5], [24.57], [26.04], [21.6], [28.8]]
)

{% endhighlight %}

앞선 강좌에서 설명으로 사용된 30 개의 데이터 세트입니다.

**배열의 형태(Shape)**는 `(30, 1)`로 **벡터(Vector)** 구조로 생성하였습니다.

예제는 `단변량(Univariate)`, `다변량(Multivariate)`에 관계 없이 적용할 수 있도록 구현되어 있습니다.

`단변량(Univariate)` 형태인 **(30,)**과 `다변량(Multivariate)` 형태인 **(30, 2)**에도 적용 가능합니다.

<br>

{% highlight Python %}

weight = 0.0
bias = 0.0
learning_rate = 0.005

{% endhighlight %}

초기 `가중치(weight)`, `편향(bias)`, `학습률(learning_rate)`을 설정합니다.

수식에서는 $$ W_{0} $$, $$ b_{0} $$, $$ \alpha $$를 의미합니다. 이 값에 따라 최적화된 값을 빠르게 찾거나 못찾을 수 있습니다.

위와 같이 모델이 학습하기 전에 영향을 미치는 값을 `하이퍼 파라미터, 초매개변수(HyperParameter)`라 부릅니다.

즉, `하이퍼 파라미터, 초매개변수(HyperParameter)`는 모델 학습 흐름을 제어할 수 있게 하는 매개 변수입니다.

이 하이퍼 파라미터 수정을 `하이퍼 파라미터 튜닝(HyperParameter Tuning)`이라 부릅니다.

- Tip : `하이퍼 파라미터 튜닝` 기법은 `탐욕적 탐색, 그리드 서치(Grid Search)`, `무작위 탐색(Random Search)` 등이 있습니다.

<br>

{% highlight Python %}

for epoch in range(10000):
    ...

{% endhighlight %}

`에폭(Epoch)`은 `인공 신경망(Neural Network)`에서 전체 데이터 세트에 대한 `순전파(Forward Propagation)`와 `역전파(Back Propagation)` 과정을 수행한 것을 의미합니다.

`순전파(Forward Propagation)`란 입력 데이터를 기반으로 신경망을 따라 `입력층(Input Layer)`부터 `출력층(Output Layer)`까지 차례대로 변수들을 계산하고 `추론(Inference)`한 결과를 의미합니다.

`역전파(Back Propagation)`란 예측값과 실젯값의 오차를 계산 결과를 기반으로 `가중치(Weight)`를 수정하여 오차가 작아지는 방향으로 수정하는 것을 의미합니다.

전체 데이트 세트에 대해 순전파/역전파 과정을 수행하면 **한 번의 학습이 진행되었다고 볼 수 있습니다.**

즉, `에폭(Epoch)`은 전체 데이트 세트를 몇 번 학습할지를 결정하게 됩니다.

이 에폭값이 너무 적을 경우 학습이 되지 않는 `과소적합(Underfitting)` 문제가 발생하고, 에폭이 너무 많을 경우 `과대적합(Overfitting)`이 될 수 있습니다.

<br>

{% highlight Python %}

y_hat = weight * x + bias

{% endhighlight %}

현재 `가중치(weight)`와 `편향(bias)`으로 $$ \hat{Y_{i}} $$을 계산합니다.

가장 처음에는 초깃값으로 설장한 값이 할당되어 $$ \hat{Y_{i}} $$을 계산하게 되고, 두 번째 학습에서는 개선된 가중치를 적용하게 됩니다.

<br>

{% highlight Python %}

cost = ((y - y_hat) ** 2).mean()

{% endhighlight %}

전체 데이터 세트를 통해 나온 $$ \hat{Y_{i}} $$으로 `비용(Cost)`을 계산합니다.

학습이 진행될 때 마다 `비용(Cost)`이 감소되는 방향으로 진행되어야 정상적으로 학습이 진행됩니다.

<br>

{% highlight Python %}

weight = weight - learning_rate * ((y_hat - y) * x).mean()
bias = bias - learning_rate * (y_hat - y).mean()

{% endhighlight %}

새로운 `가중치(weight)`와 `편향(bias)`을 계산하기 위해 `경사 하강법(Gradient Descent)` 알고리즘을 적용합니다.

이 때 적용되는 수식이 `최적화(Optimization)`를 의미합니다.

<br>

{% highlight Python %}

if (epoch + 1) % 1000 == 0:
    print(f"Epoch : {epoch+1:4d}, Weight : {weight:.3f}, Bias : {bias:.3f}, Cost : {cost:.3f}")

{% endhighlight %}
**결과**
:    
Epoch : 1000, Weight : 0.872, Bias : -0.290, Cost : 1.377<br>
Epoch : 2000, Weight : 0.877, Bias : -0.391, Cost : 1.373<br>
Epoch : 3000, Weight : 0.878, Bias : -0.422, Cost : 1.372<br>
Epoch : 4000, Weight : 0.879, Bias : -0.432, Cost : 1.372<br>
Epoch : 5000, Weight : 0.879, Bias : -0.435, Cost : 1.372<br>
Epoch : 6000, Weight : 0.879, Bias : -0.436, Cost : 1.372<br>
Epoch : 7000, Weight : 0.879, Bias : -0.436, Cost : 1.372<br>
Epoch : 8000, Weight : 0.879, Bias : -0.436, Cost : 1.372<br>
Epoch : 9000, Weight : 0.879, Bias : -0.436, Cost : 1.372<br>
Epoch : 10000, Weight : 0.879, Bias : -0.436, Cost : 1.372<br>
<br>

10,000 번을 학습하기 때문에 모든 기록 학습 결과를 출력하지 않고, 1,000 번의 학습마다 결과를 출력합니다.

출력되는 결과를 통해 학습이 진행될 때마다 `가중치(Weight)`와 `편향(Bias)`이 조정되며 `비용(Cost)`이 감소되는 것을 확인할 수 있습니다.

처음에 설정한 **초기 가중치, 편향, 학습률에서는 약 6,000회 가량 학습했을 때 더 이상 비용(Cost)이 줄어들지 않는 것을 확인할 수 있습니다.**

<br>
<br>

## 초깃값에 따른 학습 결과

이번에는 초깃값을 `weight = 0.0`, `bias = 0.0`, `learning_rate = 0.005` 값이 아닌 다른 값을 할당했을 때 어떻게 학습되는지 확인해보도록 하겠습니다.

비교를 위하여 `learning_rate = 0.005`일 때의 10회 학습 결과입니다.

```
# learning_rate = 0.005
Epoch :    1, Weight : 1.351, Bias : 0.066, Cost : 233.140
Epoch :    2, Weight : 0.568, Bias : 0.027, Cost : 79.280
Epoch :    3, Weight : 1.022, Bias : 0.049, Cost : 27.584
Epoch :    4, Weight : 0.759, Bias : 0.035, Cost : 10.215
Epoch :    5, Weight : 0.911, Bias : 0.042, Cost : 4.379
Epoch :    6, Weight : 0.823, Bias : 0.037, Cost : 2.418
Epoch :    7, Weight : 0.874, Bias : 0.039, Cost : 1.759
Epoch :    8, Weight : 0.845, Bias : 0.037, Cost : 1.538
Epoch :    9, Weight : 0.862, Bias : 0.037, Cost : 1.463
Epoch :   10, Weight : 0.852, Bias : 0.036, Cost : 1.438
```

학습이 진행될수록 `비용(Cost)`이 점점 감소하면서 적절한 가중치(Weight)와 편향(Bias)을 찾아가는 것을 확인할 수 있습니다.

앞선 [제 5강 - 최적화(Optimization)][5강]의 `학습률이 적절할 때`의 그래프로 볼 수 있습니다.

<br>

### learning_rate = 0.001

```
# learning_rate = 0.001
Epoch :    1, Weight : 0.270, Bias : 0.013, Cost : 233.140
Epoch :    2, Weight : 0.455, Bias : 0.022, Cost : 109.857
Epoch :    3, Weight : 0.582, Bias : 0.028, Cost : 52.167
Epoch :    4, Weight : 0.668, Bias : 0.032, Cost : 25.171
Epoch :    5, Weight : 0.727, Bias : 0.035, Cost : 12.538
Epoch :    6, Weight : 0.768, Bias : 0.037, Cost : 6.626
Epoch :    7, Weight : 0.795, Bias : 0.038, Cost : 3.859
Epoch :    8, Weight : 0.814, Bias : 0.039, Cost : 2.565
Epoch :    9, Weight : 0.827, Bias : 0.040, Cost : 1.959
Epoch :   10, Weight : 0.836, Bias : 0.040, Cost : 1.676
```

`learning_rate = 0.001`에서는 `learning_rate = 0.005`보다 더 천천히 `비용(Cost)`이 감소하는 것을 확인할 수 있습니다.

앞선 [제 5강 - 최적화(Optimization)][5강]의 `학습률이 낮을 때`의 그래프로 볼 수 있습니다.

<br>

### learning_rate = 0.006

```
# learning_rate = 0.006
Epoch :    1, Weight : 1.621, Bias : 0.079, Cost : 233.140
Epoch :    2, Weight : 0.169, Bias : 0.007, Cost : 187.274
Epoch :    3, Weight : 1.470, Bias : 0.070, Cost : 150.487
Epoch :    4, Weight : 0.305, Bias : 0.012, Cost : 120.981
Epoch :    5, Weight : 1.348, Bias : 0.063, Cost : 97.316
Epoch :    6, Weight : 0.414, Bias : 0.016, Cost : 78.335
Epoch :    7, Weight : 1.251, Bias : 0.057, Cost : 63.112
Epoch :    8, Weight : 0.502, Bias : 0.019, Cost : 50.901
Epoch :    9, Weight : 1.173, Bias : 0.052, Cost : 41.108
Epoch :   10, Weight : 0.572, Bias : 0.021, Cost : 33.253
```

`learning_rate = 0.006`에서는 `비용(Cost)`이 감소하긴 하지만, `가중치(Weight)`나 `편향(Bias)`이 지그재그 형태로 움직이는 것을 확인할 수 있습니다.

앞선 [제 5강 - 최적화(Optimization)][5강]의 `학습률이 높을 때`의 그래프로 볼 수 있습니다.

<br>

### learning_rate = 0.007

```
# learning_rate = 0.007
Epoch :    1, Weight : 1.892, Bias : 0.092, Cost : 233.140
Epoch :    2, Weight : -0.400, Bias : -0.021, Cost : 341.523
Epoch :    3, Weight : 2.376, Bias : 0.115, Cost : 500.602
Epoch :    4, Weight : -0.987, Bias : -0.052, Cost : 734.091
Epoch :    5, Weight : 3.088, Bias : 0.148, Cost : 1076.794
Epoch :    6, Weight : -1.849, Bias : -0.096, Cost : 1579.796
Epoch :    7, Weight : 4.132, Bias : 0.198, Cost : 2318.076
Epoch :    8, Weight : -3.114, Bias : -0.160, Cost : 3401.685
Epoch :    9, Weight : 5.665, Bias : 0.272, Cost : 4992.152
Epoch :   10, Weight : -4.971, Bias : -0.253, Cost : 7326.558
```

`learning_rate = 0.007`에서는 `비용(Cost)`이 점점 증가하고 `가중치(Weight)`와 `편향(Bias)`이 점점 적절한 값에서 멀어지게 됩니다.

앞선 [제 5강 - 최적화(Optimization)][5강]의 `학습률이 너무 높을 때`의 그래프로 볼 수 있습니다.

<br>

### weight = 100000.0, bias = 100000.0

```
# weight = 100000.0
# bias = 100000.0
# learning_rate = 0.005

Epoch :    1, Weight : -65331.982, Bias : 91750.066, Cost : 3471609980966.473
Epoch :    2, Weight : 30511.054, Bias : 96354.610, Cost : 1167846104927.318
Epoch :    3, Weight : -25035.413, Bias : 93508.296, Cost : 393797829516.361
Epoch :    4, Weight : 7170.684, Bias : 94981.065, Cost : 133720896289.252
Epoch :    5, Weight : -11488.800, Bias : 93950.498, Cost : 46333937931.186
Epoch :    6, Weight : -664.178, Bias : 94371.193, Cost : 16969339320.910
Epoch :    7, Weight : -6929.960, Bias : 93950.877, Cost : 7099764303.827
Epoch :    8, Weight : -3289.340, Bias : 94018.261, Cost : 3780358039.956
Epoch :    9, Weight : -5390.953, Bias : 93803.159, Cost : 2661760204.138
Epoch :   10, Weight : -4164.103, Bias : 93752.008, Cost : 2282622976.588
```

비교적 적절한 학습률인 `learning_rate = 0.005`일 때 초깃값을 매우 큰 값을 주는 경우, `비용(Cost)`이 감소하긴 하지만 매우 많은 학습(Epoch)을 요구하게 됩니다.

이렇듯 초깃값의 설정은 학습에 큰 영향을 끼칩니다.

**적절하지 않은 초깃값을 할당했을 때 하이퍼 파라미터 튜닝을 통해 원활한 학습을 진행할 수 있게됩니다.**

<br>
<br>

## PyTorch : 확률적 경사 하강법

### 메인 코드

{% highlight Python %}

import torch
from torch import optim


x = torch.FloatTensor([
    [1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
    [11], [12], [13], [14], [15], [16], [17], [18], [19], [20],
    [21], [22], [23], [24], [25], [26], [27], [28], [29], [30]
])
y = torch.FloatTensor([
    [0.94], [1.98], [2.88], [3.92], [3.96], [4.55], [5.64], [6.3], [7.44], [9.1],
    [8.46], [9.5], [10.67], [11.16], [14], [11.83], [14.4], [14.25], [16.2], [16.32],
    [17.46], [19.8], [18], [21.34], [22], [22.5], [24.57], [26.04], [21.6], [28.8]
])

weight = torch.zeros(1, requires_grad=True)
bias = torch.zeros(1, requires_grad=True)
learning_rate = 0.001

optimizer = optim.SGD([weight, bias], lr=learning_rate)

for epoch in range(10000):
    hypothesis = weight * x + bias
    cost = torch.mean((hypothesis - y) ** 2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch : {epoch+1:4d}, Weight : {weight.item():.3f}, Bias : {bias.item():.3f}, Cost : {cost:.3f}")


{% endhighlight %}
**결과**
:    
Epoch : 1000, Weight : 0.864, Bias : -0.138, Cost : 1.393<br>
Epoch : 2000, Weight : 0.870, Bias : -0.251, Cost : 1.380<br>
Epoch : 3000, Weight : 0.873, Bias : -0.321, Cost : 1.375<br>
Epoch : 4000, Weight : 0.875, Bias : -0.364, Cost : 1.373<br>
Epoch : 5000, Weight : 0.877, Bias : -0.391, Cost : 1.373<br>
Epoch : 6000, Weight : 0.878, Bias : -0.408, Cost : 1.372<br>
Epoch : 7000, Weight : 0.878, Bias : -0.419, Cost : 1.372<br>
Epoch : 8000, Weight : 0.878, Bias : -0.425, Cost : 1.372<br>
Epoch : 9000, Weight : 0.879, Bias : -0.429, Cost : 1.372<br>
Epoch : 10000, Weight : 0.879, Bias : -0.432, Cost : 1.372<br>
<br>

#### 세부 코드

{% highlight Python %}

import torch
from torch import optim

{% endhighlight %}

`PyTorch`를 사용하기 위해 `torch`와 `torch.optim`을 포함시킵니다.

`torch.optim`은 최적화 함수가 포함되어 있는 모듈입니다.

<br>

{% highlight Python %}

x = torch.FloatTensor([
    [1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
    [11], [12], [13], [14], [15], [16], [17], [18], [19], [20],
    [21], [22], [23], [24], [25], [26], [27], [28], [29], [30]
])
y = torch.FloatTensor([
    [0.94], [1.98], [2.88], [3.92], [3.96], [4.55], [5.64], [6.3], [7.44], [9.1],
    [8.46], [9.5], [10.67], [11.16], [14], [11.83], [14.4], [14.25], [16.2], [16.32],
    [17.46], [19.8], [18], [21.34], [22], [22.5], [24.57], [26.04], [21.6], [28.8]
])

{% endhighlight %}

`NumPy` 예제에서 사용했던 데이터와 동일한 구조의 데이터를 사용합니다.

단, `ndarray` 형식이 아닌, `FloatTensor`의 형식으로 적용합니다.

<br>

{% highlight Python %}

weight = torch.zeros(1, requires_grad=True)
bias = torch.zeros(1, requires_grad=True)
learning_rate = 0.001

{% endhighlight %}

초기 `가중치(weight)`, `편향(bias)`, `학습률(learning_rate)`을 설정합니다.

`torch.zeros` 메서드로 0의 값을 갖는 텐서를 생성합니다.

텐서의 크기는 1로 설정합니다. 이 값은 현재 `(1,)`이나 `(1, 1)`로 설정해도 동일한 결과를 반환합니다.

다음으로 `requires_grad` 매개변수는 `참(True)` 값으로 설정합니다.

`requires_grad`는 모든 텐서에 대한 연산들을 추적하며, 역전파(backward) 메서드를 호출해 `기울기(gradient)`를 계산하고 저장합니다.

<br>

{% highlight Python %}

optimizer = optim.SGD([weight, bias], lr=learning_rate)

{% endhighlight %}

최적화 함수는 `확률적 경사 하강법(optim.SGD)`을 적용합니다.

`확률적 경사 하강법(Stochastic Gradient Descent)`는 모든 데이터에 대해 연산을 진행하지 않고, 일부 데이터만 계산하여 빠르게 최적화된 값을 찾는 방식입니다.

즉, `미니 배치(Mini-Batch)`의 형태로 전체 데이터를 N 등분하여 학습을 진행합니다.

확률적 경사 하강법의 매개변수로는 `최적화하려는 변수`와 `학습률`을 입력합니다.

<br>

{% highlight Python %}

for epoch in range(10000):
    hypothesis = x * weight + bias
    cost = torch.mean((hypothesis - y) ** 2)

{% endhighlight %}

`에폭(Epoch)`은 동일하게 10,000회로 설정하고 `가설(hypothesis)`과 `비용(cost)`을 정의합니다.

`가설(hypothesis)`은 `NumPy` 코드에서 사용한 `y_hat`과 동일한 구조이며, `비용(cost)`도 `NumPy` 코드에서 사용한 `cost`와 동일한 구조입니다.

<br>

{% highlight Python %}

optimizer.zero_grad()
cost.backward()
optimizer.step()

{% endhighlight %}

`optimizer.zero_grad()` 메서드로 `optimizer` 변수에 포함시킨 매개 변수들의 `기울기(Gradient)`를 0으로 초기화 시킵니다.

텐서의 기울기는 `grad` 속성에 누적해서 더해지기 때문에 0으로 초기화해야 합니다.

이를 코드적으로 설명한다면, 기울기가 `weight = x`의 형태가 아닌, `weight += x`의 구조처럼 기울기를 저장하기 때문입니다.

그러므로, 기울기를 0으로 초기화해 중복 연산을 방지합니다.

다음으로, `cost.backward()`를 통해 `역전파(Back Propagation)`를 수행합니다.

이 연산을 통해 `optimizer` 변수에 포함시킨 매개 변수들의 기울기가 새로 계산됩니다.

이제 한 번의 학습 과정을 통해 `가중치(weight)`와 `편향(bias)`에 대한 `기울기(Gradient)`를 계산했습니다.

이 결과를 최적화 함수에 반영하기 위해 `optimizer.step()`을 진행합니다.

여기서 `학습률(lr)`의 값을 반영한 `확률적 경사 하강법(Stochastic Gradient Descent)` 연산이 적용됩니다.



<br>

{% highlight Python %}

if (epoch + 1) % 1000 == 0:
    print(f"Epoch : {epoch+1:4d}, Weight : {weight.item():.3f}, Bias : {bias.item():.3f}, Cost : {cost:.3f}")

{% endhighlight %}
**결과**
:    
Epoch : 1000, Weight : 0.864, Bias : -0.138, Cost : 1.393<br>
Epoch : 2000, Weight : 0.870, Bias : -0.251, Cost : 1.380<br>
Epoch : 3000, Weight : 0.873, Bias : -0.321, Cost : 1.375<br>
Epoch : 4000, Weight : 0.875, Bias : -0.364, Cost : 1.373<br>
Epoch : 5000, Weight : 0.877, Bias : -0.391, Cost : 1.373<br>
Epoch : 6000, Weight : 0.878, Bias : -0.408, Cost : 1.372<br>
Epoch : 7000, Weight : 0.878, Bias : -0.419, Cost : 1.372<br>
Epoch : 8000, Weight : 0.878, Bias : -0.425, Cost : 1.372<br>
Epoch : 9000, Weight : 0.879, Bias : -0.429, Cost : 1.372<br>
Epoch : 10000, Weight : 0.879, Bias : -0.432, Cost : 1.372<br>
<br>

10,000 번을 학습하기 때문에 모든 기록 학습 결과를 출력하지 않고, 1,000 번의 학습마다 결과를 출력합니다.

출력되는 결과를 통해 학습이 진행될 때마다 `가중치(Weight)`와 `편향(Bias)`가 조정되며 `비용(Cost)`이 감소되는 것을 확인할 수 있습니다.

학습률은 0.001로 기존 `경사 하강법(Stochastic Gradient Descent)`보다 낮은 학습률을 선택하였지만, 더 빠른 속도로 최적의 가중치와 편향을 찾는 것을 확인할 수 있습니다.

<br>
<br>

## zero_grad(), cost.backward(), optimizer.step() 알아보기

{% highlight Python %}

print(f"Epoch : {epoch+1:4d}")
print(f"Step [1] : Gradient : {weight.grad}, Weight : {weight.item():.5f}")

optimizer.zero_grad()
print(f"Step [2] : Gradient : {weight.grad}, Weight : {weight.item():.5f}")

cost.backward()
print(f"Step [3] : Gradient : {weight.grad}, Weight : {weight.item():.5f}")

optimizer.step()
print(f"Step [4] : Gradient : {weight.grad}, Weight : {weight.item():.5f}")

{% endhighlight %}
**결과**
:    
Epoch :    1<br>
Step [1] : Gradient : None, Weight : 0.00000<br>
Step [2] : Gradient : None, Weight : 0.00000<br>
Step [3] : Gradient : tensor([-540.4854]), Weight : 0.00000<br>
Step [4] : Gradient : tensor([-540.4854]), Weight : 0.54049<br>
Epoch :    2<br>
Step [1] : Gradient : tensor([-540.4854]), Weight : 0.54049<br>
Step [2] : Gradient : tensor([0.]), Weight : 0.54049<br>
Step [3] : Gradient : tensor([-198.9818]), Weight : 0.54049<br>
Step [4] : Gradient : tensor([-198.9818]), Weight : 0.73947<br>
...<br>
<br>

위와 같이 직접 가중치에 대한 기울기와 값을 확인해본다면 쉽게 이해할 수 있습니다.

`첫 번째 에폭(epoch=1)`에서는 계산된 기울기가 없기 때문에 `optimizer.zero_grad()` 까지는 초깃값과 동일한 형태를 출력합니다.

하지만, `cost.backward()`를 통해 `역전파(Back Propagation)`를 수행하게 되면, `기울기(-540.4854)`가 생성됩니다.

역전파를 통해 기울기는 계산했지만, `weight` 변수에는 값이 반영되지 않은 것을 알 수 있습니다.

`optimizer.step()`을 통해 `확률적 경사 하강법(Stochastic Gradient Descent)`을 수행한 결과를 `weight` 변수에 반영합니다.

`두 번째 에폭(epoch=2)`에서는 새로 계산한 기울기와 값이 있기 때문에 첫 번째 에폭에서 나온 결과로 학습을 수행합니다.

`optimizer.zero_grad()`로 기울기를 0으로 초기화합니다.

만약, `optimizer.zero_grad()`로 기울기를 초기화 해주지 않는다면 `-540.4854 + -198.9818` 연산으로 진행하게 되고, `-739.4672`의 기울기로 인해 연산이 틀어지게 됩니다.

[2강]: https://076923.github.io/posts/AI-2/
[5강]: https://076923.github.io/posts/Python-pytorch-5/
