---
layout: post
title: "Python Pytorch 강좌 : 제 7강 - 다중 선형 회귀(Multiple Linear Regression)"
tagline: "Python PyTorch Optimization"
image: /assets/images/pytorch.webp
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['PyTorch']
keywords: Python, Python Pytorch, Python torch.nn, Python Multivariate Multiple Linear Regression
ref: Python-PyTorch
category: Python
permalink: /posts/Python-pytorch-7/
comments: true
toc: true
---

## 다중 선형 회귀(Multiple Linear Regression)

`다중 선형 회귀(Multiple Linear Regression)`는 **두 개 이상의 독립 변수(Independent Variable)와 하나 이상의 종속 변수(Dependent Variable)** 사이의 선형 상관 관계를 분석하는 방법입니다.

하나의 종속 변수와의 관계를 분석하는 경우에는 `단변량 다중 선형 회귀(Univariate Multiple Linear Regression)`가 되며, 두 개 이상의 종속 변수와의 관계를 분석하는 경우에는 `다변량 다중 선형 회귀(Multivariate Multiple Linear Regression)`라 합니다.

즉, 종속 변수가 `스칼라(Scalar)` 형태($$ Y $$)를 가지면 `단변량(Univariate)`이 되며, `벡터(Vector)`의 형태($$ [Y_{1}, Y_{2}, ...] $$)를 가지면 `다변량(Multivariate)`이 됩니다.

선형 회귀에 관한 간략한 내용은 [Artificial Intelligence Theory : 지도 학습][2강]에서도 확인해 보실 수 있습니다.

- Tip : 하나의 독립 변수를 가지면 `단일(Simple)`이 되며, 두 개 이상의 독립 변수를 가지면 `다중(Multiple)`이 됩니다.

<br>
<br>

## Neural Networks 패키지(torch.nn)
 
### 메인 코드

{% highlight Python %}

import torch
from torch import nn
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

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10000):
    output = model(x)
    cost = criterion(output, y)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch : {epoch+1:4d}, Model : {list(model.parameters())}, Cost : {cost:.3f}")


{% endhighlight %}
**결과**
:    
Epoch : 1000, Model : [Parameter containing:<br>
tensor([[0.8721]], requires_grad=True), Parameter containing:<br>
tensor([-0.2968], requires_grad=True)], Cost : 1.377<br>
Epoch : 2000, Model : [Parameter containing:<br>
tensor([[0.8746]], requires_grad=True), Parameter containing:<br>
tensor([-0.3493], requires_grad=True)], Cost : 1.374<br>
Epoch : 3000, Model : [Parameter containing:<br>
tensor([[0.8762]], requires_grad=True), Parameter containing:<br>
tensor([-0.3820], requires_grad=True)], Cost : 1.373<br>
Epoch : 4000, Model : [Parameter containing:<br>
tensor([[0.8772]], requires_grad=True), Parameter containing:<br>
Epoch : 5000, Model : [Parameter containing:<br>
tensor([[0.8779]], requires_grad=True), Parameter containing:<br>
tensor([-0.4151], requires_grad=True)], Cost : 1.372<br>
Epoch : 6000, Model : [Parameter containing:<br>
tensor([[0.8783]], requires_grad=True), Parameter containing:<br>
tensor([-0.4229], requires_grad=True)], Cost : 1.372<br>
Epoch : 7000, Model : [Parameter containing:<br>
tensor([[0.8785]], requires_grad=True), Parameter containing:<br>
tensor([-0.4279], requires_grad=True)], Cost : 1.372<br>
Epoch : 8000, Model : [Parameter containing:<br>
tensor([[0.8787]], requires_grad=True), Parameter containing:<br>
tensor([-0.4309], requires_grad=True)], Cost : 1.372<br>
Epoch : 9000, Model : [Parameter containing:<br>
tensor([[0.8787]], requires_grad=True), Parameter containing:<br>
tensor([-0.4328], requires_grad=True)], Cost : 1.372<br>
Epoch : 10000, Model : [Parameter containing:<br>
tensor([[0.8788]], requires_grad=True), Parameter containing:<br>
tensor([-0.4340], requires_grad=True)], Cost : 1.372<br>

<br>

#### 세부 코드

{% highlight Python %}

from torch import nn

{% endhighlight %}

`다중 선형 회귀(Multiple Linear Regression)`를 적용해보기 전에 `Neural Networks 패키지(torch.nn)`을 적용하여 앞선 [제 6강 - 단순 선형 회귀(Simple Linear Regression)][6강]의 코드를 변경해보도록 하겠습니다.

`torch.nn` 패키지는 주로 `신경망(Neural Network)`을 구성할 때 활용됩니다.

주로 **네트워크(Net)**를 정의하거나, **자동 미분(autograd)**, **레이어(layer)** 등을 정의할 수 있는 모듈이 포함되어 있습니다.

즉, **신경망을 생성하고 학습시키는 과정을 빠르고 간편하게 구현할 수 있는 기능들이 제공됩니다.**

앞으로의 `모델(Model)` 구현은 신경망 패키지를 활용하여 구현하도록 하겠습니다.

<br>

{% highlight Python %}

model = nn.Linear(1, 1)

{% endhighlight %}

`선형 변환 함수(nn.Linear)`로 `model` 변수를 정의합니다.

선형 변환 함수는 $$ y = xA^T + b $$의 형태의 `선형 변환(Linear Transformation)`을 입력 데이터에 적용합니다.

`nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)` 구조를 갖습니다.

`입력 데이터 차원 크기(in_features)`와 `출력 데이터 차원 크기(out_features)`는 정수를 입력하며, **입/출력시에 적용되는 데이터의 차원을 의미합니다.**

`편향 유/무(bias)`는 **편향(bias) 계산 유/무를 설정합니다.** **거짓 값(False)**으로 적용한다면 편향을 계산하지 않습니다.

즉, `선형 변환 함수(nn.Linear)`의 매개 변수가 앞선 강좌의 `weight`, `bias` 변수를 대신하게 됩니다.

<br>

{% highlight Python %}

criterion = nn.MSELoss()

{% endhighlight %}

`평균 제곱 오차 클래스(nn.MSELoss)`로 `criterion` 인스턴스를 생성합니다.

`평균 제곱 오차 클래스(nn.MSELoss)`는 `비용 함수(Cost Function)`로 `criterion` 인스턴스에서 **순전파(Forward)**를 통해 나온 출력값과 실젯값을 비교하여 오차를 계산합니다.

<br>

{% highlight Python %}

optimizer = optim.SGD(model.parameters(), lr=0.001)

{% endhighlight %}

최적화 함수는 동일하게 `확률적 경사 하강법(optim.SGD)`을 적용합니다.

단, `최적화하려는 변수`는 `모델(model)`의 매개변수로 적용합니다.

모델의 매개변수는 `가중치(Weight)`와 `편향(Bias)`를 의미합니다.

<br>

{% highlight Python %}

for epoch in range(10000):
    output = model(x)
    cost = criterion(output, y)

{% endhighlight %}

`모델(model)`에 `입력값(x)`를 입력하여 **순전파(Forward)** 연산을 진행합니다.

이 연산을 통해 `예측값(output)`이 생성되고, `예측값(output)`과 `실젯값(y)`과의 오차를 계산합니다.

<br>

{% highlight Python %}

optimizer.zero_grad()
cost.backward()
optimizer.step()

if (epoch + 1) % 1000 == 0:
    print(f"Epoch : {epoch+1:4d}, Model : {list(model.parameters())}, Cost : {cost:.3f}")

{% endhighlight %}

앞선 강좌와 동일하게 **기울기 초기화**, **역전파(Back Propagation)**, **최적화 함수 적용**을 진행합니다.

`에폭(Epoch)`마다 학습이 진행되며, `모델 매개변수(model.parameters())`에 포함된 `가중치(Weight)`와 `편향(Bias)`이 계산됩니다.

<br>
<br>

## 다변량 다중 선형 회귀(Multivariate Multiple Linear Regression)

|  x1   |  x2   |  y1   |  y2   |
| :---: | :---: | :---: | :---: |
|   1   |   2   |  0.1  |  1.5  |
|   2   |   3   |  1.0  |  2.8  |
|   3   |   4   |  1.9  |  4.1  |
|   4   |   5   |  2.8  |  5.4  |
|   5   |   6   |  3.7  |  6.7  |
|   6   |   7   |  4.6  |  8.0  |

<br>

`다변량 다중 선형 회귀(Multivariate Multiple Linear Regression)`를 적용하기 위해 새로운 데이터 세트를 생성합니다.

`독립 변수(Independent Variable)`는 **x1**, **x2**를 의미하며, `종속 변수(Dependent Variable)`는 **y1**, **y2**를 의미합니다.

이를 수식으로 표현하면 다음과 같습니다.

$$
\begin{multline}
\shoveleft y_{1} = w_{1}x_{1} + w_{2}x_{2} + b_{1}\\
\shoveleft y_{2} = w_{3}x_{1} + w_{4}x_{2} + b_{2}
\end{multline}
$$

<br>

현재 데이터에서 `y1`과 `y1`를 계산하기 위해 사용된 각각의 `가중치(Weight)`와 `편향(Bias)`은 다음과 같습니다.

$$
\begin{multline}
\shoveleft y_{1} = 1.7x_{1} - 0.8x_{2} + 0\\
\shoveleft y_{2} = 1.1x_{1} + 0.2x_{2} + 0
\end{multline}
$$

<br>

그러므로, `모델 매개변수(model.parameters)`에서 반환되어야 하는 값은 다음과 같습니다.

$$
\begin{multline}
\shoveleft Weight = \begin{bmatrix}w_1 & w_2 \\w_3 & w_4 \\\end{bmatrix} = \begin{bmatrix}1.7 & -0.8 \\1.1 & 0.2 \\\end{bmatrix}\\
\shoveleft Bias = \begin{bmatrix}b_1 \\b_2 \\\end{bmatrix} = \begin{bmatrix}0 \\0 \\\end{bmatrix}
\end{multline} 
$$

<br>

위에서 미리 정의된 `가중치(Weight)`와 `편향(Bias)` 값과 동일한 값이 반환되는지 확인해보도록 하겠습니다.


### 메인 코드

{% highlight Python %}
import torch
from torch import nn
from torch import optim


x = torch.FloatTensor([
    [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]
])
y = torch.FloatTensor([
    [0.1, 1.5], [1.0, 2.8], [1.9, 4.1], [2.8, 5.4], [3.7, 6.7], [4.6, 8.0]
])

model = nn.Linear(2, 2)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10000):
    output = model(x)
    cost = criterion(output, y)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch : {epoch+1:4d}, Model : {list(model.parameters())}, Cost : {cost:.3f}")


{% endhighlight %}
**결과**
:    
Epoch : 1000, Model : [Parameter containing:<br>
tensor([[0.5695, 0.1465],<br>
&emsp;&emsp;&emsp;&emsp;[0.8522, 0.3354]], requires_grad=True), Parameter containing:<br>
tensor([-0.1738,  0.3367], requires_grad=True)], Cost : 0.079<br>
Epoch : 2000, Model : [Parameter containing:<br>
tensor([[0.6457, 0.1072],<br>
&emsp;&emsp;&emsp;&emsp;[0.8988, 0.3113]], requires_grad=True), Parameter containing:<br>
tensor([-0.2894,  0.2661], requires_grad=True)], Cost : 0.051<br>
Epoch : 3000, Model : [Parameter containing:<br>
tensor([[0.7067, 0.0757],<br>
&emsp;&emsp;&emsp;&emsp;[0.9360, 0.2921]], requires_grad=True), Parameter containing:<br>
tensor([-0.3818,  0.2097], requires_grad=True)], Cost : 0.032<br>
Epoch : 4000, Model : [Parameter containing:<br>
tensor([[0.7554, 0.0506],<br>
&emsp;&emsp;&emsp;&emsp;[0.9658, 0.2768]], requires_grad=True), Parameter containing:<br>
tensor([-0.4557,  0.1645], requires_grad=True)], Cost : 0.021<br>
Epoch : 5000, Model : [Parameter containing:<br>
tensor([[0.7944, 0.0305],<br>
&emsp;&emsp;&emsp;&emsp;[0.9896, 0.2645]], requires_grad=True), Parameter containing:<br>
tensor([-0.5147,  0.1284], requires_grad=True)], Cost : 0.013<br>
Epoch : 6000, Model : [Parameter containing:<br>
tensor([[0.8255, 0.0144],<br>
&emsp;&emsp;&emsp;&emsp;[1.0086, 0.2546]], requires_grad=True), Parameter containing:<br>
tensor([-0.5620,  0.0996], requires_grad=True)], Cost : 0.008<br>
Epoch : 7000, Model : [Parameter containing:<br>
tensor([[0.8504, 0.0015],<br>
&emsp;&emsp;&emsp;&emsp;[1.0239, 0.2468]], requires_grad=True), Parameter containing:<br>
tensor([-0.5997,  0.0765], requires_grad=True)], Cost : 0.005<br>
Epoch : 8000, Model : [Parameter containing:<br>
tensor([[ 0.8703, -0.0087],<br>
&emsp;&emsp;&emsp;&emsp;[ 1.0360,  0.2405]], requires_grad=True), Parameter containing:<br>
tensor([-0.6299,  0.0581], requires_grad=True)], Cost : 0.003<br>
Epoch : 9000, Model : [Parameter containing:<br>
tensor([[ 0.8862, -0.0170],<br>
&emsp;&emsp;&emsp;&emsp;[ 1.0457,  0.2355]], requires_grad=True), Parameter containing:<br>
tensor([-0.6540,  0.0433], requires_grad=True)], Cost : 0.002<br>
Epoch : 10000, Model : [Parameter containing:<br>
tensor([[ 0.8990, -0.0235],<br>
&emsp;&emsp;&emsp;&emsp;[ 1.0535,  0.2315]], requires_grad=True), Parameter containing:<br>
tensor([-0.6733,  0.0315], requires_grad=True)], Cost : 0.001<br>

<br>

#### 세부 코드

{% highlight Python %}

x = torch.FloatTensor([
    [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]
])
y = torch.FloatTensor([
    [0.1, 1.5], [1.0, 2.8], [1.9, 4.1], [2.8, 5.4], [3.7, 6.7], [4.6, 8.0]
])

{% endhighlight %}

`다변량 다중 선형 회귀(Multivariate Multiple Linear Regression)`으로 계산하기 위해 독립 변수와 종속 변수의 차원을 `(n, 2)`의 형태로 입력합니다.

`독립 변수(x)`에 **(x1, x2)**의 값을 순차적으로 입력하며, `종속 변수(y)`에 **(y1, y2)**의 값을 순차적으로 입력합니다.

<br>

{% highlight Python %}

model = nn.Linear(2, 2)

{% endhighlight %}

`입력 데이터 차원 크기(in_features)`은 **2**가 되며, `출력 데이터 차원 크기(out_features)`도 **2**가 됩니다.

`단변량 단일 선형 회귀(Univariate Simple Linear Regression)`에서 `다변량 다중 선형 회귀(Multivariate Multiple Linear Regression)`로 변경됐지만, 더 변경되어야 하는 코드는 없습니다.

<br>

{% highlight Python %}

Epoch : 10000, Model : [Parameter containing:
tensor([[ 0.8990, -0.0235],
        [ 1.0535,  0.2315]], requires_grad=True), Parameter containing:
tensor([-0.6733,  0.0315], requires_grad=True)], Cost : 0.001

{% endhighlight %}

10,000 번의 학습을 진행되었을 때의 `가중치(Weight)`와 `편향(Bias)`에 대한 결과입니다.

$$
\begin{multline}
\shoveleft Weight = \begin{bmatrix}w_1 & w_2 \\w_3 & w_4 \\\end{bmatrix} = \begin{bmatrix}0.8990 & -0.0235 \\1.0535 & 0.2315 \\\end{bmatrix}\\
\shoveleft Bias = \begin{bmatrix}b_1 \\b_2 \\\end{bmatrix} = \begin{bmatrix}-0.6733 \\0.0315 \\\end{bmatrix}
\end{multline} 
$$

<br>

이 값으로 예측값을 계산해 실젯값과 비교한다면 다음과 같습니다.

|  x1   |  x2   |  y1   |  y2   |  $$ \hat{y_{1}} $$  |  $$ \hat{y_{2}} $$   |
| :---: | :---: | :---: | :---: | :---: | :---: |
|   1   |   2   |  0.1  |  1.5  |  0.2  |  1.5  |
|   2   |   3   |  1.0  |  2.8  |  1.1  |  2.8  |
|   3   |   4   |  1.9  |  4.1  |  1.9  |  4.1  |
|   4   |   5   |  2.8  |  5.4  |  2.8  |  5.4  |
|   5   |   6   |  3.7  |  6.7  |  3.7  |  6.7  |
|   6   |   7   |  4.6  |  8.0  |  4.6  |  8.0  |

* 비교를 위해 $$ \hat{y_{i}} $$ 값은 반올림 처리하였습니다.

<br>

`비용(Cost)`는 **0.001**로 학습이 적절히 진행된 것을 확인할 수 있습니다.

`y1`과 `y2`를 계산하기 위해 적용한 실젯값과 예측값이 동일하거나 유사한 값으로 반환되었습니다.

`새로운 값(x1=16, x2=17 등)`을 입력해도 실젯값과 큰 차이를 보이지 않는 것을 알 수 있습니다.

위와 같이 **학습은 비용(Cost)를 줄이는 방향으로 학습이 진행**되므로 예상되는 `가중치(Weight)`와 `편향(Bias)`이 전혀 다른 값으로 생성될 수 있지만, 결괏값은 최대한 유사하게 반영됩니다.

<br>

### bias = False

{% highlight Python %}

model = nn.Linear(2, 2, bias=False)

{% endhighlight %}
**결과**
:    
Epoch : 10000, Model : [Parameter containing:<br>
tensor([[ 1.1172, -0.3289],<br>
&emsp;&emsp;&emsp;&emsp;[ 0.8934,  0.3670]], requires_grad=True)], Cost : 0.024<br>
Epoch : 50000, Model : [Parameter containing:<br>
tensor([[ 1.6671, -0.7734],<br>
&emsp;&emsp;&emsp;&emsp;[ 1.0802,  0.2160]], requires_grad=True)], Cost : 0.000<br>
<br>

이번에는 `선형 변환 함수(nn.Linear)`에서 `편향 유/무(bias)`를 `거짓(False)` 값으로 적용했을 때의 결과를 확인해보도록 하겠습니다.

각각 10,000번, 50,000 번을 학습했을 때의 결과는 위와 같습니다.

두 학습 모두 `비용(Cost)`이 낮은 것을 확인할 수 있습니다.

간단한 선형 회귀에서도 실젯값을 표현하는 예측값은 무수히 많을 수 있습니다.

더 복잡한 수식에서는 더 많은 값들이 존재하게 됩니다.

그러므로, **실제 환경에서 적용되는 데이터(학습에 사용하지 않은 데이터)를 통해 지속적으로 검증하고, 최적의 매개 변수를 찾는 방법으로 모델을 구성해야 합니다.** 


[2강]: https://076923.github.io/posts/AI-2/
[6강]: https://076923.github.io/posts/Python-pytorch-6/
