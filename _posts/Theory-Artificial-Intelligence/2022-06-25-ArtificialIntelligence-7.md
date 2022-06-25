---
layout: post
title: "Artificial Intelligence Theory : 배치 정규화(Batch Normalization)"
tagline: "Batch Normalization"
image: /assets/images/ai.jpg
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['AI']
keywords: Artificial Intelligence, Batch Normalization
ref: Theory-AI
category: Theory
permalink: /posts/AI-7/
comments: true
toc: true
plotly: true
---

## 배치 정규화(Batch Normalization)

`배치 정규화(Batch Normalization)`란 **배치(Batch)** 단위의 입력값을 **정규화(Normalization)**해 학습시 발생하는 `기울기 폭주(Exploding Gradient)`나 `기울기 소실(Vanishing Gradient)` 문제를 완화하기 위해 활용합니다.

일반적으로 기계 학습에서는 배치(Batch) 단위로 학습을 진행하게 됩니다. 이때 각 배치마다의 입력 데이터의 분포가 다르므로, 계층(Layer)마다 전달되는 데이터의 분포도 달라집니다.

이로 인해 `내부 공변량 변화(Internal Covariate Shift)`가 발생하여, `은닉층(Hidden Layer)`에서 다음 은닉층으로 전달될 때 입력값이 균일해지지 않아 `가중치(Weight)`가 제대로 갱신(Update)되지 않을 수 있습니다. 

그러므로, 배치마다 은닉층에 전달되는 데이터의 분포가 다르더라도 배치별로 값을 정규화해 학습을 안정화합니다.

`배치 정규화(Batch Normalization)`는 데이터의 분포를 **평균이 1이며 분산이 0인 값으로 정규화합니다.**

즉, 입력값이 $$ [100, 1, 1] $$ 이거나 $$ [1, 0.01, 0.01] $$ 이라면 이 두 배열의 값 모두 $$ [ 1.4142, -0.7071, -0.7071] $$의 값으로 정규화합니다.

- Tip : 배치 정규화 논문에서는 이미지 분류 모델에서 배치 정규화를 사용시 14배 더 적은 훈련 단계로 동일한 정확도를 달성하였습니다.
- Tip : 배치 정규화를 통해 더 높은 `학습률(Learning Rate)`를 사용할 수 있으며, `가중치 초기화(Weight Initialization)`에 민감하게 반응하지 않습니다. 

<br>
<br>

## 배치 정규화(Batch Normalization) 풀이

$$ y = \frac{x - \mathrm{E}[X]}{\sqrt{\mathrm{Var}[X] + \epsilon}} * \gamma + \beta $$

<br>

`배치 정규화(Batch Normalization)`는 위와 같은 수식을 활용해 출력값을 정규화합니다.

$$ x $$는 입력값을 의미하며, $$ y $$는 배치 정규화가 적용된 결괏값입니다.

$$ \mathrm{E}[X] $$는 `산술 평균(Arithmetic Mean)`을 의미하며, $$ \mathrm{Var}[X] $$는 `분산(Variance)`을 의미합니다.

$$ X $$는 **전체 모집단**을 의미하며, 배치(Batch)에서 사용된 데이터의 은닉층(Hidden Layer) 출력값을 의미합니다.

$$ \epsilon $$은 분모가 0이 되는 현상을 방지합니다. 기본값은 $$ 10^{-5}(0.00001) $$으로 사용합니다.

$$ \gamma $$와 $$ \beta $$는 학습 가능한 매개변수로서 `활성화 함수(Activation Function)`의 음수의 영역을 처리할 수 있도록 `스케일(Scale)` 값과 `시프트(Shift)` 값으로 활용됩니다.

$$ \gamma $$의 초깃값은 $$ 1 $$이며, $$ \beta $$의 초깃값은 $$ 0 $$으로 할당됩니다.

이제 다음과 같은 텐서에 배치 정규화를 적용해보겠습니다.

<br>

### 입력값

{% highlight Python %}

x = torch.FloatTensor(
    [
        [-0.6577, -0.5797, 0.6360],
        [0.7392, 0.2145, 1.523],
        [0.2432, 0.5662, 0.322]
    ]
)

{% endhighlight %}

$$ $$

$$
\begin{multline}
\shoveleft X_1 = [ -0.6577, 0.7392, 0.2432 ] \\
\shoveleft X_2 = [ -0.5797, 0.2145, 0.5662 ] \\
\shoveleft X_3 = [ 0.636, 1.523, 0.322 ]
\end{multline}
$$

<br>

### 계산 방법(Method of calculation)

$$ y_i = \frac{x_i - \mathrm{E}[X]}{\sqrt{\mathrm{Var}[X] + \epsilon}} * \gamma + \beta $$

<br>

위 수식을 활용해 $$ X_1 $$에 대한 $$ x_1, x_2, x_3 $$의 값에 `배치 정규화(Batch Normalization)`를 적용합니다.

먼저, $$ \mathrm{E}[X] $$와 $$ \mathrm{Var}[X] $$를 계산합니다.

<div style="display: flex;margin-left: 18px;">
$$ \begin{align} \mathrm{E}[X] & =\frac{-0.6577 + 0.7392 + 0.2432}{3}\\\\ & \simeq 0.1082 \end{align} $$
</div>

<div style="display: flex;margin-left: 18px;">
$$ \begin{align} \mathrm{Var}[X] & =\frac{(\mathrm{E}[X] + 0.6577)^2 + (\mathrm{E}[X] - 0.7392)^2 + (\mathrm{E}[X] - 0.2432)^2}{3} \\\\ & =\frac{(0.1082 + 0.6577)^2 + (0.1082 - 0.7392)^2 + (0.1082 - 0.2432)^2}{3} \\\\ & \simeq 0.3343 \end{align} $$
</div>

평균과 분산에 대한 계산을 완료했다면, 배치 정규화 수식을 적용해 새로운 값을 할당합니다.

먼저 $$ x_1 $$값에 대한 배치 정규화를 수행합니다.

<div style="display: flex;margin-left: 18px;">
$$ \begin{align} y_1 & = \frac{x_1 - \mathrm{E}[X]}{\sqrt{\mathrm{Var}[X] + \epsilon}} * \gamma + \beta \\\\ & = \frac{-0.6577 - 0.1082}{\sqrt{0.3343 + \epsilon}} * \gamma + \beta \\\\ & =\frac{-0.6577 - 0.1082}{\sqrt{0.3343 + 0.00001}} * 1 + 0 \\\\ & = -1.3246 \end{align} $$
</div>

위와 동일한 방법으로 $$ x_2, x_3 $$에 배치 정규화를 수행한다면, $$ X_1 $$에 대한 새로운 값($$ Y_1 $$)을 계산할 수 있습니다.

$$
\begin{multline}
\shoveleft Y_1 = [ -1.3246, 1.0912, 0.2334 ]
\end{multline}
$$

$$ Y_1 $$에 대해 다시 평균과 분산을 계산한다면 평균은 0.0, 분산은 1.0으로 정규화됩니다.

$$ X_2, X_3 $$에도 동일한 방법을 계산한다면, 최종 배치 정규화 결과는 다음과 같습니다.

$$
\begin{multline}
\shoveleft Y_1 = [ -1.3246, 1.0912, 0.2334 ] \\
\shoveleft Y_2 = [ -1.3492, 0.3077, 1.0415 ] \\
\shoveleft Y_3 = [ -0.3756, 1.3685, -0.9930 ]
\end{multline}
$$

<br>

{% highlight Python %}

x = torch.FloatTensor(
    [
        [-0.6577, -0.5797, 0.6360],
        [0.7392, 0.2145, 1.523],
        [0.2432, 0.5662, 0.322]
    ]
)

print(nn.BatchNorm1d(3)(x))

{% endhighlight %}

**결과**
:    
tensor([[-1.3246, -1.3492, -0.3756],<br>
&emsp;&emsp;&emsp;&emsp;[ 1.0912,  0.3077,  1.3685],<br>
&emsp;&emsp;&emsp;&emsp;[ 0.2334,  1.0415, -0.9930]], grad_fn=&lt;NativeBatchNormBackward&gt;)<br>
<br>

위와 같이 배치 정규화를 적용할 수 있습니다.

배치 정규화에 사용되는 $$ \gamma $$와 $$ \beta $$는 `역전파(Back Propagation)` 과정에서 값이 갱신됩니다.

<br>
<br>

* Writer by : 윤대희
