---
layout: post
title: "Artificial Intelligence Theory : 활성화 함수(Activation Function)"
tagline: "Activation Function"
image: /assets/images/ai.jpg
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['AI']
keywords: Artificial Intelligence, Activation Function, Step Function, Linear Function, Sigmoid Function, Tanh Function, Threshold Function, ReLU Function, ELU Function, PReLU Function, LeakyReLU Function, Softmax Function, LogSoftmax Function
ref: Theory-AI
category: Theory
permalink: /posts/AI-6/
comments: true
toc: true
plotly: true
---

## 활성화 함수(Activation Function)

`활성화 함수(Activation Function)`란 `인공 신경망(Neural Network)`에서 사용되는 `은닉층(Hidden Layer)`을 활성화하기 위한 함수입니다.

[순전파(Forward Propagation) & 역전파(Back Propagation)][5강]의 풀이에서 처럼 `가중치(Weight)`와 `편향(Bias)`을 **갱신(Update)**하기 위해 사용되는 주요한 함수입니다.

시스템(System)에 포함된 노드(Node)는 출력값에 동일한 영향을 미치지 않습니다. 즉, 각 노드마다 전달되어야 하는 정보량이 다르다는 의미가 됩니다.

예를 들어 **나의 출근 시간을 예측하기 위한 모델(Model)**을 구현한다고 가정해보겠습니다.

입력값이 **일어난 시간(x1), 기상 상태(x2)**라면, 기상 상태(x2)값 보다는 일어난 시간(x1)이 더 영향력이 크다는 것을 직관적으로 알 수 있습니다.

그러므로 연산 과정에서 **일어난 시간(x1)**은 더 많이 `활성화(Activate)`되어야 하며, **기상 상태(x2)**는 비교적 `비활성화(Deactivate)` 되어야 합니다.

`활성화 함수(Activation Function)`는 `비선형(Non-linear)` 구조를 가져 역전파 과정에서 미분값을 통해 학습이 진행될 수 있게 합니다.

만약, 활성화 함수가 `선형(Linear)` 구조라면, 미분 과정에서 항상 상수가 나오게 되므로 학습이 진행되지 않습니다.

- Tip : `활성화 함수(Activation Function)`는 입력을 `정규화(Normalization)`하는 과정으로 볼 수 있습니다.

<br>
<br>

## 선형 함수(Linear Function)

<center>
<div id="LinearPlot" style="width:100%;max-width:700px"></div>
<script>
var LinearX=[];var LinearY=[];for(var x=-7;x<=7;x+=0.1){LinearX.push(x);LinearY.push(eval("x"))}
Plotly.newPlot("LinearPlot",[{x:LinearX,y:LinearY,mode:"lines"}],{xaxis:{title:"x",range:[-7,7]},yaxis:{title:"y",range:[-7,7]}})
</script>
</center>

$$ Linear(x) = ax $$

<br>

`선형 함수(Linear Function)`는 `활성화 함수(Activation Function)`에 **포함되지 않습니다.**

`역전파(Back Propabation)` 과정에서 미분을 진행할 때, 항상 같은 상수($$ a $$)를 반환하게 되므로 학습이 진행되지 않습니다.

[순전파(Forward Propagation) & 역전파(Back Propagation)][5강]의 `활성화 함수(Activation Function)`를 `시그모이드(Sigmoid)`가 아닌 **선형 함수**($$ Linear(x) = x $$)로 풀이한다면 가중치가 갱신되지 않고 항상 같은 값을 반환합니다.

<br>
<br>

## 계단 함수(Step Function)

<center>
<div id="StepPlot" style="width:100%;max-width:700px"></div>
<script>
var StepX=[];var StepY=[];for(var x=-7;x<=0;x+=0.1){StepX.push(x);StepY.push(0)}
for(var x=0;x<=7;x+=0.1){StepX.push(x);StepY.push(1)}
Plotly.newPlot("StepPlot",[{x:StepX,y:StepY,mode:"lines"}],{xaxis:{title:"x",range:[-7,7]},yaxis:{title:"y",range:[-0.01,1.01]}})
</script>
</center>

$$ Step(x) = \begin{cases} \ 1 & \text{if:} \ x \geq 0 \\ \ 0 & \text{else:} \ \text{otherwise} \end{cases}$$

<br>

`계단 함수(Step Function)`는 `이진 활성화 함수(Binary Activation Function)`라고도 하며, `퍼셉트론(Perceptron)`에서 최초로 사용한 활성화 함수입니다.

`계단 함수(Step Function)` 입력값의 합이 임곗값을 넘으면 0을 출력하고, 넘지 못하면 1을 출력합니다.

`딥러닝(Deep Learning)` 모델에서는 사용되지 않는 함수로 임곗값에서 `불연속점(Point of discontinuity)`을 가지므로 미분이 불가능해 학습을 진행할 수 없습니다.

또한, `역전파(Back Propabation)` 과정에서 **데이터가 극단적으로 변경되기 때문에 적합하지 않습니다.**

<br>
<br>

## 비선형 활성화 함수(Non-linear Activations Function)

### 임곗값 함수(Threshold Function)

<center>
<div id="ThresholdPlot" style="width:100%;max-width:700px"></div>
<script>
var ThresholdX=[];var ThresholdY=[];for(var x=-7;x<=0;x+=0.1){ThresholdX.push(x);ThresholdY.push(-5)}
for(var x=0;x<=7;x+=0.1){ThresholdX.push(x);ThresholdY.push(eval("x"))}
Plotly.newPlot("ThresholdPlot",[{x:ThresholdX,y:ThresholdY,mode:"lines"}],{xaxis:{title:"x",range:[-7,7]},yaxis:{title:"y",range:[-7,7]}})
</script>
</center>

$$ Threshold(x) = \begin{cases} \ x & \text{if:} \ x > threshold \\ \ value & \text{else:} \ \text{otherwise} \end{cases}$$

<br>

`임곗값 함수(Threshold Function)`는 `임곗값(threshold)`보다 크다면 `입력값(x)`을 그대로 전달하고, `임곗값(threshold)`보다 작다면 `특정값(value)`으로 변경합니다.

`선형 함수(Linear Function)`와 `계단 함수(Step Function)`의 조합으로 볼 수 있습니다.

앞선 함수들의 문제점을 그대로 가지고 있어, 특별한 경우가 아니라면 사용되지 않는 함수입니다.

- Tip : 예제의 그래프에서 `임곗값(threshold)`은 $$ 0 $$이며, `특정값(value)`은 $$ -5 $$입니다. 

<br>

### 시그모이드 함수(Sigmoid Function)

<center>
<div id="SigmoidPlot" style="width:100%;max-width:700px"></div>
<script>
var SigmoidX=[];var SigmoidY=[];for(var x=-7;x<=7;x+=0.1){SigmoidX.push(x);SigmoidY.push(eval("1/(1+Math.exp(-x))"))}
Plotly.newPlot("SigmoidPlot",[{x:SigmoidX,y:SigmoidY,mode:"lines"}],{xaxis:{title:"x",range:[-7,7]},yaxis:{title:"y",range:[-0.01,1.01]}})
</script>
</center>

$$ Sigmoid(x) = \sigma(x) = \frac{1}{1+e^{-x}} $$

<br>

`시그모이드 함수(Sigmoid Function)`는 미분식이 단순한 형태로 입력값에 따라 출력값이 급격하게 변하지 않습니다.

출력값이 0 ~ 1의 범위를 가지므로 `기울기 폭주(Exploding Gradient)` 현상을 방지할 수 있습니다.

하지만, 출력값이 0 ~ 1의 범위를 가지는 만큼 매우 큰 입력값이 입력되어도 최대 1의 값을 갖게되어 `기울기 소실(Vanishing Gradient)`이 발생합니다.

출력값의 중심이 0이 아니므로 입력 데이터가 항상 양수인 경우라면, `기울기(Gradient)`는 모두 양수 또는 음수가 되어, 기울기가 지그재그 형태로 변동하는 문제점이 발생해 학습 효율성을 감소시킵니다.

`신경망(Neural Network)`은 `기울기(Gradient)`를 이용해 최적화된 값을 찾아 가는데, 계층(Layer)이 많아지면 점점 값이 0에 수렴되는 문제가 발생해 성능이 떨어지게 됩니다.

이러한 이유로 `은닉층(Hidden Layer)`에서는 `활성화 함수(Activation Function)`로 사용하지 않으며, 주로 `출력층(Output Layer)`에서만 사용됩니다.

<br>

### 하이퍼볼릭 탄젠트 함수(Hyperbolic Tangent Function)

<center>
<div id="TanhPlot" style="width:100%;max-width:700px"></div>
<script>
var TanhX=[];var TanhY=[];for(var x=-4;x<=4;x+=0.1){TanhX.push(x);TanhY.push(eval("(Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x))"))}
Plotly.newPlot("TanhPlot",[{x:TanhX,y:TanhY,mode:"lines"}],{xaxis:{title:"x",range:[-4,4]},yaxis:{title:"y",range:[-1.01,1.01]}})
</script>
</center>

$$ Tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$

<br>

`하이퍼볼릭 탄젠트 함수(Hyperbolic Tangent Function)`는 `시그모이드 함수(Sigmoid Function)`와 유사한 형태를 지니지만, 출력값의 중심이 0을 갖습니다.

또한, 출력값이 -1 ~ 1의 범위를 가지므로 `시그모이드 함수(Sigmoid Function)`에서는 발생하지 않는 음수값을 반환할 수 있습니다.

출력값의 범위가 더 넓고 다양한 형태로 활성화 시킬 수 있으므로 `기울기 소실(Vanishing Gradient)`이 비교적 덜 발생합니다.

하지만, `하이퍼볼릭 탄젠트 함수(Hyperbolic Tangent Function)`도 `시그모이드 함수(Sigmoid Function)`와 마찬가지로 입력값($$ x $$)가 **4보다 큰 경우 출력값이 1에 수렴하므로 동일하게 기울기 소실이 발생합니다.**

<br>

### ReLU 함수(Rectified Linear Unit Function)

<center>
<div id="ReLUPlot" style="width:100%;max-width:700px"></div>
<script>
var ReLUX=[];var ReLUY=[];for(var x=-7;x<=0;x+=0.1){ReLUX.push(x);ReLUY.push(0)}
for(var x=0;x<=7;x+=0.1){ReLUX.push(x);ReLUY.push(eval("x"))}
Plotly.newPlot("ReLUPlot",[{x:ReLUX,y:ReLUY,mode:"lines"}],{xaxis:{title:"x",range:[-4,4]},yaxis:{title:"y",range:[-1,4]}})
</script>
</center>

$$ ReLU(x) = \begin{cases} \ x & \text{if:} \ x > 0 \\ \ 0 & \text{else:} \ \text{otherwise} \end{cases}$$

`ReLU 함수(Rectified Linear Unit Function)`는 0보다 작거나 같으면 0을 반환하며, 0보다 크면 선형 함수에 값을 대입하는 구조를 갖습니다.

`시그모이드 함수(Sigmoid Function)`나 `하이퍼볼릭 탄젠트 함수(Hyperbolic Tangent Function)`는 출력값이 제한되어 `기울기 소실(Vanishing Gradient)`이 발생하지만, ReLU 함수(Rectified Linear Unit Function)는 선형 함수에 대입하므로 입력값이 양수라면 출력값이 제한되지 않아 기울기 소실이 발생하지 않습니다.

또한, 수식 또한 매우 간단해 `순전파(Forward Propagation)`나 `역전파(Back Propagation)` 과정의 연산이 매우 빠릅니다.

하지만 입력값이 음수인 경우 항상 0을 반환하므로, `가중치(Weight)`나 `편향(Bias)`이 갱신(Update)되지 않을 수 있습니다.

만약 가중치의 합이 음수가 되면, 해당 뉴런은 더 이상 값을 갱신(Update)하지 않아 `죽은 뉴런(Dead Neuron, Dying ReLU)`이 됩니다.

<br>

### LeakyReLU 함수(Leaky Rectified Linear Unit Function)

<center>
<div id="LeakyReLUPlot" style="width:100%;max-width:700px"></div>
<script>
var LeakyReLUX=[];var LeakyReLUY=[];for(var x=-7;x<=0;x+=0.1){LeakyReLUX.push(x);LeakyReLUY.push(eval("0.2*x"))}
for(var x=0;x<=7;x+=0.1){LeakyReLUX.push(x);LeakyReLUY.push(eval("x"))}
Plotly.newPlot("LeakyReLUPlot",[{x:LeakyReLUX,y:LeakyReLUY,mode:"lines"}],{xaxis:{title:"x",range:[-4,4]},yaxis:{title:"y",range:[-4,4]}})
</script>
</center>

$$ LeakyReLU(x) = \begin{cases} \ x & \text{if:} \ x > 0 \\ \ \text{negative_slope} \times x & \text{else:} \ \text{otherwise} \end{cases}$$

<br>

`LeakyReLU 함수(Leaky Rectified Linear Unit Function)`는 `음수 기울기(negative slope)`를 제어하여, Dying ReLU 현상을 방지하기 위해 사용합니다.

양수인 경우, `ReLU 함수(Rectified Linear Unit Function)`와 동일하지만 음수인 경우, 작은 값이라도 출력시켜 기울기를 갱신하게 합니다.

<br>

### PReLU 함수(Parametric Rectified Linear Unit Fucntion)

<center>
<div id="PReLUPlot" style="width:100%;max-width:700px"></div>
<script>
var PReLUX=[];var PReLUY=[];for(var x=-7;x<=0;x+=0.1){PReLUX.push(x);PReLUY.push(eval("0.2*x"))}
for(var x=0;x<=7;x+=0.1){PReLUX.push(x);PReLUY.push(eval("x"))}
Plotly.newPlot("PReLUPlot",[{x:PReLUX,y:PReLUY,mode:"lines"}],{xaxis:{title:"x",range:[-4,4]},yaxis:{title:"y",range:[-4,4]}})
</script>
</center>

$$ PReLU(x) = \begin{cases} \ x & \text{if:} \ x > 0 \\ \ a \times x & \text{else:} \ \text{otherwise} \end{cases}$$

<br>

`PReLU 함수(Parametric Rectified Linear Unit Fucntion)`는 `LeakyReLU 함수(Leaky Rectified Linear Unit Function)`와 형태가 동일하지만, `음수 기울기(negative slope)` 값을 고정값이 아닌, 학습을 통해 갱신되는 값으로 간주합니다.

즉, `PReLU 함수(Parametric Rectified Linear Unit Fucntion)`의 음수 기울기(negative slope, $$ a $$)는 지속적으로 값이 변경됩니다.

값이 지속적으로 갱신되기 매개변수이므로, 학습 데이터 세트에 영향을 받습니다.

<br>

### ELU 함수(Exponential Linear Unit Fucntion)

<center>
<div id="EReLUPlot" style="width:100%;max-width:700px"></div>
<script>
var EReLUX=[];var EReLUY=[];for(var x=-7;x<=0;x+=0.1){EReLUX.push(x);EReLUY.push(eval("1*(Math.exp(x)-1)"))}
for(var x=0;x<=7;x+=0.1){EReLUX.push(x);EReLUY.push(eval("x"))}
Plotly.newPlot("EReLUPlot",[{x:EReLUX,y:EReLUY,mode:"lines"}],{xaxis:{title:"x",range:[-4,4]},yaxis:{title:"y",range:[-4,4]}})
</script>
</center>

$$ ELU(x) = \begin{cases} \ x & \text{if:} \ x > 0 \\ \ \text{negative_slope} \times (e^{x} - 1) & \text{else:} \ \text{otherwise} \end{cases}$$

<br>

`ELU 함수(Exponential Linear Unit Fucntion)`는 지수 함수를 사용하여 부드러운 곡선의 형태를 갖습니다.

기존 `ReLU 함수(Rectified Linear Unit Function)`와 변형 함수는 0에서 끊어지게 되는데, ELU 함수(Exponential Linear Unit Fucntion)는 음의 기울기에서 비선형 구조를 갖습니다.

그러므로 입력값이 0인 경우에도 출력값이 급변하지 않아, `경사 하강법(Gradient Descent)`의 수렴 속도가 비교적 더 빠릅니다.

하지만, 더 복잡한 연산을 진행하게 되므로 학습 속도는 더 느려지게 됩니다.

- Tip : **수렴 속도**는 몇 번의 학습으로 최적화 된 값을 찾는가를 의미한다면, **학습 속도**는 학습이 완료되기 까지의 전체 속도를 의미합니다.

<br>

### 소프트맥스 함수(Softmax Fucntion)

$$ p_{k} = \frac{e^{z_k}}{\sum_{i=1}^{n} e^{z_i} } $$

`소프트맥스 함수(Softmax Function)`는 $$ n $$ 차원 벡터에서 특정 출력 값이 $$ k $$ 번째 클래스에 속할 확률을 계산합니다.

클래스에 속할 확률을 계산하는 활성화 함수이므로, `은닉층(Hidden Layer)`에서 사용하지 않고 `출력층(Output Layer)`에서 사용됩니다.

`분류 모델(Classification Model)`에서 사용되며, [제 13강 - 다중 분류(Multiclass Classification)][13강]에서 자세한 풀이를 확인해 보실 수 있습니다.

이외에도 `소프트민 함수(Softmin Function)`, `로그 소프트맥스 함수(LogSoftmax Function)` 등이 있습니다.

<br>
<br>

* Writer by : 윤대희

[5강]:https://076923.github.io/posts/AI-5/
[13강]:https://076923.github.io/posts/Python-pytorch-13/