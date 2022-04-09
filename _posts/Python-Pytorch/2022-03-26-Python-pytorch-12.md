---
layout: post
title: "Python Pytorch 강좌 : 제 12강 - 이진 분류(Binary Classification)"
tagline: "Python PyTorch Binary Classification"
image: /assets/images/pytorch.webp
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['PyTorch']
keywords: Python, Python Pytorch, Pytorch Binary Classification, Pytorch Sigmoid Function, Pytorch Binary Cross Entropy
ref: Python-PyTorch
category: Python
permalink: /posts/Python-pytorch-12/
comments: true
toc: true
plotly: true
---

## 이진 분류(Binary Classification)

`이진 분류(Binary Classification)`란 규칙에 따라 입력된 값을 **두 그룹으로 분류하는 작업을 의미합니다.**

구분하려는 결과가 **참(True)** 또는 **거짓(False)**의 형태나 **A 그룹** 또는 **B 그룹**으로 데이터를 나누는 경우를 의미합니다.

분류 결과가 맞다면 `1(True, A 그룹에 포함)`을 반환하며, 아니라면 `0(False, A 그룹에 포함되지 않음)`을 반환하는 형태가 됩니다.

즉, 결과를 이분화하는 작업을 수행합니다. 만약, 분류해야하는 그룹이 3 종류 이상이라면, `다중 분류(Multiclass Classification)`를 의미합니다.

`참(True)` 또는 `거짓(False)`으로 결과를 분류하기 때문에 `논리 회귀(Logistic Regression)` 또는 `논리 분류(Logistic Classification)`라고도 부릅니다. 

<center>
<div id="classificationPlot" style="width:100%;max-width:700px"></div>
<script>
var x=[0,1,2,3,4,5,6,7],y=[0,0,0,0,1,1,1,1],data1=[{x:x,y:y,mode:"lines",name:"Sigmoid #1",line:{width:3}}],layout1={yaxis:{range:[-.02,1.02]}};Plotly.newPlot("classificationPlot",data1,layout1);
</script>
</center>

`이진 분류(Binary Classification)`를 그래프화한다면 위 그림과 같은 형태가 됩니다.

구분하려는 값을 X, 분류된 결과를 Y라고 표현한다면 **3 이하의 값은 거짓(False, 0)**이 되며, **4 이상의 값은 값은 참(True, 1)**이 되는 형태입니다.

입력 데이터가 정수(int) 형태로 입력된다면 간단하게 구분할 수 있지만, **3.4**나 **3.7** 등 모호한 위치에 있다면 특정 그룹으로 나누기에 모호한 값이 됩니다.

하지만, X가 **3.4**일 때 Y의 값은 **0.4**가 되며, X가 **3.7**일 때 Y의 값은 **0.7**을 갖게되는 것을 알 수 있습니다.

즉, 관측치는 **0 ~ 1** 범위로 예측된 점수를 반환하며, 데이터를 0 또는 1로 분류하기 위해 임곗값을 0.5로 설정합니다.

그러므로, **Y가 0.5 보다 작은 값은 거짓이 되며, 0.5 보다 큰 값은 참이 됩니다.**

위와 같이 Y값이 0 ~ 1 범위를 갖게 하기 위해 `활성화 함수(Activation Function)` 중 하나인 `시그모이드 함수(Sigmoid Function)`를 적용합니다.

<br>
<br>

## 시그모이드 함수(Sigmoid Function)

먼저, `활성화 함수(Activation Function)`란 입력 데이터를 정해진 수식에 따라 값을 변환하는 식(Equation)을 의미합니다.

활성화 함수는 비선형으로 이뤄져 있어, 활성화 함수를 적용하면 입력값에 대한 출력값이 `비선형(Nonlinear)`으로 변환됩니다.

`시그모이드 함수(Sigmoid Function)`는 **S자형 곡선**의 모양의 형태로 반환값은 **0 ~ 1** 또는 **-1 ~ 1**의 범위를 갖습니다.

시그모이드 함수의 수식은 다음과 같이 표현합니다.

$$ Sigmoid(x) = \frac{1}{1+e^{-x}} $$

시그모이드 함수의 $$ x $$의 계수에 따라 **S자형 곡선**이 완만한 경사를 갖게 될지, 급격한 경사를 갖게 될지 설정할 수 있습니다.

<center>
<div id="sigmoidPlot" style="width:100%;max-width:700px"></div>
<script>
for(var exp1="1/(1+Math.exp(-0.5*x))",exp2="1/(1+Math.exp(-x))",exp3="1/(1+Math.exp(-2*x))",xValues=[],yValues1=[],yValues2=[],yValues3=[],x=-10;x<=10;x+=.1)xValues.push(x),yValues1.push(eval(exp1)),yValues2.push(eval(exp2)),yValues3.push(eval(exp3));var data2=[{x:xValues,y:yValues1,mode:"lines",name:"Sigmoid #1"},{x:xValues,y:yValues2,mode:"lines",name:"Sigmoid #2"},{x:xValues,y:yValues3,mode:"lines",name:"Sigmoid #3"}],layout2={xaxis:{title:"X"},yaxis:{title:"Y"}};Plotly.newPlot("sigmoidPlot",data2,layout2);
</script>
</center>

$$
\begin{multline}
\shoveleft Sigmoid\ \text{#1}(x) = \frac{1}{1+e^{-0.5x}}\\ 
\shoveleft Sigmoid\ \text{#2}(x) = \frac{1}{1+e^{-x}}\\
\shoveleft Sigmoid\ \text{#3}(x) = \frac{1}{1+e^{-2x}}
\end{multline}
$$

<br>

`시그모이드 함수(Sigmoid Function)`의 $$ x $$**의 계수가 0에 가까워질수록 완만한 경사를 갖게되며, 0에서 멀어질수록 급격한 경사를 갖게됩니다.**

`시그모이드 함수(Sigmoid Function)`는 주로 `로지스틱 회귀(Logistic Regression)`에 사용됩니다.

로지스틱 회귀는 독립 변수(x)의 선형 결합을 활용하여 결과를 예측합니다.

종속 변수(Y)를 범주형 데이터를 대상으로 계산하기 때문에 해당 데이터의 결과가 특정 분류로 나뉘게 됩니다.

즉, 로지스틱 회귀는 `분류(Classification)`에서도 사용될 수 있습니다. 

그러므로, `시그모이드 함수(Sigmoid Function)`를 통해 나온 출력값이 **0.5**보다 낮다면 `거짓(False)`으로 분류하며, **0.5**보다 크다면 `참(True)`으로 분류합니다.

시그모이드 함수는 유연한 미분 값 가지므로, 입력에 따라 값이 급격하게 변하지 않는 장점이 있습니다.

또한, 출력값의 범위가 0 ~ 1 사이로 제한됨으로써 정규화 중 `기울기 폭주(Exploding Gradient)` 문제를 방지하고 미분식이 단순한 형태 지닙니다.

하지만, 기울기 폭주를 방지하는 대신 `기울기 소실(Vanishing Gradient)` 문제 발생합니다.

`신경망(Neural Network)`은 `기울기(Gradient)`를 이용해 최적화된 값을 찾아 가는데, 층(Layer)이 많아지면 점점 값이 0에 수렴되는 문제가 발생해 성능이 떨어지게 됩니다.

그 외에도 Y값의 의 중심이 0이 아니므로 입력 데이터가 항상 양수인 경우라면, `기울기(Gradient)`는 모두 양수 또는 음수가 되어, 기울기가 지그재그 형태로 변동하는 문제점이 발생해 학습 효율성을 감소시킵니다.

<br>
<br>

## 이진 교차 엔트로피(Binary Cross Entropy)

`이진 분류(Binary Classification)`에서 사용되는 `시그모이드 함수(Sigmoid Function)`의 예측값은 0 ~ 1의 범위를 가지며, 실젯값도 0 ~ 1의 범위를 갖습니다.

앞선 예제에서는 `비용 함수(Cost Function)`를 `평균 제곱 오차(Mean Squared Error, MSE)` 사용하여 오차를 계산했습니다.

`평균 제곱 오차(Mean Squared Error, MSE)` 함수를 `이진 분류(Binary Classification)`에서도 사용한다면 **좋은 결과를 얻기 어렵습니다.**

$$
\begin{multline}
\shoveleft MSE = (\hat{Y_{i}} - Y_{i})^2\\
\shoveleft MSE = (0.999999999999 - 1)^2 \simeq 0\\ 
\shoveleft MSE = (0.000000000001 - 0)^2 \simeq 0\\
\shoveleft MSE = (0.000000000001 - 1)^2 \simeq 1
\end{multline}
$$

* Tip : `시그모이드 함수(Sigmoid Function)`의 반환값은 $$ 0 < \hat{Y_{i}} < 1 $$을 갖습니다.

<br>

위 샘플에서 확인할 수 있듯이 `평균 제곱 오차(Mean Squared Error, MSE)`를 적용하게 되면, `극솟값(Local Minimum)`이 산발적으로 발생합니다.

이러한 경우를 방지하고자, `이진 교차 엔트로피(Binary Cross Entropy)`를 오차 함수로 사용합니다.

<center>
<div id="bcePlot" style="width:100%;max-width:700px"></div>
<script>
for(var xValues2=[],yValues4=[],yValues5=[],x=0;x<=1;x+=.01)xValues2.push(x),yValues4.push(eval("-0.25*Math.log(x)")),yValues5.push(eval("-0.25*Math.log(1-x)"));var data3=[{x:xValues2,y:yValues4,mode:"lines",name:"BCE #1"},{x:xValues2,y:yValues5,mode:"lines",name:"BCE #2"}],layout3={xaxis:{title:"H(x)"},yaxis:{title:"Cost",range:[0,1]}};Plotly.newPlot("bcePlot",data3,layout3);</script>
</center>

$$
\begin{multline}
\shoveleft BCE \ \text{#1} = - Y_{i} \cdot log (\hat{Y_{i}}) \\ 
\shoveleft BCE \ \text{#2} = - (1 - Y_{i}) \cdot log (1 - \hat{Y_{i}}) \\ \\
\shoveleft BCE \ loss = BCE \ \text{#1} + BCE \ \text{#2} \\
\shoveleft BCE \ loss = - ( Y_{i} \cdot log (\hat{Y_{i}}) + (1 - Y_{i}) \cdot log (1 - \hat{Y_{i}}))
\end{multline}
$$

<br>

`이진 교차 엔트로피(Binary Cross Entropy)`는 로그 함수를 활용하여 오차 함수를 구현합니다.

두 가지의 로그 함수를 교차하여 오차를 계산합니다.

`BCE #1` 수식은 **실젯값**($$ Y_{i} = 1$$)이 **1**일 때 적용하는 수식이며, `BCE #2` 수식은 **실젯값**($$ Y_{i} = 0$$)이 **0**일 때 적용하는 수식입니다.

$$
\begin{multline}
\shoveleft BCE \ \text{#1} = - Y_{i} \cdot log (\hat{Y_{i}}) \\ 
\shoveleft BCE \ \text{#1} = - 1 \cdot log(0.999999999999) \simeq 0\\ 
\shoveleft BCE \ \text{#1} = - 1 \cdot log(0.000000000001) \simeq 12
\end{multline}
$$

$$
\begin{multline}
\shoveleft BCE \ \text{#2} = - (1 - Y_{i}) \cdot log (1 - \hat{Y_{i}}) \\
\shoveleft BCE \ \text{#2} = - (1 - 0) \cdot log(0.999999999999) = 0\\ 
\shoveleft BCE \ \text{#2} = - (1 - 0) \cdot log(0.000000000001) \simeq 12
\end{multline}
$$

<br>

로그 함수는 **로그의 진수가 0에 가까워질수록 무한대로 발산합니다.**

기존의 `평균 제곱 오차(Mean Squared Error, MSE)` 함수는 **명확하게 불일치 하는 경우에도 높은 손실(Loss) 값을 반환하지 않습니다.**

하지만, **로그 함수의 경우 불일치하는 비중이 높을수록 높은 손실(Loss) 값을 반환합니다.**

로그 함수의 경우 한 쪽 방향으로는 무한대로 이동하며 다른 한 쪽 방향으로는 0에 가까워지기 때문에 기울기가 0이 되는 지점을 찾기 위해 두 가지의 로그 함수를 하나로 합쳐 사용합니다.

그래프에서 확인할 수 있듯이, `BCE #1`과 `BCE #2`를 하나의 수식으로 합친다면, 기울기가 0이되는 지점을 찾을 수 있게 됩니다.

최종으로 반환되는 `이진 교차 엔트로피(Binary Cross Entropy)` 함수는 `오차(Error)` 계산하기 위해 **각 손실(Loss) 값의 평균을 반환합니다.**

$$ BCE = - \frac{1}{n} \sum_{i=1}^{n} ( Y_{i} \cdot log (\hat{Y_{i}}) + (1 - Y_{i}) \cdot log (1 - \hat{Y_{i}})) $$

<br>
<br>

## 데이터 세트

학습에 사용된 `dataset.csv`는 아래 링크에서 다운로드할 수 있습니다.

> `Dataset 다운로드` : [다운로드][Dataset]

[Dataset]: https://github.com/076923/076923.github.io/raw/master/download/datasets/pytorch-12/dataset.csv

`dataset.csv`는 다음과 같은 형태로 구성되어 있습니다.

|    x   |    y   |    z   |   pass  |
| :----: | :----: | :----: | :-----: |
|   86   |   22   |   1    |  False  |
|   81   |   75   |   91   |  True   |
|   54   |   85   |   78   |  True   |
|   5    |   58   |   4    |  False  |
|   53   |   93   |   100  |  True   |
|   73   |   95   |   70   |  True   |
|   23   |   73   |   88   |  False  |
|   74   |   46   |   28   |  False  |
|  ...   |  ...   |   ...  |   ...   |


`x`, `y`, `z`와 `pass`의 관계는 `x`, `y`, `z`가 **모두 40 이상이며, 평균이 60 이상일 때** `True`를 반환합니다.

이 데이터를 활용하여 `이진 분류(Binary Classification)`를 `모델(Model)`로 구현해보도록 하겠습니다.

<br>
<br>

## 메인 코드

{% highlight Python %}

import torch
import pandas as pd
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        self.x1 = df.iloc[:, 0].values
        self.x2 = df.iloc[:, 1].values
        self.x3 = df.iloc[:, 2].values
        self.y = df.iloc[:, 3].values
        self.length = len(df)

    def __getitem__(self, index):
        x = torch.FloatTensor([self.x1[index], self.x2[index], self.x3[index]])
        y = torch.FloatTensor([self.y[index]])
        return x, y

    def __len__(self):
        return self.length


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layer = nn.Sequential(
          nn.Linear(3, 1),
          nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer(x)
        return x


train_dataset = CustomDataset("./dataset.csv")
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CustomModel().to(device)
criterion = nn.BCELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.0001)

for epoch in range(10000):
    cost = 0.0

    for x, y in train_dataloader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss

    cost = cost / len(train_dataloader)

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch : {epoch+1:4d}, Model : {list(model.parameters())}, Cost : {cost:.3f}")


with torch.no_grad():
    model.eval()
    inputs = torch.FloatTensor(
        [[89, 92, 75], [75, 64, 50], [38, 58, 63], [33, 42, 39], [23, 15, 32]]
    ).to(device)
    outputs = model(inputs)

    print("---------")
    print(outputs)
    print(outputs >= torch.FloatTensor([0.5]).to(device))


{% endhighlight %}
**결과**
:    
Epoch : 1000, Model : [Parameter containing:<br>
tensor([[0.0067, 0.0040, 0.0061]], device='cuda:0', requires_grad=True), Parameter containing:<br>
tensor([-0.5287], device='cuda:0', requires_grad=True)], Cost : 0.612<br>
Epoch : 2000, Model : [Parameter containing:<br>
tensor([[0.0073, 0.0043, 0.0058]], device='cuda:0', requires_grad=True), Parameter containing:<br>
tensor([-0.5963], device='cuda:0', requires_grad=True)], Cost : 0.608<br>
Epoch : 3000, Model : [Parameter containing:<br>
tensor([[0.0074, 0.0044, 0.0062]], device='cuda:0', requires_grad=True), Parameter containing:<br>
tensor([-0.6629], device='cuda:0', requires_grad=True)], Cost : 0.604<br>
Epoch : 4000, Model : [Parameter containing:<br>
tensor([[0.0079, 0.0046, 0.0067]], device='cuda:0', requires_grad=True), Parameter containing:<br>
tensor([-0.7286], device='cuda:0', requires_grad=True)], Cost : 0.591<br>
Epoch : 5000, Model : [Parameter containing:<br>
tensor([[0.0075, 0.0057, 0.0072]], device='cuda:0', requires_grad=True), Parameter containing:<br>
tensor([-0.7934], device='cuda:0', requires_grad=True)], Cost : 0.588<br>
Epoch : 6000, Model : [Parameter containing:<br>
tensor([[0.0083, 0.0058, 0.0074]], device='cuda:0', requires_grad=True), Parameter containing:<br>
tensor([-0.8573], device='cuda:0', requires_grad=True)], Cost : 0.582<br>
Epoch : 7000, Model : [Parameter containing:<br>
tensor([[0.0083, 0.0062, 0.0077]], device='cuda:0', requires_grad=True), Parameter containing:<br>
tensor([-0.9203], device='cuda:0', requires_grad=True)], Cost : 0.580<br>
Epoch : 8000, Model : [Parameter containing:<br>
tensor([[0.0089, 0.0062, 0.0082]], device='cuda:0', requires_grad=True), Parameter containing:<br>
tensor([-0.9825], device='cuda:0', requires_grad=True)], Cost : 0.571<br>
Epoch : 9000, Model : [Parameter containing:<br>
tensor([[0.0087, 0.0064, 0.0078]], device='cuda:0', requires_grad=True), Parameter containing:<br>
tensor([-1.0438], device='cuda:0', requires_grad=True)], Cost : 0.566<br>
Epoch : 10000, Model : [Parameter containing:<br>
tensor([[0.0093, 0.0068, 0.0082]], device='cuda:0', requires_grad=True), Parameter containing:<br>
tensor([-1.1043], device='cuda:0', requires_grad=True)], Cost : 0.562<br>
\-\-\-\-\-\-\-\-\-<br>
tensor([[0.7232],<br>
&emsp;&emsp;&emsp;&emsp;[0.6073],<br>
&emsp;&emsp;&emsp;&emsp;[0.5394],<br>
&emsp;&emsp;&emsp;&emsp;[0.4518],<br>
&emsp;&emsp;&emsp;&emsp;[0.3713]], device='cuda:0')<br>
tensor([[ True],<br>
&emsp;&emsp;&emsp;&emsp;[ True],<br>
&emsp;&emsp;&emsp;&emsp;[ True],<br>
&emsp;&emsp;&emsp;&emsp;[False],<br>
&emsp;&emsp;&emsp;&emsp;[False]], device='cuda:0')<br>
<br>

### 세부 코드

{% highlight Python %}

class CustomDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        self.x1 = df.iloc[:, 0].values
        self.x2 = df.iloc[:, 1].values
        self.x3 = df.iloc[:, 2].values
        self.y = df.iloc[:, 3].values
        self.length = len(df)

    def __getitem__(self, index):
        x = torch.FloatTensor([self.x1[index], self.x2[index], self.x3[index]])
        y = torch.FloatTensor([self.y[index]])
        return x, y

    def __len__(self):
        return self.length

{% endhighlight %}

`데이터 세트(Dataset)` 클래스를 상속받아 `사용자 정의 데이터 세트(CustomDataset)`를 정의합니다.

`초기화 메서드(__init__)`에서 CSV 파일의 경로를 입력받을 수 있게 `file_path`를 정의합니다.

`self.x1`, `self.x2`, `self.x3`에 `x`, `y`, `z` 값을 할당하며, `self.y`에 `pass` 값을 할당합니다.

`호출 메서드(__getitem__)`에서 `x`는 **입력값(x1, x2, x3)**를 할당하고, `y`는 **실젯값(y)**를 반환합니다.

<br>

{% highlight Python %}

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layer = nn.Sequential(
          nn.Linear(3, 1),
          nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer(x)
        return x

{% endhighlight %}

`모듈(Module)` 클래스를 상속받아 `사용자 정의 모델(CustomModel)`를 정의합니다.

`super()` 함수를 통해 `Module` 클래스의 속성을 초기화하고, 사용할 `계층(Layer)`을 정의합니다.

`시퀀셜(Sequential)`을 활용하여 여러 `계층(Layer)`을 하나로 묶습니다.

묶어진 계층은 순차적으로 실행되며, `가독성(Readability)`을 높일 수 있습니다.

`선형 변환 함수(nn.Linear)`의 `입력 데이터 차원 크기(in_features)`는 `3`을 입력하고, `출력 데이터 차원 크기(out_features)`는 `1`을 입력합니다.

또한, `시그모이드 함수(Sigmoid Function)`을 적용할 예정이므로, `시그모이드 함수(nn.Sigmoid)`를 `선형 변환 함수(nn.Linear)` 뒤에 연결합니다.

<br>

{% highlight Python %}

criterion = nn.BCELoss().to(device)

{% endhighlight %}

`이진 교차 엔트로피 클래스(nn.BCELoss)`로 `criterion` 인스턴스를 생성합니다.

`이진 교차 엔트로피 클래스(nn.BCELoss)`는 `비용 함수(Cost Function)`로 criterion 인스턴스에서 **순전파(Forward)**를 통해 나온 출력값과 실젯값을 비교하여 오차를 계산합니다.

<br>

{% highlight Python %}

with torch.no_grad():
    model.eval()
    inputs = torch.FloatTensor(
        [[89, 92, 75], [75, 64, 50], [38, 58, 63], [33, 42, 39], [23, 15, 32]]
    ).to(device)
    outputs = model(inputs)

    print("---------")
    print(outputs)
    print(outputs >= torch.FloatTensor([0.5]).to(device))

{% endhighlight %}

`참(True)` 또는 `거짓(False)`으로 반환될 수 있도록 예측값이 `0.5` 이상의 값을 지닌다면, `참(True)` 값으로 반환합니다.

예측값은 **0 ~ 1** 범위를 가지므로, 1에 가까울 수록 `참(True)`일 확률이 높아집니다.

<br>

### 출력 결과

{% highlight Python %}

tensor([[0.7232],
        [0.6073],
        [0.5394],
        [0.4518],
        [0.3713]], device='cuda:0')
tensor([[ True],
        [ True],
        [ True],
        [False],
        [False]], device='cuda:0')

{% endhighlight %}

|    입력값    |  평균 | 최솟값 | 실젯값 | 예측값(1) | 예측값(2)
| :---------: | :----: | :---: | :----: |:----: |:----: |
| [89, 92, 75] | 85.33 | 75 | True | 0.7232 | True |
| [75, 64, 50] | 63.00 | 50 | True | 0.6073 | True |
| [38, 58, 63] | 53.00 | 38 | False | 0.5394 | True |
| [33, 42, 39] | 38.00 | 33 | False | 0.4518 | False |
| [23, 15, 32] | 23.33 | 15 | False | 0.3713 | False |

검증을 위해 무작위로 설정한 값은 표와 같습니다.

첫 번째 데이터는 명확하게 `참(True)`으로 판단할 수 있는 데이터 구조를 갖기 때문에 예측값이 **0.7232**로 높게 계산되었습니다.

반대로, 마지막 데이터는 `거짓(False)`으로 판단하기 쉬운 데이터 구조를 갖기 때문에 예측값이 **0.3713**으로 낮게 계산되었습니다.

판단을 하기 어려운 중간 데이터는 참으로 판단되었지만, **0.5**에 근사한 예측값을 갖는 **0.5394**로 계산되었습니다.

결과를 이분화하기 때문에, 참 또는 거짓으로 나눌 수 있지만, 0.5에 가까워질수록 결과가 정확하지 않습니다.

`예측값(1)`을 활용하여 정확성을 판단하여 결과를 예측하는 것이 좋습니다.

