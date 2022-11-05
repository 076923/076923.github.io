---
layout: post
title: "Python Pytorch 강좌 : 제 9강 - 모델(Model)"
tagline: "Python PyTorch Model"
image: /assets/images/pytorch.webp
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['PyTorch']
keywords: Python, Python Pytorch, Pytorch Model, Pytorch nn.Module, Pytorch CustomDataset, Pytorch CustomModel
ref: Python-PyTorch
category: Python
permalink: /posts/Python-pytorch-9/
comments: true
toc: true
---

## 모델(Model)

PyTorch에서 사용되는 `모델(Model)`은 모든 **인공 신경망(Neural Network)** 모듈의 기본 클래스입니다.

`모델(Model)`은 데이터에 대한 연산을 수행하는 `계층(Layer)`을 정의하고, `순방향(Forward)` 연산을 수행합니다.

모델 클래스를 활용해 **복잡한 구조의 인공 신경망을 모듈화해 빠르게 구축하고 관리하기 용이한 상태로 만듭니다.**


<br>
<br>

## PyTorch의 모듈(Module)

`모듈(Module)` 클래스는 **인공 신경망(Neural Network)**의 기본 클래스입니다.

새로운 모델 클래스를 생성하려면, `모듈(Module)` 클래스를 상속받아 임의의 `서브 클래스(Sub Class)`를 생성합니다.

이 클래스는 다른 `모듈(Module)` 클래스를 포함할 수 있으며, **트리 구조(Tree structure)**로 중첩할 수 있습니다.

<br>

### 모듈(Module) 기본형

{% highlight Python %}

import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

{% endhighlight %}

`초기화 메서드(__init__ )`는 신경망에 사용될 `계층(Layer)`을 초기화합니다.

먼저, `super()` 함수를 통해 `Module` 클래스의 속성을 초기화합니다.

초기화 이후에는 **학습에 사용되는 계층을 초기화 메서드에서 선언합니다.**

`순방향 메서드(forward)`는 모델이 **데이터(x)**를 입력받아 학습을 진행하는 일련의 과정을 정의합니다.

모델 객체를 호출하는 순간 `순방향 메서드(forward)`의 정의 순서대로 학습이 수행됩니다.

<br>

## 데이터 세트

이번 강좌에서는 `사용자 정의 데이터 세트`, `사용자 정의 모델`, `GPU 연산 적용`, `모델 예측`을 진행해보도록 하겠습니다.

학습에 사용된 `dataset.csv`는 아래 링크에서 다운로드할 수 있습니다.

> `Dataset 다운로드` : [다운로드][Dataset]

[Dataset]: https://github.com/076923/076923.github.io/raw/master/download/datasets/pytorch-9/dataset.csv

`dataset.csv`는 다음과 같은 형태로 구성되어 있습니다.

|    x   |    y   |
| :----: | :----: |
|  -10.0 | 327.79 |
|  -9.9  | 321.39 |
|  -9.8  | 314.48 |
|  -9.7  | 308.51 |
|  -9.6  | 302.86 |
|  -9.5  | 296.41 |
|  -9.4  | 290.48 |
|  -9.3  |  284.7 |
|  ...   |   ...  | 

`x` 데이터와 `y` 데이터의 관계는 $$ y = 3.1x^2 - 1.7x + \text{random}(0.01, 0.99) $$의 관계를 갖습니다.

이 데이터를 활용하여 `비선형 회귀(Nonlinear Regression)`를 `모델(Model)`로 구현해보도록 하겠습니다.

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
        self.x = df.iloc[:, 0].values
        self.y = df.iloc[:, 1].values
        self.length = len(df)

    def __getitem__(self, index):
        x = torch.FloatTensor([self.x[index] ** 2, self.x[index]])
        y = torch.FloatTensor([self.y[index]])
        return x, y

    def __len__(self):
        return self.length


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, x):
        x = self.layer(x)
        return x


train_dataset = CustomDataset("./dataset.csv")
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CustomModel().to(device)
criterion = nn.MSELoss().to(device)
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
    inputs = torch.FloatTensor([[1 ** 2, 1], [5 **2, 5], [11**2, 11]]).to(device)
    outputs = model(inputs)
    print(outputs)


{% endhighlight %}
**결과**
:    
Epoch : 1000, Model : [Parameter containing:<br>
tensor([[ 3.1034, -1.7008]], device='cuda:0', requires_grad=True), Parameter containing:<br>
tensor([0.2861], device='cuda:0', requires_grad=True)], Cost : 0.095<br>
Epoch : 2000, Model : [Parameter containing:<br>
tensor([[ 3.1030, -1.7033]], device='cuda:0', requires_grad=True), Parameter containing:<br>
tensor([0.3037], device='cuda:0', requires_grad=True)], Cost : 0.089<br>
Epoch : 3000, Model : [Parameter containing:<br>
tensor([[ 3.1027, -1.7030]], device='cuda:0', requires_grad=True), Parameter containing:<br>
tensor([0.3200], device='cuda:0', requires_grad=True)], Cost : 0.089<br>
Epoch : 4000, Model : [Parameter containing:<br>
tensor([[ 3.1026, -1.7032]], device='cuda:0', requires_grad=True), Parameter containing:<br>
tensor([0.3349], device='cuda:0', requires_grad=True)], Cost : 0.085<br>
Epoch : 5000, Model : [Parameter containing:<br>
tensor([[ 3.1025, -1.7034]], device='cuda:0', requires_grad=True), Parameter containing:<br>
tensor([0.3485], device='cuda:0', requires_grad=True)], Cost : 0.087<br>
Epoch : 6000, Model : [Parameter containing:<br>
tensor([[ 3.1023, -1.7031]], device='cuda:0', requires_grad=True), Parameter containing:<br>
tensor([0.3609], device='cuda:0', requires_grad=True)], Cost : 0.078<br>
Epoch : 7000, Model : [Parameter containing:<br>
tensor([[ 3.1018, -1.7031]], device='cuda:0', requires_grad=True), Parameter containing:<br>
tensor([0.3722], device='cuda:0', requires_grad=True)], Cost : 0.078<br>
Epoch : 8000, Model : [Parameter containing:<br>
tensor([[ 3.1018, -1.7032]], device='cuda:0', requires_grad=True), Parameter containing:<br>
tensor([0.3826], device='cuda:0', requires_grad=True)], Cost : 0.089<br>
Epoch : 9000, Model : [Parameter containing:<br>
tensor([[ 3.1015, -1.7030]], device='cuda:0', requires_grad=True), Parameter containing:<br>
tensor([0.3922], device='cuda:0', requires_grad=True)], Cost : 0.079<br>
Epoch : 10000, Model : [Parameter containing:<br>
tensor([[ 3.1015, -1.7032]], device='cuda:0', requires_grad=True), Parameter containing:<br>
tensor([0.4008], device='cuda:0', requires_grad=True)], Cost : 0.078<br>
tensor([[  1.7992],<br>
&emsp;&emsp;&emsp;&emsp;[ 69.4222],<br>
&emsp;&emsp;&emsp;&emsp;[356.9461]], device='cuda:0')<br>

<br>

### 세부 코드

{% highlight Python %}

import torch
import pandas as pd
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

{% endhighlight %}

`PyTorch`를 사용하기 위해 `torch`와 관련 모듈을 포함시킵니다.

`pandas`는 `dataset.csv`를 읽기 위해 사용합니다.

<br>

#### 사용자 정의 데이터 세트

{% highlight Python %}

class CustomDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        self.x = df.iloc[:, 0].values
        self.y = df.iloc[:, 1].values
        self.length = len(df)

    def __getitem__(self, index):
        x = torch.FloatTensor([self.x[index] ** 2, self.x[index]])
        y = torch.FloatTensor([self.y[index]])
        return x, y

    def __len__(self):
        return self.length

{% endhighlight %}

`데이터 세트(Dataset)` 클래스를 상속받아 `사용자 정의 데이터 세트(CustomDataset)`를 정의합니다.

`초기화 메서드(__init__)`에서 CSV 파일의 경로를 입력받을 수 있게 `file_path`를 정의합니다.

`self.x`와 `self.y`에 각각 `x` 값과 `y` 값을 할당합니다.

`호출 메서드(__getitem__)`에서 `x`값과 `y`값을 반환합니다.

결괏값은 이차 방정식($$ y = W_{1}x^2 + W_{2}x + b $$) 형태이므로, 반환되는 `x` 값은 $$ x^2, x $$의 구조로 반환합니다.

`길이 반환 메서드(__len__ )`로 현재 데이터의 길이를 반환합니다.

<br>

#### 사용자 정의 모델

{% highlight Python %}

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, x):
        x = self.layer(x)
        return x

{% endhighlight %}

`모듈(Module)` 클래스를 상속받아 `사용자 정의 모델(CustomModel)`를 정의합니다.

`super()` 함수를 통해 `Module` 클래스의 속성을 초기화하고, 사용할 `계층(Layer)`을 정의합니다.

`선형 변환 함수(nn.Linear)`의 `입력 데이터 차원 크기(in_features)`는 이차 다항식이므로 `2`를 입력하고, `출력 데이터 차원 크기(out_features)`는 `1`을 입력합니다.

`순방향 메서드(forward)`로 학습 과정을 정의합니다.

이 메서드로 `계층(Layer)`에 데이터가 입력돼 처리된 결과가 반환됩니다.

<br>

#### 인스턴스 생성

{% highlight Python %}

train_dataset = CustomDataset("./dataset.csv")
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)

{% endhighlight %}

`훈련 데이터 세트(train_dataset)`로 `CSV` 파일에서 데이터를 불러옵니다.

이후, `훈련 데이터 로더(train_dataloader)`로 `배치 크기(batch_size)`는 **128**, `데이터 순서 변경(shuffle)`, 마지막 배치 제거(drop_last)를 **참(True)** 값으로 할당합니다.

<br>

#### GPU 연산 적용

{% highlight Python %}

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CustomModel().to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.0001)

{% endhighlight %}

이번 강좌에서는 `GPU`를 통해 학습을 진행할 예정이므로 `장치(device)`를 설정합니다.

`GPU` 환경이 지원되지 않는 경우, `CPU` 환경에서 실행됩니다.

`모델(model)`을 선언하고 `사용자 정의 모델(CustomModel)`를 할당합니다.

모델 인스턴스에 `GPU` 연산이 적용될 수 있도록 `to` 메서드를 적용합니다.

`비용 함수(criterion)`도 `GPU` 연산을 적용합니다.

`최적화 함수(optimizer)`의 `최적화하려는 변수`에 `모델 매개 변수(model.parameters)`로 할당하여 모델에 사용된 매개 변수들을 최적화합니다.

<br>

#### 학습 진행

{% highlight Python %}

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

{% endhighlight %}

앞선 강좌와 동일한 방법으로 학습이 진행됩니다.

차이점으로는 학습에 사용되는 `x`와 `y`에 `GPU` 장치를 적용하기 위해 `to` 메서드를 적용합니다.

<br>

#### 학습 결과

{% highlight Python %}

Epoch : 1000, Model : [Parameter containing:
tensor([[ 3.1034, -1.7008]], device='cuda:0', requires_grad=True), Parameter containing:
tensor([0.2861], device='cuda:0', requires_grad=True)], Cost : 0.095

Epoch : 2000, Model : [Parameter containing:
tensor([[ 3.1030, -1.7033]], device='cuda:0', requires_grad=True), Parameter containing:
tensor([0.3037], device='cuda:0', requires_grad=True)], Cost : 0.089

Epoch : 3000, Model : [Parameter containing:
tensor([[ 3.1027, -1.7030]], device='cuda:0', requires_grad=True), Parameter containing:
tensor([0.3200], device='cuda:0', requires_grad=True)], Cost : 0.089

Epoch : 4000, Model : [Parameter containing:
tensor([[ 3.1026, -1.7032]], device='cuda:0', requires_grad=True), Parameter containing:
tensor([0.3349], device='cuda:0', requires_grad=True)], Cost : 0.085

Epoch : 5000, Model : [Parameter containing:
tensor([[ 3.1025, -1.7034]], device='cuda:0', requires_grad=True), Parameter containing:
tensor([0.3485], device='cuda:0', requires_grad=True)], Cost : 0.087

Epoch : 6000, Model : [Parameter containing:
tensor([[ 3.1023, -1.7031]], device='cuda:0', requires_grad=True), Parameter containing:
tensor([0.3609], device='cuda:0', requires_grad=True)], Cost : 0.078

Epoch : 7000, Model : [Parameter containing:
tensor([[ 3.1018, -1.7031]], device='cuda:0', requires_grad=True), Parameter containing:
tensor([0.3722], device='cuda:0', requires_grad=True)], Cost : 0.078

Epoch : 8000, Model : [Parameter containing:
tensor([[ 3.1018, -1.7032]], device='cuda:0', requires_grad=True), Parameter containing:
tensor([0.3826], device='cuda:0', requires_grad=True)], Cost : 0.089

Epoch : 9000, Model : [Parameter containing:
tensor([[ 3.1015, -1.7030]], device='cuda:0', requires_grad=True), Parameter containing:
tensor([0.3922], device='cuda:0', requires_grad=True)], Cost : 0.079

Epoch : 10000, Model : [Parameter containing:
tensor([[ 3.1015, -1.7032]], device='cuda:0', requires_grad=True), Parameter containing:
tensor([0.4008], device='cuda:0', requires_grad=True)], Cost : 0.078

{% endhighlight %}

총 10,000번의 학습을 진행하였을 때 `가중치(Weight)`는 각각 `3.1015`, `-1.7032`로 계산되며 `편향(Bias)`은 `0.4008`의 값을 반환하는 것을 확인할 수 있습니다.

데이터 세트는 $$ y = 3.1x^2 - 1.7x + \text{random}(0.01, 0.99) $$로 구성되어 있으므로, 학습이 원활하게 진행된 것을 확인할 수 있습니다.

<br>

#### 모델 평가

{% highlight Python %}

with torch.no_grad():
    model.eval()
    inputs = torch.FloatTensor([[1 ** 2, 1], [5 **2, 5], [11**2, 11]]).to(device)
    outputs = model(inputs)
    print(outputs)

{% endhighlight %}

이제 학습된 `모델(model)`에 값을 입력하여 결과를 확인해보도록 하겠습니다.

테스트 데이터 세트가 존재하지 않으므로 임의의 값을 입력하여 결과를 확인합니다.

`torch.no_grad`를 통해 기울기 계산을 비활성화 합니다.

`자동 미분(autograd)` 기능을 사용하지 않도록 설정하여 메모리 사용량을 줄여 `추론(inference)`에 적합한 상태로 변경합니다.

`model.eval()`을 호출하여 모델을 평가 모드로 변경합니다.

모델을 평가 모드로 변경하지 않는다면, 추론 결과가 일관성 없는 결과를 반환합니다.

추론에 사용할 `테스트 데이터(inputs)`를 정의합니다.

`테스트 데이터(inputs)`는 `모델(model)`에서 요구하는 입력 차원과 동일한 구조를 가져야 합니다.

즉, `훈련 데이터(train_data)`와 동일한 형태를 가져야 합니다.

<br>

{% highlight Python %}

tensor([[  1.7992],
        [ 69.4222],
        [356.9461]], device='cuda:0')

{% endhighlight %}

출력 결과에서 확인할 수 있듯이 실젯값과 유사하게 나오는 것을 확인할 수 있습니다.

`평가 모드(model.eval)`를 진행한 다음 다시 학습을 진행하려는 경우 `학습 모드`로 변경해야 합니다.

`model.train()`를 호출한 다음 학습을 다시 진행할 수 있습니다.
