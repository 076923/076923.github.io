---
layout: post
title: "Python Pytorch 강좌 : 제 9강 - 모델(Model)"
tagline: "Python PyTorch Model"
image: /assets/images/pytorch.webp
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
        super().__init__()
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

`순방향 메서드(forward)`는 모델이 **데이터(x)**를 입력받아 결과를 계산하는 일련의 과정을 정의합니다.

모델 객체를 `model(x)`처럼 호출하는 순간 `순방향 메서드(forward)`의 정의 순서대로 연산이 수행됩니다.

- Tip : `model.forward(x)`를 직접 호출하지 않고 `model(x)`로 호출해야 모듈에 등록된 `훅(Hook)` 등이 함께 실행됩니다.

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
        self.x = df.iloc[:, 0].to_numpy()
        self.y = df.iloc[:, 1].to_numpy()
        self.length = len(df)

    def __getitem__(self, index):
        x = torch.tensor([self.x[index] ** 2, self.x[index]], dtype=torch.float32)
        y = torch.tensor([self.y[index]], dtype=torch.float32)
        return x, y

    def __len__(self):
        return self.length


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, x):
        x = self.layer(x)
        return x


torch.manual_seed(42)

train_dataset = CustomDataset("./dataset.csv")
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
elif torch.xpu.is_available():
    device = "xpu"
else:
    device = "cpu"

model = CustomModel().to(device)
criterion = nn.MSELoss()
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

        cost += loss.item()

    cost = cost / len(train_dataloader)

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch : {epoch+1:4d}, Model : {list(model.parameters())}, Cost : {cost:.3f}")


model.eval()
with torch.inference_mode():
    inputs = torch.tensor([[1 ** 2, 1], [5 ** 2, 5], [11 ** 2, 11]], dtype=torch.float32).to(device)
    outputs = model(inputs)
    print(outputs)


{% endhighlight %}
**결과**
:    
Epoch : 1000, Model : [Parameter containing:<br>
tensor([[ 3.1092, -1.6997]], requires_grad=True), Parameter containing:<br>
tensor([-0.0733], requires_grad=True)], Cost : 0.209<br>
Epoch : 2000, Model : [Parameter containing:<br>
tensor([[ 3.1088, -1.7030]], requires_grad=True), Parameter containing:<br>
tensor([-0.0249], requires_grad=True)], Cost : 0.204<br>
Epoch : 3000, Model : [Parameter containing:<br>
tensor([[ 3.1077, -1.7028]], requires_grad=True), Parameter containing:<br>
tensor([0.0192], requires_grad=True)], Cost : 0.190<br>
Epoch : 4000, Model : [Parameter containing:<br>
tensor([[ 3.1068, -1.7027]], requires_grad=True), Parameter containing:<br>
tensor([0.0596], requires_grad=True)], Cost : 0.159<br>
Epoch : 5000, Model : [Parameter containing:<br>
tensor([[ 3.1066, -1.7030]], requires_grad=True), Parameter containing:<br>
tensor([0.0965], requires_grad=True)], Cost : 0.145<br>
Epoch : 6000, Model : [Parameter containing:<br>
tensor([[ 3.1056, -1.7028]], requires_grad=True), Parameter containing:<br>
tensor([0.1304], requires_grad=True)], Cost : 0.130<br>
Epoch : 7000, Model : [Parameter containing:<br>
tensor([[ 3.1054, -1.7032]], requires_grad=True), Parameter containing:<br>
tensor([0.1614], requires_grad=True)], Cost : 0.129<br>
Epoch : 8000, Model : [Parameter containing:<br>
tensor([[ 3.1052, -1.7030]], requires_grad=True), Parameter containing:<br>
tensor([0.1898], requires_grad=True)], Cost : 0.117<br>
Epoch : 9000, Model : [Parameter containing:<br>
tensor([[ 3.1047, -1.7028]], requires_grad=True), Parameter containing:<br>
tensor([0.2156], requires_grad=True)], Cost : 0.108<br>
Epoch : 10000, Model : [Parameter containing:<br>
tensor([[ 3.1042, -1.7031]], requires_grad=True), Parameter containing:<br>
tensor([0.2391], requires_grad=True)], Cost : 0.106<br>
tensor([[  1.6402],<br>
&emsp;&emsp;&emsp;&emsp;[ 69.3279],<br>
&emsp;&emsp;&emsp;&emsp;[357.1102]])<br>
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
        self.x = df.iloc[:, 0].to_numpy()
        self.y = df.iloc[:, 1].to_numpy()
        self.length = len(df)

    def __getitem__(self, index):
        x = torch.tensor([self.x[index] ** 2, self.x[index]], dtype=torch.float32)
        y = torch.tensor([self.y[index]], dtype=torch.float32)
        return x, y

    def __len__(self):
        return self.length

{% endhighlight %}

`데이터 세트(Dataset)` 클래스를 상속받아 `사용자 정의 데이터 세트(CustomDataset)`를 정의합니다.

`초기화 메서드(__init__)`에서 CSV 파일의 경로를 입력받을 수 있게 `file_path`를 정의합니다.

`self.x`와 `self.y`에 각각 `x` 값과 `y` 값을 할당합니다.

`to_numpy()` 메서드는 `pandas`의 `Series`를 `NumPy` 배열로 변환하며, `pandas`에서 `values` 속성보다 권장되는 방식입니다.

`호출 메서드(__getitem__)`에서 `x`값과 `y`값을 반환합니다.

결괏값은 이차 방정식($$ y = W_{1}x^2 + W_{2}x + b $$) 형태이므로, 반환되는 `x` 값은 $$ x^2, x $$의 구조로 반환합니다.

반환값은 `torch.tensor` 함수의 `자료형(dtype)`을 `torch.float32`로 지정해 모델의 매개변수와 같은 자료형으로 생성합니다.

`길이 반환 메서드(__len__ )`로 현재 데이터의 길이를 반환합니다.

<br>

#### 사용자 정의 모델

{% highlight Python %}

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
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

torch.manual_seed(42)

train_dataset = CustomDataset("./dataset.csv")
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)

{% endhighlight %}

가중치 초기화와 `데이터 순서 변경(shuffle)`에 난수가 사용되므로, `torch.manual_seed`로 시드를 고정해 실행할 때마다 같은 결과가 나오도록 합니다.

`훈련 데이터 세트(train_dataset)`로 `CSV` 파일에서 데이터를 불러옵니다.

이후, `훈련 데이터 로더(train_dataloader)`로 `배치 크기(batch_size)`는 **128**, `데이터 순서 변경(shuffle)`, 마지막 배치 제거(drop_last)를 **참(True)** 값으로 할당합니다.

<br>

#### GPU 연산 적용

{% highlight Python %}

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
elif torch.xpu.is_available():
    device = "xpu"
else:
    device = "cpu"

model = CustomModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)

{% endhighlight %}

이번 강좌에서는 `GPU`를 통해 학습을 진행할 예정이므로 `장치(device)`를 설정합니다.

NVIDIA GPU는 `cuda`, Apple Silicon의 GPU는 `mps`, Intel GPU는 `xpu` 장치를 사용하며, 사용 가능한 `GPU`가 없는 경우 `CPU` 환경에서 실행됩니다.

`모델(model)`을 선언하고 `사용자 정의 모델(CustomModel)`를 할당합니다.

모델 인스턴스에 `GPU` 연산이 적용될 수 있도록 `to` 메서드를 적용합니다.

`비용 함수(nn.MSELoss)`는 학습 가능한 매개변수가 없으므로 `to` 메서드를 적용하지 않아도 되며, 입력된 텐서의 장치에서 연산됩니다.

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

        cost += loss.item()

    cost = cost / len(train_dataloader)

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch : {epoch+1:4d}, Model : {list(model.parameters())}, Cost : {cost:.3f}")

{% endhighlight %}

앞선 강좌와 동일한 방법으로 학습이 진행됩니다.

차이점으로는 학습에 사용되는 `x`와 `y`에 `GPU` 장치를 적용하기 위해 `to` 메서드를 적용합니다.

`비용(cost)`에는 `loss.item()`으로 꺼낸 Python 실수를 누적하여, 배치마다 생성되는 연산 그래프가 남지 않도록 합니다.

<br>

#### 학습 결과

{% highlight Python %}

Epoch : 1000, Model : [Parameter containing:
tensor([[ 3.1092, -1.6997]], requires_grad=True), Parameter containing:
tensor([-0.0733], requires_grad=True)], Cost : 0.209

Epoch : 2000, Model : [Parameter containing:
tensor([[ 3.1088, -1.7030]], requires_grad=True), Parameter containing:
tensor([-0.0249], requires_grad=True)], Cost : 0.204

Epoch : 3000, Model : [Parameter containing:
tensor([[ 3.1077, -1.7028]], requires_grad=True), Parameter containing:
tensor([0.0192], requires_grad=True)], Cost : 0.190

Epoch : 4000, Model : [Parameter containing:
tensor([[ 3.1068, -1.7027]], requires_grad=True), Parameter containing:
tensor([0.0596], requires_grad=True)], Cost : 0.159

Epoch : 5000, Model : [Parameter containing:
tensor([[ 3.1066, -1.7030]], requires_grad=True), Parameter containing:
tensor([0.0965], requires_grad=True)], Cost : 0.145

Epoch : 6000, Model : [Parameter containing:
tensor([[ 3.1056, -1.7028]], requires_grad=True), Parameter containing:
tensor([0.1304], requires_grad=True)], Cost : 0.130

Epoch : 7000, Model : [Parameter containing:
tensor([[ 3.1054, -1.7032]], requires_grad=True), Parameter containing:
tensor([0.1614], requires_grad=True)], Cost : 0.129

Epoch : 8000, Model : [Parameter containing:
tensor([[ 3.1052, -1.7030]], requires_grad=True), Parameter containing:
tensor([0.1898], requires_grad=True)], Cost : 0.117

Epoch : 9000, Model : [Parameter containing:
tensor([[ 3.1047, -1.7028]], requires_grad=True), Parameter containing:
tensor([0.2156], requires_grad=True)], Cost : 0.108

Epoch : 10000, Model : [Parameter containing:
tensor([[ 3.1042, -1.7031]], requires_grad=True), Parameter containing:
tensor([0.2391], requires_grad=True)], Cost : 0.106

{% endhighlight %}

총 10,000번의 학습을 진행하였을 때 `가중치(Weight)`는 각각 `3.1042`, `-1.7031`로 계산되며 `편향(Bias)`은 `0.2391`의 값을 반환하는 것을 확인할 수 있습니다.

데이터 세트는 $$ y = 3.1x^2 - 1.7x + \text{random}(0.01, 0.99) $$로 구성되어 있으므로, 학습이 원활하게 진행된 것을 확인할 수 있습니다.

`편향(Bias)`은 학습률이 낮아 천천히 증가하고 있으므로, 학습을 더 진행하면 무작위 값의 평균에 더 가까워집니다.

- Tip : 위 결과는 `CPU`에서 실행한 결과입니다. `GPU`에서 실행하면 매개변수에 `device='cuda:0'`과 같은 장치 정보가 함께 출력되며, 난수 생성 방식과 연산 방식의 차이로 결괏값이 조금 다를 수 있습니다.

<br>

#### 모델 평가

{% highlight Python %}

model.eval()
with torch.inference_mode():
    inputs = torch.tensor([[1 ** 2, 1], [5 ** 2, 5], [11 ** 2, 11]], dtype=torch.float32).to(device)
    outputs = model(inputs)
    print(outputs)

{% endhighlight %}

이제 학습된 `모델(model)`에 값을 입력하여 결과를 확인해보도록 하겠습니다.

테스트 데이터 세트가 존재하지 않으므로 임의의 값을 입력하여 결과를 확인합니다.

먼저 `model.eval()`을 호출하여 모델을 평가 모드로 변경합니다.

평가 모드는 `드롭아웃(Dropout)`이나 `배치 정규화(Batch Normalization)`처럼 학습과 추론에서 다르게 동작하는 계층을 추론용 동작으로 바꿉니다.

현재 모델은 `선형 변환 함수(nn.Linear)`만 사용하므로 결과가 달라지지 않지만, 이러한 계층이 포함된 모델에서 평가 모드로 변경하지 않으면 추론할 때마다 일관성 없는 결과를 반환할 수 있습니다.

다음으로 `torch.inference_mode`를 통해 기울기 계산을 비활성화 합니다.

`자동 미분(autograd)` 기능을 사용하지 않도록 설정하여 메모리 사용량을 줄여 `추론(inference)`에 적합한 상태로 변경합니다.

`torch.inference_mode`는 `torch.no_grad`와 같이 기울기 계산을 비활성화하며, 추가로 텐서의 버전 추적 등을 생략해 추론에 더 최적화되어 있습니다.

단, `torch.inference_mode`에서 생성된 텐서는 이후 `자동 미분(autograd)`이 기록되는 연산에 사용할 수 없으므로, 추론 결과를 다시 학습에 사용해야 한다면 `torch.no_grad`를 사용합니다.

추론에 사용할 `테스트 데이터(inputs)`를 정의합니다.

`테스트 데이터(inputs)`는 `모델(model)`에서 요구하는 입력 차원과 동일한 구조를 가져야 합니다.

즉, `훈련 데이터(train_data)`와 동일한 형태를 가져야 합니다.

<br>

{% highlight Python %}

tensor([[  1.6402],
        [ 69.3279],
        [357.1102]])

{% endhighlight %}

출력 결과에서 확인할 수 있듯이 실젯값과 유사하게 나오는 것을 확인할 수 있습니다.

`평가 모드(model.eval)`를 진행한 다음 다시 학습을 진행하려는 경우 `학습 모드`로 변경해야 합니다.

`model.train()`를 호출한 다음 학습을 다시 진행할 수 있습니다.
