---
layout: post
title: "Python Pytorch 강좌 : 제 10강 - 모델 저장/불러오기(Model Save/Load)"
tagline: "Python PyTorch Model Save/Load"
image: /assets/images/pytorch.webp
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['PyTorch']
keywords: Python, Python Pytorch, Pytorch Model, Pytorch Model Save, Pytorch Model Load, Pytorch Checkpoint
ref: Python-PyTorch
category: Python
permalink: /posts/Python-pytorch-10/
comments: true
toc: true
---

## 모델 저장/불러오기(Model Save/Load)

PyTorch의 `모델(Model)`은 `직렬화(Serialize)`와 `역직렬화(Deserialize)`를 통해 객체를 저장하고 불러올 수 있습니다.

`모델(Model)`을 저장하는 방법은 Python의 `피클(Pickle)`을 활용하여 파이썬 객체 구조를 **바이너리 프로토콜(Binary Protocols)**로 직렬화합니다.

모델에 사용된 `텐서(Tensor)`나 `매개 변수(Dictionary)`를 저장합니다.

`모델(Model)`을 불러오는 방법은 저장된 객체 파일을 역직렬화 하여 현재 프로세스의 메모리에 업로드합니다.

이를 통해 모델을 통해 계산된 `텐서(Tensor)`나 `매개 변수(Dictionary)`를 불러올 수 있습니다.

모델을 저장하는 경우에는 모델 학습이 모두 완료된 이후에 작성하거나, 특정 에폭이 끝날 때마다 저장합니다.

모델 파일 확장자는 주로 `*.pt`나 `*.pth` 의 확장자를 사용하여 저장합니다.

<br>
<br>

## 모델 전체 저장/불러오기

모델 전체를 저장하는 경우에는 학습에 사용된 모델 클래스의 구조와 학습 상태 등을 모두 저장합니다.

모델의 **계층(Layer) 구조**, **매개 변수(model.parameters)** 등이 모두 기록된 상태로 저장하기 때문에 모델 파일로도 동일한 구조를 구현할 수 있습니다.

<br>

### 모델 저장

{% highlight Python %}

torch.save(model, f'./model.pt')

{% endhighlight %}

`모델 저장 함수(torch.save)`를 활용해 모델을 저장합니다.

`torch.save(model, path)`는 `모델(model)`의 정보를 `경로(path)`에 저장합니다.

<br>

### 모델 불러오기

{% highlight Python %}

import torch
from torch import nn


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, x):
        x = self.layer(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load("model.pt", map_location=device)
print(model)

with torch.no_grad():
    model.eval()
    inputs = torch.FloatTensor([[1 ** 2, 1], [5 **2, 5], [11**2, 11]]).to(device)
    outputs = model(inputs)
    print(outputs)


{% endhighlight %}
**결과**
:    
CustomModel(<br>
&emsp;(layer): Linear(in_features=2, out_features=1, bias=True)<br>
)<br>
tensor([[  1.4342],<br>
&emsp;&emsp;&emsp;&emsp;[ 69.2052],<br>
&emsp;&emsp;&emsp;&emsp;[357.3152]])<br>
<br>

단, 모델을 불러오는 경우에도 동일한 형태의 클래스가 선언되어 있어야 합니다.

또한, 학습이 `GPU` 상태에서 진행되었는지 `CPU` 상태에서 진행되었는지 상관 없이 활용할 수 있도록 `map_location` 매개 변수 통해 장치(device)를 적용합니다.

위와 같이 `CustomModel` 클래스가 동일한 구조로 선언되어있다면, 동일하게 `추론(inference)`을 진행할 수 있습니다.

만약 다음과 같은 형태로 불러온다면 `AttributeError` 오류가 발생하여 모델을 불러올 수 없습니다.

<br>

### 모델 클래스의 구조를 알 수 없는 경우

{% highlight Python %}

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load("model.pth", map_location=device)
print(model)

{% endhighlight %}
**결과**
:    
예외가 발생했습니다. AttributeError       (note: full exception trace is shown but execution is paused at: <module>)<br>
Can't get attribute 'CustomModel' on <module '__main__' from 'source code path'><Br>
<br>

만약, 모델 전체 파일은 가지고 있으나 모델 구조를 확인하지 못하는 경우에는 모델 구조를 확인하여 해결할 수 있습니다.

위 경고 메세지에서 `CustomModel` 속성을 가져오지 못했다고 알려주므로 동일한 명칭의 클래스를 생성합니다.

{% highlight Python %}

import torch
from torch import nn

class CustomModel(nn.Module):
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load("model.pth", map_location=device)
print(model)

{% endhighlight %}
**결과**
:    
CustomModel(<br>
&emsp;(layer): Linear(in_features=2, out_features=1, bias=True)<br>
)<br>
<br>

모델의 구조를 확인할 수 있으므로, `CustomModel` 클래스에 동일한 형태로 구현합니다.

주의 사항으로는 **변수의 명칭(layer)**까지 동일한 형태로 구현해야 합니다.

<br>
<br>

## 모델 상태 저장/불러오기

모델 전체를 저장하는 경우에는 모든 정보를 저장하므로 모델 상태만 저장하는 것보다 더 많은 저장 공간을 요구하게 됩니다.

그러므로, 모델의 매개변수만 저장하여 활용해보도록 하겠습니다.

<br>

### 모델 저장

{% highlight Python %}

torch.save(model.state_dict(), "./model_state_dict.pt")

{% endhighlight %}

`모델 상태(torch.state_dict)`만을 가져와 모델 상태를 저장합니다.

`모델 상태(torch.state_dict)`는 모델에서 학습이 가능한 매개변수를 `순서가 있는 사전(OrderedDict)` 형식으로 반환합니다.

현재 모델 상태는 다음과 같은 형태로 반환합니다.

<br>

{% highlight Python %}

OrderedDict(
    [
        (
            'layer.weight', tensor([[ 3.1076, -1.7026]], device='cuda:0')
        ),
        (
            'layer.bias', tensor([0.0293], device='cuda:0')
        )
    ]
)

{% endhighlight %}

학습에 사용된 `CustomModel` 클래스의 `layer` 변수의 **가중치(Weight)**와 **편향(Bias)**이 저장되어 있습니다.

즉, `추론(inference)`에 필요한 데이터만 가져오는 방식으로 이해할 수 있습니다.

<br>

### 모델 불러오기

{% highlight Python %}

import torch
from torch import nn


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, x):
        x = self.layer(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CustomModel().to(device)

model_state_dict = torch.load("./model_state_dict.pt", map_location=device)
model.load_state_dict(model_state_dict)

with torch.no_grad():
    model.eval()
    inputs = torch.FloatTensor([[1 ** 2, 1], [5 **2, 5], [11**2, 11]]).to(device)
    outputs = model(inputs)

{% endhighlight %}

모델의 상태만 저장했기 때문에 `CustomModel`에 학습 결과를 반영합니다.

모델 상태만 불러오는 경우에는 모델 구조를 알 수 없으므로, `CustomModel` 클래스가 동일하게 구현되어 있어야 합니다.

모델 상태 파일인 `model_state_dict.pt`도 `torch.load` 함수를 통해 불러옵니다.

단, 모델에 적용해야 하기 때문에 `model` 인스턴스의 `load_state_dict` 메서드로 모델 상태를 반영합니다.

<br>
<br>

## 체크포인트 저장/불러오기

`체크포인트(Checkpoint)`는 학습을 과정을 저장하는 과정을 의미합니다.

`빅 데이터(Big data)`를 `깊은 레이어(Deep Layer)` 구조의 모델로 학습시킨다면 오랜 시간이 소요됩니다.

이 학습 과정에서 한 번에 전체 에폭(Epoch)을 반복할 수 없거나 모종의 이유로 학습이 중단될 수 있습니다.

이러한 현상을 방지하기 위해 **일정 에폭(Epooch)마다 학습된 결과를 저장하여 나중에 이어서 학습할 수 있습니다.**

<br>

### 체크포인트 저장

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

checkpoint = 1
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
        torch.save(
            {
                "model": "CustomModel",
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cost": cost,
                "description": f"CustomModel 체크포인트-{checkpoint}",
            },
            f"./checkpoint-{checkpoint}.pt",
        )
        checkpoint += 1

{% endhighlight %}

`체크포인트(Checkpoint)`도 `모델 저장 함수(torch.save)`를 활용해 여러 상태를 저장합니다.

단, 다양한 정보를 저장하기 위해 `사전(Dictionary)` 형식으로 값을 할당합니다.

학습을 이어서 진행하기 위한 목적이므로, `에폭(Epoch)`, `모델 상태(model.state_dict)`, `최적화 상태(optimizer.state_dict)` 등은 필수로 포함되어야 합니다.

**정수형**, **실수형**, **문자열** 등도 함께 저장할 수 있으므로, 부수적인 정보도 함께 포함시킬 수 있습니다.

<br>

### 체크포인트 불러오기

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

checkpoint = torch.load('./checkpoint-6.pt')
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
checkpoint_epoch = checkpoint["epoch"]
checkpoint_description = checkpoint["description"]

print(checkpoint_description)

for epoch in range(checkpoint_epoch + 1, 10000):
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
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch : {epoch+1:4d}, Model : {list(model.parameters())}, Cost : {cost:.3f}")


{% endhighlight %}
**결과**
:    
CustomModel 체크포인트-6<br>
Epoch : 7000, Model : [Parameter containing:<br>
tensor([[ 3.1012, -1.7030]], device='cuda:0', requires_grad=True), Parameter containing:<br>
tensor([0.4412], device='cuda:0', requires_grad=True)], Cost : 0.081<br>
Epoch : 8000, Model : [Parameter containing:<br>
tensor([[ 3.1007, -1.7031]], device='cuda:0', requires_grad=True), Parameter containing:<br>
tensor([0.4457], device='cuda:0', requires_grad=True)], Cost : 0.080<br>
Epoch : 9000, Model : [Parameter containing:<br>
tensor([[ 3.1006, -1.7032]], device='cuda:0', requires_grad=True), Parameter containing:<br>
tensor([0.4499], device='cuda:0', requires_grad=True)], Cost : 0.072<br>
Epoch : 10000, Model : [Parameter containing:<br>
tensor([[ 3.1007, -1.7033]], device='cuda:0', requires_grad=True), Parameter containing:<br>
tensor([0.4538], device='cuda:0', requires_grad=True)], Cost : 0.070<br>
<br>

이어서 학습을 진행하기 위해 `모델(model)`과 `최적화 함수(optimizer)`에 각각 `load_state_dict` 메서드로 저장된 값을 불러옵니다.

원활하게 학습을 이어서 진행할 수 있도록 `에폭(Epoch)`도 반복문 시작값에 적용시킵니다.

출력 결과에서 확인할 수 있듯이 `가중치(Weight)`와 `편향(Bias)`가 **체크포인트-6** 상태에서 이어서 진행된 것을 확인할 수 있습니다.

