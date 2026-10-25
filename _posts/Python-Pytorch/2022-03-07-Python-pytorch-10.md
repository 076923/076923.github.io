---
layout: post
title: "Python Pytorch 강좌 : 제 10강 - 모델 저장/불러오기(Model Save/Load)"
tagline: "Python PyTorch Model Save/Load"
image: /assets/images/pytorch.webp
tags: ['PyTorch']
keywords: Python, Python Pytorch, Pytorch Model, Pytorch Model Save, Pytorch Model Load, Pytorch Checkpoint, Pytorch state_dict, Pytorch weights_only
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

모델을 저장하는 방법은 크게 **모델 상태(state_dict)만 저장하는 방법**과 **모델 전체를 저장하는 방법**이 있으며, 일반적으로 `모델 상태(state_dict)`만 저장하는 방법을 권장합니다.

이번 강좌의 예제는 [제 9강 - 모델(Model)][9강]에서 학습한 `CustomModel`을 저장하고 불러옵니다.

<br>
<br>

## 모델 상태 저장/불러오기

`모델 상태(state_dict)`는 모델의 매개변수만 저장하는 방식입니다.

모델 클래스의 구조는 코드로 관리하고, 학습된 결과인 매개변수만 파일로 저장하므로 저장된 파일이 특정 코드 구조에 덜 의존하며 안전하게 불러올 수 있습니다.

<br>

### 모델 저장

{% highlight Python %}

torch.save(model.state_dict(), "./model_state_dict.pt")
print(model.state_dict())

{% endhighlight %}
**결과**
:    
OrderedDict({'layer.weight': tensor([[ 3.1042, -1.7031]]), 'layer.bias': tensor([0.2391])})<br>
<br>

`모델 상태(model.state_dict)`만을 가져와 모델 상태를 저장합니다.

`모델 상태(model.state_dict)`는 모델의 매개변수와 버퍼를 `순서가 있는 사전(OrderedDict)` 형식으로 반환합니다.

학습에 사용된 `CustomModel` 클래스의 `layer` 변수의 **가중치(Weight)**와 **편향(Bias)**이 저장되어 있습니다.

즉, `추론(inference)`에 필요한 데이터만 가져오는 방식으로 이해할 수 있습니다.

<br>

### 모델 불러오기

{% highlight Python %}

import torch
from torch import nn


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, x):
        x = self.layer(x)
        return x


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
elif torch.xpu.is_available():
    device = "xpu"
else:
    device = "cpu"

model = CustomModel().to(device)

model_state_dict = torch.load("./model_state_dict.pt", map_location=device, weights_only=True)
model.load_state_dict(model_state_dict)

model.eval()
with torch.inference_mode():
    inputs = torch.tensor([[1 ** 2, 1], [5 ** 2, 5], [11 ** 2, 11]], dtype=torch.float32).to(device)
    outputs = model(inputs)
    print(outputs)

{% endhighlight %}
**결과**
:    
tensor([[  1.6402],<br>
&emsp;&emsp;&emsp;&emsp;[ 69.3279],<br>
&emsp;&emsp;&emsp;&emsp;[357.1102]])<br>
<br>

모델의 상태만 저장했기 때문에 `CustomModel` 인스턴스를 먼저 생성한 다음 학습 결과를 반영합니다.

모델 상태만 불러오는 경우에는 모델 구조를 알 수 없으므로, `CustomModel` 클래스가 동일하게 구현되어 있어야 합니다.

모델 상태 파일인 `model_state_dict.pt`도 `torch.load` 함수를 통해 불러옵니다.

학습이 `GPU` 상태에서 진행되었는지 `CPU` 상태에서 진행되었는지 상관 없이 활용할 수 있도록 `map_location` 매개 변수를 통해 장치(device)를 적용합니다.

`weights_only` 매개변수는 **텐서, 사전, 리스트, 기본 자료형 등 허용된 형식만 역직렬화**하도록 제한합니다. PyTorch 2.6 이상에서는 기본값이 `참(True)`이지만, 의도를 명확히 하기 위해 직접 지정하였습니다.

불러온 모델 상태는 `model` 인스턴스의 `load_state_dict` 메서드로 모델에 반영합니다.

추론 전에는 `model.eval()`로 평가 모드로 변경하고, `torch.inference_mode()`로 기울기 계산을 비활성화합니다.

출력 결과에서 확인할 수 있듯이 [제 9강 - 모델(Model)][9강]의 학습 직후 추론 결과와 동일한 값을 반환합니다.

<br>
<br>

## 모델 전체 저장/불러오기

모델 전체를 저장하는 경우에는 모델 인스턴스 자체를 `피클(Pickle)`로 직렬화합니다.

모델의 **계층(Layer) 구조**, **매개 변수(model.parameters)** 등이 함께 기록되지만, **모델 클래스의 코드는 저장되지 않고 클래스의 경로(`__main__.CustomModel`)만 저장됩니다.**

그러므로 모델을 불러오는 환경에도 동일한 경로의 클래스가 선언되어 있어야 합니다.

<br>

### 모델 저장

{% highlight Python %}

torch.save(model, "./model.pt")

{% endhighlight %}

`모델 저장 함수(torch.save)`를 활용해 모델을 저장합니다.

`torch.save(model, path)`는 `모델(model)`의 정보를 `경로(path)`에 저장합니다.

<br>

### weights_only 기본값 변경

PyTorch 2.6 이상에서는 `torch.load` 함수의 `weights_only` 매개변수 기본값이 `거짓(False)`에서 `참(True)`으로 변경되었습니다.

그러므로 이전 버전처럼 모델 전체를 저장한 파일을 `torch.load("model.pt")`로 불러오면 다음과 같은 오류가 발생합니다.

{% highlight Python %}

import torch
from torch import nn


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, x):
        x = self.layer(x)
        return x


model = torch.load("./model.pt", map_location="cpu")

{% endhighlight %}
**결과**
:    
\_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint.<br>
&emsp;(1) In PyTorch 2.6, we changed the default value of the \`weights\_only\` argument in \`torch.load\` from \`False\` to \`True\`. Re-running \`torch.load\` with \`weights\_only\` set to \`False\` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.<br>
&emsp;(2) Alternatively, to load with \`weights\_only=True\` please check the recommended steps in the following error message.<br>
&emsp;WeightsUnpickler error: Unsupported global: GLOBAL \_\_main\_\_.CustomModel was not an allowed global by default. Please use \`torch.serialization.add\_safe\_globals([\_\_main\_\_.CustomModel])\` or the \`torch.serialization.safe\_globals([\_\_main\_\_.CustomModel])\` context manager to allowlist this global if you trust this class/function.<br>
<br>

`weights_only=True`에서는 허용된 형식만 역직렬화하므로, 허용 목록에 없는 `__main__.CustomModel` 클래스를 만나면 불러오기를 중단합니다.

`피클(Pickle)`은 역직렬화 과정에서 파일에 기록된 **임의의 Python 객체를 생성하고 함수를 호출할 수 있습니다.**

즉, 악의적으로 만들어진 모델 파일을 `weights_only=False`로 불러오면 파일을 여는 것만으로 **임의의 코드가 실행**될 수 있습니다.

이러한 위험을 막기 위해 기본값이 `참(True)`으로 변경되었습니다.

<br>

### 모델 불러오기

{% highlight Python %}

import torch
from torch import nn


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, x):
        x = self.layer(x)
        return x


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
elif torch.xpu.is_available():
    device = "xpu"
else:
    device = "cpu"

model = torch.load("./model.pt", map_location=device, weights_only=False)
print(model)

model.eval()
with torch.inference_mode():
    inputs = torch.tensor([[1 ** 2, 1], [5 ** 2, 5], [11 ** 2, 11]], dtype=torch.float32).to(device)
    outputs = model(inputs)
    print(outputs)


{% endhighlight %}
**결과**
:    
CustomModel(<br>
&emsp;(layer): Linear(in_features=2, out_features=1, bias=True)<br>
)<br>
tensor([[  1.6402],<br>
&emsp;&emsp;&emsp;&emsp;[ 69.3279],<br>
&emsp;&emsp;&emsp;&emsp;[357.1102]])<br>
<br>

모델 전체를 불러오려면 `weights_only=False`를 명시해야 합니다.

**`weights_only=False`는 직접 저장했거나 출처를 신뢰할 수 있는 파일에만 사용해야 합니다.** 인터넷에서 내려받은 모델 파일처럼 출처를 확인할 수 없는 파일은 `weights_only=True`로 불러올 수 있는 `모델 상태(state_dict)` 형식을 사용합니다.

또한, 모델을 불러오는 경우에도 동일한 형태의 클래스가 선언되어 있어야 합니다.

위와 같이 `CustomModel` 클래스가 동일한 구조로 선언되어있다면, 동일하게 `추론(inference)`을 진행할 수 있습니다.

- Tip : 신뢰할 수 있는 클래스라면 `with torch.serialization.safe_globals([CustomModel, nn.Linear]):`처럼 허용 목록에 클래스를 등록한 다음, `weights_only=True` 상태로 모델 전체를 불러올 수도 있습니다.

<br>

### 모델 클래스의 구조를 알 수 없는 경우

{% highlight Python %}

import torch

device = "cpu"
model = torch.load("./model.pt", map_location=device, weights_only=False)
print(model)

{% endhighlight %}
**결과**
:    
AttributeError: module '\_\_main\_\_' has no attribute 'CustomModel'<br>
<br>

`CustomModel` 클래스를 선언하지 않고 모델 전체를 불러오면 `AttributeError` 오류가 발생하여 모델을 불러올 수 없습니다.

모델 파일에는 클래스의 코드가 아닌 클래스의 경로만 저장되어 있기 때문입니다.

만약, 모델 전체 파일은 가지고 있으나 모델 구조를 확인하지 못하는 경우에는 모델 구조를 확인하여 해결할 수 있습니다.

위 오류 메세지에서 `CustomModel` 속성을 가져오지 못했다고 알려주므로 동일한 명칭의 클래스를 생성합니다.

{% highlight Python %}

import torch
from torch import nn


class CustomModel(nn.Module):
    pass


device = "cpu"
model = torch.load("./model.pt", map_location=device, weights_only=False)
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

## 체크포인트 저장/불러오기

`체크포인트(Checkpoint)`는 학습 과정을 저장하는 것을 의미합니다.

`빅 데이터(Big data)`를 `깊은 레이어(Deep Layer)` 구조의 모델로 학습시킨다면 오랜 시간이 소요됩니다.

이 학습 과정에서 한 번에 전체 에폭(Epoch)을 반복할 수 없거나 모종의 이유로 학습이 중단될 수 있습니다.

이러한 현상을 방지하기 위해 **일정 에폭(Epoch)마다 학습된 결과를 저장하여 나중에 이어서 학습할 수 있습니다.**

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

        cost += loss.item()

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

위 예제는 1,000 에폭마다 `checkpoint-1.pt`부터 `checkpoint-10.pt`까지 10개의 체크포인트 파일을 생성합니다.

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

checkpoint = torch.load("./checkpoint-6.pt", map_location=device, weights_only=True)
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

        cost += loss.item()

    cost = cost / len(train_dataloader)

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch : {epoch+1:4d}, Model : {list(model.parameters())}, Cost : {cost:.3f}")


{% endhighlight %}
**결과**
:    
CustomModel 체크포인트-6<br>
Epoch : 7000, Model : [Parameter containing:<br>
tensor([[ 3.1053, -1.7029]], requires_grad=True), Parameter containing:<br>
tensor([0.1614], requires_grad=True)], Cost : 0.118<br>
Epoch : 8000, Model : [Parameter containing:<br>
tensor([[ 3.1051, -1.7032]], requires_grad=True), Parameter containing:<br>
tensor([0.1898], requires_grad=True)], Cost : 0.123<br>
Epoch : 9000, Model : [Parameter containing:<br>
tensor([[ 3.1045, -1.7029]], requires_grad=True), Parameter containing:<br>
tensor([0.2157], requires_grad=True)], Cost : 0.120<br>
Epoch : 10000, Model : [Parameter containing:<br>
tensor([[ 3.1038, -1.7028]], requires_grad=True), Parameter containing:<br>
tensor([0.2394], requires_grad=True)], Cost : 0.106<br>
<br>

이어서 학습을 진행하기 위해 `모델(model)`과 `최적화 함수(optimizer)`에 각각 `load_state_dict` 메서드로 저장된 값을 불러옵니다.

체크포인트 파일은 텐서, 사전, 정수, 실수, 문자열로만 구성되어 있으므로 `weights_only=True` 상태로 안전하게 불러올 수 있습니다.

원활하게 학습을 이어서 진행할 수 있도록 `에폭(Epoch)`도 반복문 시작값에 적용시킵니다.

출력 결과에서 확인할 수 있듯이 `가중치(Weight)`와 `편향(Bias)`이 **체크포인트-6** 상태에서 이어져 진행된 것을 확인할 수 있습니다.

- Tip : 체크포인트에는 난수 생성기의 상태가 저장되지 않으므로, `데이터 순서 변경(shuffle)` 순서가 중단 없이 학습한 경우와 달라져 결괏값이 조금 다를 수 있습니다. 완전히 같은 결과가 필요하다면 `torch.get_rng_state()`로 난수 생성기 상태를 함께 저장하고 `torch.set_rng_state()`로 복원합니다.

[9강]: https://076923.github.io/posts/Python-pytorch-9/
