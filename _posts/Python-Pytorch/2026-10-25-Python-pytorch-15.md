---
layout: post
title: "Python Pytorch 강좌 : 제 15강 - 합성곱 신경망(Convolutional Neural Network)"
tagline: "Python PyTorch Convolutional Neural Network"
image: /assets/images/pytorch.webp
tags: ['PyTorch']
keywords: Python, Python Pytorch, Pytorch CNN, Pytorch Convolutional Neural Network, Pytorch Conv2d, Pytorch MaxPool2d, Pytorch Flatten, torchvision, FashionMNIST, Pytorch Image Classification
ref: Python-PyTorch
category: Python
permalink: /posts/Python-pytorch-15/
comments: true
toc: true
---

## 합성곱 신경망(Convolutional Neural Network)

[퍼셉트론(Perceptron)][14강]에서는 입력을 1차원 벡터로 받아 `nn.Linear` 계층을 쌓은 `다층 퍼셉트론(Multi Layer Perceptron)`으로 문제를 해결했습니다. 그러나 이미지는 가로와 세로로 픽셀이 배치된 **2차원 구조**를 가지며, 가까운 픽셀끼리 의미 있는 패턴을 이룹니다. 이미지를 한 줄로 펼쳐 다층 퍼셉트론에 넣으면 이 구조가 사라지고, 입력 크기에 비례해 가중치의 수가 크게 늘어납니다.

`합성곱 신경망(Convolutional Neural Network, CNN)`은 작은 크기의 `커널(Kernel)`을 이미지 위에서 이동시키며 같은 가중치로 지역 패턴을 찾는 신경망입니다. 커널 하나가 이미지의 모든 위치에서 재사용되므로 매개변수가 적고, 같은 패턴이 어느 위치에 있든 같은 방식으로 찾아낼 수 있습니다. 합성곱 신경망의 원리는 [합성곱 신경망 이론][AI-12강]에서 자세히 다루었으며, 이번 강좌에서는 PyTorch로 직접 구현하는 방법을 다룹니다.

이번 강좌에서는 `torchvision`이 제공하는 `FashionMNIST` 데이터 세트로 의류 이미지를 10가지 범주로 분류하는 모델을 만듭니다. 지금까지는 CSV 파일을 직접 읽어 `Dataset`을 구현했지만, `torchvision`에는 자주 쓰이는 이미지 데이터 세트가 미리 구현되어 있어 한 줄로 내려받아 사용할 수 있습니다.

<br>
<br>

## 데이터 세트

`FashionMNIST`는 독일의 전자 상거래 기업 Zalando가 공개한 의류 이미지 데이터 세트로, 학습용 60,000장과 시험용 10,000장으로 이루어져 있습니다. 각 이미지는 가로세로 28 픽셀의 흑백 이미지이며, 티셔츠, 바지, 스웨터, 드레스, 코트, 샌들, 셔츠, 운동화, 가방, 앵클부츠의 10개 범주 중 하나에 속합니다. 손글씨 숫자 데이터 세트인 `MNIST`와 크기와 형식이 같지만, 분류가 더 어려워 모델의 성능 차이가 잘 드러납니다.

{% highlight Python %}

from torchvision import datasets, transforms

train_dataset = datasets.FashionMNIST("./data", train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.FashionMNIST("./data", train=False, download=True, transform=transforms.ToTensor())

x, y = train_dataset[0]
print(len(train_dataset), len(test_dataset))
print(x.shape, x.dtype, x.min().item(), x.max().item())
print(y, train_dataset.classes[y])
print(train_dataset.classes)

{% endhighlight %}

**결과**
:    
60000 10000<br>
torch.Size([1, 28, 28]) torch.float32 0.0 1.0<br>
9 Ankle boot<br>
['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']<br>
<br>

`datasets.FashionMNIST(root, train, download, transform)`은 `root` 경로에 데이터를 내려받고 `Dataset` 객체를 반환합니다. `train=True`이면 학습용, `False`이면 시험용 데이터를 불러오며, `download=True`이면 파일이 없을 때 자동으로 내려받습니다. 반환된 객체는 [데이터 세트(Data Set)][8강]에서 직접 구현한 `Dataset`과 같은 방식으로 `len()`과 색인으로 사용할 수 있습니다.

`transform`은 데이터를 꺼낼 때마다 적용할 변환입니다. 원본 이미지는 0에서 255 사이의 정수로 된 이미지 객체이므로, `transforms.ToTensor()`로 **0에서 1 사이의 실수 텐서**로 바꿉니다. 결과에서 확인할 수 있듯 변환된 텐서의 형태는 `[1, 28, 28]`로, 맨 앞의 1은 `채널(Channel)`의 수입니다. 흑백 이미지는 채널이 하나이고, 컬러 이미지라면 빨강, 초록, 파랑의 세 채널이 됩니다.

첫 번째 학습 데이터의 정답은 9이며, `classes` 속성에서 9번째 이름인 `Ankle boot`(앵클부츠)에 해당합니다.

- Tip : PyTorch의 합성곱 계층은 이미지를 `[배치 크기, 채널, 높이, 너비]` 순서의 4차원 텐서로 받습니다. 이 순서를 `NCHW`라 부르며, 이미지 라이브러리에서 흔히 쓰는 `[높이, 너비, 채널]` 순서와 다르므로 직접 불러온 이미지를 사용할 때는 순서를 바꿔야 합니다.

<br>
<br>

## 모델 구조

합성곱 신경망은 보통 **합성곱 계층으로 특징을 추출**하고, **완전 연결 계층으로 분류**하는 구조를 가집니다. 이번 모델은 합성곱 블록 두 개와 완전 연결 계층 하나로 구성합니다.

{% highlight Python %}

import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


model = CNN()
x = torch.zeros(64, 1, 28, 28)

print(x.shape)
x = model.layer1(x)
print(x.shape)
x = model.layer2(x)
print(x.shape)
x = torch.flatten(x, start_dim=1)
print(x.shape)
x = model.fc(x)
print(x.shape)

{% endhighlight %}

**결과**
:    
torch.Size([64, 1, 28, 28])<br>
torch.Size([64, 16, 14, 14])<br>
torch.Size([64, 32, 7, 7])<br>
torch.Size([64, 1568])<br>
torch.Size([64, 10])<br>
<br>

`nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)`는 2차원 합성곱 계층입니다. `in_channels`는 입력 채널의 수, `out_channels`는 사용할 커널의 수이자 출력 채널의 수입니다. 첫 번째 합성곱은 흑백 이미지(1채널)를 받아 커널 16개를 적용하므로 16채널의 `특징 맵(Feature Map)`을 출력합니다. `kernel_size=3`은 $$ 3 \times 3 $$ 크기의 커널을, `padding=1`은 가장자리를 0으로 한 칸씩 둘러싸 **입력과 출력의 가로세로 크기를 같게** 유지한다는 뜻입니다.

`nn.ReLU()`는 활성화 함수로, 합성곱의 결과에서 음수를 0으로 바꿉니다. 활성화 함수가 없으면 합성곱 계층을 여러 개 쌓아도 하나의 선형 변환과 같아지므로, 계층 사이마다 활성화 함수를 넣습니다.

`nn.MaxPool2d(kernel_size=2)`는 $$ 2 \times 2 $$ 구역마다 가장 큰 값 하나를 남기는 `최대 풀링(Max Pooling)`입니다. 가로세로가 절반으로 줄어들어, 결과에서 $$ 28 \times 28 $$이 첫 번째 블록 뒤에 $$ 14 \times 14 $$로, 두 번째 블록 뒤에 $$ 7 \times 7 $$로 바뀌는 것을 확인할 수 있습니다. 크기가 줄어드는 만큼 채널은 1, 16, 32로 늘어나며, 이미지의 공간 정보가 점차 **채널 방향의 특징 정보**로 바뀝니다.

마지막 합성곱 블록의 출력은 `[64, 32, 7, 7]` 형태입니다. 완전 연결 계층은 1차원 벡터를 입력으로 받으므로, `torch.flatten(x, start_dim=1)`으로 배치 차원을 제외한 나머지를 $$ 32 \times 7 \times 7 = 1568 $$개의 값으로 펼칩니다. `start_dim=1`을 지정하지 않으면 배치 차원까지 하나로 펼쳐져 모든 이미지가 하나의 벡터로 합쳐지므로 주의해야 합니다. 마지막으로 `nn.Linear(1568, 10)`이 10개 범주의 점수를 출력합니다.

- Tip : 완전 연결 계층의 입력 크기 `32 * 7 * 7`을 직접 계산하기 번거롭다면, 위와 같이 0으로 채운 가짜 입력을 계층에 차례로 통과시켜 형태를 출력해 보는 것이 가장 확실합니다. 입력 크기를 자동으로 추론하는 `nn.LazyLinear`를 사용하는 방법도 있습니다.

<br>

### 매개변수 수 비교

{% highlight Python %}

for name, param in model.named_parameters():
    print(name, tuple(param.shape), param.numel())

print("total :", sum(p.numel() for p in model.parameters()))

{% endhighlight %}

**결과**
:    
layer1.0.weight (16, 1, 3, 3) 144<br>
layer1.0.bias (16,) 16<br>
layer2.0.weight (32, 16, 3, 3) 4608<br>
layer2.0.bias (32,) 32<br>
fc.weight (10, 1568) 15680<br>
fc.bias (10,) 10<br>
total : 20490<br>
<br>

`named_parameters()`는 모델의 모든 학습 가능한 매개변수를 이름과 함께 반환하며, `numel()`은 텐서의 원소 개수를 반환합니다. 첫 번째 합성곱 계층의 가중치는 `(16, 1, 3, 3)` 형태로, **출력 채널, 입력 채널, 커널 높이, 커널 너비** 순서입니다. 두 번째 합성곱 계층은 16채널의 입력 각각에 $$ 3 \times 3 $$ 커널을 적용하는 것을 32번 반복하므로 $$ 32 \times 16 \times 3 \times 3 = 4608 $$개의 가중치를 가집니다.

모델 전체의 매개변수는 20,490개입니다. 같은 이미지를 펼쳐서 은닉층 128개의 다층 퍼셉트론에 넣는다면, 첫 번째 계층에만 $$ 784 \times 128 = 100{,}352 $$개의 가중치가 필요합니다. 합성곱 계층은 이미지 크기와 무관하게 커널 크기와 채널 수만으로 매개변수가 정해지므로, 훨씬 적은 매개변수로 이미지의 특징을 추출할 수 있습니다. 이번 모델에서는 오히려 매개변수의 대부분이 마지막 완전 연결 계층에 있다는 점도 확인할 수 있습니다.

<br>
<br>

## 메인 코드

{% highlight Python %}

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    with torch.inference_mode():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            correct += (output.argmax(dim=1) == y).sum().item()
    return correct / len(dataloader.dataset)


torch.manual_seed(42)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=256)

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
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
    accuracy = evaluate(model, test_dataloader, device)
    print(f"Epoch : {epoch + 1}, Cost : {cost:.4f}, Accuracy : {accuracy:.4f}")

x, y = test_dataset[0]
model.eval()
with torch.inference_mode():
    probability = torch.softmax(model(x.unsqueeze(0).to(device)), dim=1)

print(test_dataset.classes[y], test_dataset.classes[probability.argmax().item()], f"{probability.max().item():.4f}")

{% endhighlight %}

**결과**
:    
Epoch : 1, Cost : 0.5236, Accuracy : 0.8661<br>
Epoch : 2, Cost : 0.3449, Accuracy : 0.8769<br>
Epoch : 3, Cost : 0.3061, Accuracy : 0.8872<br>
Epoch : 4, Cost : 0.2824, Accuracy : 0.8840<br>
Epoch : 5, Cost : 0.2655, Accuracy : 0.8941<br>
Ankle boot Ankle boot 0.9894<br>
<br>

결과는 PyTorch 2.14.1의 CPU 환경에서 실행한 값입니다. GPU에서 실행하면 연산 순서의 차이로 소수점 아래 값이 조금 달라질 수 있습니다.

학습이 진행될수록 학습 데이터의 `비용(Cost)`은 0.5236에서 0.2655로 꾸준히 줄어듭니다. 시험 데이터의 정확도는 첫 에폭에 이미 86.61%에 도달하고, 다섯 번째 에폭에서 89.41%가 됩니다. 네 번째 에폭에서는 정확도가 88.72%에서 88.40%로 조금 떨어졌는데, 시험 정확도는 학습 비용처럼 매번 줄어드는 값이 아니라 **에폭마다 오르내릴 수 있다**는 점을 보여 줍니다.

마지막 줄은 시험 데이터의 첫 번째 이미지를 모델에 넣은 결과입니다. 정답인 `Ankle boot`를 98.94%의 확률로 맞혔습니다.

<br>

### 세부 코드

{% highlight Python %}

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

{% endhighlight %}

합성곱 연산은 계산량이 많아 GPU를 사용하면 학습 속도가 크게 빨라집니다. 이전 강좌와 같이 `torch.accelerator`로 사용 가능한 가속 장치를 확인해, NVIDIA GPU가 있으면 `cuda`를, Apple Silicon Mac이면 `mps`를, Intel GPU면 `xpu`를, 가속 장치가 없으면 `cpu`를 사용합니다. 위의 결과를 CPU에서 얻는 데는 수 분이 걸리며, GPU를 사용하면 훨씬 짧아집니다.

<br>

{% highlight Python %}

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=256)

{% endhighlight %}

학습용 데이터 로더는 `shuffle=True`로 에폭마다 순서를 섞어, 모델이 데이터의 순서를 외우지 않도록 합니다. 시험용 데이터 로더는 기울기를 계산하지 않아 메모리를 적게 쓰므로 배치 크기를 크게 잡아도 되고, 순서를 섞을 필요도 없습니다.

<br>

{% highlight Python %}

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

{% endhighlight %}

10개 범주의 다중 분류이므로 [다중 분류(Multiclass Classification)][13강]에서 다룬 `nn.CrossEntropyLoss`를 사용합니다. 모델의 마지막 계층은 소프트맥스를 적용하지 않은 점수인 `로짓(Logit)`을 출력하며, `nn.CrossEntropyLoss`가 내부에서 소프트맥스와 로그를 함께 계산합니다. 정답 `y`는 원-핫 벡터가 아니라 **범주 번호**를 그대로 사용합니다.

최적화 함수는 `Adam`을 사용합니다. 지금까지 사용한 `SGD`보다 학습률에 덜 민감하고 빠르게 수렴하므로, 새로운 모델을 시도할 때 기본값으로 자주 사용됩니다. 최적화 함수의 차이는 [최적화 알고리즘 이론][AI-10강]에서 자세히 다루었습니다.

<br>

{% highlight Python %}

for epoch in range(5):
    model.train()
    cost = 0.0

    for x, y in train_dataloader:
        ...
        cost += loss.item()

    cost = cost / len(train_dataloader)
    accuracy = evaluate(model, test_dataloader, device)

{% endhighlight %}

학습 반복문의 구조는 이전 강좌와 같습니다. 에폭을 시작할 때 `model.train()`으로 학습 모드를 켜는 이유는, 뒤에서 호출하는 `evaluate` 함수가 `model.eval()`로 평가 모드로 바꾸기 때문입니다. 지금 모델에는 학습과 평가에서 동작이 달라지는 계층이 없지만, 다음 강좌에서 다룰 드롭아웃이나 배치 정규화를 추가하면 이 전환이 반드시 필요합니다.

손실을 더할 때는 `loss.item()`으로 파이썬 숫자로 바꿔 더합니다. 텐서 그대로 더하면 매 배치의 계산 그래프가 에폭이 끝날 때까지 남아 메모리를 계속 차지합니다.

<br>

{% highlight Python %}

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    with torch.inference_mode():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            correct += (output.argmax(dim=1) == y).sum().item()
    return correct / len(dataloader.dataset)

{% endhighlight %}

평가 함수는 시험 데이터 전체에 대한 정확도를 계산합니다. `torch.inference_mode()` 안에서는 기울기를 계산하지 않으므로 더 빠르고 메모리를 적게 씁니다. `output.argmax(dim=1)`은 각 이미지에서 점수가 가장 높은 범주 번호를 고르며, 정답과 같은 개수를 모두 더한 뒤 데이터 수로 나누어 정확도를 구합니다.

<br>

{% highlight Python %}

x, y = test_dataset[0]
model.eval()
with torch.inference_mode():
    probability = torch.softmax(model(x.unsqueeze(0).to(device)), dim=1)

{% endhighlight %}

이미지 한 장을 예측할 때는 `unsqueeze(0)`으로 맨 앞에 배치 차원을 추가해 `[1, 1, 28, 28]` 형태로 만들어야 합니다. 모델은 항상 배치 단위의 4차원 입력을 기대하기 때문입니다. 출력된 로짓에 `torch.softmax`를 적용하면 10개 범주의 확률이 되며, 가장 높은 확률과 그 범주를 확인할 수 있습니다.

<br>
<br>

## 합성곱 계층 정리

| 계층 | 주요 매개변수 | 역할 | 출력 형태 변화 |
|---|---|---|---|
| `nn.Conv2d` | `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding` | 커널로 지역 패턴을 찾아 특징 맵 생성 | 채널 수 변경, 패딩에 따라 크기 유지 또는 감소 |
| `nn.ReLU` | 없음 | 음수를 0으로 바꾸는 활성화 함수 | 변화 없음 |
| `nn.MaxPool2d` | `kernel_size`, `stride` | 구역의 최댓값으로 요약 | 가로세로 감소 |
| `torch.flatten` | `start_dim` | 배치를 제외한 차원을 1차원으로 펼침 | `[N, C, H, W]` → `[N, C×H×W]` |
| `nn.Linear` | `in_features`, `out_features` | 펼친 특징으로 범주 점수 계산 | `[N, 입력]` → `[N, 범주 수]` |

합성곱 신경망의 출력 크기는 다음 식으로 계산할 수 있습니다. 입력 크기가 $$ W $$, 커널 크기가 $$ K $$, 패딩이 $$ P $$, 스트라이드가 $$ S $$일 때 출력 크기는 $$ \lfloor (W - K + 2P) / S \rfloor + 1 $$입니다. 이번 모델의 합성곱 계층은 $$ (28 - 3 + 2) / 1 + 1 = 28 $$로 크기를 유지하고, 풀링 계층이 크기를 절반으로 줄입니다.

이번 모델은 학습 데이터 6만 장으로 학습해 시험 데이터에서 약 89%의 정확도를 얻었습니다. 그러나 학습 데이터가 적거나 모델이 커지면, 학습 데이터에서는 정확도가 계속 오르는데 시험 데이터에서는 오히려 떨어지는 `과대적합(Overfitting)`이 일어나기 쉽습니다. 다음 강좌에서는 이 문제를 줄이는 기법들을 같은 데이터 세트로 실험합니다.

<br>
<br>

[8강]: https://076923.github.io/posts/Python-pytorch-8/
[13강]: https://076923.github.io/posts/Python-pytorch-13/
[14강]: https://076923.github.io/posts/Python-pytorch-14/
[AI-10강]: https://076923.github.io/posts/AI-10/
[AI-12강]: https://076923.github.io/posts/AI-12/
