---
layout: post
title: "Python Pytorch 강좌 : 제 16강 - 과대적합 방지(Regularization)"
tagline: "Python PyTorch Regularization"
image: /assets/images/pytorch.webp
tags: ['PyTorch']
keywords: Python, Python Pytorch, Pytorch Overfitting, Pytorch Regularization, Pytorch Dropout, Pytorch BatchNorm2d, Pytorch Weight Decay, Pytorch AdamW, Pytorch Early Stopping, Pytorch Data Augmentation, torchvision transforms, Pytorch Validation
ref: Python-PyTorch
category: Python
permalink: /posts/Python-pytorch-16/
comments: true
toc: true
---

## 과대적합(Overfitting)

[합성곱 신경망(Convolutional Neural Network)][15강]에서는 학습 데이터 60,000장으로 FashionMNIST 분류 모델을 학습해 시험 데이터에서 약 89%의 정확도를 얻었습니다. 그러나 실제 문제에서는 이만큼 많은 데이터를 확보하기 어려운 경우가 많습니다. 데이터가 적은데 모델의 표현력이 크면, 모델은 데이터에 담긴 규칙 대신 **학습 데이터 자체를 외우기** 시작합니다.

이처럼 학습 데이터에서는 성능이 계속 좋아지지만 새로운 데이터에서는 오히려 나빠지는 현상을 `과대적합(Overfitting)`이라 합니다. 과대적합을 줄이기 위해 모델의 자유도를 의도적으로 제한하는 기법을 통틀어 `정칙화(Regularization)`라 하며, 원리는 [과대적합과 정칙화 이론][AI-9강]에서 자세히 다루었습니다.

이번 강좌에서는 과대적합이 쉽게 일어나도록 **학습 데이터를 1,000장으로 줄인** 상황에서, PyTorch가 제공하는 정칙화 기법을 하나의 모델에 적용하고 결과를 비교합니다. 다루는 기법은 `드롭아웃(Dropout)`, `배치 정규화(Batch Normalization)`, `가중치 감쇠(Weight Decay)`, `데이터 증강(Data Augmentation)`, `조기 종료(Early Stopping)`입니다.

<br>
<br>

## 검증 데이터 세트

과대적합이 일어나고 있는지 알려면 **학습에 사용하지 않은 데이터**로 모델을 평가해야 합니다. [데이터 세트 분리(Data Set Split)][11강]에서 다룬 것처럼, 학습 중에 모델의 상태를 점검하는 데이터를 `검증 데이터(Validation Data)`라 합니다. 시험 데이터는 모든 결정이 끝난 뒤 최종 성능을 확인하는 용도로만 사용하고, 학습을 언제 멈출지 같은 결정은 검증 데이터로 내립니다.

{% highlight Python %}

indices = torch.randperm(60000, generator=torch.Generator().manual_seed(42))
train_indices, val_indices = indices[:1000], indices[1000:6000]

{% endhighlight %}

FashionMNIST의 학습용 데이터 60,000장의 순서를 무작위로 섞은 뒤, 앞의 1,000장을 학습 데이터로, 다음 5,000장을 검증 데이터로 사용합니다. `torch.Generator().manual_seed(42)`로 난수 생성기를 고정하면 실행할 때마다 같은 데이터가 선택됩니다.

이번에는 `random_split` 대신 색인을 직접 나누고 `Subset`으로 데이터 세트를 만듭니다. 뒤에서 다룰 데이터 증강은 **학습 데이터에만** 적용해야 하는데, `random_split`으로 나눈 데이터 세트들은 원본의 `transform`을 함께 쓰기 때문입니다. 같은 원본 데이터에 서로 다른 `transform`을 지정한 두 개의 데이터 세트를 만들고, 같은 색인 집합으로 잘라 내면 학습용과 검증용에 다른 변환을 적용할 수 있습니다.

<br>
<br>

## 드롭아웃(Dropout)

`드롭아웃(Dropout)`은 학습할 때마다 계층의 출력 중 일부를 무작위로 0으로 만드는 기법입니다. 매번 다른 노드가 꺼지므로, 모델은 특정 노드 몇 개에 의존하지 않고 여러 경로로 정보를 전달하도록 학습됩니다.

{% highlight Python %}

import torch
from torch import nn

torch.manual_seed(42)
dropout = nn.Dropout(p=0.5)
x = torch.ones(1, 8)

dropout.train()
print(dropout(x))
print(dropout(x))

dropout.eval()
print(dropout(x))

{% endhighlight %}

**결과**
:    
tensor([[2., 2., 2., 2., 0., 2., 0., 0.]])<br>
tensor([[2., 2., 2., 2., 0., 0., 2., 0.]])<br>
tensor([[1., 1., 1., 1., 1., 1., 1., 1.]])<br>
<br>

`nn.Dropout(p=0.5)`는 각 값을 50%의 확률로 0으로 만듭니다. 학습 모드에서 같은 입력을 두 번 넣었지만, 0이 된 위치가 매번 다른 것을 확인할 수 있습니다.

살아남은 값이 1이 아니라 2가 된 점에 주목해야 합니다. 드롭아웃은 살아남은 값을 $$ 1 / (1 - p) $$배로 키워, 일부를 꺼도 **출력의 평균적인 크기가 유지**되도록 합니다. 덕분에 평가 모드에서는 아무것도 끄지 않고 입력을 그대로 내보내도 학습 때와 같은 크기의 출력을 얻습니다. 세 번째 결과에서 평가 모드의 드롭아웃이 입력을 그대로 통과시키는 것을 확인할 수 있습니다.

- Tip : 드롭아웃은 `model.train()`과 `model.eval()`에 따라 동작이 달라집니다. 평가할 때 `model.eval()`을 호출하지 않으면 예측할 때마다 일부 노드가 무작위로 꺼져, 같은 입력에도 매번 다른 결과가 나옵니다.

<br>
<br>

## 배치 정규화(Batch Normalization)

`배치 정규화(Batch Normalization)`는 계층의 출력을 미니 배치 단위로 **평균 0, 분산 1이 되도록 정규화**하는 기법입니다. 학습을 안정적이고 빠르게 만드는 것이 주된 목적이지만, 배치마다 평균과 분산이 조금씩 달라지며 잡음이 섞이므로 부수적으로 정칙화 효과도 냅니다. 원리는 [배치 정규화 이론][AI-7강]에서 다루었습니다.

{% highlight Python %}

batchnorm = nn.BatchNorm1d(3)
data = torch.tensor([[1.0, 2.0, 3.0], [3.0, 6.0, 9.0]])

batchnorm.train()
print(batchnorm(data))
print(batchnorm.running_mean)

batchnorm.eval()
print(batchnorm(data))

{% endhighlight %}

**결과**
:    
tensor([[-1.0000, -1.0000, -1.0000],<br>
&emsp;&emsp;&emsp;&emsp;[ 1.0000,  1.0000,  1.0000]], grad_fn=&lt;NativeBatchNormBackward0&gt;)<br>
tensor([0.2000, 0.4000, 0.6000])<br>
tensor([[0.7628, 1.2271, 1.4606],<br>
&emsp;&emsp;&emsp;&emsp;[2.6697, 4.2950, 5.1121]], grad_fn=&lt;NativeBatchNormBackward0&gt;)<br>
<br>

학습 모드에서는 **지금 들어온 배치의 평균과 분산**으로 정규화합니다. 첫 번째 열의 값 1과 3은 평균이 2이므로, 정규화하면 -1과 1이 됩니다. 다른 열도 같은 비율로 커졌을 뿐이므로 모두 -1과 1이 됩니다.

동시에 배치 정규화는 평가 때 사용할 평균을 `running_mean`에 조금씩 누적합니다. 초깃값 0에서 이번 배치의 평균 2, 4, 6을 10%만 반영해 0.2, 0.4, 0.6이 되었습니다.

평가 모드에서는 배치의 통계 대신 이렇게 **누적한 평균과 분산**을 사용합니다. 그래서 같은 데이터를 넣어도 학습 모드와 결과가 전혀 다릅니다. 평가 모드에서는 이미지 한 장을 넣어도 배치에 관계없이 항상 같은 결과가 나와야 하기 때문입니다. 위의 결과는 한 번만 학습한 상태여서 누적값이 아직 실제 통계와 거리가 멀지만, 학습이 진행되면 누적값이 데이터 전체의 통계에 가까워집니다.

합성곱 계층의 출력에는 채널마다 정규화하는 `nn.BatchNorm2d(채널 수)`를 사용하며, 보통 합성곱 계층과 활성화 함수 사이에 넣습니다.

<br>
<br>

## 가중치 감쇠(Weight Decay)

`가중치 감쇠(Weight Decay)`는 가중치가 큰 값을 갖지 않도록 매 갱신마다 가중치를 조금씩 0 쪽으로 줄이는 기법입니다. 가중치가 작게 유지되면 모델이 특정 입력에 과도하게 반응하지 않는 완만한 함수를 학습합니다.

{% highlight Python %}

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)

{% endhighlight %}

PyTorch에서는 최적화 함수의 `weight_decay` 인자로 적용합니다. `Adam`에도 `weight_decay` 인자가 있지만, `Adam`은 이를 기울기에 더하는 방식이라 적응적 학습률과 섞이면서 의도한 효과가 약해집니다. 가중치 감쇠를 사용할 때는 **기울기와 분리해 가중치에서 직접 줄이는** `AdamW`를 사용하는 것이 권장됩니다. 두 방식의 차이는 [최적화 알고리즘 이론][AI-10강]에서 다루었습니다.

<br>
<br>

## 데이터 증강(Data Augmentation)

`데이터 증강(Data Augmentation)`은 모델을 제한하는 대신 **학습 데이터를 다양하게 만들어** 과대적합을 줄이는 기법입니다. 이미지를 조금 뒤집거나 회전하거나 이동해도 그 이미지가 무엇인지는 바뀌지 않으므로, 이런 변형을 무작위로 적용하면 같은 이미지라도 에폭마다 조금씩 다른 이미지로 학습하게 됩니다.

{% highlight Python %}

augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])

{% endhighlight %}

`transforms.Compose`는 여러 변환을 순서대로 적용합니다. `RandomHorizontalFlip()`은 50%의 확률로 이미지를 좌우로 뒤집고, `RandomAffine(degrees=10, translate=(0.1, 0.1))`은 최대 10도 회전하고 가로세로로 최대 10%까지 이동합니다. 마지막으로 `ToTensor()`로 텐서로 바꿉니다.

이 변환은 데이터를 **꺼낼 때마다** 새로 적용되므로, 데이터를 미리 늘려 저장할 필요가 없습니다. 같은 1,000장이라도 에폭마다 다른 모습으로 학습에 사용됩니다.

- Tip : 증강은 **정답이 바뀌지 않는 범위**에서만 적용해야 합니다. 옷은 좌우로 뒤집어도 같은 옷이지만, 숫자 이미지에서 6을 180도 회전하면 9가 됩니다. 또한 증강은 학습 데이터에만 적용하고, 검증과 시험 데이터에는 적용하지 않습니다.

<br>
<br>

## 메인 코드

{% highlight Python %}

import copy
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class CNN(nn.Module):
    def __init__(self, regularize=False):
        super().__init__()
        self.layer1 = self.block(1, 32, regularize)
        self.layer2 = self.block(32, 64, regularize)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5) if regularize else nn.Identity(),
            nn.Linear(256, 10),
        )

    def block(self, in_channels, out_channels, regularize):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        if regularize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers += [nn.ReLU(), nn.MaxPool2d(kernel_size=2)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x)
        return x


def run_epoch(model, dataloader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct = 0.0, 0

    with torch.set_grad_enabled(training):
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(x)
            correct += (output.argmax(dim=1) == y).sum().item()

    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)


def train(regularize, augment, epochs=50, patience=None):
    torch.manual_seed(42)
    train_transform = augment_transform if augment else transforms.ToTensor()
    train_dataset = Subset(datasets.FashionMNIST("./data", train=True, transform=train_transform), train_indices)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = CNN(regularize).to(device)
    criterion = nn.CrossEntropyLoss()
    if regularize:
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_loss, best_state, best_epoch, wait = float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        train_loss, _ = run_epoch(model, train_dataloader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_dataloader, criterion, device)

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch : {epoch:2d}, Train Loss : {train_loss:.4f}, Val Loss : {val_loss:.4f}, Val Acc : {val_acc:.4f}")

        if val_loss < best_loss:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if patience is not None and wait >= patience:
                print(f"Early Stopping : Epoch {epoch}")
                break

    last_acc = run_epoch(model, test_dataloader, criterion, device)[1]
    model.load_state_dict(best_state)
    best_acc = run_epoch(model, test_dataloader, criterion, device)[1]
    print(f"Best Epoch : {best_epoch}, Best Val Loss : {best_loss:.4f}")
    print(f"Test Acc (Last) : {last_acc:.4f}, Test Acc (Best) : {best_acc:.4f}")


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

indices = torch.randperm(60000, generator=torch.Generator().manual_seed(42))
train_indices, val_indices = indices[:1000], indices[1000:6000]

augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])

val_dataset = Subset(datasets.FashionMNIST("./data", train=True, download=True, transform=transforms.ToTensor()), val_indices)
test_dataset = datasets.FashionMNIST("./data", train=False, download=True, transform=transforms.ToTensor())
val_dataloader = DataLoader(val_dataset, batch_size=256)
test_dataloader = DataLoader(test_dataset, batch_size=256)

for name, regularize, augment in [("Baseline", False, False), ("Regularization", True, False), ("Regularization + Augmentation", True, True)]:
    print(f"[{name}]")
    train(regularize, augment)

{% endhighlight %}

**결과**
:    
[Baseline]<br>
Epoch :  1, Train Loss : 1.8609, Val Loss : 1.1427, Val Acc : 0.5588<br>
Epoch : 10, Train Loss : 0.4347, Val Loss : 0.5913, Val Acc : 0.7990<br>
Epoch : 20, Train Loss : 0.2264, Val Loss : 0.5311, Val Acc : 0.8216<br>
Epoch : 30, Train Loss : 0.1012, Val Loss : 0.5742, Val Acc : 0.8308<br>
Epoch : 40, Train Loss : 0.0516, Val Loss : 0.7107, Val Acc : 0.8282<br>
Epoch : 50, Train Loss : 0.0160, Val Loss : 0.7820, Val Acc : 0.8326<br>
Best Epoch : 16, Best Val Loss : 0.4940<br>
Test Acc (Last) : 0.8328, Test Acc (Best) : 0.8206<br>
[Regularization]<br>
Epoch :  1, Train Loss : 1.4752, Val Loss : 1.6406, Val Acc : 0.5774<br>
Epoch : 10, Train Loss : 0.3223, Val Loss : 0.4911, Val Acc : 0.8204<br>
Epoch : 20, Train Loss : 0.1052, Val Loss : 0.5069, Val Acc : 0.8396<br>
Epoch : 30, Train Loss : 0.0403, Val Loss : 0.5748, Val Acc : 0.8418<br>
Epoch : 40, Train Loss : 0.0370, Val Loss : 0.6787, Val Acc : 0.8390<br>
Epoch : 50, Train Loss : 0.0119, Val Loss : 0.7219, Val Acc : 0.8374<br>
Best Epoch : 14, Best Val Loss : 0.4706<br>
Test Acc (Last) : 0.8398, Test Acc (Best) : 0.8349<br>
[Regularization + Augmentation]<br>
Epoch :  1, Train Loss : 1.8672, Val Loss : 1.7153, Val Acc : 0.5142<br>
Epoch : 10, Train Loss : 0.7611, Val Loss : 0.6275, Val Acc : 0.7596<br>
Epoch : 20, Train Loss : 0.6784, Val Loss : 0.5608, Val Acc : 0.7880<br>
Epoch : 30, Train Loss : 0.5979, Val Loss : 0.5362, Val Acc : 0.8018<br>
Epoch : 40, Train Loss : 0.5449, Val Loss : 0.5031, Val Acc : 0.8086<br>
Epoch : 50, Train Loss : 0.4821, Val Loss : 0.5015, Val Acc : 0.8178<br>
Best Epoch : 41, Best Val Loss : 0.4986<br>
Test Acc (Last) : 0.8211, Test Acc (Best) : 0.8115<br>
<br>

결과는 PyTorch 2.14.1의 CPU 환경에서 실행한 값입니다. 같은 모델과 데이터에 세 가지 설정을 적용해 50 에폭씩 학습했으며, `Test Acc (Last)`는 마지막 에폭의 모델, `Test Acc (Best)`는 검증 손실이 가장 낮았던 에폭의 모델로 계산한 시험 정확도입니다.

<br>

### 기본 모델의 과대적합

`Baseline`은 정칙화 기법을 하나도 쓰지 않은 모델입니다. 학습 손실은 1.8609에서 0.0160까지 계속 줄어들어, 학습 데이터 1,000장은 거의 완벽하게 맞히게 됩니다. 그러나 검증 손실은 16 에폭의 0.4940을 최저점으로 다시 올라가기 시작해, 50 에폭에는 0.7820이 됩니다. **학습 손실은 줄어드는데 검증 손실은 늘어나는** 이 모습이 과대적합의 전형적인 신호입니다.

흥미로운 점은 검증 손실이 크게 늘어나는 동안에도 검증 정확도는 0.82~0.83 정도로 거의 떨어지지 않는다는 것입니다. 이는 과대적합된 모델이 **틀리는 데이터에 대해서도 지나치게 확신**하게 되기 때문입니다. 정답을 맞히는 비율은 비슷하지만, 틀린 예측에 매우 높은 확률을 주면 [손실 함수 이론][AI-11강]에서 다룬 교차 엔트로피가 크게 늘어납니다. 손실이 커진 모델은 예측 확률을 그대로 믿기 어려운 모델이 된 것입니다.

<br>

### 드롭아웃, 배치 정규화, 가중치 감쇠

`Regularization`은 드롭아웃, 배치 정규화, 가중치 감쇠를 함께 적용한 모델입니다. 최저 검증 손실이 0.4706으로 기본 모델의 0.4940보다 낮고, 검증 정확도도 에폭에 따라 0.5~2%포인트 정도 높습니다. 마지막 모델의 시험 정확도도 83.28%에서 83.98%로 올랐습니다.

다만 이 모델도 14 에폭 이후에는 검증 손실이 다시 늘어납니다. 학습 데이터가 1,000장뿐이라, 이 정도의 정칙화로는 과대적합을 **완전히 막지는 못하고 늦추고 줄이는** 정도입니다.

학습 손실은 드롭아웃이 켜진 학습 모드에서 계산한 값이므로, 실제로 노드를 모두 사용하는 평가 모드보다 조금 크게 나온다는 점도 알아 두어야 합니다.

<br>

### 데이터 증강

`Regularization + Augmentation`은 여기에 데이터 증강까지 더한 모델입니다. 결과가 앞의 두 모델과 뚜렷하게 다릅니다. 50 에폭이 지나도 학습 손실은 0.4821로 높게 남아 있고, 검증 손실 0.5015와의 차이가 거의 없습니다. 모델이 매번 조금씩 다른 이미지를 보므로 **학습 데이터를 외울 수 없게 된** 것입니다.

그 결과 검증 손실은 50 에폭까지 계속 줄어들고 있으며, 과대적합의 신호가 보이지 않습니다. 대신 50 에폭 시점의 시험 정확도는 82.11%로 다른 두 모델보다 낮습니다. 모델이 아직 충분히 학습되지 않은 `과소적합(Underfitting)` 상태에 가깝기 때문입니다. 증강을 사용하면 학습이 더 어려워지므로 **더 많은 에폭이 필요**하며, 검증 손실이 계속 줄고 있다는 것은 더 학습하면 성능이 오를 여지가 있다는 뜻입니다.

정칙화는 이처럼 과대적합과 과소적합 사이의 균형을 조절하는 도구입니다. 약하게 적용하면 과대적합을 막지 못하고, 강하게 적용하면 같은 학습 시간 안에 충분히 학습하지 못합니다. 어느 정도가 적절한지는 학습 곡선을 보며 정해야 합니다.

<br>

### 세부 코드

{% highlight Python %}

nn.Dropout(p=0.5) if regularize else nn.Identity(),

{% endhighlight %}

`nn.Identity()`는 입력을 아무것도 바꾸지 않고 그대로 내보내는 계층입니다. 설정에 따라 계층을 넣거나 빼야 할 때, `nn.Sequential`의 구조를 바꾸지 않고 자리만 채우는 용도로 사용합니다.

<br>

{% highlight Python %}

def run_epoch(model, dataloader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    ...
    with torch.set_grad_enabled(training):

{% endhighlight %}

학습과 평가를 하나의 함수로 처리합니다. `optimizer`를 넘기면 학습, 넘기지 않으면 평가로 동작합니다. `model.train(False)`는 `model.eval()`과 같으며, 드롭아웃과 배치 정규화가 평가 모드로 바뀝니다. `torch.set_grad_enabled(False)`는 기울기 계산을 끄는 것으로, 평가할 때 `torch.no_grad()`와 같은 역할을 합니다.

손실은 `loss.item() * len(x)`로 배치 크기를 곱해 더한 뒤 마지막에 전체 데이터 수로 나눕니다. 마지막 배치는 다른 배치보다 작을 수 있으므로, 배치마다의 평균을 단순히 평균하면 마지막 배치의 비중이 실제보다 커집니다.

<br>

{% highlight Python %}

if val_loss < best_loss:
    best_loss, best_epoch, wait = val_loss, epoch, 0
    best_state = copy.deepcopy(model.state_dict())

{% endhighlight %}

검증 손실이 지금까지의 최저값보다 낮아지면, 그 시점의 가중치를 따로 저장합니다. `model.state_dict()`는 모델의 가중치를 **참조**로 반환하므로, 그대로 저장하면 이후 학습으로 가중치가 바뀔 때 저장해 둔 값도 함께 바뀝니다. 그래서 `copy.deepcopy`로 그 순간의 값을 복사해 둡니다. 파일로 저장하는 방법은 [모델 저장/불러오기(Model Save/Load)][10강]에서 다루었습니다.

<br>
<br>

## 조기 종료(Early Stopping)

`조기 종료(Early Stopping)`는 검증 성능이 더 이상 나아지지 않으면 학습을 멈추는 기법입니다. 기본 모델의 결과에서 보았듯 과대적합은 특정 시점 이후의 학습에서 발생하므로, 그 시점에서 멈추면 불필요한 학습 시간도 줄이고 과대적합도 피할 수 있습니다.

{% highlight Python %}

train(regularize=False, augment=False, epochs=100, patience=5)

{% endhighlight %}

**결과**
:    
Epoch :  1, Train Loss : 1.8609, Val Loss : 1.1427, Val Acc : 0.5588<br>
Epoch : 10, Train Loss : 0.4347, Val Loss : 0.5913, Val Acc : 0.7990<br>
Epoch : 20, Train Loss : 0.2264, Val Loss : 0.5311, Val Acc : 0.8216<br>
Early Stopping : Epoch 21<br>
Best Epoch : 16, Best Val Loss : 0.4940<br>
Test Acc (Last) : 0.8277, Test Acc (Best) : 0.8206<br>
<br>

메인 코드의 마지막 반복문 대신 위의 한 줄을 실행한 결과입니다. 검증 손실이 연속으로 개선되지 않은 횟수를 `wait`로 세고, 그 횟수가 `patience`에 도달하면 학습을 멈춥니다. 검증 손실은 16 에폭에서 최저였고, 이후 다섯 번 연속으로 개선되지 않자 21 에폭에서 학습이 멈췄습니다. 최대 100 에폭으로 설정했지만 실제로는 21 에폭만 학습했습니다.

검증 손실은 매 에폭 일정하게 줄어들지 않고 오르내리므로, 한 번 나빠졌다고 바로 멈추지 않고 `인내(Patience)`만큼 기다립니다. 인내를 너무 작게 잡으면 잠깐 흔들린 것을 과대적합으로 오인해 너무 일찍 멈추고, 너무 크게 잡으면 과대적합이 진행된 뒤에야 멈춥니다.

이 실험에서는 한 가지 더 확인할 점이 있습니다. 검증 손실이 가장 낮았던 16 에폭 모델의 시험 정확도는 82.06%로, 21 에폭 모델의 82.77%보다 오히려 낮습니다. 앞에서 살펴본 것처럼 **손실과 정확도가 항상 같은 방향으로 움직이지는 않기** 때문입니다. 조기 종료의 기준은 실제로 중요하게 여기는 지표로 정해야 합니다. 예측 확률을 그대로 사용하는 서비스라면 검증 손실이, 맞힌 비율만 중요하다면 검증 정확도가 더 적절한 기준이 될 수 있습니다.

- Tip : 조기 종료로 멈춘 시점의 모델이 아니라, **기준 지표가 가장 좋았던 시점의 모델**을 최종 모델로 사용합니다. 멈춘 시점은 이미 인내만큼 성능이 나빠진 상태이기 때문입니다. 위의 `train` 함수도 학습이 끝난 뒤 저장해 둔 `best_state`를 불러와 평가합니다.

<br>
<br>

## 정칙화 기법 정리

| 기법 | PyTorch 구현 | 적용 위치 | 학습 / 평가 동작 차이 |
|---|---|---|---|
| 드롭아웃 | `nn.Dropout(p)` | 주로 완전 연결 계층 사이 | 학습 시에만 일부 노드를 끔 |
| 배치 정규화 | `nn.BatchNorm1d`, `nn.BatchNorm2d` | 합성곱 · 선형 계층과 활성화 함수 사이 | 학습 시 배치 통계, 평가 시 누적 통계 사용 |
| 가중치 감쇠 | `optim.AdamW(..., weight_decay)` | 최적화 함수 | 없음 |
| 데이터 증강 | `transforms.RandomHorizontalFlip` 등 | 학습 데이터의 `transform` | 학습 데이터에만 적용 |
| 조기 종료 | 검증 지표를 기록하는 학습 반복문 | 학습 반복문 | 없음 |

정칙화 기법은 서로 배타적이지 않아 함께 사용할 수 있지만, 이번 실험처럼 **기법을 더할수록 성능이 항상 좋아지는 것은 아닙니다.** 정칙화를 강하게 걸수록 과대적합은 줄어들지만 학습은 느려지므로, 학습 손실과 검증 손실을 함께 기록하며 두 곡선의 간격과 추세를 보고 판단해야 합니다.

이번 강좌의 결과는 학습 데이터 1,000장이라는 극단적인 조건에서 얻은 것입니다. 정칙화가 해결할 수 있는 범위에는 한계가 있으며, 과대적합의 가장 근본적인 해결책은 여전히 **더 많은 데이터를 확보하는 것**입니다. [합성곱 신경망][15강]에서 채널 수가 더 적은 비슷한 구조의 모델을 60,000장으로 학습했을 때 정칙화 없이도 약 89%의 정확도를 얻은 것과 비교해 보면 그 차이를 확인할 수 있습니다.

<br>
<br>

[10강]: https://076923.github.io/posts/Python-pytorch-10/
[11강]: https://076923.github.io/posts/Python-pytorch-11/
[15강]: https://076923.github.io/posts/Python-pytorch-15/
[AI-7강]: https://076923.github.io/posts/AI-7/
[AI-9강]: https://076923.github.io/posts/AI-9/
[AI-10강]: https://076923.github.io/posts/AI-10/
[AI-11강]: https://076923.github.io/posts/AI-11/
