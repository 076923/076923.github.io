---
layout: post
title: "Python Pytorch 강좌 : 제 8강 - 데이터 세트(Data Set)"
tagline: "Python PyTorch Data Set"
image: /assets/images/pytorch.webp
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['PyTorch']
keywords: Python, Python Pytorch, Pytorch Dataset, Pytorch TensorDataset, Pytorch DataLoader
ref: Python-PyTorch
category: Python
permalink: /posts/Python-pytorch-8/
comments: true
toc: true
---

## 데이터 세트(Data Set)

`데이터 세트(Data Set)`는 데이터의 집합을 의미하며, `입력값(X)`과 `결괏값(Y)`에 대한 정보를 제공하거나 **일련의 데이터 묶음**을 제공합니다.

데이터 세트의 구조는 일반적으로 `데이터베이스(Database)`의 `테이블(Table)`과 같은 형태로 구성되어 있습니다.

데이터 세트의 한 패턴을 테이블의 `행(Row)`으로 간주한다면, **이 행에서 데이터를 불러와 학습을 진행합니다.**

즉, 다음과 같은 테이블이 하나의 데이터 세트가 됩니다.

<br>

### 입력값과 결괏값을 제공하는 데이터 세트

| | path | class |	
| :---: | -------- | ---- | 
| 1 | 2f35ab7d-6d28-4f7f-adf1-51cb065aaf38.jpg | dog |
| 2 | 4f515235-eb81-4a0a-b65e-ff24fe2de3f3.jpg | cat |
| 3 | c93f02c1-a015-4120-9857-40b8811d27ca.jpg | dog |
| 4 | d4dadc3d-054d-48e1-b709-5486c9b84b3a.jpg | human |

<br>

### 일련의 데이터 묶음을 제공하는 데이터 세트

| | location | date |	variant | num_sequences | perc_sequences | num_sequences_total |
| :---: | --- | --- | --- | --- | --- | --- |
| 1 | Angola | 2020-12-21 | B.1.160 | 0 | 0 | 93 |
| 2 | Angola | 2020-12-21 | B.1.620 | 2 | 0 | 93 |
| 3 | Angola | 2020-12-21 | B.1.258 | 0 | 1 | 93 |
| 4 | Angola | 2020-12-21 | B.1.221 | 0 | 0 | 93 |


위 데이터 세트에서 확인할 수 있듯이 제공되는 데이터의 구조나 패턴은 매우 다양합니다.

만약 학습해야하는 데이터가 `파일 경로`로 제공되거나, 활용하기 위해 `전처리` 단계가 필요한 경우도 존재합니다.

또한 다양한 데이터가 포함된 데이터 세트에서는 **특정한 필드의 값을 사용하거나 사용하지 않을 수 있습니다.**

위와 같은 데이터를 변형하고 매핑하는 코드를 학습 과정에 직접 반영하게 되면 `모듈화(modularization)`, `재사용성(reusable)`, `가독성(Readability)` 등을 떨어뜨리는 주요한 원인이 됩니다.

이와 같은 현상을 방지하고 코드를 구조적으로 설계할 수 있도록 `데이터 세트(Dataset)`와 `데이터 로더(DataLoader)`를 사용합니다.

<br>
<br>

## PyTorch의 데이터 세트(Dataset)

PyTorch에서 사용되는 `데이터 세트(Dataset)`는 학습에 필요한 `데이터 샘플(Sample)`을 정제하고 `정답(Label)`을 저장하는 기능을 제공합니다.

`데이터 세트(Dataset)`는 `클래스(Class)` 형태로 제공되며, 세 가지의 `초기화 메서드(__init__ )`, `호출 메서드(__getitem__)`, `길이 반환 메서드(__len__ )` 를 재정의하여 활용합니다.

<br>

### 데이터 세트(Dataset) 기본형

{% highlight Python %}

class Dataset:

    def __init__(self, data, *arg, **kwargs):
        self.data = data

    def __getitem__(self, index):
        return tuple(data[index] for data in data.tensors)

    def __len__(self):
        return self.data[0].size(0)


{% endhighlight %}

`초기화 메서드(__init__ )`는 입력된 데이터의 전처리를 과정을 수행하는 메서드입니다.

새로운 인스턴스가 생성될 때 학습에 사용될 데이터를 선언하고, **학습에 필요한 형태로 변형하는 과정을 진행합니다.**

예를 들어, 입력된 데이터가 파일 경로의 형태로 제공된다면 초기화 메서드에서 파일을 불러와 활용 가능한 형태로 변형하는 과정을 진행합니다.

`호출 메서드(__getitem__)`는 학습을 진행할 때 사용되는 하나의 `행(Row)`을 불러오는 과정으로 볼 수 있습니다.

입력된 `색인(index)`에 해당하는 데이터 샘플을 불러오고 반환합니다.

초기화 메서드에서 변형되거나 개선된 데이터를 가져오며, `데이터 샘플(Sample)`과 `정답(Label)`을 반환합니다.

`길이 반환 메서드(__len__ )`는 학습에 사용된 **전체 데이터 세트의 개수를 반환합니다.**

이 메서드를 통해 몇 개의 데이터로 학습이 진행되는지 확인할 수 있습니다.

<br>
<br>

## PyTorch의 데이터 로더(DataLoader)

PyTorch에서 사용되는 `데이터 로더(DataLoader)`는 `데이터 세트(Dataset)`에 저장된 데이터를 어떠한 방식으로 불러와 활용할지 정의합니다.

학습을 조금 더 원활하게 진행할 수 있도록 `배치 크기(Batch Size)`, `데이터 순서 변경(Shuffle)`, `데이터 로드 프로세스 수(num_workers)` 등의 기능을 제공합니다.

`배치 크기(Batch Size)`는 학습에 사용되는 데이터의 개수가 매우 많아 한 번의 `에폭(Epoch)`에서 **모든 데이터를 메모리에 올릴 수 없을 때 데이터를 나누는 역할을 합니다.**

전체 데이터 세트에서 `배치 크기(Batch Size)`만큼 데이터 샘플을 나누게 되고, 모든 `배치(Batch)`를 대상으로 학습을 완료하면 한 번의 `에폭(Epoch)` 완료되는 구조로 볼 수 있습니다.

즉, **1,000**개의 데이터 샘플이 `데이터 세트(Dataset)`의 전체 길이일 때, `배치 크기(Batch Size)`를 **100**으로 할당한다면 **10**번의 배치가 완료될 때 **1**번의 에폭(Epoch)이 진행되었다 볼 수 있습니다.

`데이터 순서 변경(Shuffle)`은 `모델(Model)`이 데이터의 관계가 아닌, **데이터의 순서로 학습되는 것을 방지하고자 수행하는 기능입니다.**

**데이터 샘플(Sample)**과 **정답(Label)**의 매핑 관계는 변경되지 않으며, `행(Row)`의 순서를 변경하는 개념입니다.

`데이터 로드 프로세스 수(num_workers)`는 데이터를 불러올 때 프로세스의 개수를 의미합니다.

학습을 제외한 코드에서는 데이터를 불러오는 시간이 가장 오래 소요됩니다. 이를 최소화 하고자 **데이터 로드에 필요한 프로세스의 수를 늘릴 수 있습니다.**

<br>

PyTorch에서는 `데이터 세트(Dataset)`와 `데이터 로더(Dataloader)`를 통해 학습에 필요한 데이터 구조를 생성합니다.

일반적으로 `데이터 세트(Dataset)`를 재정의하여 가장 많이 사용하며, `데이터 로더(Dataloader)`에서는 주로 `배치 크기(Batch Size)`를 조절해가며 현재 학습 환경에 맞는 구조로 할당합니다.

<br>
<br>

## 데이터 세트(Dataset)/데이터 로더(Dataloader) 적용

### 메인 코드

{% highlight Python %}

import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader 


train_x = torch.FloatTensor([
    [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]
])
train_y = torch.FloatTensor([
    [0.1, 1.5], [1, 2.8], [1.9, 4.1], [2.8, 5.4], [3.7, 6.7], [4.6, 8]
])

train_dataset = TensorDataset(train_x, train_y)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)

model = nn.Linear(2, 2, bias=False)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(20000):
    cost = 0.0
    
    for batch in train_dataloader:
        x, y = batch
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

<br>

#### 세부 코드

{% highlight Python %}

from torch.utils.data import TensorDataset, DataLoader 

{% endhighlight %}

`데이터 세트`와 `데이터 로더`를 사용하기 위해 `TensorDataset`과 `DataLoader`을 포함시킵니다.

`텐서 데이터 세트(TensorDataset)`는 `기본 데이터 세트(Dataset)` 클래스를 상속받아 재정의된 클래스입니다.

지금 예제에서는 `데이터 세트(Dataset)`를 재정의하지 않고 제공되는 `텐서 데이터 세트(TensorDataset)`를 사용하도록 하겠습니다.

<br>

{% highlight Python %}

train_dataset = TensorDataset(train_x, train_y)

{% endhighlight %}

`텐서 데이터 세트(TensorDataset)`를 활용하여 훈련용 데이터 세트를 생성합니다.

텐서 데이터 세트는 초기화값을 `*args` 형태로 입력받기 때문에 여러 개의 데이터를 입력받을 수 있습니다.

<br>

{% highlight Python %}

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)

{% endhighlight %}

`데이터 로더(DataLoader)`를 활용하여 `훈련용 데이터 세트(train_dataset)`를 불러옵니다.

데이터 로더에 사용되는 `배치 크기(batch_size)`를 2로 선언하여 한 번의 배치마다 두 개의 데이터 샘플과 정답을 가져오게 합니다.

`데이터 순서 변경(shuffle)`을 `참(True)` 값으로 적용하여 불러오는 데이터의 순서를 무작위로 변경합니다.

`마지막 배치 제거(drop_last)`는 `배치 크기(batch_size)`로 전체 데이터를 나눈다면 마지막 배치는 배치 크기와 같은 크기를 가질 수 없습니다.

예를 들어, 전체 데이터 세트의 크기가 5일 때, 배치 크기가 2라면 마지막 배치의 크기는 1이 됩니다.

배치 크기가 1인 배치를 `불완전한 배치(Incomplete Batch)`라고 합니다.

이 불완전한 배치를 사용할지, 사용하지 않을지를 설정하게 됩니다.

<br>

{% highlight Python %}

for epoch in range(20000):
    cost = 0.0
    
    for batch in train_dataloader:
        x, y = batch
        output = model(x)

{% endhighlight %}

`비용(Cost)`을 다시 계산하기 위해 `에폭(Epoch)`마다 0으로 초기화합니다.

이제 전체 데이터 세트의 크기가 아닌 `배치(Batch)` 크기로 데이터를 학습하기 때문에 `손실(Loss)`을 계산하게 됩니다.

`훈련용 데이터 로더(train_dataloader)`를 반복하여 `배치(batch)`를 반환합니다.

이 `배치(batch)`에는 `텐서 데이터 세트(TensorDataset)`에 입력한 순서로 데이터가 반환됩니다.

즉, `배치(batch)`에는 `입력값(x)`과 `결괏값(y)`이 포함되어 있습니다.

<br>

{% highlight Python %}

    ...
    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    cost += loss

cost = cost / len(train_dataloader)

{% endhighlight %}

`손실(loss)` 값을 계산하고, 배치마다 `비용(cost)`에 `손실(loss)` 값을 누적해서 더합니다.

`비용(cost)`의 평균값을 계산하기 위해 `훈련용 데이터 로더(train_dataloader)`의 길이 만큼 나눕니다.

`데이터 로더(Dataloader)`를 활용하게 되면 자연스럽게 `배치(batch)` 구조로 코드가 변경되게 됩니다.

학습에 사용되는 데이터의 구조나 형태가 변경되더라도, 실제 학습에 사용되는 코드는 변경되지 않게 되어 각 모듈에 집중할 수 있게 됩니다.
