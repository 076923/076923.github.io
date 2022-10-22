---
layout: post
title: "Python Pytorch 강좌 : 제 2강 - 텐서(Tensor)"
tagline: "Python PyTorch Tensor Structure"
image: /assets/images/pytorch.webp
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['PyTorch']
keywords: Python, Python Pytorch, Pytorch Tensor, Pytorch Tensor Structure
ref: Python-PyTorch
category: Python
permalink: /posts/Python-pytorch-2/
comments: true
toc: true
---

## 텐서(Tensor)

`텐서(Tensor)`란 `NumPy`의 `ndarray` 클래스와 유사한 구조로 `배열(Array)`이나 `행렬(Matrix)`과 유사한 자료구조(자료형)입니다.

`PyTorch`에서는 텐서를 사용하여 모델의 입/출력뿐만 아니라 모델의 매개변수를 `부호화(Encode)`하고 `GPU`를 활용해 연산을 가속화할 수 있습니다.

`NumPy`와 공통점은 `수학 계산`, `선형 대수` 연산을 비롯해 `전치(Transposing)`, `인덱싱(Indexing)`, `슬라이싱(slicing)`, `임의 샘플링(random sampling)` 등 다양한 텐서 연산을 진행할 수 있습니다.

`NumPy`와의 차이점은 `CPU`에서 사용하는 텐서와 `GPU`에서 사용하는 텐서의 선언 방식의 차이가 있습니다.

`GPU 가속(GPU Acceleration)`을 적용할 수 있으므로 `CPU 텐서`와 `GPU 텐서`로 나눠지고, 각각의 텐서를 **상호 변환**하거나 **GPU 사용 유/무**를 설정합니다.

<br>
<br>

## N차원 텐서

<img data-src="{{ site.images }}/assets/posts/Python/PyTorch/lecture-2/1.webp" class="lazyload" width="100%" height="100%"/>

`텐서(Tensor)`는 앞선 설명과 같이 `NumPy`의 `ndarray`와 비슷한 방식으로 구현될 수 있습니다.

어떤 `차원(Rank)`으로 구성되어있느냐에 따라 텐서의 형태를 이해할 수 있습니다.

<br>

`스칼라(Scalar)`는 크기만 있는 물리량이지만 **0차원 텐서**라고도 부릅니다.

**모든 값들의 기본 형태로 볼 수 있으며, 차원은 없습니다.**

<br>

`벡터(vector)`는 `[1, 2, 3]`과 같은 형태로 Python에서 많이 사용하는 `1차원 목록(List)`와 비슷한 형태입니다.

**스칼라 값들을 하나로 묶은 형태**로 간주할 수 있으며, `(N, )`의 차원을 갖습니다.

- Tip : 행렬과 구분하기 위해 `(N, 1)`이 아닌, `(N, )`으로 표현합니다.

<br>

`행렬(matrix)`은 `[[1, 2, 3], [4, 5, 6]]`과 같은 형태로 `회색조(Grayscale)` 이미지를 표현하거나 `좌표계(Coordinate System)`로도 활용될 수 있습니다.

**벡터 값들을 하나로 묶은 형태**로 간주할 수 있으며, `(N, M)`으로 표현합니다.

<br>

`배열(Array)`은 3차원 이상의 배열을 모두 지칭하며, 각각의 차원을 구별하기 위해 `N차원 배열` 또는 `N차원 텐서`로 표현합니다.

배열의 경우 `이미지(Image)`를 표현하기에 가장 적합한 형태를 띕니다.

즉, `행렬(matrix)`를 **세 개 생성하여 겹쳐놓은 구조로 볼 수 있습니다.**

이 행렬의 의미는 각각 `R, G, B`로 표현될 수 있습니다.

행렬 값들을 하나로 묶은 형태로 간주할 수 있으며, `(N, M, K)`으로 표현합니다.

이미지의 경우 `(C, H, W)`로 표현합니다. `C`은 채널, `H`는 이미지의 높이, `W`는 이미지의 너비가 됩니다.

<br>

`4차원 배열(4D Array, 4D Tensor)`은 3차원 배열들을 하나로 묶은 형태이므로, **여러 개의 이미지(Image)들의 묶음으로 볼 수 있습니다.**

`PyTorch`를 통해 **이미지 데이터를 학습시킬 때 주로 4차원 배열 구조의 형태로 가장 많이 사용합니다.**

이미지의 경우 `(N, C, H, W)`로 표현합니다. `N`의 경우 이미지의 개수를 의미합니다.

<br>
<br>

## 텐서 유형

| 데이터 형식 | 의미 | 자료형 | CPU 텐서 | GPU 텐서 |
| :--------: | :---: | :---: | :-------: | :------: |
| Byte | 부호 없는 8 비트 정수	| torch.uint8 | torch.ByteTensor | torch.cuda.ByteTensor |
| Short | 부호 있는 16 비트 정수 | torch.short | torch.ShortTensor | torch.cuda.ShortTensor |
| Int | 부호 있는 32 비트 정수 | torch.int | torch.IntTensor | torch.cuda.IntTensor |
| Long | 부호 있는 64 비트 정수 | torch.long | torch.LongTensor | torch.cuda.LongTensor |
| Binary16 | 16 비트 부동 소수점 | torch.half | torch.HalfTensor | torch.cuda.HalfTensor |
| Brain Floating Point | 16 비트 부동 소수점 | torch.bfloat16 | torch.BFloat16Tensor | torch.cuda.BFloat16Tensor |
| Float | 32 비트 부동 소수점 | torch.float | torch.FloatTensor | torch.cuda.FloatTensor |
| Double | 32 비트 부동 소수점 | torch.double | torch.DoubleTensor | torch.cuda.DoubleTensor |
| Boolean | 논리 형식 | torch.bool | torch.BoolTensor | torch.cuda.BoolTensor |
| Int8 | 부호 있는 8 비트 정수 | torch.int8 | torch.CharTensor | torch.CharTensor |
| Int16 | 부호 있는 16 비트 정수 | torch.int16 | torch.ShortTensor | torch.cuda.ShortTensor |
| Int32 | 부호 있는 32 비트 정수 | torch.int32 | torch.IntTensor | torch.cuda.IntTensor |
| Int64 | 부호 있는 64 비트 정수 | torch.int64 | torch.LongTensor | torch.cuda.LongTensor |
| Float16 | 16 비트 부동 소수점 | torch.float16 | torch.HalfTensor | torch.cuda.HalfTensor |
| Float32 | 32 비트 부동 소수점 | torch.float32 | torch.FloatTensor | torch.cuda.FloatTens |
| Float64 | 64 비트 부동 소수점 | torch.float64 | torch.DoubleTensor | torch.cuda.DoubleTensor |
| Complex32 | 32 비트 복합 형식 | torch.complex32 | | |
| Complex64 | 64 비트 복합 형식 | torch.complex64 | | |
| Complex128 | 128 비트 복합 형식 | torch.complex128 | | |
| Complex128 | 128 비트 복합 형식 | torch.cdouble | | |
| Quantized Int | 양자화된 부호 있는 4비트 정수 | torch.quint4x2 | torch.ByteTensor | |
| Quantized Int | 양자화된 부호 없는 8 비트 정수 | torch.quint8 | torch.ByteTensor | |
| Quantized Int | 양자화된 부호 있는 8 비트 정수 | torch.qint8 | torch.CharTensor | |
| Quantized Int | 양자화된 부호 있는 32 비트 정수 | torch.qfint32 | torch.IntTensor | |

`PyTorch`에서 지원하는 텐서의 유형은 위와 같습니다.

그 중, `IntTensor`, `LongTensor`, `FloatTensor`가 가장 많이 활용됩니다.

또한, `torch.Tensor`의 기본 유형은 `torch.FloatTensor(torch.float32)`를 따릅니다.

<br>
<br>

## 텐서 사용하기

### 텐서 생성

{% highlight Python %}

import torch

print(torch.tensor([1, 2, 3]))
print(torch.Tensor([[1, 2, 3], [4, 5, 6]]))
print(torch.LongTensor([1, 2, 3]))
print(torch.FloatTensor([1, 2, 3]))

{% endhighlight %}
**결과**
:    
tensor([1, 2, 3])<br>
tensor([[1., 2., 3.],<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[4., 5., 6.]])<br>
tensor([1, 2, 3])<br>
tensor([1., 2., 3.])<br>
<br>

`PyTorch`에서 `텐서(Tensor)`는 `torch.tensor()` 또는 `torch.Tensor()`를 통해 생성할 수 있습니다.

`torch.tensor()`는 **입력된 데이터를 복사하여 텐서(Tensor)로 변환하는 함수입니다.**

즉, 데이터를 복사하기 때문에 값이 무조건 존재해야 하며 입력된 데이터의 형식에 가장 적합한 텐서 자료형으로 변환합니다.

`torch.Tensor()`는 **텐서(Tensor)의 기본형으로 텐서 인스턴스를 생성하는 클래스입니다.**

인스턴스를 생성하기 때문에 값을 입력하지 않는 경우, 비어있는 텐서를 생성합니다.

- Tip : 가능한 자료형이 명확하게 표현되는 클래스 형태의 `torch.Tensor()`를 사용하는 것이 좋습니다.
- Tip : `torch.tensor()`는 비어있는 구조로 생성이 되지 않으며 자동으로 자료형을 할당하기 때문에 의도하지 않는 자료형으로 변경될 수 있습니다.

<br>

### 텐서 속성

{% highlight Python %}

import torch

tensor = torch.rand(1, 2)

print(tensor)
print(tensor.shape)
print(tensor.dtype)
print(tensor.device)

{% endhighlight %}
**결과**
:    
tensor([[0.8522, 0.3964]])<br>
torch.Size([1, 2])<br>
torch.float32<br>
cpu<br>
<br>

텐서의 `속성(Attribute)`은 크게 `형태(shape)`, `자료형(dtype)`, `장치(device)`로 나눌 수 있습니다.

`형태(shape)`는 **텐서의 차원을 의미합니다.**

`자료형(dtype)`은 **텐서의 데이터 구조를 의미합니다.**

`장치(device)`는 **텐서의 GPU 가속 유/무를 의미합니다.**

텐서 연산을 진행할 때, 위 세 가지 속성 중 하나라도 맞지 않는다면 동작하지 않습니다.

<br>

### 텐서 차원 변환

{% highlight Python %}

import torch

tensor = torch.rand(1, 2)
print(tensor)

tensor = tensor.reshape(2, 1)
print(tensor)

{% endhighlight %}
**결과**
:    
tensor([[0.6499, 0.3419]])<br>
tensor([[0.6499],<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[0.3419]])<br>
<br>

텐서의 차원 변환은 `reshape()` 메서드를 활용하여 변환이 가능합니다.

기존 `NumPy`에서 차원의 형태를 변경하는 방법과 동일합니다.

<br>

### 자료형 설정

{% highlight Python %}

import torch

print(torch.rand((3, 3), dtype=torch.float))

{% endhighlight %}
**결과**
:    
tensor([[0.6837, 0.7457, 0.9212],<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[0.3221, 0.9590, 0.1553],<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[0.7908, 0.4360, 0.7417]])<br>

텐서 함수나 메서드 등을 통해 텐서를 생성할 때는 `데이터 형식(dtype)`을 선언하여 명확하게 데이터를 표현합니다.

인자로 주어진 **값의 속성(형태, 자료형 등)**을 유지하지만, 위 예시와 같이 모호한 경우에는 명확하게 구현하려는 데이터 형태를 표현하는 것이 좋습니다.

<br>

### 장치 설정

{% highlight Python %}

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cpu = torch.FloatTensor([1, 2, 3])
gpu = torch.cuda.FloatTensor([1, 2, 3])
tensor = torch.rand((1, 1), device=device)

print(device)
print(cpu)
print(gpu)
print(tensor)

{% endhighlight %}
**결과**
:    
cuda<br>
tensor([1., 2., 3.])<br>
tensor([1., 2., 3.], device='cuda:0')<br>
tensor([[0.1998]], device='cuda:0')<br>

텐서의 `장치(device)`를 설정하는 방법으로는 `device` 매개변수에 장치 속성을 할당합니다.

`torch.cuda.is_available()` 함수로 `CUDA` 사용 유/무를 확인할 수 있으므로, 해당 변수를 선언해서 장치를 통일시킬 수 있습니다.

`Tensor` 클래스 선언의 경우 `device` 매개변수가 존재하지만, `CUDA`용 클래스가 별도로 존재하므로 `torch.cuda.Tensor` 클래스를 사용합니다.

특정 데이터에서 복사하여 텐서를 적용하는 경우 `device` 매개변수에 장치값을 할당합니다.

- Tip : `device` 속성에는 cpu, cuda, mkldnn, opengl, opencl, ideep, hip, msnpu, xla 등이 있습니다.

<br>

### 장치 변환

{% highlight Python %}

import torch

cpu = torch.FloatTensor([1, 2, 3])
gpu = cpu.cuda()
gpu2cpu = gpu.cpu()

print(cpu)
print(gpu)
print(gpu2cpu)

{% endhighlight %}
**결과**
:    
tensor([1., 2., 3.])<br>
tensor([1., 2., 3.], device='cuda:0')<br>
tensor([1., 2., 3.])<br>

`장치(device)`간 상호 변환은 `cuda()`와 `cpu()` 메서드를 통해 변환할 수 있습니다.

`cuda()` 메서드는 `cpu` 장치로 선언된 값을 `gpu`로 변환합니다.

`cpu()` 메서드는 `gpu` 장치로 선언된 값을 `cpu`로 변환합니다.

<br>

### NumPy 변환

{% highlight Python %}

import torch
import numpy as np

print(torch.tensor(np.array([1, 2, 3], dtype=np.uint8)))
print(torch.Tensor(np.array([1, 2, 3], dtype=np.uint8)))
print(torch.from_numpy(np.array([1, 2, 3], dtype=np.uint8)))

{% endhighlight %}
**결과**
:    
tensor([1, 2, 3], dtype=torch.uint8)<br>
tensor([1., 2., 3.])<br>
tensor([1, 2, 3], dtype=torch.uint8)<br>
<br>

`NumPy`의 `ndarray` 데이터를 그대로 입력하여 텐서로 변경할 수 있습니다.

`NumPy`와 매우 친화적인 구조를 가지므로 `NumPy`의 데이터를 변환없이 적용할 수 있습니다.

`torch.Tensor()` 클래스나 `torch.tensor()`, `torch.from_numpy()` 함수를 통해서 `텐서(Tensor)`로 변환할 수 있습니다. 

<br>

### Tensor 변환

{% highlight Python %}

import torch

tensor = torch.cuda.FloatTensor([1, 2, 3])
ndarray = tensor.detach().cpu().numpy()

print(ndarray)

{% endhighlight %}
**결과**
:    
[1. 2. 3.]<br>
<br>

`텐서(Tensor)`를 `NumPy`로 변환할 때는 `detach()` 메서드를 적용한 후에 변환하는 것이 좋습니다.

`detach()` 메서드는 현재 `그래프(graph)`에서 분리된 **새로운 텐서(Tensor)를 반환합니다.**

텐서는 모든 연산을 추적해 기록하는데, 이 기록을 통해 `역전파(Backpropagation)` 등과 같은 연산이 진행됩니다.

`detach()` 메서드는 이 연산을 분리하여 텐서로 변환합니다.


