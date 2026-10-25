---
layout: post
title: "Python Pytorch 강좌 : 제 2강 - 텐서(Tensor)"
tagline: "Python PyTorch Tensor Structure"
image: /assets/images/pytorch.webp
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

`NumPy`와의 차이점은 텐서가 `장치(device)` 속성을 가진다는 점입니다.

`GPU 가속(GPU Acceleration)`을 적용할 수 있으므로 텐서가 `CPU`와 `GPU` 중 어느 장치에 있는지 지정하고, `to()` 메서드로 장치 간에 **상호 변환**합니다.

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

| 데이터 형식 | 의미 | 자료형(dtype) | 별칭 |
| :--------: | :---: | :---: | :---: |
| Boolean | 논리 형식 | torch.bool | |
| UInt8 | 부호 없는 8 비트 정수 | torch.uint8 | |
| Int8 | 부호 있는 8 비트 정수 | torch.int8 | |
| Int16 | 부호 있는 16 비트 정수 | torch.int16 | torch.short |
| Int32 | 부호 있는 32 비트 정수 | torch.int32 | torch.int |
| Int64 | 부호 있는 64 비트 정수 | torch.int64 | torch.long |
| Float16 | 16 비트 부동 소수점 | torch.float16 | torch.half |
| Brain Floating Point | 16 비트 부동 소수점 | torch.bfloat16 | |
| Float32 | 32 비트 부동 소수점 | torch.float32 | torch.float |
| Float64 | 64 비트 부동 소수점 | torch.float64 | torch.double |
| Complex32 | 32 비트 복소수 | torch.complex32 | torch.chalf |
| Complex64 | 64 비트 복소수 | torch.complex64 | torch.cfloat |
| Complex128 | 128 비트 복소수 | torch.complex128 | torch.cdouble |
| Quantized UInt4 | 양자화된 부호 없는 4 비트 정수 | torch.quint4x2 | |
| Quantized UInt8 | 양자화된 부호 없는 8 비트 정수 | torch.quint8 | |
| Quantized Int8 | 양자화된 부호 있는 8 비트 정수 | torch.qint8 | |
| Quantized Int32 | 양자화된 부호 있는 32 비트 정수 | torch.qint32 | |

`PyTorch`에서 지원하는 대표적인 텐서의 자료형은 위와 같습니다.

그 중, `torch.int64`, `torch.float32`가 가장 많이 활용되며, 정수 목록으로 텐서를 만들면 `torch.int64`, 실수 목록으로 만들면 `torch.float32`가 할당됩니다.

또한, 부동 소수점 텐서의 기본 자료형은 `torch.float32`를 따르며, `torch.get_default_dtype()`으로 확인할 수 있습니다.

`GPU`에서는 메모리 사용량과 연산량을 줄이기 위해 `torch.float16`이나 `torch.bfloat16`도 많이 사용됩니다.

- Tip : 예전에는 `torch.FloatTensor`, `torch.cuda.FloatTensor`처럼 자료형과 장치마다 별도의 텐서 클래스를 사용했지만, 현재는 `torch.tensor(..., dtype=...)`로 **자료형**을, `device` 매개변수나 `to()` 메서드로 **장치**를 지정합니다.
- Tip : 양자화 자료형은 모델 `양자화(Quantization)`에서 사용하는 특수한 자료형입니다.

<br>
<br>

## 텐서 사용하기

### 텐서 생성

{% highlight Python %}

import torch

print(torch.tensor([1, 2, 3]))
print(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32))
print(torch.tensor([1, 2, 3], dtype=torch.int64))
print(torch.tensor([1, 2, 3], dtype=torch.float32))
print(torch.Tensor([1, 2, 3]))

{% endhighlight %}
**결과**
:    
tensor([1, 2, 3])<br>
tensor([[1., 2., 3.],<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[4., 5., 6.]])<br>
tensor([1, 2, 3])<br>
tensor([1., 2., 3.])<br>
tensor([1., 2., 3.])<br>
<br>

`PyTorch`에서 `텐서(Tensor)`는 `torch.tensor()` 또는 `torch.Tensor()`를 통해 생성할 수 있습니다.

`torch.tensor()`는 **입력된 데이터를 복사하여 텐서(Tensor)로 변환하는 함수입니다.**

즉, 데이터를 복사하기 때문에 값이 무조건 존재해야 하며, `dtype`을 지정하지 않으면 입력된 데이터의 형식에 가장 적합한 자료형으로 변환합니다.

`dtype` 매개변수에 자료형을 할당하면 원하는 자료형으로 텐서를 생성할 수 있습니다.

`torch.Tensor()`는 **텐서(Tensor)의 기본형으로 텐서 인스턴스를 생성하는 클래스입니다.**

입력된 데이터의 형식과 관계없이 기본 자료형(`torch.float32`)으로 생성되며, 값을 입력하지 않는 경우 비어있는 텐서를 생성합니다.

- Tip : 데이터로 텐서를 생성할 때는 `torch.tensor()`에 `dtype`을 명시하는 방법을 권장합니다.
- Tip : `dtype`을 생략하면 입력 데이터에 따라 자료형이 자동으로 정해지므로, 정수 목록을 입력하면 의도와 달리 `torch.int64`가 될 수 있습니다.

<br>

### 텐서 속성

{% highlight Python %}

import torch

torch.manual_seed(0)
tensor = torch.rand(1, 2)

print(tensor)
print(tensor.shape)
print(tensor.dtype)
print(tensor.device)

{% endhighlight %}
**결과**
:    
tensor([[0.4963, 0.7682]])<br>
torch.Size([1, 2])<br>
torch.float32<br>
cpu<br>
<br>

텐서의 `속성(Attribute)`은 크게 `형태(shape)`, `자료형(dtype)`, `장치(device)`로 나눌 수 있습니다.

`형태(shape)`는 **텐서의 차원을 의미합니다.**

`자료형(dtype)`은 **텐서의 데이터 구조를 의미합니다.**

`장치(device)`는 **텐서가 저장되고 연산되는 장치(CPU, GPU 등)를 의미합니다.**

텐서 연산을 진행할 때, 위 세 가지 속성이 서로 맞지 않는다면 오류가 발생할 수 있습니다. 특히 **서로 다른 장치에 있는 텐서끼리는 연산할 수 없습니다.**

- Tip : `torch.manual_seed()`로 난수 생성기의 시드를 고정하면 매번 같은 난수 결과를 얻을 수 있습니다.

<br>

### 텐서 차원 변환

{% highlight Python %}

import torch

torch.manual_seed(0)
tensor = torch.rand(1, 2)
print(tensor)

tensor = tensor.reshape(2, 1)
print(tensor)

{% endhighlight %}
**결과**
:    
tensor([[0.4963, 0.7682]])<br>
tensor([[0.4963],<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[0.7682]])<br>
<br>

텐서의 차원 변환은 `reshape()` 메서드를 활용하여 변환이 가능합니다.

기존 `NumPy`에서 차원의 형태를 변경하는 방법과 동일합니다.

<br>

### 자료형 설정

{% highlight Python %}

import torch

torch.manual_seed(0)
print(torch.rand((3, 3), dtype=torch.float32))
print(torch.tensor([1, 2, 3]).to(torch.float64))

{% endhighlight %}
**결과**
:    
tensor([[0.4963, 0.7682, 0.0885],<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[0.1320, 0.3074, 0.6341],<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[0.4901, 0.8964, 0.4556]])<br>
tensor([1., 2., 3.], dtype=torch.float64)<br>
<br>

텐서 함수나 메서드 등을 통해 텐서를 생성할 때는 `데이터 형식(dtype)`을 선언하여 명확하게 데이터를 표현합니다.

인자로 주어진 **값의 속성(형태, 자료형 등)**을 유지하지만, 위 예시와 같이 모호한 경우에는 명확하게 구현하려는 데이터 형태를 표현하는 것이 좋습니다.

이미 생성된 텐서의 자료형은 `to()` 메서드에 자료형을 전달하여 변환할 수 있습니다.

- Tip : `torch.float32`, `torch.int64`, `torch.bool`이 아닌 텐서는 출력할 때 `dtype`이 함께 표시됩니다.

<br>

### 장치 설정

{% highlight Python %}

import torch

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

cpu = torch.tensor([1, 2, 3], dtype=torch.float32)
tensor = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
rand = torch.rand((1, 1), device=device)

print(device)
print(cpu)
print(tensor)
print(rand.device)

{% endhighlight %}
**결과**
:    
cpu<br>
tensor([1., 2., 3.])<br>
tensor([1., 2., 3.])<br>
cpu<br>
<br>

텐서의 `장치(device)`를 설정하는 방법으로는 `device` 매개변수에 장치 속성을 할당합니다.

`torch.cuda.is_available()` 함수로 `CUDA` 사용 유/무를, `torch.backends.mps.is_available()` 함수로 Apple Silicon의 `MPS` 사용 유/무를 확인할 수 있으므로, 사용 가능한 장치를 변수에 저장해서 장치를 통일시킬 수 있습니다.

위 결과는 **GPU가 없는 환경에서 실행한 결과입니다.** `CUDA`를 사용할 수 있는 환경이라면 `device`가 `cuda`가 되며, `tensor`는 `tensor([1., 2., 3.], device='cuda:0')`, `rand.device`는 `cuda:0`으로 출력됩니다.

`cuda:0`의 `0`은 장치의 번호를 의미하며, GPU가 여러 개인 경우 `cuda:1`과 같이 특정 GPU를 지정할 수 있습니다.

`torch.tensor()`나 `torch.rand()`처럼 텐서를 생성하는 함수는 대부분 `device` 매개변수를 지원하므로, 생성할 때부터 원하는 장치에 텐서를 할당할 수 있습니다.

- Tip : `PyTorch 2.6` 이상에서는 `torch.accelerator.current_accelerator()`로 현재 사용 가능한 가속 장치(`cuda`, `mps`, `xpu` 등)를 확인할 수도 있습니다. 가속 장치가 없다면 `None`을 반환합니다.
- Tip : `torch.device`는 장치 종류로 `cpu`, `cuda`, `mps`, `xpu`, `meta`, `xla`, `hip`, `ipu`, `hpu`, `mtia`, `maia`, `lazy`, `vulkan`, `privateuseone` 등을 받습니다. 단, 이름을 받는다고 해서 해당 장치를 사용할 수 있는 것은 아니며, 일반적으로는 `cpu`, `cuda`, `mps`, `xpu`를 사용합니다.

<br>

### 장치 변환

{% highlight Python %}

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

cpu = torch.tensor([1, 2, 3], dtype=torch.float32)
moved = cpu.to(device)
back = moved.to("cpu")

print(cpu)
print(moved)
print(back)

{% endhighlight %}
**결과**
:    
tensor([1., 2., 3.])<br>
tensor([1., 2., 3.])<br>
tensor([1., 2., 3.])<br>
<br>

`장치(device)`간 상호 변환은 `to()` 메서드를 통해 변환할 수 있습니다.

`to(device)`는 텐서를 지정한 장치로 옮긴 **새로운 텐서를 반환하며**, 이미 해당 장치에 있다면 원래 텐서를 그대로 반환합니다.

위 결과는 **GPU가 없는 환경에서 실행하여** 세 텐서가 모두 `cpu`에 있습니다. `CUDA`를 사용할 수 있는 환경이라면 `moved`는 `tensor([1., 2., 3.], device='cuda:0')`로 출력되고, `back`은 다시 `cpu` 텐서가 됩니다.

`cuda()`와 `cpu()` 메서드로도 변환할 수 있지만, `to(device)`를 사용하면 장치에 따라 코드를 바꾸지 않아도 되므로 `to()` 메서드를 주로 사용합니다.

- Tip : `to()` 메서드는 `tensor.to(device, dtype=torch.float16)`처럼 장치와 자료형을 한 번에 변환할 수도 있습니다.

<br>

### NumPy 변환

{% highlight Python %}

import torch
import numpy as np

ndarray = np.array([1, 2, 3], dtype=np.uint8)

print(torch.tensor(ndarray))
print(torch.Tensor(ndarray))
print(torch.from_numpy(ndarray))

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

`torch.tensor()`와 `torch.from_numpy()`는 `ndarray`의 자료형을 유지하지만, `torch.Tensor()`는 기본 자료형(`torch.float32`)으로 변환합니다.

- Tip : `torch.tensor()`는 데이터를 복사하지만, `torch.from_numpy()`는 `ndarray`와 메모리를 공유합니다. 그러므로 원본 `ndarray`의 값을 수정하면 `from_numpy()`로 만든 텐서의 값도 함께 바뀝니다.

<br>

### Tensor 변환

{% highlight Python %}

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

tensor = torch.tensor([1, 2, 3], dtype=torch.float32, device=device, requires_grad=True)
ndarray = tensor.detach().cpu().numpy()

print(ndarray)
print(type(ndarray))

{% endhighlight %}
**결과**
:    
[1. 2. 3.]<br>
&lt;class 'numpy.ndarray'&gt;<br>
<br>

`텐서(Tensor)`를 `NumPy`로 변환할 때는 `detach()` 메서드를 적용한 후에 변환하는 것이 좋습니다.

`detach()` 메서드는 현재 `그래프(graph)`에서 분리된 **새로운 텐서(Tensor)를 반환합니다.**

텐서는 모든 연산을 추적해 기록하는데, 이 기록을 통해 `역전파(Backpropagation)` 등과 같은 연산이 진행됩니다.

`detach()` 메서드는 이 연산을 분리하여 텐서로 변환합니다.

`requires_grad=True`로 연산을 기록하는 텐서에 바로 `numpy()` 메서드를 호출하면 `RuntimeError`가 발생합니다.

또한, `NumPy`는 `CPU` 메모리의 데이터만 다룰 수 있으므로 `GPU` 텐서는 `cpu()` 메서드로 `CPU`로 옮긴 다음 변환합니다.
