---
layout: post
title: "Python Pytorch 강좌 : 제 1강 - PyTorch 소개 및 설치"
tagline: "Python PyTorch Install"
image: /assets/images/pytorch.webp
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['PyTorch']
keywords: Python, Python Pytorch, Pytorch install, Pytorch CPU Install, Pytorch GPU Install
ref: Python-PyTorch
category: Python
permalink: /posts/Python-pytorch-1/
comments: true
toc: true
---

## PyTorch

<img data-src="{{ site.images }}/assets/posts/Python/PyTorch/lecture-1/1.webp" class="lazyload" width="100%" height="100%"/>

PyTorch는 **오픈소스 머신 러닝 라이브러리입니다.**

`Torch`를 기반으로 하여 `Facebook`의 인공지능 연구팀이 개발했습니다.

자연어 처리를 비롯해 이미지 프로세싱과 같은 애플리케이션을 위해 사용됩니다. 

Python에 친화적인 라이브러리로 간결하고 구현이 빨리되며, 텐서플로우보다 사용자가 익히기 훨씬 쉽습니다.

학습 및 추론 속도가 빠르고 `Define by Run` 방식을 기반으로 하여 실시간 결괏값을 시각화할 수 있습니다.

`자동 미분 모듈`, `최적화 모듈`, `이미지 처리 모듈`, `오디오 처리 모듈` 등 머신 러닝 및 딥 러닝을 위한 다양한 모듈을 제공합니다.

또한, `PyTorch`는 **클라우드 플랫폼(Amazon Web Services, Google Cloud Platform 등)**에서도 손쉽게 적용할 수 있습니다.

<br>
<br>

## 아나콘다(Anaconda) 설치

<img data-src="{{ site.images }}/assets/posts/Python/PyTorch/lecture-1/2.webp" class="lazyload" width="100%" height="100%"/>

`아나콘다(Anaconda)`를 패키지 관리자로 사용하여 운영체제 내에 가상 Python 환경을 설정합니다. 

아나콘다는 전 세계적으로 2,500만 명이 넘는 사용자를 보유한 가장 인기 있는 Python 배포 플랫폼입니다.

클라우드 기반 저장소를 검색하여 7,500개 이상의 **데이터 과학 및 머신 러닝, 딥 러닝 패키지를 쉽게 설치할 수 있습니다.**

**아나콘다를 사용하면 서로 간섭 없이 개별적으로 유지 관리하고 실행할 수 있는 여러 실행 환경을 쉽게 관리할 수 있습니다.**

> [Anaconda Individual Edition 설치하기][Conda]

<br>
<br>

## PyTorch CPU 설치

`아나콘다(Anaconda)`를 통해 PyTorch를 설치할 수 있습니다.

만약, 아나콘다를 사용하지 않는 환경이라면 `PIP(Package Manager)`를 통해서도 설치가 가능합니다.

pytorch는 다양한 모듈을 제공하며, 대표적으로 다음과 같은 라이브러리가 있습니다.

1. `pytorch` : 자동 미분 시스템에 구축된 심층 신경망 라이브러리

2. `torchvision` : 컴퓨터 비전을 위한 이미지 변환 라이브러리

3. `torchaudio` : 오디오 및 신호 처리를 위한 라이브러리

`PyTorch CPU(cpuonly)`는 다음과 같은 명령어를 통해 설치가 가능합니다.

<br>

### Conda Install

```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

<br>

### PIP Install

```
pip install pytorch torchvision torchaudio cpuonly
```

<br>
<br>

## PyTorch GPU 설치

`PyTorch GPU`는 `아나콘다(Anaconda)` 이외에도 `CUDA`를 설치해야 활용이 가능합니다.

또한, `CUDA` 설치 조건에는 특정 조건을 만족하는 `GPU`만 사용이 가능합니다.

현재 PyTorch를 설치하려는 GPU의 사양을 확인합니다.

`CUDA Compute Capability`가 3.5 이상의 GPU를 사용하고 있는지 확인합니다.

만약, **3.0** 이상의 GPU라면, `CUDA 10.2`까지 적용이 가능합니다.

그 미만의 `CUDA Compute Capability`라면 `PyTorch GPU`를 적용할 수 없습니다.

또한, 현재 사용하고 있는 `GPU`의 드라이버를 최신 버전으로 업데이트합니다.

> [GPU 사양 확인하기][Compute]

> [NVIDIA 드라이버 업데이트하기][NVIDIA]

<br>

### CUDA Toolkit

`CUDA Toolkit`은 GPU 가속화 애플리케이션 개발에 필요한 라이브러리를 제공합니다.

`GPU 가속화 라이브러리`, `디버깅 및 최적화 툴`, `컴파일러` 등을 제공합니다.

이 라이브러리를 활용하여 딥 러닝 알고리즘들을 사용하기 쉽게 도와줍니다.

`CUDA Toolkit`을 설치하기 전에, `PyTorch GPU`에서 지원하는 `CUDA Toolkit` 버전을 확인해야 합니다.

> [PyTorch의 CUDA Toolkit 버전 확인하기][PyTorch]

<br>

<img data-src="{{ site.images }}/assets/posts/Python/PyTorch/lecture-1/3.webp" class="lazyload" width="100%" height="100%"/>

현재, `PyTorch GPU`에서 지원하는 `CUDA Toolkit`은 `CUDA 10.2`, `CUDA 11.3`입니다.

GPU의 `CUDA Compute Capability`에 맞는 `CUDA Toolkit`으로 설치합니다.

> [CUDA Toolkit 설치하기][CUDA]

<br>

### NVIDIA cuDNN

`NVIDIA CUDA 심층 신경망 라이브러리(cuDNN)`는 심층 신경망을 위한 **GPU 가속 프리미티브(GPU-accelerated library of primitives)** 라이브러리입니다.

압축 파일을 다운로드하여, `NVIDIA GPU Computing Toolkit`이 설치된 경로로 파일을 덮어 씌웁니다.

`NVIDIA GPU Computing Toolkit/CUDA/{Version}`의 경로입니다.

파일을 모두 덮어 씌웠다면 `환경 변수`에 `경로(Path)`를 등록합니다.

<br>

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\extras\CUPTI\libx64
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\include
```

만약, `C:\Program Files` 경로에 `11.3` 버전으로 설치했다면 위와 같이 경로를 추가합니다.

총 **세 개**의 경로를 환경 변수에 추가합니다.

<br>

```
setx path "%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\bin"
setx path "%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\extras\CUPTI\libx64"
setx path "%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\include"
```

`커맨드 창(cmd)`에서 경로를 등록하는 경우 위와 같이 입력할 수 있습니다.

커맨드 창에 `path`를 입력하여 경로가 정상적으로 등록됬는지 확인합니다.

경로가 정상적으로 등록됐다면, `path` 명령어에서 출력되는 결괏값에서 추가된 경로를 확인할 수 있습니다.

<br>

### Conda Install

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

`cudatoolkit={Version}`의 형태로 `PyTorch GPU`를 설치합니다.

`Version`은 현재 컴퓨터에 설치한 `CUDA Toolkit` 버전을 추가합니다.

<br>
<br>

## 결과 확인

{% highlight Python %}

import torch

print(torch.__version__)
print(torch.cuda.is_available())

{% endhighlight %}

**결과**
:    
1.8.1<br>
True


현재 `PyTorch`의 **버전**과 `GPU` **사용 가능 여부**가 출력됩니다.

만약, `GPU` 버전으로 설치하였는데 `torch.cuda.is_available()`의 값이 **거짓(False) 값**으로 나온다면 커맨드 창에서 `nvcc --version`을 입력합니다.

마지막 줄의 `Cuda compilation tools, release {Version}`을 확인하여 `CUDA`가 정상적으로 설치되었는지 확인합니다.

또한, `cuDNN`의 패치 경로와 환경 변수를 확인합니다.

[Conda]: https://www.anaconda.com/products/individual
[Compute]: https://developer.nvidia.com/cuda-gpus#compute
[NVIDIA]: https://www.nvidia.com/Download/index.aspx?lang=kr
[PyTorch]: https://pytorch.org/
[CUDA]: https://developer.nvidia.com/cuda-toolkit-archive
