---
layout: post
title: "Python Pytorch 강좌 : 제 1강 - PyTorch 소개 및 설치"
tagline: "Python PyTorch Install"
image: /assets/images/pytorch.webp
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

`Torch`를 기반으로 하여 `Facebook(현 Meta)`의 인공지능 연구팀이 개발했습니다.

2022년에는 프로젝트가 `리눅스 재단(Linux Foundation)` 산하의 `PyTorch 재단(PyTorch Foundation)`으로 이관되어, 현재는 특정 기업이 아닌 재단 중심으로 운영되고 있습니다.

자연어 처리를 비롯해 이미지 프로세싱과 같은 애플리케이션을 위해 사용됩니다. 

Python에 친화적인 라이브러리로 간결하고 구현이 빨리되며, 사용자가 익히기 쉽습니다.

`Define by Run` 방식을 기반으로 하여 연산을 실행하는 시점에 그래프가 구성되므로, 중간 결괏값을 바로 확인하고 디버깅할 수 있습니다.

`자동 미분 모듈`, `최적화 모듈`, `이미지 처리 모듈`, `오디오 처리 모듈` 등 머신 러닝 및 딥 러닝을 위한 다양한 모듈을 제공합니다.

또한, `PyTorch 2.0` 이상에서는 `torch.compile()` 함수로 기존 코드를 거의 그대로 둔 채 모델을 컴파일하여 실행 속도를 높일 수 있습니다.

`PyTorch`는 **클라우드 플랫폼(Amazon Web Services, Google Cloud Platform 등)**에서도 손쉽게 적용할 수 있습니다.

<br>
<br>

## 가상 환경 설정

PyTorch는 프로젝트마다 버전이 달라지는 경우가 많으므로, `가상 환경(Virtual Environment)`을 만들어 설치하는 것이 좋습니다.

**가상 환경을 사용하면 서로 간섭 없이 개별적으로 유지 관리하고 실행할 수 있는 여러 실행 환경을 쉽게 관리할 수 있습니다.**

Python에 기본으로 포함된 `venv` 모듈로 가상 환경을 생성할 수 있습니다.

PyTorch 2.14.1은 **Python 3.10 이상**을 요구합니다.

<br>

{% highlight bash %}

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux, macOS
source .venv/bin/activate

{% endhighlight %}

`python -m venv {경로}`로 가상 환경을 생성한 다음, 운영체제에 맞는 명령어로 가상 환경을 활성화합니다.

`아나콘다(Anaconda)`나 `Miniconda`를 사용한다면 `conda create`로 가상 환경을 만들어도 됩니다.

단, 가상 환경만 conda로 만들고 **PyTorch는 아래의 pip 명령어로 설치합니다.**

- Tip : 아나콘다의 `pytorch` 채널은 2.5.1 버전에서 갱신이 멈췄으므로, `conda install ... -c pytorch` 명령어로는 최신 버전의 PyTorch를 설치할 수 없습니다.

<br>
<br>

## PyTorch CPU 설치

PyTorch는 `pip(Package Installer for Python)`를 통해 설치합니다.

PyTorch는 다양한 모듈을 제공하며, 대표적으로 다음과 같은 라이브러리가 있습니다.

1. `torch` : 자동 미분 시스템에 구축된 심층 신경망 라이브러리

2. `torchvision` : 컴퓨터 비전을 위한 데이터 세트, 모델, 이미지 변환 라이브러리

3. `torchaudio` : 오디오 및 신호 처리를 위한 라이브러리

pip에서 PyTorch 본체의 패키지 이름은 `pytorch`가 아닌 **`torch`**입니다.

`PyTorch CPU` 버전은 다음과 같은 명령어를 통해 설치가 가능합니다.

<br>

{% highlight bash %}

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

{% endhighlight %}

`--index-url`은 패키지를 내려받을 저장소를 지정합니다.

PyTorch는 연산 장치별로 저장소를 나누어 휠(Wheel)을 배포하며, `.../whl/cpu`는 **CPU 전용 휠**을 제공하는 저장소입니다.

- Tip : `pip install pytorch torchvision torchaudio cpuonly`는 잘못된 명령어입니다. pip에서는 `torch`가 올바른 패키지 이름이며, `cpuonly`는 conda 전용 패키지이므로 pip로 설치할 수 없습니다.
- Tip : `torchaudio`는 별도의 버전 체계로 배포되며, 2026년 10월 기준 PyPI의 최신 버전은 2.11.0입니다. 오디오 처리가 필요한 경우에만 추가로 설치합니다.

<br>
<br>

## PyTorch GPU 설치

`PyTorch GPU`는 `NVIDIA GPU`와 `CUDA`를 활용해 연산을 가속합니다.

pip로 배포되는 `CUDA` 버전 휠에는 실행에 필요한 `CUDA` 라이브러리가 함께 제공되므로, **별도로 `CUDA Toolkit`이나 `cuDNN`을 설치하고 환경 변수를 등록할 필요가 없습니다.**

대신 현재 사용하고 있는 `GPU`의 **드라이버를 최신 버전으로 업데이트**해야 합니다.

GPU가 지원하는 `CUDA Compute Capability`는 아래 링크에서 확인할 수 있습니다.

> [GPU 사양 확인하기][Compute]

> [NVIDIA 드라이버 업데이트하기][NVIDIA]

<br>

### CUDA 버전 선택

PyTorch 2.14.1은 `CUDA 12.6`, `CUDA 13.0`, `CUDA 13.2` 버전의 휠을 제공합니다.

설치할 `CUDA` 버전에 맞는 저장소를 `--index-url`로 지정합니다.

| CUDA 버전 | 저장소 주소 |
| :-------: | :--------: |
| CUDA 12.6 | https://download.pytorch.org/whl/cu126 |
| CUDA 13.0 | https://download.pytorch.org/whl/cu130 |
| CUDA 13.2 | https://download.pytorch.org/whl/cu132 |
| CPU | https://download.pytorch.org/whl/cpu |

<br>

{% highlight bash %}

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

{% endhighlight %}

위 명령어는 `CUDA 13.0` 버전의 휠을 설치합니다.

`cu130`을 `cu126`이나 `cu132`로 바꾸면 해당 `CUDA` 버전의 휠이 설치됩니다.

GPU 드라이버가 지원하는 `CUDA` 버전은 커맨드 창에서 `nvidia-smi`를 입력해 출력되는 `CUDA Version` 항목으로 확인할 수 있습니다.

**드라이버가 지원하는 `CUDA` 버전보다 높지 않은 휠을 선택합니다.**

- Tip : 지원 버전과 설치 명령어는 PyTorch 버전마다 달라지므로, [PyTorch 공식 홈페이지][PyTorch]의 설치 선택 화면에서 함께 확인합니다.
- Tip : Intel GPU용 `xpu` 휠도 `https://download.pytorch.org/whl/xpu` 저장소로 제공됩니다.

<br>
<br>

## 결과 확인

{% highlight Python %}

import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.backends.mps.is_available())

{% endhighlight %}

**결과**
:    
2.14.1+cpu<br>
False<br>
False<br>
<br>

현재 `PyTorch`의 **버전**과 `GPU` **사용 가능 여부**가 출력됩니다.

위 결과는 `CPU` 버전을 설치한 Windows 환경의 결과입니다. 버전 뒤의 `+cpu`는 설치한 휠의 종류를 의미하며, `CUDA 13.0` 휠을 설치했다면 `2.14.1+cu130`의 형태로 출력됩니다.

`torch.cuda.is_available()`은 `NVIDIA GPU(CUDA)`를 사용할 수 있는지 반환합니다.

`torch.backends.mps.is_available()`은 Apple Silicon Mac의 `MPS(Metal Performance Shaders)` 가속을 사용할 수 있는지 반환합니다.

만약, `GPU` 버전으로 설치하였는데 `torch.cuda.is_available()`의 값이 **거짓(False) 값**으로 나온다면 먼저 `torch.__version__`이 `+cpu`로 끝나는지 확인합니다.

`+cpu`로 끝난다면 CPU 전용 휠이 설치된 것이므로, `pip uninstall torch torchvision`으로 제거한 다음 `--index-url`을 `CUDA` 저장소로 지정해 다시 설치합니다.

`+cu130`과 같이 `CUDA` 휠이 설치되어 있다면 `nvidia-smi`로 드라이버가 정상적으로 설치되었는지, 드라이버가 지원하는 `CUDA` 버전이 휠의 버전 이상인지 확인합니다.

[Compute]: https://developer.nvidia.com/cuda-gpus
[NVIDIA]: https://www.nvidia.com/Download/index.aspx?lang=kr
[PyTorch]: https://pytorch.org/get-started/locally/
