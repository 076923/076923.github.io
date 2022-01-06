---
layout: post
title: "Python OpenCV 강좌 : 제 1강 - OpenCV 설치"
tagline: "Python OpenCV 4.1"
image: /assets/images/opencv.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['OpenCV']
keywords: Python, Python OpenCV, OpenCV install
ref: Python-OpenCV
category: Python
permalink: /posts/Python-opencv-1/
comments: true
toc: true
---

## OpenCV

OpenCV는 **Open Source Computer Vision Library**의 약어로 오픈소스 컴퓨터 비전 라이브러리입니다.

실시간 영상 처리에 중점을 둔 영상 처리 라이브러리로서, Apache 2.0 라이선스하에 배포되어 학술적 용도 외에도 상업적 용도로도 사용할 수 있습니다.

OpenCV는 계산 효율성과 실시간 처리에 중점을 두고 설계되었습니다.

500가지가 넘는 알고리즘이 최적화돼 있으며 이 알고리즘을 구성하거나 지원하는 함수는 알고리즘 수의 10배가 넘습니다.

물체 인식, 얼굴 인식, 제스처 인식을 비롯해 자율주행 자동차, OCR 판독기, 불량 검사기 등에 활용할 수 있습니다. 

본 강좌는 `Python-OpenCV 4.5.1.48`에 맞추어져 있습니다.

<br>
<br>

## OpenCV 설치

Python OpenCV는 다음과 같은 네 종류의 패키지를 제공합니다.

```
opencv-python
opencv-contrib-python
opencv-python-headless
opencv-contrib-python-headless
```

`contrib`가 포함된 패키지는 **확장 모듈**이 포함된 패키지이며, 추가 모듈이 포함된 OpenCV를 설치합니다

`headless`가 포함된 패키지는 GUI 라이브러리 종속성이 없어 **서버 환경(Docker, Cloud)**에서 사용할 수 있는 OpenCV를 설치합니다.

특별한 경우가 아니라면, 일반적으로 `opencv-python` 패키지를 사용합니다.

`OpenCV`는 `pip`를 통하여 설치할 수 있습니다.

명령 프롬프트나 터미널에서 `python -m pip install opencv-python` 명령어로 설치할 수 있습니다.

- `PIP로 패키지 설치하기` : [28강 바로가기][28강]

<br>

{% highlight Python %}

import cv2
print(cv2.__version__)

{% endhighlight %}

**결과**
:    
4.5.1<br>
<br>

정상적으로 설치가 완료되었다면 설치한 OpenCV의 버전인 `4.5.1`이 출력됩니다.

<br>
<br>

## Python 플랫폼

<img data-src="{{ site.images }}/assets/posts/Python/OpenCV/lecture-1/1.webp" class="lazyload" width="100%" height="100%"/>

본 `Python-OpenCV` 강좌에서 사용될 이미지의 경로는 위와 같습니다.

`D:\Python\Image` 폴더 안에 **이미지 및 동영상을 저장하여 사용합니다.**

<br>

<img data-src="{{ site.images }}/assets/posts/Python/OpenCV/lecture-1/2.webp" class="lazyload" width="100%" height="100%"/>

`IDLE`를 사용할 경우, `상대 경로`를 이용하여 `"Image/파일명"`으로 이미지를 불러올 수 있습니다.

<br>

<img data-src="{{ site.images }}/assets/posts/Python/OpenCV/lecture-1/3.webp" class="lazyload" width="100%" height="100%"/>

`Visual Studio`를 사용할 경우, `절대 경로`를 이용하여 `"D:/Python/Image/파일명"`으로 이미지를 불러올 수 있습니다.

`Anaconda`를 이용하는 경우에도 동일합니다.

[28강]: https://076923.github.io/posts/Python-28/
