---
layout: post
title: "Python OpenCV 강좌 : 제 1강 - OpenCV 설치"
tagline: "Python OpenCV 4.1"
image: /assets/images/opencv_logo.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-OpenCV']
keywords: Python, Python OpenCV, OpenCV install
ref: Python-OpenCV
category: posts
permalink: /posts/Python-opencv-1/
comments: true
---

## OpenCV ##
----------

OpenCV (Open Source Computer Vision)은 `오픈 소스 컴퓨터 비전 라이브러리`입니다.

`객체ㆍ얼굴ㆍ행동 인식`, `독순`, `모션 추적` 등의 응용 프로그램에서 사용합니다.

본 강좌는 `Python-OpenCVSharp 4.1.0.25`에 맞추어져 있습니다.

<br>
<br>

## OpenCV 설치 ##
----------

`OpenCV` 모듈은 `pip`를 통하여 설치할 수 있습니다.

설치 명령어는 `python -m pip install opencv-python` 입니다.

`OpenCV 설치하기` : [28강 바로가기][28강]

<br>

{% highlight Python %}

import cv2
print(cv2.__version__)

{% endhighlight %}

**결과**
:    
4.1.0<br>
<br>

정상적으로 설치가 완료되었다면 위와 같이 `4.1.0`가 출력되어야합니다.

<br>
<br>

## Python 플랫폼 ##
----------

![1]({{ site.images }}/assets/images/Python/opencv/ch1/1.png)

본 `Python-OpenCV` 강좌에서 사용될 이미지의 경로는 위와 같습니다.

`D:\Python\Image` 폴더 안에 **이미지 및 동영상을 저장하여 사용합니다.**

<br>
<br>

![2]({{ site.images }}/assets/images/Python/opencv/ch1/2.png)

`IDLE`를 사용할 경우, `상대 경로`를 이용하여 `"Image/파일명"`으로 이미지를 불러올 수 있습니다.

<br>
<br>

![3]({{ site.images }}/assets/images/Python/opencv/ch1/3.png)

`Visual Studio`를 사용할 경우, `절대 경로`를 이용하여 `"D:/Python/Image/파일명"`으로 이미지를 불러올 수 있습니다.

<br>
<br>

`Anaconda`를 이용하는 경우에도 동일합니다.


[28강]: https://076923.github.io/posts/Python-28/
