---
bg: "opencv.png"
layout: post
comments: true
title:  "Python OpenCV 강좌 : 제 1강 - OpenCV 설치"
crawlertitle: "Python OpenCV 강좌 : 제 1강 - OpenCV 설치"
summary: "Python OpenCV 3.4"
date: 2018-08-12
categories: posts
tags: ['Python-OpenCV']
author: 윤대희
star: true
---

### OpenCV ###
----------
OpenCV (Open Source Computer Vision)은 `오픈 소스 컴퓨터 비전 라이브러리`입니다. `객체ㆍ얼굴ㆍ행동 인식`, `독순`, `모션 추적` 등의 응용 프로그램에서 사용합니다.

<br>

본 강좌는 `Python-OpenCVSharp 3.4.2`에 맞추어져 있습니다.

<br>

### OpenCV 설치 ###
----------

<br>

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
3.4.2<br>
<br>

정상적으로 설치가 완료되었다면 위와 같이 `3.4.2`가 출력되어야합니다.


### Python 플랫폼 ###
----------

[![1]({{ site.images }}/Python/opencv/ch1/1.png)]({{ site.images }}/Python/opencv/ch1/1.png)

본 `Python-OpenCV` 강좌에서 사용될 이미지의 경로는 위와 같습니다.

`D:\Python\Image` 폴더 안에 **이미지 및 동영상을 저장하여 사용합니다.**

<br>

[![2]({{ site.images }}/Python/opencv/ch1/2.png)]({{ site.images }}/Python/opencv/ch1/2.png)

`IDLE`를 사용할 경우, `상대 경로`를 이용하여 `"Image/파일명"`으로 이미지를 불러올 수 있습니다.

<br>

[![3]({{ site.images }}/Python/opencv/ch1/3.png)]({{ site.images }}/Python/opencv/ch1/3.png)

`Visual Studio`를 사용할 경우, `절대 경로`를 이용하여 `"D:/Python/Image/파일명"`으로 이미지를 불러올 수 있습니다.

<br>

`Anaconda`를 이용하는 경우에도 동일합니다.


[28강]: https://076923.github.io/posts/Python-28/
