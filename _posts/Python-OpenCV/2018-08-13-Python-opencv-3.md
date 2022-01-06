---
layout: post
title: "Python OpenCV 강좌 : 제 3강 - 이미지 출력"
tagline: "Python OpenCV Using Image"
image: /assets/images/opencv.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['OpenCV']
keywords: Python, Python OpenCV, OpenCV Using Image
ref: Python-OpenCV
category: Python
permalink: /posts/Python-opencv-3/
comments: true
toc: true
---

## 이미지 출력

<img data-src="{{ site.images }}/assets/posts/Python/OpenCV/lecture-3/1.webp" class="lazyload" width="100%" height="100%"/>

OpenCV는 **래스터 그래픽스 이미지 파일 포맷**을 쉽게 불러올 수 있는 별도의 함수를 제공합니다.

이 함수는 불러온 압축 해제된 이미지 데이터 구조에 필요한 메모리 할당과 같은 복잡한 작업을 처리하며, **파일 시그니처(File Signature)**를 읽어 적절한 코덱을 결정합니다.

OpenCV에서 이미지를 불러올 때는 확장자를 확인하는 방식이 아닌 파일 시그니처를 읽어 파일의 포맷을 분석합니다.

파일 시그니처는 `파일 매직 넘버(File Magic Number)`라고도 하며, 각 파일 형식마다 몇 개의 바이트가 지정되어 있습니다.

예를 들어, **PNG** 확장자의 경우 **89 50 4E 47 …** 형태로 파일 헤더에 포함되어 있습니다.

이미지 입력 함수는 운영체제의 코덱을 사용해 운영체제 별로 픽셀값이 다를 수 있습니다. 

<br>
<br>

## 메인 코드

{% highlight Python %}

import cv2

image = cv2.imread("Image/lunar.jpg", cv2.IMREAD_ANYCOLOR)
cv2.imshow("Moon", image)
cv2.waitKey()
cv2.destroyAllWindows()

{% endhighlight %}

<br>

### 세부 코드

{% highlight Python %}

image = cv2.imread("Image/lunar.jpg", cv2.IMREAD_ANYCOLOR)

{% endhighlight %}

`이미지 입력 함수(cv2.imread)`를 통해 **로컬 경로**의 이미지 파일을 읽어올 수 있습니다.

`image = cv2.imread(fileName, flags)`는 `파일 경로(fileName)`의 이미지 파일을 `플래그(flags)` 설정에 따라 불러옵니다.

`파일 경로(fileName)`는 **상대 경로** 또는 **절대 경로**를 사용하여 이미지를 불러옵니다.

`flags`은 이미지를 초기에 불러올 때 적용할 **초기 상태**를 의미합니다.

* flags
    - `cv2.IMREAD_UNCHANGED` : 원본 사용
    - `cv2.IMREAD_GRAYSCALE` : 1 채널, 그레이스케일 적용
    - `cv2.IMREAD_COLOR` : 3 채널, BGR 이미지 사용
    - `cv2.IMREAD_ANYDEPTH` : 이미지에 따라 정밀도를 16/32비트 또는 8비트로 사용
    - `cv2.IMREAD_ANYCOLOR` : 가능한 3 채널, 색상 이미지로 사용
    - `cv2.IMREAD_REDUCED_GRAYSCALE_2` : 1 채널, 1/2 크기, 그레이스케일 적용
    - `cv2.IMREAD_REDUCED_GRAYSCALE_4` : 1 채널, 1/4 크기, 그레이스케일 적용
    - `cv2.IMREAD_REDUCED_GRAYSCALE_8` : 1 채널, 1/8 크기, 그레이스케일 적용
    - `cv2.IMREAD_REDUCED_COLOR_2` : 3 채널, 1/2 크기, BGR 이미지 사용
    - `cv2.IMREAD_REDUCED_COLOR_4` : 3 채널, 1/4 크기, BGR 이미지 사용
    - `cv2.IMREAD_REDUCED_COLOR_8` : 3 채널, 1/8 크기, BGR 이미지 사용

<br>

{% highlight Python %}

cv2.imshow("Moon", image)
cv2.waitKey()
cv2.destroyAllWindows()

{% endhighlight %}

`이미지 표시 함수(cv2.imshow)`와 `키 입력 대기 함수(cv2.waitkey)`로 윈도우 창에 이미지를 띄울 수 있습니다.

키 입력 대기 함수를 사용하지 않을 경우, 윈도우 창이 유지되지 않고 프로그램이 종료됩니다.

키 입력 이후, `모든 윈도우 창 제거 함수(cv2.destroyAllWindows)`를 이용하여 모든 윈도우 창을 닫습니다.

<br>
<br>

## 추가 정보

{% highlight Python %}

height, width channel = image.shape
print(height, width , channel)

{% endhighlight %}

**결과**
:    
1920 1280 3<br>
<br>

`height, width , channel = image.shape`를 이용하여 해당 이미지의 `높이`, `너비`, `채널`의 값을 확인할 수 있습니다.

이미지의 속성은 `크기`, `정밀도`, `채널`을 주요한 속성으로 사용합니다.

<br>

* `크기` : 이미지의 **높이**와 **너비**를 의미합니다.

* `정밀도` : 이미지의 처리 결과의 **정밀성**을 의미합니다.

* `채널` : 이미지의 **색상 정보**를 의미합니다. 

- Tip : **유효 비트가 많을 수록 더 정밀해집니다.**

- Tip : 채널이 `3`일 경우, **다색 이미지**입니다. 채널이 `1`일 경우, **단색 이미지**입니다.

<br>
<br>

## 출력 결과

<img data-src="{{ site.images }}/assets/posts/Python/OpenCV/lecture-3/2.webp" class="lazyload" width="100%" height="100%"/>
