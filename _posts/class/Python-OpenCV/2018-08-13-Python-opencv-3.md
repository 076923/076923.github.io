---
layout: post
title: "Python OpenCV 강좌 : 제 3강 - Image 출력"
tagline: "Python OpenCV Using Image"
image: /assets/images/opencv_logo.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-OpenCV']
keywords: Python, Python OpenCV, OpenCV Using Image
ref: Python-OpenCV
category: posts
permalink: /posts/Python-opencv-3/
comments: true
---

## Image 출력 ##
----------

![1]({{ site.images }}/assets/images/Python/opencv/ch3/1.jpg)
`컴퓨터에 저장된 이미지`를 얻어와 Python에서 출력할 수 있습니다.

<br>
<br>

## Main Code ##
----------

{% highlight Python %}

import cv2

image = cv2.imread("Image/lunar.jpg", cv2.IMREAD_ANYCOLOR)
cv2.imshow("Moon", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

{% endhighlight %}

<br>
<br>

## Detailed Code ##
----------

{% highlight Python %}

image = cv2.imread("Image/lunar.jpg", cv2.IMREAD_ANYCOLOR)

{% endhighlight %}

`image = cv2.imread("경로", mode)`을 이용하여 이미지를 불러와 변수에 저장할 수 있습니다.

`경로`는 **상대 경로** 또는 **절대 경로**를 사용하여 이미지를 불러올 수 있습니다.

`mode`은 이미지를 초기에 불러올 때 적용할 **초기 상태**를 의미합니다.

* mode
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
cv2.waitKey(0)
cv2.destroyAllWindows()

{% endhighlight %}

`cv2.imshow("윈도우 창 제목", 이미지)`를 이용하여 **윈도우 창**에 **이미지**를 띄웁니다.

`cv2.waitkey(time)`이며 `time`마다 키 입력상태를 받아옵니다. `0`일 경우, 지속적으로 검사하여 **해당 구문을 넘어가지 않습니다.**

`cv2.destroyAllWindows()`를 이용하여 **모든 윈도우창을 닫습니다.**

<br>
<br>

## Additional Information ##
----------

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

* Tip : **유효 비트가 많을 수록 더 정밀해집니다.**
* Tip : 채널이 `3`일 경우, **다색 이미지**입니다. 채널이 `1`일 경우, **단색 이미지**입니다.

<br>
<br>

## Result ##
----------

![2]({{ site.images }}/assets/images/Python/opencv/ch3/2.png)