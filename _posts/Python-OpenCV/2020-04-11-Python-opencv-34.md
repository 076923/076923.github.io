---
layout: post
title: "Python OpenCV 강좌 : 제 34강 - 픽셀 접근"
tagline: "Python OpenCV Pixel Access"
image: /assets/images/opencv.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['OpenCV']
keywords: Python, Python OpenCV, OpenCV Pixel Access
ref: Python-OpenCV
category: Python
permalink: /posts/Python-opencv-34/
comments: true
toc: true
---

## 픽셀 접근(Pixel Access)

![1]({{ site.images }}/assets/posts/Python/OpenCV/lecture-34/1.webp){: width="100%" height="100%"}

`픽셀 접근`은 이미지 배열에서 특정 좌표에 대한 값을 받아오거나, 변경할 때 사용합니다.

`Numpy 배열`의 요소 접근 방식과 동일하며, 직접 값을 변경하거나 할당할 수 있습니다.

OpenCV의 Mat 클래스는 `Numpy 배열`을 사용하므로 **문자열, 리스트, 튜플** 등에 사용되는 `슬라이싱`을 동일하게 사용할 수 있습니다.

<br>
<br>

## 메인 코드

{% highlight Python %}

import cv2
import numpy as np

gray = np.linspace(0, 255, num=90000, endpoint=True, retstep=False, dtype=np.uint8).reshape(300, 300, 1)
color = np.zeros((300, 300, 3), np.uint8)
color[0:150, :, 0] = gray[0:150, :, 0]
color[:, 150:300, 2] = gray[:, 150:300, 0]

x, y, c = 200, 100, 0
access_gray = gray[y, x, c]
access_color_blue = color[y, x, c]
access_color = color[y, x]

print(access_gray)
print(access_color_blue)
print(access_color)

cv2.imshow("gray", gray)
cv2.imshow("color", color)
cv2.waitKey(0)
cv2.destroyAllWindows()

{% endhighlight %}

<br>

### 세부 코드

{% highlight Python %}

gray = np.linspace(0, 255, num=90000, endpoint=True, retstep=False, dtype=np.uint8).reshape(300, 300, 1)

{% endhighlight %}

회색 그라데이션 이미지인 `gray`를 선언합니다.

그라데이션 이미지는 `등간격(numpy.linspace)`을 활용해 구현할 수 있습니다.

등간격 배열을 생성하면, 이미지 배열이 아니므로, `차원 변경(numpy.reshape)` 함수를 활용해 **단일 채널 이미지**로 변경합니다.

- `등간격 사용하기` : [Python Numpy 4강 바로가기][Python Numpy 4강]

- Tip : `300×300×1` 크기이 이미지를 생성할 예정이므로, `num`은 90000을 갖습니다.

<br>

{% highlight Python %}

color = np.zeros((300, 300, 3), np.uint8)
color[0:150, :, 0] = gray[0:150, :, 0]
color[:, 150:300, 2] = gray[:, 150:300, 0]

{% endhighlight %}

색상 그라데이션 이미지인 `color`를 선언합니다.

`Blue 채널`의 **0 ~ 150 행**에 `gray`의 값을 할당하고, `Red 채널`의 **150 ~ 300 열**에 `gray` 값을 할당합니다.

배열의 접근 방식은 `배열[행 시작:행 끝, 열 시작: 열 끝, 차원 시작:차원 끝]`의 구조를 갖습니다.

슬라이싱 방법과 동일하게 접근하며 **행, 열, 차원**의 순서입니다.

- `슬라이싱 사용하기` : [Python Numpy 5강 바로가기][Python Numpy 5강]

<br>

{% highlight Python %}

x, y, c = 200, 100, 0
access_gray = gray[y, x, c]
access_color_blue = color[y, x, c]
access_color = color[y, x]

print(access_gray)
print(access_color_blue)
print(access_color)

{% endhighlight %}

**결과**
:<br>
85<br>
85<br>
[85  0 85]<br>
<br>

요소 접근시 배열 접근은 `행, 열, 차원` 구조이므로, `y, x, 차원`의 형태로 접근해야 합니다.

그러므로, `y, x, c`의 형태로 이미지 요소에 접근합니다.

만약, 차원을 포함하지 않는 경우 `세 차원` 모두 반환하므로, `Numpy 배열` 형식의 값을 반환합니다.

요소 접근에도 `콜론(:)`을 통해 범위 형태로 값을 접근할 수 있습니다.

또한, `콜론(:)`을 한번 더 포함한다면 `간격(step)`으로도 값을 뛰엄뛰엄 받아올 수 있습니다.

<br>
<br>

## 출력 결과

![2]({{ site.images }}/assets/posts/Python/OpenCV/lecture-34/2.webp){: width="100%" height="100%"}

[Python Numpy 4강]: https://076923.github.io/posts/Python-numpy-4/
[Python Numpy 5강]: https://076923.github.io/posts/Python-numpy-5/
