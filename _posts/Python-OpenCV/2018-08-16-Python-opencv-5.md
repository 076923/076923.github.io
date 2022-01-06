---
layout: post
title: "Python OpenCV 강좌 : 제 5강 - 대칭"
tagline: "Python OpenCV Flip(Symmetry)"
image: /assets/images/opencv.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['OpenCV']
keywords: Python, Python OpenCV, OpenCV Flip, OpenCV Symmetry
ref: Python-OpenCV
category: Python
permalink: /posts/Python-opencv-5/
comments: true
toc: true
---

## 대칭 (Flip, Symmetry)

<img data-src="{{ site.images }}/assets/posts/Python/OpenCV/lecture-5/1.webp" class="lazyload" width="100%" height="100%"/>

`대칭(Flip)`은 기하학적인 측면에서 **반사(reflection)**의 의미를 갖습니다.

2차원 유클리드 공간에서의 기하학적인 변환의 하나로 $$ R^2 $$(2차원 유클리드 공간) 위의 선형 변환을 진행합니다.

대칭은 변환할 행렬(이미지)에 대해 **2×2 행렬을 왼쪽 곱셈을 진행합니다.** 즉, 'p' 형태의 물체에 Y축 대칭을 적용한다면 'q' 형태를 갖게 됩니다.

그러므로, 원본 행렬(이미지)에 각 축에 대한 대칭을 적용했을 때, 단순히 원본 행렬에서 **축에 따라 재매핑**을 적용하면 대칭된 행렬을 얻을 수 있습니다. 

<br>
<br>

## 메인 코드

{% highlight Python %}

import cv2

src = cv2.imread("Image/glass.jpg", cv2.IMREAD_COLOR)
dst = cv2.flip(src, 0)

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()

{% endhighlight %}

<br>

### 세부 코드

{% highlight Python %}

src = cv2.imread("Image/glass.jpg", cv2.IMREAD_COLOR)

{% endhighlight %}

`이미지 입력 함수(cv2.imread)`를 통해 원본 이미지로 사용할 `src`를 선언하고 **로컬 경로**에서 이미지 파일을 읽어 옵니다.

<br>

{% highlight Python %}

dst = cv2.flip(src, 0)

{% endhighlight %}

`대칭 함수(cv2.flip)`로 이미지를 대칭할 수 있습니다.

`dst = cv2.flip(src, flipCode)`는 `원본 이미지(src)`에 `대칭 축(flipCode)`을 기준으로 대칭한 `출력 이미지(dst)`를 반환합니다.

**대칭 축**은 **상수**를 입력해 대칭할 축을 설정할 수 있습니다.

`flipCode < 0`은 **XY 축 대칭(상하좌우 대칭)**을 적용합니다.

`flipCode = 0`은 **X 축 대칭(상하 대칭)**을 적용합니다. 

`flipCode > 0`은 **Y 축 대칭(좌우 대칭)**을 적용합니다.

<br>

{% highlight Python %}

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()

{% endhighlight %}

`이미지 표시 함수(cv2.imshow)`와 `키 입력 대기 함수(cv2.waitkey)`로 윈도우 창에 이미지를 띄울 수 있습니다.

이미지 표시 함수는 여러 개의 윈도우 창을 띄울 수 있으며, 동일한 이미지도 여러 개의 윈도우 창으로도 띄울 수 있습니다.

단, **윈도우 창의 제목은 중복되지 않게 작성합니다.** 

키 입력 대기 함수로 키가 입력될 때 까지 윈도우 창이 유지되도록 구성합니다.

키 입력 이후, `모든 윈도우 창 제거 함수(cv2.destroyAllWindows)`를 이용하여 모든 윈도우 창을 닫습니다.

<br>
<br>

## 출력 결과

<img data-src="{{ site.images }}/assets/posts/Python/OpenCV/lecture-5/2.webp" class="lazyload" width="100%" height="100%"/>
