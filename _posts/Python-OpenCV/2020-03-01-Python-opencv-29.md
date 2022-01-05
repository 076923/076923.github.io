---
layout: post
title: "Python OpenCV 강좌 : 제 29강 - 원 검출"
tagline: "Python OpenCV Circle Detection"
image: /assets/images/opencv.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['OpenCV']
keywords: Python, Python OpenCV, OpenCV Hough Transform, OpenCV HoughCircles, OpenCV Two stage Hough Transform, OpenCV HoughCircles, OpenCV HOUGH_GRADIENT
ref: Python-OpenCV
category: Python
permalink: /posts/Python-opencv-29/
comments: true
toc: true
---

## 원 검출(Circle Detection)

![1]({{ site.images }}/assets/posts/Python/OpenCV/lecture-29/1.webp){:class="lazyload" width="100%" height="100%"}

원 검출 알고리즘도 허프 변환 알고리즘 중 하나인 `허프 원 변환(Hough Circle Transform) 알고리즘`을 활용해 원을 검출합니다.

허프 원 변환 알고리즘은 앞서 배운 허프 선 변환 알고리즘과 비슷한 방식으로 동작합니다.

허프 원 변환 알고리즘은 2차원이 아닌 **3차원 누산 평면**으로 검출합니다.

각 차원은 `원의 중심점 x`, `원의 중심점 y`, `원의 반경 r`을 활용해 누산 평면을 구성합니다.

누산 평면은 **2차원 공간(x, y**)에서 **3차원 공간(a, b, r)**으로 변환됩니다.

허프 원 변환의 동작 방식은 이미지에서 가장자리를 검출합니다.

<br>

![2]({{ site.images }}/assets/posts/Python/OpenCV/lecture-29/2.webp){:class="lazyload" width="100%" height="100%"}

3차원 히스토그램에서 돗수가 높은 **(a, b, r)**을 선택합니다. 하지만, 이 방법은 이미지에서 가장 긴 변의 길이가 N이라면 $$ N^3 $$바이트의 메모리를 필요로 합니다.

이 방식은 필요한 메모리가 너무 많아 **비효율적**이므로, 메모리 문제와 느린 처리 속도를 해결하기 위해 `2차원 방식`을 사용합니다.

이러한 문제로 인해 2단계로 나눠 계산합니다.

먼저 가장자리에 그레이디언트 방법을 이용해 **원의 중심점(a, b)에 대한 2차원 히스토그램을 선정**합니다.

모든 점에 대해 최소 거리에서 최대 거리까지 기울기의 선분을 따라 **누산 평면의 모든 점을 증가**시킵니다.

또한 중심점을 선택하기 위해 중심점 후보군에서 임곗값보다 크고 인접한 점보다 큰 점을 중심점으로 사용합니다.

선정된 **중심점(a, b)와 가장자리의 좌표**를 원의 방정식에 대입해 `반지름 r의 1차원 히스토그램으로 판단`하게 됩니다.

히스토그램에 필요한 메모리가 줄어들어 이미지에서 가장 긴 변의 길이가 N이라면 $$ N^2 + N $$바이트의 메모리를 필요로 합니다.

OpenCV 원 검출 함수는 `2단계 허프 변환(Two stage Hough Transform)` 방법을 활용해 원을 검출합니다.

<br>
<br>

## 메인 코드

{% highlight Python %}

import cv2

src = cv2.imread("colorball.jpg")
dst = src.copy()
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1 = 250, param2 = 10, minRadius = 80, maxRadius = 120)

for i in circles[0]:
    cv2.circle(dst, (i[0], i[1]), i[2], (255, 255, 255), 5)

cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

{% endhighlight %}

<br>

### 세부 코드

{% highlight Python %}

src = cv2.imread("colorball.jpg")
dst = src.copy()
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

{% endhighlight %}

이미지에서 직선을 검출하기 위해서, 전처리 작업을 진행합니다.

`원본 이미지(src)`와 `결과 이미지(dst)`를 선언합니다.

전처리를 진행하기 위해 `그레이스케일 이미지(gray)`를 사용합니다.

<br>

{% highlight Python %}

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1 = 250, param2 = 10, minRadius = 80, maxRadius = 120)

{% endhighlight %}

`cv2.HoughCircles(검출 이미지, 검출 방법, 해상도 비율, 최소 거리, 캐니 엣지 임곗값, 중심 임곗값, 최소 반지름, 최대 반지름)`를 이용하여 **원 검출을 진행합니다.**

`검출 방법`은 항상 **2단계 허프 변환 방법(21HT, 그레이디언트)**만 사용합니다.

`해상도 비율`은 원의 중심을 검출하는 데 사용되는 **누산 평면의 해상도**를 의미합니다.

인수를 **1**로 지정할 경우 입력한 이미지와 동일한 해상도를 가집니다. 즉, 입력 이미지 너비와 높이가 동일한 누산 평면이 생성됩니다.

또한 인수를 **2**로 지정하면 누산 평면의 해상도가 절반으로 줄어 입력 이미지의 크기와 반비례합니다.

`최소 거리`는 일차적으로 **검출된 원과 원 사이의 최소 거리**입니다. 이 값은 원이 여러 개 검출되는 것을 줄이는 역할을 합니다.

`캐니 엣지 임곗값`은 허프 변환에서 자체적으로 캐니 엣지를 적용하게 되는데, 이때 사용되는 **상위 임곗값**을 의미합니다.

**하위 임곗값**은 자동으로 할당되며, **상위 임곗값의 절반에 해당하는 값**을 사용합니다. 

`중심 임곗값`은 그레이디언트 방법에 적용된 중심 `히스토그램(누산 평면)`에 대한 임곗값입니다. **이 값이 낮을 경우 더 많은 원이 검출됩니다.**

`최소 반지름`과 `최대 반지름`은 검출될 원의 반지름 범위입니다. `0`을 입력할 경우 검출할 수 있는 반지름에 제한 조건을 두지 않습니다.

최소 반지름과 최대 반지름에 각각 0을 입력할 경우 반지름을 고려하지 않고 검출하며, 최대 반지름에 `음수`를 입력할 경우 검출된 원의 중심만 반환합니다.

<br>

{% highlight Python %}

for i in circles[0]:
    cv2.circle(dst, (i[0], i[1]), i[2], (255, 255, 255), 5)

{% endhighlight %}

검출을 통해 반환되는 `circles` 변수는 (1, N, 3)차원 형태를 갖습니다.

내부 차원의 요소로는 검출된 `중심점(x, y)`과 `반지름(r)`이 저장돼 있습니다.

반복문을 활용해 `circles` 배열에서 **중심점**과 **반지름**을 반환할 수 있습니다.

검출된 정보는 소수점을 포함합니다. 원 그리기 함수는 소수점이 포함되어도 사용할 수 있으므로, 형변환을 진행하지 않습니다.

원 그리기 함수를 활용해 `(x, y, r)`의 원을 표시합니다.

<br>
<br>

## 출력 결과

![3]({{ site.images }}/assets/posts/Python/OpenCV/lecture-29/3.webp){:class="lazyload" width="100%" height="100%"}
