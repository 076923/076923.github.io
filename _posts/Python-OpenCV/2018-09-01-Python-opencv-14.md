---
layout: post
title: "Python OpenCV 강좌 : 제 14강 - 가장자리 검출"
tagline: "Python OpenCV Edge Image"
image: /assets/images/opencv.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['OpenCV']
keywords: Python, Python OpenCV, OpenCV Edge Image, OpenCV Canny, OpenCV Sobel, OpenCV Laplacian
ref: Python-OpenCV
category: Python
permalink: /posts/Python-opencv-14/
comments: true
toc: true
---

## 가장자리 검출(Edge)

![1]({{ site.images }}/assets/posts/Python/OpenCV/lecture-14/1.webp){: width="100%" height="100%"}

`가장자리(Edge)`는 가장 바깥 부분의 둘레를 의미하며, 객체의 테두리로 볼 수 있습니다.

이미지 상에서 가장자리는 `전경(Foreground)`과 `배경(Background)`이 구분되는 지점이며, 전경과 배경 사이에서 **밝기가 큰 폭으로 변하는 지점**이 객체의 가장자리가 됩니다.

그러므로 가장자리는 픽셀의 밝기가 급격하게 변하는 부분으로 간주할 수 있습니다.

가장자리를 찾기 위해 **미분(Derivative)**과 **기울기(Gradient)** 연산을 수행하며, 이미지 상에서 픽셀의 밝기 변화율이 높은 경계선을 찾습니다. 

<br>
<br>

## 메인 코드

{% highlight Python %}

import cv2

src = cv2.imread("Image/wheat.jpg", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)
laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
canny = cv2.Canny(src, 100, 255)

cv2.imshow("sobel", sobel)
cv2.imshow("laplacian", laplacian)
cv2.imshow("canny", canny)
cv2.waitKey()
cv2.destroyAllWindows()

{% endhighlight %}

<br>

### 세부 코드 (Sobel)

{% highlight Python %}

sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)

{% endhighlight %}

`소벨 함수(cv2.Sobel)`로 입력 이미지에서 가장자리를 검출할 수 있습니다.

미분 값을 구할 때 가장 많이 사용되는 연산자이며, 인접한 픽셀들의 차이로 **기울기(Gradient)의 크기**를 구합니다.

이때 인접한 픽셀들의 기울기를 계산하기 위해 컨벌루션 연산을 수행합니다. 

`dst = cv2.Sobel(src, ddepth, dx, dy, ksize, scale, delta, borderType)`은 `입력 이미지(src)`에 `출력 이미지 정밀도(ddepth)`를 설정하고 `dx(X 방향 미분 차수)`, `dy(Y 방향 미분 차수)`, `커널 크기(ksize)`, `비율(scale)`, `오프셋(delta)`, `테두리 외삽법(borderType)`을 설정하여 `결과 이미지(dst)`를 반환합니다.

`출력 이미지 정밀도`는 반환되는 결과 이미지의 정밀도를 설정합니다. 

`X 방향 미분 차수`는 이미지에서 `X 방향`으로 미분할 차수를 설정합니다.

`Y 방향 미분 차수`는 이미지에서 `Y 방향`으로 미분할 차수를 설정합니다.

`커널 크기`는 소벨 마스크의 크기를 설정합니다. `1`, `3`, `5`, `7` 등의 홀수 값을 사용하며, **최대 31**까지 설정할 수 있습니다.

`비율`과 `오프셋`은 출력 이미지를 반환하기 전에 적용되며, 주로 시각적으로 확인하기 위해 사용합니다.

`픽셀 외삽법`은 이미지 가장자리 부분의 처리 방식을 설정합니다.

- Tip :  `X 방향 미분 차수`와 `Y 방향 미분 차수`는 합이 1 이상이여야 하며, 0의 값은 해당 방향으로 미분하지 않음을 의미합니다.

<br>

### 세부 코드 (Laplacian)

{% highlight Python %}

laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)

{% endhighlight %}

`라플라시안 함수(cv2.Laplacian)`로 입력 이미지에서 가장자리를 검출할 수 있습니다.

라플라시안은 2차 미분의 형태로 가장자리가 밝은 부분에서 발생한 것인지, 어두운 부분에서 발생한 것인지 알 수 있습니다.

2차 미분 방식은 X 축과 Y 축을 따라 2차 미분한 합을 의미합니다.

`dst = cv2.laplacian(src, ddepth, ksize, scale, delta, borderType)`은 `입력 이미지(src)`에 `출력 이미지 정밀도(ddepth)`를 설정하고 `커널 크기(ksize)`, `비율(scale)`, `오프셋(delta)`, `테두리 외삽법(borderType)`을 설정하여 `결과 이미지(dst)`를 반환합니다.

`출력 이미지 정밀도`는 반환되는 결과 이미지의 정밀도를 설정합니다. 

`커널 크기`는 라플라시안 필터의 크기를 설정합니다. `커널`의 값이 1일 경우, **중심값이 -4인 3 x 3 Aperture Size**를 사용합니다.

`비율`과 `오프셋`은 출력 이미지를 반환하기 전에 적용되며, 주로 시각적으로 확인하기 위해 사용합니다.

`픽셀 외삽법`은 이미지 가장자리 부분의 처리 방식을 설정합니다.

<br>

### 세부 코드 (Canny)

{% highlight Python %}

canny = cv2.Canny(src, 100, 255)

{% endhighlight %}

`캐니 함수(cv2.Canny)`로 입력 이미지에서 가장자리를 검출할 수 있습니다.

캐니 엣지는 라플라스 필터 방식을 개선한 방식으로 x와 y에 대해 1차 미분을 계산한 다음, 네 방향으로 미분합니다.

네 방향으로 미분한 결과로 극댓값을 갖는 지점들이 가장자리가 됩니다.

앞서 설명한 가장자리 검출기보다 성능이 월등히 좋으며 노이즈에 민감하지 않아 강한 가장자리를 검출하는 데 목적을 둔 알고리즘입니다.

`dst = cv2.Canny(src, threshold1, threshold2, apertureSize, L2gradient)`는 `입력 이미지(src)`를 `하위 임곗값(threshold1)`, `상위 임곗값(threshold2)`, `소벨 연산자 마스크 크기(apertureSize)`, `L2 그레이디언트(L2gradient)`을 설정하여 `결과 이미지(dst)`를 반환합니다.

`하위 임곗값`과 `상위 임곗값`으로 픽셀이 갖는 최솟값과 최댓값을 설정해 검출을 진행합니다. 

픽셀이 상위 임곗값보다 큰 기울기를 가지면 픽셀을 가장자리로 간주하고, 하위 임곗값보다 낮은 경우 가장자리로 고려하지 않습니다.

`소벨 연산자 마스크 크기`는 소벨 연산을 활용하므로, 소벨 마스크의 크기를 설정합니다.

`L2 그레이디언트`는 L2-norm으로 방향성 그레이디언트를 정확하게 계산할지, 정확성은 떨어지지만 속도가 더 빠른 L1-norm으로 계산할지를 선택합니다.

<br>

`L1그라디언트` : $$ L_{1} = \left \|  \frac{dI}{dx}  \right \| + \left \|  \frac{dI}{dy}  \right \| $$

`L2그라디언트` : $$ \sqrt{(dI/dx)^2 + (dI/dy)^2} $$

<br>
<br>

## 추가 정보

### 픽셀 외삽법 종류

|          속성          |                의미                |
|:----------------------:|:----------------------------------:|
|   cv2.BORDER_CONSTANT  |       iiiiii \| abcdefgh \| iiiiiii      |
|  cv2.BORDER_REPLICATE  |       aaaaaa \| abcdefgh \| hhhhhhh      |
|   cv2.BORDER_REFLECT   |       fedcba \| abcdefgh \| hgfedcb      |
|     cv2.BORDER_WRAP    |       cdefgh \| abcdefgh \| abcdefg      |
| cv2.BORDER_REFLECT_101 |       gfedcb \| abcdefgh \| gfedcba      |
|  cv2.BORDER_REFLECT101 |       gfedcb \| abcdefgh \| gfedcba      |
|   cv2.BORDER_DEFAULT   |       gfedcb \| abcdefgh \| gfedcba      |
| cv2.BORDER_TRANSPARENT |       uvwxyz \| abcdefgh \| ijklmno      |
|   cv2.BORDER_ISOLATED  | 관심 영역 (ROI) 밖은 고려하지 않음 |

<br>
<br>

## 출력 결과

### Sobel

![3]({{ site.images }}/assets/posts/Python/OpenCV/lecture-14/3.webp){: width="100%" height="100%"}

<br>

### Laplacian

![4]({{ site.images }}/assets/posts/Python/OpenCV/lecture-14/4.webp){: width="100%" height="100%"}

<br>

### Canny

![5]({{ site.images }}/assets/posts/Python/OpenCV/lecture-14/2.webp){: width="100%" height="100%"}
