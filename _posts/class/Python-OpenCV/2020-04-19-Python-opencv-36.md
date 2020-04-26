---
layout: post
title: "Python OpenCV 강좌 : 제 36강 - 적응형 이진화"
tagline: "Python OpenCV Adaptive Threshold"
image: /assets/images/opencv_logo.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-OpenCV']
keywords: Python, Python OpenCV, OpenCV Adaptive Threshold, OpenCV ADAPTIVE_THRESH_MEAN_C, OpenCV ADAPTIVE_THRESH_GAUSSIAN_C
ref: Python-OpenCV
category: posts
permalink: /posts/Python-opencv-36/
comments: true
---

## 적응형 이진화(Adaptive Threshold) ##
----------

![1]({{ site.images }}/assets/images/Python/opencv/ch36/1.jpg)
적응형 이진화 알고리즘은 입력 이미지에 따라 `임곗값`이 **스스로 다른 값을 할당할 수 있도록 구성된 이진화 알고리즘**입니다.

이미지에 따라 어떠한 임곗값을 주더라도 **이진화 처리가 어려운 이미지가 존재**합니다.

예를 들어, 조명의 변화나 반사가 심한 경우 이미지 내의 **밝기 분포가 달라 국소적으로 임곗값을 적용해야 하는 경우**가 있습니다.

이러한 경우 적응형 이진화 알고리즘을 적용한다면 우수한 결과를 얻을 수 있습니다. 

<br>
<br>

## Main Code ##
----------

{% highlight Python %}

import cv2

src = cv2.imread("tree.jpg")
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 467, 37)

cv2.imshow("binary", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

{% endhighlight %}

<br>
<br>

## Detailed Code ##
----------

{% highlight Python %}

src = cv2.imread("tree.jpg")
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

{% endhighlight %}

`원본 이미지(src)`를 선언하고, `그레이스케일(gray)`을 적용합니다.

<br>
<br>

{% highlight Python %}

binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 467, 37)

{% endhighlight %}

`적응형 이진화 함수(cv2.adaptiveThreshold)`로 그레이스케일 이미지를 이진화를 적용합니다.

`cv2.adaptiveThreshold(입력 이미지, 최댓값, 적응형 이진화 플래그, 임곗값 형식, 블록 크기, 감산값)`을 의미합니다.

`입력 이미지`는 8비트의 **단일 채널 이미지**를 사용합니다.

`최댓값`과 `임곗값 형식`은 **기존 이진화 함수와 동일한 역할**을 합니다.

`적응형 이진화 플래그`는 블록 크기 내의 연산 방법을 의미합니다.

<br>

### 적응형 이진화 플래그 ###

|   플래그   |               설명               |
|:----------:|:--------------------------------:|
| cv2.ADAPTIVE_THRESH_MEAN_C | blockSize 영역의 모든 픽셀에 평균 가중치를 적용 |
| cv2.ADAPTIVE_THRESH_GAUSSIAN_C | blockSize 영역의 모든 픽셀에 중심점으로부터의 거리에 대한 가우시안 가중치 적용 |

<br>

플래그는 두 종류이지만, 연산 방법은 세 가지가 있습니다.

`평균 가중치`, `가우시안 가중치`, `혼합`이 있습니다.

혼합은 `평균 가중치`와 `가우시안 가중치`를 `OR` 연산해 사용할 수 있습니다.

<br>

* Tip : `cv2.ADAPTIVE_THRESH_MEAN_C | cv2.ADAPTIVE_THRESH_GAUSSIAN_C`의 형태로 사용합니다.

<br>

다음으로, `블록 크기`와 `감산값`은 결괏값을 계산하는 수식에서 활용됩니다.

<br>

$$ T(x, y) = \frac{1}{blockSize^2} \sum_{x_i} \sum_{y_i} I(x+x_i, y+y_i) - C $$

<br>

블록 크기는 $$ blockSize $$를 의미하며, 감산값은 $$C$$를 의미합니다.

수식에서 확인할 수 있듯이 주변 영역의 크기인 `blockSize`와 상수 `C`에 따라 설정되는 임곗값의 결과가 크게 달라집니다.

`blockSize`는 중심점이 존재할 수 있게 **홀수**만 가능하며 상수 `C`는 **일반적으로 양수의 값을 사용**하지만 경우에 따라 0이나 음수도 사용 가능합니다.

<br>

* Tip : blockSize가 클수록 연산 시간이 오래걸리게 됩니다.

<br>
<br>

## Result ##
----------

![2]({{ site.images }}/assets/images/Python/opencv/ch36/2.jpg)