---
layout: post
title: "Python OpenCV 강좌 : 제 18강 - 도형 그리기"
tagline: "Python OpenCV Drawing"
image: /assets/images/opencv.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['OpenCV']
keywords: Python, Python OpenCV, OpenCV Drawing
ref: Python-OpenCV
category: Python
permalink: /posts/Python-opencv-18/
comments: true
toc: true
---

## 도형 그리기(Drawing)

<img data-src="{{ site.images }}/assets/posts/Python/OpenCV/lecture-18/1.webp" class="lazyload" width="100%" height="100%"/>

`도형 그리기(Drawing)`는 영상이나 이미지 위에 그래픽을 그려 검출 결과를 시각적으로 표시합니다.

또한, 이미지 위에 검출 결과를 새롭게 그려 결괏값을 변형하거나 보정하기 위해서도 사용합니다

도형 그리기는 **직선**, **사각형**, **원**, **다각형** 등을 그릴 수 있습니다.

도형 그리기는 `선형 타입(Line Types)`, `비트 시프트(Bit Shift)`에 따라 결과가 달라질 수 있습니다.

<br>
<br>

## 선형 타입(Line Types)

선형 타입은 도형을 그릴 때 **어떤 유형의 선으로 그릴지 결정합니다.**

선형 타입으로는 `브레젠험 알고리즘(Bresenham's algorithm)`, `안티 에일리어싱(Anti-Aliasing)` 방식이 있습니다.

선은 점들의 연속으로 이뤄진 형태로 두 점 사이의 직선을 그린다면 시작점과 도착점 사이에 연속한 점을 두게 되어 직선을 그리게 됩니다.

일반적으로 직선의 방정식을 사용한다면 두 점 사이에 있는 모든 좌표를 알 수 있습니다.

하지만 이 방식은 실수 형태로 **소수점이 발생합니다.**

이미지는 래스터 형식의 사각형 격자 구조로 이뤄진 행렬이며, 점의 좌표는 모두 정수의 값으로 이뤄져 있습니다.

`브레젠험 알고리즘`은 실수 연산을 하지 않고 **정수 연산으로만 선을 그릴 수 있도록 개발된 알고리즘입니다.**

브레젠험 알고리즘은 `4 연결 방식`과 `8 연결 방식`이 있습니다.

4 연결 방식의 경우 선분에 픽셀을 할당할 때 다음에 할당될 위치로 **오른쪽, 왼쪽, 위쪽, 아래쪽 영역만 고려합니다.**

8 연결 방식의 경우 **대각선 방향까지 추가돼 총 여덟 개의 위치를 고려합니다.**

`안티 에일리어싱`은 영상 신호의 결함을 없애기 위한 기법으로서 이미지나 객체의 **가장자리 부분에서 발생하는 계단 현상을 없애고 계단을 부드럽게 보이도록 하는 방식입니다.**

안티 에일리어싱 방식은 가우스 필터링을 사용하며, 넓은 선의 경우 항상 끝이 둥글게 그려집니다.

<br>
<br>

## 비트 시프트(Bit Shift)

도형 그리기 함수에서 사용되는 값은 일반적으로 정숫값입니다.

하지만 비트 시프트를 활용하면 **소수점 이하의 값이 포함된 실숫값 좌표로도 도형 그리기 함수를 사용할 수 있습니다.**

비트 시프트는 `서브 픽셀(sub pixel)` 정렬을 지원해서 소수점 이하 자릿수를 표현할 수 있습니다.

소수점은 도형 그리기 함수에서 표현할 수 없으므로 비트 시프트의 값으로 지정합니다. 

<br>
<br>

## 메인 코드

{% highlight Python %}

import cv2
import numpy as np

src = np.zeros((768, 1366, 3), dtype=np.uint8)

src = cv2.line(src, (100, 100), (1200, 100), (0, 0, 255), 3, cv2.LINE_AA)
src = cv2.circle(src, (300, 300), 50, (0, 255, 0), cv2.FILLED, cv2.LINE_4)
src = cv2.rectangle(src, (500, 200), (1000, 400), (255, 0, 0), 5, cv2.LINE_8)
src = cv2.ellipse(src, (1200, 300), (100, 50), 0, 90, 180, (255, 255, 0), 2)

pts1 = np.array([[100, 500], [300, 500], [200, 600]])
pts2 = np.array([[600, 500], [800, 500], [700, 600]])
src = cv2.polylines(src, [pts1], True, (0, 255, 255), 2)
src = cv2.fillPoly(src, [pts2], (255, 0, 255), cv2.LINE_AA)

src = cv2.putText(src, "YUNDAEHEE", (900, 600), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 3)

cv2.imshow("src", src)
cv2.waitKey()
cv2.destroyAllWindows()

{% endhighlight %}

<br>

### 세부 코드

{% highlight Python %}

src = cv2.line(src, (100, 100), (1200, 100), (0, 0, 255), 3, cv2.LINE_AA)

{% endhighlight %}

`직선 그리기 함수(cv2.line)`로 입력 이미지에 직선을 그릴 수 있습니다.

`dst = cv2.line(src, pt1, pt2, color, thickness, lineType, shift)`는 `입력 이미지(src)`에 `시작 좌표(pt1)`부터 `도착 좌표(pt2)`를 지나는 특정 `색상(color)`과 `두께(thickness)`의 직선을 그립니다. 추가로 `선형 타입(lineType)`, `비트 시프트(shift)`를 설정할 수 있습니다.

설정된 입력값으로 그려진 직선이 포함된 `출력 이미지(dst)`을 생성합니다.

<br>

{% highlight Python %}

src = cv2.circle(src, (300, 300), 50, (0, 255, 0), cv2.FILLED, cv2.LINE_4)

{% endhighlight %}

`원 그리기 함수(cv2.circle)`로 입력 이미지에 원을 그릴 수 있습니다.

`dst = cv2.circle(src, center, radius, color, thickness, lineType, shift)`는 `입력 이미지(src)`에 `중심점(center)`으로부터 `반지름(radius)`크기의 특정 `색상(color)`과 `두께(thickness)`의 원을 그립니다. 추가로 `선형 타입(lineType)`, `비트 시프트(shift)`를 설정할 수 있습니다.

만약, 내부가 채워진 원을 그리는 경우, 두께에 `cv2.FILLED`을 사용해 내부를 채울 수 있습니다.

설정된 입력값으로 그려진 원이 포함된 `출력 이미지(dst)`을 생성합니다.

<br>

{% highlight Python %}

src = cv2.rectangle(src, (500, 200), (1000, 400), (255, 0, 0), 5, cv2.LINE_8)

{% endhighlight %}

`사각형 그리기 함수(cv2.rectangle)`로 입력 이미지에 원을 그릴 수 있습니다.

`dst = cv2.rectangle(src, pt1, pt2, color, thickness, lineType, shift)`는 `입력 이미지(src)`에 `좌측 상단 모서리 좌표(pt1)`부터 `우측 하단 모서리 좌표(pt2)`로 구성된 특정 `색상(color)`과 `두께(thickness)`의 사각형을 그립니다. 추가로 `선형 타입(lineType)`, `비트 시프트(shift)`를 설정할 수 있습니다.

설정된 입력값으로 그려진 사각형이 포함된 `출력 이미지(dst)`을 생성합니다.

<br>

{% highlight Python %}

src = cv2.ellipse(src, (1200, 300), (100, 50), 0, 90, 180, (255, 255, 0), 2)

{% endhighlight %}

`호 그리기 함수(cv2.ellipse)`로 입력 이미지에 원을 그릴 수 있습니다.

`dst = cv2.ellipse(src, center, axes, angle, startAngle, endAngle, color, thickness, lineType, shift)`는 `입력 이미지(src)`에 `중심점(center)`으로부터 `장축과 단축(axes)` 크기를 갖는 특정 `색상(color)`과 `두께(thickness)`의 호를 그립니다. 

`각도(angle)`는 장축이 기울어진 각도를 의미하며, `시작 각도(startAngle)`와 `도착 각도(endAngle)`를 설정해 호의 형태를 구성합니다. 추가로 `선형 타입(lineType)`, `비트 시프트(shift)`를 설정할 수 있습니다.

설정된 입력값으로 그려진 호가 포함된 `출력 이미지(dst)`을 생성합니다.

<br>

{% highlight Python %}

pts1 = np.array([[100, 500], [300, 500], [200, 600]])
pts2 = np.array([[600, 500], [800, 500], [700, 600]])

{% endhighlight %}

`poly` 함수를 사용하는 경우, `numpy` 형태로 저장된 `위치 좌표`들이 필요합니다.

`n`개의 점이 저장된 경우, `n 각형`을 그릴 수 있습니다.

<br>

{% highlight Python %}

src = cv2.polylines(src, [pts1], True, (0, 255, 255), 2)

{% endhighlight %}

`내부가 채워지지 않은 다각형 그리기 함수(cv2.polylines)`로 입력 이미지에 다각형을 그릴 수 있습니다.

`dst = cv2.ellipse(src, pts, isClosed, color, thickness, lineType, shift)`는 `입력 이미지(src)`에 `선들의 묶음(pts)` 이뤄진 N개의 내부가 채워지지 않은 다각형을 그립니다.  

`닫힘 여부(isClosed)`를 설정해 처음 좌표와 마지막 좌표의 연결 여부를 설정하며, 설정한 `색상(color)`과 `두께(thickness)`의 다각형이 그려집니다.

추가로 `선형 타입(lineType)`, `비트 시프트(shift)`를 설정할 수 있습니다.

설정된 입력값으로 그려진 다각형이 포함된 `출력 이미지(dst)`을 생성합니다.

<br>

{% highlight Python %}

src = cv2.fillPoly(src, [pts2], (255, 0, 255), cv2.LINE_AA)

{% endhighlight %}

`내부가 채워진 다각형 그리기 함수(cv2.fillPoly)`로 입력 이미지에 다각형을 그릴 수 있습니다.

`dst = cv2.ellipse(src, pts, color, thickness, lineType, shift, offset)`는 `입력 이미지(src)`에 `선들의 묶음(pts)` 이뤄진 N개의 내부가 채워지지 않은 다각형을 그립니다.  

설정한 `색상(color)`과 `두께(thickness)`의 다각형이 그려집니다. 

추가로 `선형 타입(lineType)`, `비트 시프트(shift)`, `오프셋(offset)`을 설정할 수 있습니다.

오프셋은 좌표를 (x, y)만큼 이동시켜 표시할 수 있습니다.

설정된 입력값으로 그려진 다각형이 포함된 `출력 이미지(dst)`을 생성합니다.

<br>

{% highlight Python %}

src = cv2.putText(src, "YUNDAEHEE", (900, 600), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 3)

{% endhighlight %}

`문자 그리기 함수(cv2.putText)`로 입력 이미지에 문자를 그릴 수 있습니다.

`dst = cv2.putText(src, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin)`는 `입력 이미지(src)`에 `문자열(text)`을 텍스트 박스의 `좌측 상단 모서리(org)`를 기준으로 문자가 그려집니다.

설정한 `글꼴(fontFace)`과 `글자 크기(fontScale)`, `색상(color)`과 `두께(thickness)`의 다각형이 그려집니다. 

추가로 `선형 타입(lineType)`, `기준 좌표(bottomLeftOrigin)`를 설정할 수 있습니다.

기준 좌표는 텍스트 박스 좌측 상단 모서리가 아닌 **텍스트 박스 좌측 하단 모서리**를 사용할 경우 기준 좌표에 true를 지정합니다.

설정된 입력값으로 그려진 문자가 포함된 `출력 이미지(dst)`을 생성합니다.

<br>
<br>

## 추가 정보

### 선형 타입 종류

|     속성    |      의미     |
|:-----------:|:-------------:|
|  cv2.FILLED |  내부 채우기  |
|  cv2.LINE_4 | 4점 이웃 연결 |
|  cv2.LINE_8 | 8점 이웃 연결 |
| cv2.LINE_AA |   AntiAlias  |

<br>

### 글꼴 종류

|               속성              |            의미           |  비고  |
|:-------------------------------:|:-------------------------:|:------:|
|     cv2.FONT_HERSHEY_SIMPLEX    | 보통 크기의 산세리프 글꼴 |    -   |
|      cv2.FONT_HERSHEY_PLAIN     | 작은 크기의 산세리프 글꼴 |    -   |
|     cv2.FONT_HERSHEY_DUPLEX     | 보통 크기의 산세리프 글꼴 | 정교함 |
|     cv2.FONT_HERSHEY_COMPLEX    |  보통 크기의 세리프 글꼴  |    -   |
|     cv2.FONT_HERSHEY_TRIPLEX    |  보통 크기의 세리프 글꼴  | 정교함 |
|  cv2.FONT_HERSHEY_COMPLEX_SMALL |  작은 크기의 손글씨 글꼴  |    -   |
| cv2.FONT_HERSHEY_SCRIPT_SIMPLEX |  보통 크기의 손글씨 글꼴  |    -   |
| cv2.FONT_HERSHEY_SCRIPT_COMPLEX |  보통 크기의 손글씨 글꼴  | 정교함 |
|         cv2.FONT_ITALIC         |         기울임 꼴         |    -   |

<br>
<br>

## 출력 결과

<img data-src="{{ site.images }}/assets/posts/Python/OpenCV/lecture-18/1.webp" class="lazyload" width="100%" height="100%"/>
