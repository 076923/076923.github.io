---
layout: post
title: "Python OpenCV 강좌 : 제 30강 - 이미지 연산 (1)"
tagline: "Python OpenCV Image Calculation (1)"
image: /assets/images/opencv_logo.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-OpenCV']
keywords: Python, Python OpenCV, OpenCV Calculation, OpenCV add, OpenCV subtract, OpenCV multiply, OpenCV divide
ref: Python-OpenCV
category: posts
permalink: /posts/Python-opencv-30/
comments: true
---

## 이미지 연산(Image Calculation) ##
----------

![1]({{ site.images }}/assets/images/Python/opencv/ch30/1.jpg)
이미지 연산은 하나 또는 둘 이상의 이미지에 대해 수학적인 연산을 수행합니다.

Numpy 클래스의 배열 연산과 동일하거나 비슷한 의미와 결과를 갖습니다.

또한, `대수적 표현(+, - 등)`을 통해 Mat 클래스 간의 연산을 수행할 수 있습니다.

<br>
<br>

## Main Code ##
----------

{% highlight Python %}

import numpy as np
import cv2

src = cv2.imread("pencils.jpg")
number1 = np.ones_like(src) * 127
number2 = np.ones_like(src) * 2

add = cv2.add(src, number1)
sub = cv2.subtract(src, number1)
mul = cv2.multiply(src, number2)
div = cv2.divide(src, number2)

src = np.concatenate((src, src, src, src), axis = 1)
number = np.concatenate((number1, number1, number2, number2), axis = 1)
dst = np.concatenate((add, sub, mul, div), axis = 1)

dst = np.concatenate((src, number, dst), axis = 0)

cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

{% endhighlight %}

<br>
<br>

## Detailed Code ##
----------

{% highlight Python %}

src = cv2.imread("pencils.jpg")
number1 = np.ones_like(src) * 127
number2 = np.ones_like(src) * 2

{% endhighlight %}

`원본 이미지(src)`와 `연산 값(number1, number2)`을 선언합니다.

연산 이미지는 `회색 이미지(127, 127, 127)`과 `검은색 이미지(2, 2, 2)`를 사용합니다.

<br>
<br>

{% highlight Python %}

add = cv2.add(src, number1)
sub = cv2.subtract(src, number1)
mul = cv2.multiply(src, number2)
div = cv2.divide(src, number2)

{% endhighlight %}

`cv2.Calc(연산 이미지1, 연산 이미지2)`를 이용하여 **이미지 연산을 진행합니다.**

`덧셈(add)`, `뺄셈(subtract)`, `곱셈(multiply)`, `나눗셈(divide)` 등으로 연산이 가능합니다.

결괏값이 **0보다 작다면, 0으로 반환**되며, 결괏값이 **255보다 크다면, 255로 반환**됩니다.

만약, `대수적 표현(+, - 등)`을 통해 연산을 진행한다면, **오버플로우(Overflow)**나 **언더플로우(Underflow)**가 발생합니다.

즉, `0 - 2`를 진행한다면 `-1`이 아닌, `255`값이 됩니다.

이미지는 `uint8`로, 256개의 공간(0 ~ 255)을 갖고 있습니다.

`..., 253, 254, 255, 0, 1, 2, 3, ...`이므로, **255**값을 반환합니다.

<br>
<br>

{% highlight Python %}

src = np.concatenate((src, src, src, src), axis = 1)
number = np.concatenate((number1, number1, number2, number2), axis = 1)
dst = np.concatenate((add, sub, mul, div), axis = 1)

dst = np.concatenate((src, number, dst), axis = 0)

{% endhighlight %}

`연결 함수(np.concatenate)`로 이미지들을 연결합니다.

결과 이미지는 다음과 같이 구성됩니다.

<br>

| src | src | src | src |
| number1 | number1 | number2 | number2 |
| add | sub | mul | div |

<br>
<br>

## Additional Information ##
----------

{% highlight Python %}

src = cv2.imread("pencils.jpg")
number1 = 127 ## np.ones_like(src) * 127
number2 = 2   ## np.ones_like(src) * 2

{% endhighlight %}

여기서 연산값을 `np.ones_like(src) * N`이 아닌 `N`으로 선언해도 연산이 가능합니다.

단, 이 연산은 `브로드캐스팅(Broadcasting)`이 적용돼, `[src.height, src.width, 1]`이 됩니다.

즉, 단일 채널 이미지가 되며 원본 이미지의 첫 번째 채널에만 연산됩니다.

number1가 N이라면 $$ [1, 0, 182] + N = [1 + N, 0, 182] $$ 가 됩니다.

<br>
<br>

## Result ##
----------

![2]({{ site.images }}/assets/images/Python/opencv/ch30/2.jpg)