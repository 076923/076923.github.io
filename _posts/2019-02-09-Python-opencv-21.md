---
bg: "opencv.png"
layout: post
comments: true
title:  "Python OpenCV 강좌 : 제 21강 - 윤곽선 검출"
crawlertitle: "Python OpenCV 강좌 : 제 21강 - 윤곽선 검출"
summary: "Python OpenCV Contour"
date: 2019-02-09
categories: posts
tags: ['Python-OpenCV']
author: 윤대희
star: true
---

### 윤곽선 (Contour) ###
----------
[![1]({{ site.images }}/Python/opencv/ch21/1.png)]({{ site.images }}/Python/opencv/ch21/1.png)
영상이나 이미지의 `윤곽선(컨투어)을 검출`하기 위해 사용합니다. 영상이나 이미지에서 `외곽과 내곽`의 **윤곽선(컨투어)을 검출** 할 수 있습니다.

<br>
<br>

### Main Code ###
----------

{% highlight Python %}

import cv2

src = cv2.imread("Image/contours.png", cv2.IMREAD_COLOR)

gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
binary = cv2.bitwise_not(binary)

img, contours, hierachy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

for i in range(len(contours)):
    cv2.drawContours(src, [contours[i]], 0, (0, 0, 255), 2)
    cv2.putText(src, str(i), tuple(contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
    print(i, hierachy[0][i])
    cv2.imshow("src", src)
    cv2.waitKey(0)

cv2.destroyAllWindows()

{% endhighlight %}

<br>
<br>

### Detailed Code ###
----------

{% highlight Python %}

gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
binary = cv2.bitwise_not(binary)

{% endhighlight %}

윤곽선(컨투어)를 검출하는 주된 요소는 `하얀색`의 객체를 검출합니다. 그러므로 배경은 `검은색`이며 검출하려는 물체는 `하얀색`의 성질을 띄게끔 변형합니다. `이진화` 처리 후, `반전`시켜 검출하려는 물체를 `하얀색`의 성질을 띄도록 변환합니다.

<br>
<br>

{% highlight Python %}

img, contours, hierachy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

{% endhighlight %}

`cv2.findContours()`를 이용하여 `이진화` 이미지에서 윤곽선(컨투어)를 검색합니다. `cv2.findContours(이진화 이미지, 검색 방법, 근사화 방법)`을 의미합니다.

반환 값으로 **검출 이미지**, **윤곽선**, **계층 구조**를 반환합니다.

`검출 이미지`는 검출을 시도한 `binary`와 동일한 이미지가 담겨있습니다.

`윤곽선`은 Numpy 구조의 배열로 검출된 `윤곽선의 지점`들이 담겨있습니다.

`계층 구조`는 윤곽선의 계층 구조를 의미합니다. 각 윤곽선에 해당하는 `속성 정보`들이 담겨있습니다.

<br>
<br>

{% highlight Python %}

for i in range(len(contours)):
    cv2.drawContours(src, [contours[i]], 0, (0, 0, 255), 2)
    cv2.putText(src, str(i), tuple(contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
    print(i, hierachy[0][i])
    cv2.imshow("src", src)
    cv2.waitKey(0)

{% endhighlight %}

**반복문**을 사용하여 검출된 윤곽선을 그리며 해당 윤곽선의 **계층 구조**를 표시합니다.

`cv2.drawContours()`을 이용하여 검출된 윤곽선을 그립니다. `cv2.drawContours(이미지, [윤곽선], 윤곽선 인덱스, (B, G, R), 두께, 선형 타입)`을 의미합니다.

`윤곽선`은 검출된 윤곽선들이 저장된 Numpy 배열입니다.

`윤곽선 인덱스`는 검출된 윤곽선 배열에서 몇 번째 인덱스의 윤곽선을 그릴지를 의미합니다.

<br>

* Tip : 윤곽선 인덱스를 0으로 사용할 경우 `0 번째` 인덱스의 윤곽선을 그리게 됩니다. 하지만, 윤곽선 인수를 대괄호로 다시 묶을 경우, 0 번째 인덱스가 최댓값인 배열로 변경됩니다.

* Tip : 동일한 방식으로 `[윤곽선], 0`과 `윤곽선, -1`은 동일한 의미를 갖습니다. (-1은 윤곽선 배열 모두를 의미)

<br>
<br>

### Additional Information ###
----------

#### 검색 방법 ####

`cv2.RETR_EXTERNAL` : 외곽 윤곽선만 검출하며, 계층 구조를 구성하지 않습니다.
`cv2.RETR_LIST` : 모든 윤곽선을 검출하며, 계층 구조를 구성하지 않습니다.
`cv2.RETR_CCOMP` : 모든 윤곽선을 검출하며, 계층 구조는 2단계로 구성합니다.
`cv2.RETR_TREE` : 모든 윤곽선을 검출하며, 계층 구조를 모두 형성합니다. (Tree 구조)

<br>

#### 근사화 방법 ####

`cv2.CHAIN_APPROX_NONE` : 윤곽점들의 모든 점을 반환합니다.
`cv2.CHAIN_APPROX_SIMPLE` :  윤곽점들 단순화 수평, 수직 및 대각선 요소를 압축하고 끝점만 남겨 둡니다.
`cv2.CHAIN_APPROX_TC89_L1` : 프리먼 체인 코드에서의 윤곽선으로 적용합니다.
`cv2.CHAIN_APPROX_TC89_KCOS` : 프리먼 체인 코드에서의 윤곽선으로 적용합니다.

<br>

#### 계층 구조 ####

계층 구조는 윤곽선을 `포함 관계`의 여부를 나타냅니다. 즉 **외곽 윤곽선**, **내곽 윤곽선**, **같은 계층 구조**를 구별할 수 있습니다. 이 정보는 `hierachy`에 담겨있습니다.

`hierachy`를 출력할 경우 다음과 같은 결과를 반환합니다.

{% highlight Python %}

[[[ 2 -1  1 -1]
  [-1 -1 -1  0]
  [ 4  0  3 -1]
  [-1 -1 -1  2]
  [ 6  2  5 -1]
  [-1 -1 -1  4]
  [ 8  4  7 -1]
  [-1 -1 -1  6]
  [ 9  6 -1 -1]
  [10  8 -1 -1]
  [11  9 -1 -1]
  [-1 10 -1 -1]]]

{% endhighlight %}

첫 번째 계층 구조는 `[ 2 -1  1 -1]`의 값을 갖습니다. `[다음 윤곽선, 이전 윤곽선, 내곽 윤곽선, 외곽 윤곽선]`에 대한 인덱스 정보를 포함하고 있습니다.

인덱스 0의 윤곽선의 `다음 윤곽선`은 **인덱스 2의 윤곽선**을 의미하며 `이전 윤곽선`은 **존재하지 않다**는 것을 의미합니다.

**내곽 윤곽선**은 인덱스 1에 해당하는 윤곽선을 **자식 윤곽선**으로 두고 있다는 의미입니다. 즉, 인덱스 0 윤곽선 내부에 `인덱스 1의 윤곽선이 포함`되어 있습니다.

**외곽 윤곽선**은 -1의 값을 갖고 있으므로 `외곽 윤곽선`은 **존재하지 않습니다.**

다음 검출 결과를 통하여 쉽게 이해할 수 있습니다.

<br>

[![2]({{ site.images }}/Python/opencv/ch21/2.png)]({{ site.images }}/Python/opencv/ch21/2.png)

<br>

{% highlight Python %}

0 [ 2 -1  1 -1]
1 [-1 -1 -1  0]
2 [ 4  0  3 -1]

{% endhighlight %}

`print(i, hierachy[0][i])`을 통하여 3개의 윤곽선을 출력한 결과입니다.

다음 윤곽선과 이전 윤곽선의 정보가 `-1`의 값이 아니라면 서로 동등한 계층의 윤곽선입니다.

**0번 인덱스의 윤곽선**과 `동등한 계층`에 있는 윤곽선은 **2번 인덱스의 윤곽선**입니다.

**0번 인덱스의 윤곽선**은 **1번 인덱스의 윤곽선**을 `내부 윤곽선`으로 갖고 있습니다.

**1번 인덱스의 윤곽선**은 `동등한 계층`에 있는 윤곽선이 없으므로 `다음 윤곽선`과 `이전 윤곽선`의 값이 `-1`입니다.

**1번 인덱스의 윤곽선**은 내곽 윤곽선이 없으며, 외곽 윤곽선만 존재하여 **0번 인덱스의 윤곽선**을 외곽 윤곽선으로 반환합니다.

<br>

* Tip : 해당 예제는 `cv2.RETR_CCOMP`로 `2 단계 계층 구조`로만 표시합니다.

* Tip : `계층 구조`를 사용하여 **내곽, 외곽 윤곽선을 구분**할 수 있습니다.

<br>
<br>

### Result ###
----------

[![3]({{ site.images }}/Python/opencv/ch21/3.png)]({{ site.images }}/Python/opencv/ch21/3.png)

{% highlight Python %}

0 [ 2 -1  1 -1]
1 [-1 -1 -1  0]
2 [ 4  0  3 -1]
3 [-1 -1 -1  2]
4 [ 6  2  5 -1]
5 [-1 -1 -1  4]
6 [ 8  4  7 -1]
7 [-1 -1 -1  6]
8 [ 9  6 -1 -1]
9 [10  8 -1 -1]
10 [11  9 -1 -1]
11 [-1 10 -1 -1]

{% endhighlight %}
