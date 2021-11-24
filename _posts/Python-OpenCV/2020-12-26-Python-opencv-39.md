---
layout: post
title: "Python OpenCV 강좌 : 제 39강 - 마우스 콜백"
tagline: "Python OpenCV Mouse Callback"
image: /assets/images/opencv.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['OpenCV']
keywords: Python, Python OpenCV, OpenCV Mouse Callback, OpenCV setMouseCallback, OpenCV Mouse Event
ref: Python-OpenCV
category: Python
permalink: /posts/Python-opencv-39/
comments: true
toc: true
---

## 마우스 콜백(Mouse Callback)

![1]({{ site.images }}/assets/posts/Python/OpenCV/lecture-39/1.jpg)

`콜백(Callback)` 함수는 매개 변수를 통해 다른 함수를 전달 받고, 이벤트가 발생할 때 **매개 변수에 전달된 함수를 호출**하는 역할을 합니다.

즉, 특정한 이벤트가 발생하면 다른 함수를 실행하는 함수입니다.

**마우스 콜백**은 윈도우에 마우스 이벤트가 발생했을 때, 특정한 함수에 이벤트를 전달해 실행합니다.

<br>
<br>

## 메인 코드

{% highlight Python %}

import cv2
import numpy as np

def mouse_event(event, x, y, flags, param):
    global radius
    
    if event == cv2.EVENT_FLAG_LBUTTON:    
        cv2.circle(param, (x, y), radius, (255, 0, 0), 2)
        cv2.imshow("draw", src)

    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            radius += 1
        elif radius > 1:
            radius -= 1

radius = 3
src = np.full((500, 500, 3), 255, dtype=np.uint8)

cv2.imshow("draw", src)
cv2.setMouseCallback("draw", mouse_event, src)
cv2.waitKey()

{% endhighlight %}

<br>

### 세부 코드

{% highlight Python %}

radius = 3
src = np.full((500, 500, 3), 255, dtype=np.uint8)

{% endhighlight %}

`반지름(radius)`를 저장할 변수와 `원본 이미지(src)`를 선언합니다.

**radius**는 마우스 콜백 함수에서 마우스 스크롤에 따라, 값을 증가하거나 감소할 수 있습니다.

<br>

{% highlight Python %}

cv2.imshow("draw", src)
cv2.setMouseCallback("draw", mouse_event, src)
cv2.waitKey()

{% endhighlight %}

먼저, `cv2.namedWindow()` 함수나 `cv2.imshow()` 함수를 활용하여 윈도우를 생성합니다.

윈도우가 생성되었다면, `마우스 콜백 설정 함수(cv2.setMouseCallback)`로 마우스 콜백을 설정합니다.

`cv2.setMouseCallback(윈도우, 콜백 함수, 사용자 정의 데이터)`을 의미합니다.

`윈도우`는 미리 생성되어 있는 윈도우의 이름을 의미합니다.

`콜백 함수`는 마우스 이벤트가 발생했을 때, 전달할 함수를 의미합니다.

`사용자 정의 데이터`는 마우스 이벤트로 전달할 때, 함께 전달할 사용자 정의 데이터를 의미합니다.

<br>

{% highlight Python %}

def mouse_event(event, x, y, flags, param):
    global radius
    
    if event == cv2.EVENT_FLAG_LBUTTON:    
        cv2.circle(param, (x, y), radius, (255, 0, 0), 2)
        cv2.imshow("draw", src)

    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            radius += 1
        elif radius > 1:
            radius -= 1

{% endhighlight %}

콜백 함수에서 사용되는 매개변수는 `event`, `x`, `y`, `flags`, `param` 입니다.

`event`는 윈도우에서 발생하는 이벤트를 의미합니다.

`x`, `y`는 마우스의 좌표를 의미합니다.

`flags`는 `event`와 함께 활용되는 역할로 특수한 상태를 확인하는 용도입니다.

`param`은 마우스 콜백 설정 함수에서 함께 전달되는 사용자 정의 데이터를 의미합니다.

<br>

콜백 함수를 선언하고, `radius`를 전역변수로 선언합니다.

분기문(if)을 통해 `event`가 **왼쪽 마우스 클릭**이 발생했을 때 윈도우에 파란색 원을 그립니다.

만약, `event`가 **마우스 스크롤**을 조작했다면, 다시 하위 분기문(if)을 생성하여 나눕니다.

`event`가 마우스 스크롤 이벤트일 때, `flag`는 마우스 스크롤의 방향을 나타냅니다.

`flag`가 양수라면 **스크롤 업**이며, 음수라면 **스크롤 다운**입니다.

마우스 스크롤 업 이벤트일 때는 **반지름(radius)**를 증가시키고, 낮을 때에는 반지름을 감소시킵니다.

단, 반지름이 1보다 작지 않게 설정하기 위해, `radius > 1` 조건으로 검사합니다. 

<br>
<br>

## Event

| 이름 | 의미 |
|:----------:|:------------------------------------------:|
| EVENT_MOUSEMOVE | 마우스 포인터가 윈도우 위에서 움직일 때 |
| EVENT_LBUTTONDOWN | 마우스 왼쪽 버튼을 누를 때 |
| EVENT_MBUTTONDOWN | 마우스 가운데 버튼을 누를 때 |
| EVENT_RBUTTONDOWN | 마우스 오른쪽 버튼을 누를 때 |
| EVENT_LBUTTONUP | 마우스 왼쪽 버튼을 뗄 때 |
| EVENT_MBUTTONUP | 마우스 가운데 버튼을 뗄 때 |
| EVENT_RBUTTONUP | 마우스 오른쪽 버튼을 뗄 때 |
| EVENT_LBUTTONDBLCLK | 마우스 왼쪽 버튼을 더블 클릭할 때 |
| EVENT_MBUTTONDBLCLK | 마우스 가운데 버튼을 더블 클릭할 때 |
| EVENT_RBUTTONDBLCLK | 마우스 오른쪽 버튼을 더블 클릭할 때 |
| EVENT_MOUSEWHEEL | 마우스 상하 스크롤을 사용할 때 |
| EVENT_MOUSEHWHEEL | 마우스 좌우 스크롤을 사용할 때 |

<br>
<br>

## Flags

| 이름 | 의미 |
|:----------:|:------------------------------------------:|
| EVENT_FLAG_LBUTTON | 마우스 왼쪽 버튼이 눌러져 있음 |
| EVENT_FLAG_MBUTTON | 마우스 가운데 버튼이 눌러져 있음 |
| EVENT_FLAG_RBUTTON | 마우스 오른쪽 버튼이 눌러져 있음 |
| EVENT_FLAG_CTRLKEY | 컨트롤(Ctrl) 키가 눌러져 있음 |
| EVENT_FLAG_SHIFTKEY | 쉬프트(Shift) 키가 눌러져 있음 |
| EVENT_FLAG_ALTKEY | 알트(Alt) 키가 눌러져 있음 |
| flags > 0 | 마우스 스크롤 이벤트의 윗 방향 또는 오른쪽 방향 |
| flags < 0 | 마우스 스크롤 이벤트의 아랫 방향 또는 왼쪽 방향 |

<br>
<br>

## 출력 결과

![2]({{ site.images }}/assets/posts/Python/OpenCV/lecture-39/2.png)
