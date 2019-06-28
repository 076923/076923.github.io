---
layout: post
title: "Python OpenCV 강좌 : 제 2강 - CAMERA 출력"
tagline: "Python OpenCV Using Camera"
image: /assets/images/opencv_logo.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-OpenCV']
keywords: Python, Python OpenCV, OpenCV Using Camera
ref: Python-OpenCV
category: posts
permalink: /posts/Python-opencv-2/
comments: true
---


## Camera 출력 ##
----------

**내장 카메라** 또는 **외장 카메라**에서 이미지를 얻어와 프레임을 재생할 수 있습니다.

<br>
<br>

## Main Code ##
----------

{% highlight Python %}

import cv2

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)
    if cv2.waitKey(1) > 0: break

capture.release()
cv2.destroyAllWindows()

{% endhighlight %}

<br>
<br>

## Detailed Code ##
----------

{% highlight Python %}

capture = cv2.VideoCapture(0)

{% endhighlight %}

`cv2.VideoCapture(n)`을 이용하여 **내장 카메라** 또는 **외장 카메라**에서 영상을 받아옵니다.

`n`은 **카메라의 장치 번호**를 의미합니다. 노트북을 이용할 경우, 내장 카메라가 존재하므로 카메라의 장치 번호는 `0`이 됩니다.

카메라를 추가적으로 연결하여 **외장 카메라**를 사용하는 경우, 장치 번호가 `1~n`까지 변화합니다.

<br>

{% highlight Python %}

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

{% endhighlight %}

`capture.set(option, n)`을 이용하여 카메라의 속성을 설정할 수 있습니다.

`option`은 **프레임의 너비와 높이**등의 속성을 설정할 수 있습니다.

`n`의 경우 해당 **너비와 높이의 값**을 의미합니다.

<br>

{% highlight Python %}

while True:
    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)
    if cv2.waitKey(1) > 0: break

{% endhighlight %}

`while`문을 이용하여 **영상 출력을 반복합니다.**

`capture.read()`를 이용하여 `카메라의 상태` 및 `프레임`을 받아옵니다.

`ret`은 카메라의 상태가 저장되며 정상 작동할 경우 `True`를 반환합니다. 작동하지 않을 경우 `False`를 반환합니다.

`frame`에 현재 프레임이 저장됩니다.

`cv2.imshow("윈도우 창 제목", 이미지)`를 이용하여 **윈도우 창**에 **이미지**를 띄웁니다.

`if`문을 이용하여 **키 입력**이 있을 때 까지 `while`문을 반복합니다.

`cv2.waitkey(time)`이며 `time`마다 키 입력상태를 받아옵니다. 

키가 입력될 경우, 해당 키의 `아스키 값`을 반환합니다.

즉, 어떤 키라도 누를 경우, `break`하여 `while`문을 종료합니다.

* Tip : `time`이 `0`일 경우, 지속적으로 검사하여 **프레임이 넘어가지 않습니다.**
* Tip : `if cv2.waitKey(1) == ord('q'): break`으로 사용할 경우, `q`가 입력될 때 `while`문을 종료합니다.

<br>

{% highlight Python %}

capture.release()
cv2.destroyAllWindows()

{% endhighlight %}

`capture.relase()`를 통하여 카메라 장치에서 받아온 **메모리를 해제합니다.**

`cv2.destroyAllWindows()`를 이용하여 **모든 윈도우창을 닫습니다.**

* Tip : `cv2.destroyWindow("윈도우 창 제목")`을 이용하여 **특정 윈도우 창만 닫을 수 있습니다.**
