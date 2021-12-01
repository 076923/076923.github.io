---
layout: post
title: "Python tkinter 강좌 : 제 13강 - Frame"
tagline: "Python tkinter Frame"
image: /assets/images/tkinter.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Tkinter']
keywords: Python, Python tkinter, tkinter Frame
ref: Python-Tkinter
category: Python
permalink: /posts/Python-tkinter-13/
comments: true
toc: true
---

## Frame(프레임)

![1]({{ site.images }}/assets/posts/Python/Tkinter/lecture-13/1.png)

`Frame`을 이용하여 **다른 위젯들을 포함**하기 위한 `프레임`을 생성할 수 있습니다.

<br>
<br>

## Frame 사용

{% highlight Python %}
import tkinter

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

frame1=tkinter.Frame(window, relief="solid", bd=2)
frame1.pack(side="left", fill="both", expand=True)

frame2=tkinter.Frame(window, relief="solid", bd=2)
frame2.pack(side="right", fill="both", expand=True)

button1=tkinter.Button(frame1, text="프레임1")
button1.pack(side="right")

button2=tkinter.Button(frame2, text="프레임2")
button2.pack(side="left")

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

frame1=tkinter.Frame(window, relief="solid", bd=2)
frame1.pack(side="left", fill="both", expand=True)

frame2=tkinter.Frame(window, relief="solid", bd=2)
frame2.pack(side="right", fill="both", expand=True)

button1=tkinter.Button(frame1, text="프레임1")
button1.pack(side="right")

button2=tkinter.Button(frame2, text="프레임2")
button2.pack(side="left")

{% endhighlight %}

`tkinter.Frame(윈도우 창, 매개변수1, 매개변수2, 매개변수3, ...)`을 사용하여 해당 `윈도우 창`에 표시할 `프레임의 속성`을 설정할 수 있습니다.

`매개변수`를 사용하여 `프레임의 속성`을 설정합니다.

위젯의 매개변수 중 `윈도우 창`에서 `프레임 이름`을 이용하여 해당 프레임에 위젯을 `포함`시킬 수 있습니다.

<br>
<br>

## Frame Parameter

### 프레임 형태 설정

|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
|      width     |            [프레임의 너비](#reference-1)                |         0        |                    상수                    |
|     height     |            [프레임의 높이](#reference-1)                |         0        |                    상수                    |
|     relief     |        프레임의 테두리 모양        |       flat       | flat, groove, raised, ridge, solid, sunken |
|  background=bg |           프레임의 배경 색상        | SystemButtonFace |                    color                 |
|      padx      | 프레임의 테두리와 내용의 가로 여백 |         1        |                    상수                    |
|      pady      | 프레임의 테두리와 내용의 세로 여백 |         1        |                    상수                    |

<br>

### 프레임 형식 설정

|   이름   |                           의미                          |     기본값    |                                          속성                                          |
|:--------:|:-------------------------------------------------------:|:-------------:|:--------------------------------------------------------------------------------------:|
|  cursor  |                 프레임의 마우스 커서 모양                 |       -       |                                    [커서 속성](#reference-2)                                   |
|   class_   |           클래스 설정            | - |          -          |
|   visual   |           시각적 정보 설정            | - |          -          |
|   colormap |            256 색상을 지정하는 색상 맵 설정            | - |          new          |

<br>

### 프레임 하이라이트 설정

|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    highlightcolor   |    프레임이 선택되었을 때 색상   | SystemWindowFrame |  color  |
| highlightbackground | 프레임이 선택되지 않았을 때 색상 |  SystemButtonFace |  color  |
|  highlightthickness |    프레임이 선택되었을 때 두께 [(두께 설정)](#reference-3)     |         0         | 상수 |

<br>

### 프레임 동작 설정

|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    takefocus |    Tab 키를 이용하여 위젯 이동 허용 여부  | False |  Boolean |
|    container  |   [응용 프로그램이 포함될 컨테이너로 사용](#reference-4)   | False |  Boolean |

<br>

<a id="reference-1"></a>

### 참고

<a id="reference-2"></a>

* 내부에 위젯이 존재할 경우, `width`와 `height` 설정을 무시하고 `크기 자동 조절`

<a id="reference-3"></a>

* cursor 매개변수

    - arrow, based_arrow_down, based_arrow_up, boat, bogosity, bottom_left_corner, bottom_right_corner, bottom_side, bottom_tee, box_spiral, center_ptr, circle, clock,	coffee_mug, cross, cross_reverse, crosshair, diamond_cross, dot, dotbox, double_arrow, draft_large, draft_small, draped_box, exchange, fleur, gobbler, gumby, hand1, hand2, heart, icon, iron_cross, left_ptr, left_side, left_tee, leftbutton, ll_angle, lr_angle, man, middlebutton, mouse, pencil, pirate, plus, question_arrow, right_ptr, right_side, right_tee, rightbutton, rtl_logo, sailboat, sb_down_arrow, sb_h_double_arrow, sb_left_arrow, sb_right_arrow, sb_up_arrow, sb_v_double_arrow, shuttle, sizing, spider, spraycan, star, target, tcross, top_left_arrow, top_left_corner, top_right_corner, top_side, top_tee, trek, ul_angle, umbrella, ur_angle, watch, wait, xterm, X_cursor


<a id="reference-4"></a>

* `highlightbackground`를 설정하였을 경우, 프레임이 선택되지 않았을 때에도 두께가 표시됨

* `container`를 `True`로 설정하였을 경우, 프레임의 내부에 `위젯`이 포함되어 있지 않아야 함
