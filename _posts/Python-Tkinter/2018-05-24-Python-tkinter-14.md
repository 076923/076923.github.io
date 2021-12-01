---
layout: post
title: "Python tkinter 강좌 : 제 14강 - Message"
tagline: "Python tkinter Message"
image: /assets/images/tkinter.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Tkinter']
keywords: Python, Python tkinter, tkinter Message
ref: Python-Tkinter
category: Python
permalink: /posts/Python-tkinter-14/
comments: true
toc: true
---

## Message(메세지)

`Message`를 이용하여 **여러줄의 문자열을 포함**하기 위한 `메세지`를 생성할 수 있습니다.

<br>
<br>

## Message 사용

{% highlight Python %}

import tkinter

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

message=tkinter.Message(window, text="메세지입니다.", width=100, relief="solid")
message.pack()

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

message=tkinter.Message(window, text="메세지입니다.", width=100, relief="solid"
message.pack()

{% endhighlight %}

`tkinter.Message(윈도우 창, 매개변수1, 매개변수2, 매개변수3, ...)`을 사용하여 해당 `윈도우 창`에 표시할 `메세지의 속성`을 설정할 수 있습니다.

`매개변수`를 사용하여 `메세지의 속성`을 설정합니다.

<br>
<br>

## Message Parameter

### 메세지 문자열 설정

|     이름     |                    의미                   | 기본값 |                 속성                |
|:------------:|:-----------------------------------------:|:------:|:-----------------------------------:|
|     text     |            메세지에 표시할 문자열           |    -   |                  -                  |
| textvariable |     메세지에 표시할 문자열을 가져올 변수    |    -   |                  -                  |
|    anchor    |     메세지안의 문자열의 위치    | center | n, ne, e, se, s, sw, w, nw, center  |
|    justify   | 메세지의 문자열이 여러 줄 일 경우 정렬 방법 | center |         center, left, right         |

<br>

### 메세지 형태 설정

|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
|      width     |            메세지의 최대 허용 너비           |         0        |                    상수                    |
|     height     |            메세지의 최대 허용 높이           |         0        |                    상수                    |
|     relief     |        메세지의 테두리 모양        |       flat       | flat, groove, raised, ridge, solid, sunken |
|  borderwidth=bd |           메세지의 테두리 두께        | 1 |                    상수 |
|  background=bg |           메세지의 배경 색상        | SystemButtonFace |                    color                 |
|  foreground=fg |          메세지의 문자열 색상         | SystemButtonFace |                    color                   |
|      padx      | 메세지의 테두리와 내용의 가로 여백 |         -1        |                    상수                    |
|      pady      | 메세지의 테두리와 내용의 세로 여백 |         -1       |                    상수                    |
|      aspect  | [메시지의 높이에 대한 너비 비율](#reference-1)  |         150       |                    상수                    |

<br>

### 메세지 형식 설정

|   이름   |                           의미                          |     기본값    |                                          속성                                          |
|:--------:|:-------------------------------------------------------:|:-------------:|:--------------------------------------------------------------------------------------:|
|   font   |                메세지의 문자열 글꼴 설정               | TkDefaultFont |                                          font                                          |
|  cursor  |                 메세지의 마우스 커서 모양                 |       -       |                                    [커서 속성](#reference-2)                                   |

<br>

### 메세지 하이라이트 설정

|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    highlightcolor   |    메세지가 선택되었을 때 색상   | SystemWindowFrame |  color  |
| highlightbackground | 메세지가 선택되지 않았을 때 색상 |  SystemButtonFace |  color  |
|  highlightthickness |    메세지가 선택되었을 때 두께 [(두께 설정)](#reference-3)     |         0         | 상수 |

<br>

### 메세지 동작 설정

|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    takefocus |    Tab 키를 이용하여 위젯 이동 허용 여부  | True |  Boolean |

<br>

<a id="reference-1"></a>

### 참고

<a id="reference-2"></a>

* `aspect`를 150으로 설정하였을 경우, `100:150=높이:너비`를 의미함, width나 height가 설정된 경우 `aspect` 설정 값은 **무시됨**

<a id="reference-3"></a>

* cursor 매개변수

    - arrow, based_arrow_down, based_arrow_up, boat, bogosity, bottom_left_corner, bottom_right_corner, bottom_side, bottom_tee, box_spiral, center_ptr, circle, clock,	coffee_mug, cross, cross_reverse, crosshair, diamond_cross, dot, dotbox, double_arrow, draft_large, draft_small, draped_box, exchange, fleur, gobbler, gumby, hand1, hand2, heart, icon, iron_cross, left_ptr, left_side, left_tee, leftbutton, ll_angle, lr_angle, man, middlebutton, mouse, pencil, pirate, plus, question_arrow, right_ptr, right_side, right_tee, rightbutton, rtl_logo, sailboat, sb_down_arrow, sb_h_double_arrow, sb_left_arrow, sb_right_arrow, sb_up_arrow, sb_v_double_arrow, shuttle, sizing, spider, spraycan, star, target, tcross, top_left_arrow, top_left_corner, top_right_corner, top_side, top_tee, trek, ul_angle, umbrella, ur_angle, watch, wait, xterm, X_cursor

* `highlightbackground`를 설정하였을 경우, 메세지가 선택되지 않았을 때에도 두께가 표시됨
