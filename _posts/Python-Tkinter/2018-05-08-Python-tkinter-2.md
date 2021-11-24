---
layout: post
title: "Python tkinter 강좌 : 제 2강 - Label"
tagline: "Python tkinter Label"
image: /assets/images/tkinter.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Tkinter']
keywords: Python, Python tkinter, tkinter Label
ref: Python-Tkinter
category: Python
permalink: /posts/Python-tkinter-2/
comments: true
toc: true
---

## Label(라벨)

`Label`을 이용하여 삽입한 이미지나 도표, 그림 등에 사용되는 `주석문`을 생성할 수 있습니다.

<br>
<br>

## Label 사용

{% highlight Python %}

import tkinter

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

label=tkinter.Label(window, text="파이썬", width=10, height=5, fg="red", relief="solid")
label.pack()

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

label=tkinter.Label(window, text="파이썬", width=10, height=5, fg="red", relief="solid")
label.pack()

{% endhighlight %}

`tkinter.Label(윈도우 창, 매개변수1, 매개변수2, 매개변수3, ...)`을 사용하여 해당 `윈도우 창`에 표시할 `라벨의 속성`을 설정할 수 있습니다.

`매개변수`를 사용하여 `라벨의 속성`을 설정합니다.

<br>
<br>

## Label Parameter

### 라벨 문자열 설정

|     이름     |                    의미                   | 기본값 |                 속성                |
|:------------:|:-----------------------------------------:|:------:|:-----------------------------------:|
|     text     |            라벨에 표시할 문자열           |    -   |                  -                  |
| textvariable |     라벨에 표시할 문자열을 가져올 변수    |    -   |                  -                  |
|    anchor    |     라벨안의 문자열 또는 이미지의 위치    | center | n, ne, e, se, s, sw, w, nw, center  |
|    justify   | 라벨의 문자열이 여러 줄 일 경우 정렬 방법 | center |         center, left, right         |
|  wraplength  |           자동 줄내림 설정 너비           |    0   |                 상수                |

<br>

### 라벨 형태 설정

|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
|      width     |            라벨의 너비           |         0        |                    상수                    |
|     height     |            라벨의 높이           |         0        |                    상수                    |
|     relief     |        라벨의 테두리 모양        |       flat       | flat, groove, raised, ridge, solid, sunken |
| borderwidth=bd |        라벨의 테두리 두께        |         2        |                    상수                    |
|  background=bg |           라벨의 배경 색상          | SystemButtonFace |                    color                   |
|  foreground=fg |          라벨의 문자열 색상         | SystemButtonFace |                    color                   |
|      padx      | 라벨의 테두리와 내용의 가로 여백 |         1        |                    상수                    |
|      pady      | 라벨의 테두리와 내용의 세로 여백 |         1        |                    상수                    |

<br>

### 라벨 형식 설정

|   이름   |                           의미                          |     기본값    |                                          속성                                          |
|:--------:|:-------------------------------------------------------:|:-------------:|:--------------------------------------------------------------------------------------:|
|  bitmap  |                라벨에 포함할 기본 이미지                |       -       | info, warning, error, question,   questhead, hourglass, gray12, gray25, gray50, gray75 |
|   image  |                라벨에 포함할 임의 이미지                |       -       |                                            -                                           |
| compound | 라벨에 문자열과 이미지를 동시에 표시할 때 이미지의 위치 |      none     |                         bottom, center, left, none, right, top                         |
|   font   |                라벨의 문자열 글꼴 설정               | TkDefaultFont |                                          font                                          |
|  cursor  |                 라벨의 마우스 커서 모양                 |       -       |                                    [커서 속성](#reference-1)                                   |

<br>

### 라벨 상태 설정

|        이름        |                   의미                   |       기본값       |           속성           |
|:------------------:|:----------------------------------------:|:------------------:|:------------------------:|
|        state       |                 상태 설정                 |       normal       | [normal](#reference-2), active, disabled |
|  activebackground  |   active 상태일 때 라벨의 배경 색상   |  SystemButtonFace  |           color          |
|  activeforeground  |  active 상태일 때 라벨의 문자열 색상  |  SystemButtonText  |           color          |
| disabledforeground | disabeld 상태일 때 라벨의 문자열 색상 | SystemDisabledText |           color          |

<br>

### 라벨 하이라이트 설정

|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    highlightcolor   |    라벨이 선택되었을 때 색상   | SystemWindowFrame |  color  |
| highlightbackground | 라벨이 선택되지 않았을 때 색상 |  SystemButtonFace |  color  |
|  highlightthickness |    라벨이 선택되었을 때 두께 [(두께 설정)](#reference-3)     |         0         | 상수 |

<br>

### 참고

<a id="reference-1"></a>

* cursor 매개변수

    - arrow, based_arrow_down, based_arrow_up, boat, bogosity, bottom_left_corner, bottom_right_corner, bottom_side, bottom_tee, box_spiral, center_ptr, circle, clock,	coffee_mug, cross, cross_reverse, crosshair, diamond_cross, dot, dotbox, double_arrow, draft_large, draft_small, draped_box, exchange, fleur, gobbler, gumby, hand1, hand2, heart, icon, iron_cross, left_ptr, left_side, left_tee, leftbutton, ll_angle, lr_angle, man, middlebutton, mouse, pencil, pirate, plus, question_arrow, right_ptr, right_side, right_tee, rightbutton, rtl_logo, sailboat, sb_down_arrow, sb_h_double_arrow, sb_left_arrow, sb_right_arrow, sb_up_arrow, sb_v_double_arrow, shuttle, sizing, spider, spraycan, star, target, tcross, top_left_arrow, top_left_corner, top_right_corner, top_side, top_tee, trek, ul_angle, umbrella, ur_angle, watch, wait, xterm, X_cursor


<a id="reference-2"></a>

* 기본 설정은 `normal` 상태의 설정을 의미함 (`bg`, `fg` 등의 설정)


<a id="reference-3"></a>

* `highlightbackground`를 설정하였을 경우, 라벨이 선택되지 않았을 때에도 두께가 표시됨
