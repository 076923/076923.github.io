---
layout: post
title: "Python tkinter 강좌 : 제 15강 - Canvas"
tagline: "Python tkinter Canvas"
image: /assets/images/tkinter.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Tkinter']
keywords: Python, Python tkinter, tkinter Canvas
ref: Python-Tkinter
category: Python
permalink: /posts/Python-tkinter-15/
comments: true
toc: true
---

## Canvas(캔버스)

![1]({{ site.images }}/assets/posts/Python/Tkinter/lecture-15/1.webp){:class="lazyload" width="100%" height="100%"}

`Canvas`을 이용하여 `선`, `다각형`, `원`등을 그리기 위한 `캔버스`을 생성할 수 있습니다.

<br>
<br>

## Canvas 사용

{% highlight Python %}

import tkinter

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

canvas=tkinter.Canvas(window, relief="solid", bd=2)

line=canvas.create_line(10, 10, 20, 20, 20, 130, 30, 140, fill="red")
polygon=canvas.create_polygon(50, 50, 170, 170, 100, 170, outline="yellow")
oval=canvas.create_oval(100, 200, 150, 250, fill="blue", width=3)
arc=canvas.create_arc(100, 100, 300, 300, start=0, extent=150, fill='red')

canvas.pack()

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

canvas=tkinter.Canvas(window, relief="solid", bd=2)

line=canvas.create_line(10, 10, 20, 20, 20, 130, 30, 140, fill="red")
polygon=canvas.create_polygon(50, 50, 170, 170, 100, 170, outline="yellow")
oval=canvas.create_oval(100, 200, 150, 250, fill="blue", width=3)
arc=canvas.create_arc(100, 100, 300, 300, start=0, extent=150, fill='red')

canvas.pack()

{% endhighlight %}

`tkinter. Canvas(윈도우 창, 매개변수1, 매개변수2, 매개변수3, ...)`을 사용하여 해당 `윈도우 창`에 표시할 `캔버스의 속성`을 설정할 수 있습니다.

`매개변수`를 사용하여 `캔버스의 속성`을 설정합니다.

`메서드`를 이용하여 캔버스에 `도형`을 그릴 수 있습니다.

<br>
<br>

## Canvas Method

|    이름    |  의미  |       설명      |
|:----------------------------------------------------:|:------:|:-------------------------------------------------------------------------------------:|
|   create_line(x1, y1, x2, y2, ... , xn, yn, option)  |   선   |  (x1, y1), (x2, y2), ..., (xn, yn) 까지 연결되는 선 생성  |
|   create_line(x1, y1, x2, y2, option)  | 사각형 |  (x1, y1)에서 (x2, y2)의 크기를 갖는 사각형 생성  |
| create_polygon(x1, y1, x2, y2, ... , xn, yn, option) | 다각형 |      (x1, y1), (x2, y2), ..., (xn, yn) 의 꼭지점을 같는 다각형 생성     |
|   create_oval(x1, y1, x2, y2, option)  |   원   |       (x1, y1)에서 (x2, y2)의 크기를 갖는 원 생성       |
|   create_arc(x1, y1, x2, y2, start, extent, option)  |   호   | (x1, y1)에서 (x2, y2)의 크기를 가지며 `start` 각도부터 `extent`의 각을 지니는 호 생성 |
|    create_image(x, y, image, option)   | 이미지 |          (x, y) 위치의 `image` 생성         |

<br>

* option

    - `fill` : 배경 색상
 
    - `outline` : 두께 색상

    - `width` : 두께

    - `fill` : 배경 색상
 
    - `anchor` : 위치 지정

<br>

- Tip : 이외에도 여러 `option`이 존재
 
<br>
<br>

## Entry Parameter

### 캔버스 형태 설정

| 이름 | 의미 | 기본값 | 속성 |
|:------:|:---------:|:--------:|:-----:|
| width | 캔버스의 너비 | 378 | 상수 |
| height | 캔버스의 높이 | 265  | 상수 |
| relief | 캔버스의 테두리 모양 | flat | flat, groove, raised, ridge, solid, sunken |
| borderwidth=bd | 캔버스의 테두리 두께 | 0 | 상수 |
| background=bg | 캔버스의 배경 색상 | SystemButtonFace | color |
| offset | 캔버스의 오프셋 설정  | 0,0 | x, y, n, e, w, s, ne, nw, se, sw |

<br>

### 캔버스 형식 설정

|   이름   |      의미     |     기본값    |       속성       |
|:--------:|:-------------------------------------------------------:|:-------------:|:--------------------------------------------------------------------------------------:|
|  cursor  |   캔버스의 마우스 커서 모양   |       -       |        [커서 속성](#reference-1)       |
|   xscrollcommand   |  캔버스의 가로스크롤 위젯 적용        | - |       Scrollbar위젯.set       |
|   yscrollcommand   |  캔버스의 세로스크롤 위젯 적용        | - |       Scrollbar위젯.set       |
|   xscrollincrement |  캔버스 가로스크롤의 증가량        | 0 |   상수  |
|   yscrollincrement |  캔버스 가로스크롤의 증가량        | 0 |   상수  |
|   scrollregion  |  캔버스 스크롤 영역 크기 설정       | - |   n, e, w, s  |
|   confine   |  캔버스의 스크롤 영역 내 제한       | True |   Boolean  |

<br>

### 캔버스 하이라이트 설정

|  이름 |       의미       |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    highlightcolor   |    캔버스가 선택되었을 때 색상   | SystemWindowFrame |  color  |
| highlightbackground | 캔버스가 선택되지 않았을 때 색상 |  SystemButtonFace |  color  |
|  highlightthickness |    캔버스가 선택되었을 때 두께 [(두께 설정)](#reference-2)     |  0  | 상수 |

<br>

### 캔버스 동작 설정

|  이름 |       의미       |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    takefocus |    Tab 키를 이용하여 위젯 이동 허용 여부  | True |  Boolean |

<br>

<a id="reference-1"></a>

### 참고

<a id="reference-2"></a>

* cursor 매개변수

    - arrow, based_arrow_down, based_arrow_up, boat, bogosity, bottom_left_corner, bottom_right_corner, bottom_side, bottom_tee, box_spiral, center_ptr, circle, clock,	coffee_mug, cross, cross_reverse, crosshair, diamond_cross, dot, dotbox, double_arrow, draft_large, draft_small, draped_box, exchange, fleur, gobbler, gumby, hand1, hand2, heart, icon, iron_cross, left_ptr, left_side, left_tee, leftbutton, ll_angle, lr_angle, man, middlebutton, mouse, pencil, pirate, plus, question_arrow, right_ptr, right_side, right_tee, rightbutton, rtl_logo, sailboat, sb_down_arrow, sb_h_double_arrow, sb_left_arrow, sb_right_arrow, sb_up_arrow, sb_v_double_arrow, shuttle, sizing, spider, spraycan, star, target, tcross, top_left_arrow, top_left_corner, top_right_corner, top_side, top_tee, trek, ul_angle, umbrella, ur_angle, watch, wait, xterm, X_cursor

* `highlightbackground`를 설정하였을 경우, 캔버스가 선택되지 않았을 때에도 두께가 표시됨
