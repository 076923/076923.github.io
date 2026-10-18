---
layout: post
title: "Python tkinter 강좌 : 제 20강 - PanedWindow"
tagline: "Python tkinter PanedWindow"
image: /assets/images/tkinter.png
tags: ['Tkinter']
keywords: Python, Python tkinter, tkinter PanedWindow, tkinter 내부 윈도우, tkinter 분할 창, tkinter sash, tkinter ttk PanedWindow
ref: Python-Tkinter
category: Python
permalink: /posts/Python-tkinter-20/
comments: true
toc: true
---

## PanedWindow(내부 윈도우) 

<img data-src="{{ site.images }}/assets/posts/Python/Tkinter/lecture-20/1.webp" class="lazyload" width="100%" height="100%"/>

`PanedWindow`를 이용하여 **다른 위젯들을 포함**하고 **구역을 나눌 수 있는** `내부 윈도우`를 생성할 수 있습니다.

<br>
<br>

## PanedWindow 사용

{% highlight Python %}

import tkinter

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(True, True)

panedwindow1=tkinter.PanedWindow(relief="raised", bd=2)
panedwindow1.pack(expand=True)
 
left=tkinter.Label(panedwindow1, text="내부윈도우-1 (좌측)")
panedwindow1.add(left)

panedwindow2=tkinter.PanedWindow(panedwindow1, orient="vertical", relief="groove", bd=3)
panedwindow1.add(panedwindow2)

right=tkinter.Label(panedwindow1, text="내부윈도우-2 (우측)")
panedwindow1.add(right)

top=tkinter.Label(panedwindow2, text="내부윈도우-3 (상단)")
panedwindow2.add(top)

bottom=tkinter.Label(panedwindow2, text="내부윈도우-4 (하단)")
panedwindow2.add(bottom)

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}
panedwindow1=tkinter.PanedWindow(relief="raised", bd=2)
panedwindow1.pack(expand=True)
 
left=tkinter.Label(panedwindow1, text="내부윈도우-1 (좌측)")
panedwindow1.add(left)

panedwindow2=tkinter.PanedWindow(panedwindow1, orient="vertical", relief="groove", bd=3)
panedwindow1.add(panedwindow2)

right=tkinter.Label(panedwindow1, text="내부윈도우-2 (우측)")
panedwindow1.add(right)

top=tkinter.Label(panedwindow2, text="내부윈도우-3 (상단)")
panedwindow2.add(top)

bottom=tkinter.Label(panedwindow2, text="내부윈도우-4 (하단)")
panedwindow2.add(bottom)

{% endhighlight %}

`tkinter.PanedWindow(윈도우 창, 매개변수1, 매개변수2, 매개변수3, ...)`을 사용하여 해당 `윈도우 창`에 표시할 `내부 윈도우의 속성`을 설정할 수 있습니다.

`매개변수`를 사용하여 `내부 윈도우의 속성`을 설정합니다.

`내부 윈도우의 새시`를 이동하여 컨테이너를 움직일 수 있습니다.

`새시(sash)`는 구역과 구역 사이의 **경계선**이며, 마우스로 드래그하여 각 구역의 크기를 조절할 수 있습니다.

예제의 `panedwindow1`은 `orient`를 설정하지 않았으므로 기본값인 `horizontal`이 적용되어 위젯이 **좌우**로 나뉘며, `panedwindow2`는 `orient="vertical"`이므로 위젯이 **상하**로 나뉩니다.

- Tip : 예제의 `tkinter.PanedWindow(relief="raised", bd=2)`처럼 첫 번째 매개변수인 `윈도우 창`을 생략하면 **기본 윈도우 창(Tk)**에 생성됩니다.

- Tip : 내부 윈도우에 포함할 위젯은 `pack`, `grid`, `place`로 배치하지 않고, 반드시 `add` 메서드로 추가해야 합니다.

- Tip : `tkinter.ttk.PanedWindow`도 동일한 용도로 사용할 수 있으며, `add(위젯, weight=1)`과 같이 `weight`로 창 크기가 변경될 때 각 구역이 늘어나는 비율을 설정합니다.

<br>
<br>

## PanedWindow Method

|              이름              |       의미       |                       설명                      |
|:------------------------------:|:----------------:|:-----------------------------------------------:|
|     add(위젯, option)    |    위젯 추가   |       해당 `위젯`을 내부 윈도우에 추가       |
|     remove(위젯) / forget(위젯)    |    위젯 제거   |       해당 `위젯`을 내부 윈도우에서 제거       |
|     paneconfig(위젯, option)    |    위젯 설정 변경   |       추가된 `위젯`의 `option`을 변경       |
|     panes()    |    위젯 목록 반환   |       내부 윈도우에 추가된 위젯의 목록을 반환       |
|     sash_coord(index)    |    새시 좌표 반환   |       `index` 번째 새시의 (x, y) 좌표를 반환       |
|     sash_place(index, x, y)    |    새시 위치 변경   |       `index` 번째 새시를 (x, y) 위치로 이동       |

<br>

* option

  - after : 해당 위젯의 **뒤**에 배치, `panedwindow2.add(top, after=bottom)`은 `bottom` 뒤에 `top`을 배치
  - before : 해당 위젯의 **앞**에 배치, `panedwindow2.add(bottom, before=top)`은 `top` 앞에 `bottom`을 배치
  - width : 내부 윈도우 자식의 너비
  - height : 내부 윈도우 자식의 높이
  - minsize : 해당 위젯 구역의 최소 크기, 새시를 이 크기보다 작게 이동할 수 없음
  - sticky : 할당된 공간 내에서의 위치 조정, `n, e, w, s`의 조합 `(기본값 = nesw)`
  - stretch : 창 크기가 변경될 때 늘어나는 구역 설정, `always, first, last, middle, never` `(기본값 = last)`
  - hide : True일 경우, 해당 위젯을 내부 윈도우에서 숨김
  - padx : 내부 윈도우의 가로 여백	
  - pady : 내부 윈도우의 세로 여백	

- Tip : 위젯이 **추가된 순서**에 따라 `내부 윈도우`에서 위젯의 **배열 순서를 결정**

<br>
<br>

## PanedWindow Parameter

### 내부 윈도우 형태 설정


|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
| width | [내부 윈도우의 너비](#reference-1) | 0 | 상수 |
| height | [내부 윈도우의 높이](#reference-1) | 0 | 상수 |
| relief | 내부 윈도우의 테두리 모양 | flat | flat, groove, raised, ridge, solid, sunken |
| borderwidth=bd | 내부 윈도우의 테두리 두께 | 1 | 상수 |
| background=bg | 내부 윈도우의 배경 색상 | SystemButtonFace | color |
| orient | 내부 윈도우의 표시 방향 | horizontal | vertical, horizontal |
| sashwidth | 내부 윈도우 새시의 너비 | 3 | 상수 |
| sashrelief | 내부 윈도우 새시의 테두리 모양 | flat | flat, groove, raised, ridge, solid, sunken |
| sashpad | 내부 윈도우 새시의 여백 | 0 | 상수 |
| showhandle | 내부 윈도우 새시의 손잡이 표시 유/무 | False | Boolean |
| handlesize | 내부 윈도우 새시의 손잡이 크기 | 8 | 상수 |
| handlepad | 내부 윈도우 새시의 손잡이 위치 | 8 | 상수 |
| opaqueresize | [새시를 드래그하는 동안 구역 크기를 즉시 변경할지 여부](#reference-3) | True | Boolean |

<br>

### 내부 윈도우 형식 설정

|   이름   |                           의미                          |     기본값    |                                          속성                                          |
|:--------:|:-------------------------------------------------------:|:-------------:|:--------------------------------------------------------------------------------------:|
|  cursor  |      내부 윈도우의 마우스 커서 모양                 |       -       |   [커서 속성](#reference-2)  |
|  sashcursor |      내부 윈도우 새시의 마우스 커서 모양                 |       -       |   [커서 속성](#reference-2)  |

<br>

<a id="reference-1"></a>

### 참고

<a id="reference-2"></a>

* 미 입력시, `width`와 `height`를 `자동 조절`

<a id="reference-3"></a>

* `opaqueresize`가 `False`일 경우, 드래그하는 동안에는 새시의 위치만 표시되고 **마우스를 뗄 때** 구역의 크기가 변경됨

* cursor 매개변수

    - arrow, based_arrow_down, based_arrow_up, boat, bogosity, bottom_left_corner, bottom_right_corner, bottom_side, bottom_tee, box_spiral, center_ptr, circle, clock,	coffee_mug, cross, cross_reverse, crosshair, diamond_cross, dot, dotbox, double_arrow, draft_large, draft_small, draped_box, exchange, fleur, gobbler, gumby, hand1, hand2, heart, icon, iron_cross, left_ptr, left_side, left_tee, leftbutton, ll_angle, lr_angle, man, middlebutton, mouse, pencil, pirate, plus, question_arrow, right_ptr, right_side, right_tee, rightbutton, rtl_logo, sailboat, sb_down_arrow, sb_h_double_arrow, sb_left_arrow, sb_right_arrow, sb_up_arrow, sb_v_double_arrow, shuttle, sizing, spider, spraycan, star, target, tcross, top_left_arrow, top_left_corner, top_right_corner, top_side, top_tee, trek, ul_angle, umbrella, ur_angle, watch, wait, xterm, X_cursor
