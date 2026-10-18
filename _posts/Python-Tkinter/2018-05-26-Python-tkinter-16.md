---
layout: post
title: "Python tkinter 강좌 : 제 16강 - Scrollbar"
tagline: "Python tkinter Scrollbar"
image: /assets/images/tkinter.png
tags: ['Tkinter']
keywords: Python, Python tkinter, tkinter Scrollbar, tkinter 스크롤바, tkinter yscrollcommand, tkinter yview, tkinter ttk Scrollbar
ref: Python-Tkinter
category: Python
permalink: /posts/Python-tkinter-16/
comments: true
toc: true
---

## Scrollbar(스크롤 바)

<img data-src="{{ site.images }}/assets/posts/Python/Tkinter/lecture-16/1.webp" class="lazyload" width="100%" height="100%"/>

`Scrollbar`를 이용하여 `위젯`에 `스크롤`을 적용하기 위한 `스크롤바`를 생성할 수 있습니다.

<br>
<br>

## Scrollbar 사용

{% highlight Python %}

import tkinter

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

frame=tkinter.Frame(window)

scrollbar=tkinter.Scrollbar(frame)
scrollbar.pack(side="right", fill="y")

listbox=tkinter.Listbox(frame, yscrollcommand = scrollbar.set)
for line in range(1,1001):
   listbox.insert(line, str(line) + "/1000")
listbox.pack(side="left")

scrollbar["command"]=listbox.yview

frame.pack()

window.mainloop()
{% endhighlight %}

<br>

{% highlight Python %}

frame=tkinter.Frame(window)

scrollbar=tkinter.Scrollbar(frame)
scrollbar.pack(side="right", fill="y")

listbox=tkinter.Listbox(frame, yscrollcommand = scrollbar.set)
for line in range(1,1001):
   listbox.insert(line, str(line) + "/1000")
listbox.pack(side="left")

scrollbar["command"]=listbox.yview

frame.pack()

{% endhighlight %}

`tkinter.Scrollbar(윈도우 창, 매개변수1, 매개변수2, 매개변수3, ...)`을 사용하여 해당 `윈도우 창`에 표시할 `스크롤바의 속성`을 설정할 수 있습니다.

`매개변수`를 사용하여 `스크롤바의 속성`을 설정합니다.

`스크롤바`를 생성한 후, 스크롤을 적용할 위젯의 `yscrollcommand` 매개변수에 `스크롤바.set`을 연결합니다.

이 후, 스크롤바의 `command` 매개변수를 `적용할 위젯.yview`와 연결합니다.

`스크롤바`와 `위젯`은 **서로를 양방향으로 연결**해야 정상적으로 동작합니다.

- `위젯.yscrollcommand = 스크롤바.set` : 위젯의 내용이 스크롤되면 스크롤바의 위치를 갱신합니다.

- `스크롤바.command = 위젯.yview` : 스크롤바를 움직이면 위젯의 내용을 스크롤합니다.

- Tip : `스크롤바`와 `연결된 위젯`은 각각의 객체이므로 `프레임`으로 **연결하여 사용하는 것을 권장합니다.**

- Tip : 가로 스크롤바는 `orient="horizontal"`로 생성하며, 위젯의 `xscrollcommand`와 스크롤바의 `command=위젯.xview`를 연결합니다.

- Tip : `Listbox`, `Text`, `Canvas` 등 `xscrollcommand`, `yscrollcommand` 매개변수가 있는 위젯에 스크롤바를 연결할 수 있습니다.

- Tip : `tkinter.ttk.Scrollbar`는 운영 체제의 테마가 적용된 스크롤바를 생성하며, 연결 방법은 동일합니다.

<br>

## Scrollbar Method

|           이름          |            의미           |                          설명                          |
|:-----------------------:|:-------------------------:|:------------------------------------------------------:|
|           set(first, last)           |        슬라이더 위치 설정        |  전체 내용 중 현재 보이는 영역의 시작 비율(first)과 끝 비율(last)로 슬라이더의 위치와 크기를 설정, 위젯의 `yscrollcommand`에 연결하여 사용  |
|          get()          | (first, last) 반환 |  현재 슬라이더의 시작 비율과 끝 비율을 반환  |

- Tip : `first`와 `last`는 `0.0 ~ 1.0` 사이의 비율이며, 예를 들어 `(0.0, 0.2)`는 전체 내용의 **앞부분 20%가 표시되고 있음**을 의미합니다.
        
<br>
<br>

## Scrollbar Parameter

### 스크롤바 형태 설정

|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
|      width     |            스크롤바의 너비           |         17        |                    상수                    |
|     relief     |        스크롤바의 테두리 모양        |       sunken       | flat, groove, raised, ridge, solid, sunken |
| borderwidth=bd |        스크롤바의 테두리 두께        |         0       |                    상수                    |
|  background=bg |           스크롤바의 배경 색상          | SystemButtonFace |                    color                   |
|  elementborderwidth |           스크롤 요소의 테두리 두께   | -1 |     상수  |
|  orient |           스크롤의 표시 방향   | vertical |   vertical, horizontal  |

<br>

### 스크롤바 형식 설정

|   이름   |                           의미                          |     기본값    |                                          속성                                          |
|:--------:|:-------------------------------------------------------:|:-------------:|:--------------------------------------------------------------------------------------:|
|  cursor  |                 스크롤바의 마우스 커서 모양                 |       -       |                                    [커서 속성](#reference-1)                                   |

<br>

### 스크롤바 상태 설정

|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    activebackground |    active 상태일 때 스크롤바의 배경 색상	 | SystemButtonFace |  color  |
| activerelief  | active 상태일 때 스크롤바의 테두리 모양 |  raised |   flat, groove, raised, ridge, solid, sunken |

<br>

### 스크롤바 하이라이트 설정

|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    highlightcolor   |    스크롤바가 선택되었을 때 색상   | SystemWindowFrame |  color  |
| highlightbackground | 스크롤바가 선택되지 않았을 때 색상 |  SystemButtonFace |  color  |
|  highlightthickness |    스크롤바가 선택되었을 때 두께 [(두께 설정)](#reference-2)     |         0         | 상수 |

<br>

### 스크롤바 동작 설정

|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    takefocus |    Tab 키를 이용하여 위젯 이동 허용 여부  | - |  Boolean |
|    command |    스크롤바를 움직일 때 실행하는 메서드(함수), 주로 `위젯.yview` 또는 `위젯.xview` 연결   | - |  메서드, 함수 |
|    jump  | True일 경우, 슬라이더를 드래그하는 중에는 command를 호출하지 않고 **마우스를 뗄 때** 한 번만 호출   | False |  Boolean |
| repeatdelay | [버튼이 눌러진 상태에서 command 실행까지의 대기 시간](#reference-3)   |  300 |  상수(ms) |
|  repeatinterval |    [버튼이 눌러진 상태에서 command 실행의 반복 시간](#reference-4)    |         100         | 상수(ms) |

<br>

<a id="reference-1"></a>

### 참고

<a id="reference-2"></a>

* cursor 매개변수

    - arrow, based_arrow_down, based_arrow_up, boat, bogosity, bottom_left_corner, bottom_right_corner, bottom_side, bottom_tee, box_spiral, center_ptr, circle, clock,	coffee_mug, cross, cross_reverse, crosshair, diamond_cross, dot, dotbox, double_arrow, draft_large, draft_small, draped_box, exchange, fleur, gobbler, gumby, hand1, hand2, heart, icon, iron_cross, left_ptr, left_side, left_tee, leftbutton, ll_angle, lr_angle, man, middlebutton, mouse, pencil, pirate, plus, question_arrow, right_ptr, right_side, right_tee, rightbutton, rtl_logo, sailboat, sb_down_arrow, sb_h_double_arrow, sb_left_arrow, sb_right_arrow, sb_up_arrow, sb_v_double_arrow, shuttle, sizing, spider, spraycan, star, target, tcross, top_left_arrow, top_left_corner, top_right_corner, top_side, top_tee, trek, ul_angle, umbrella, ur_angle, watch, wait, xterm, X_cursor

<a id="reference-3"></a>

* `highlightbackground`를 설정하였을 경우, 스크롤바가 선택되지 않았을 때에도 두께가 표시됨

<a id="reference-4"></a>

* `repeatdelay=100` 일 경우, **누르고 있기 시작한 0.1초 후**에 `command`가 실행됨

* `repeatdelay=1000`, `repeatinterval=100` 일 경우, **1초 후에 command가 실행되며 0.1초마다 버튼을 뗄 때까지** `command`가 계속 실행됨
