---
layout: post
title: "Python tkinter 강좌 : 제 16강 - Scrollbar"
tagline: "Python tkinter Scrollbar"
image: /assets/images/tkinter.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Tkinter']
keywords: Python, Python tkinter, tkinter Scrollbar
ref: Python-Tkinter
category: Python
permalink: /posts/Python-tkinter-16/
comments: true
toc: true
---

## Scrollbar(스크롤 바)

![1]({{ site.images }}/assets/posts/Python/Tkinter/lecture-16/1.webp){: width="100%" height="100%"}

`Scrollbar`을 이용하여 `위젯`에 `스크롤`을 적용하기 위한 `스크롤바`을 생성할 수 있습니다.

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

`스크롤바`를 생성 후, `스크롤바의 객체`를 생성하여 `위젯과 위젯을 연결`합니다.

이 후, `command` 매개변수를 `적용할 위젯`과 연결합니다.

- Tip : `스크롤바`와 `연결된 위젯`은 각각의 객체이므로 `프레임`으로 **연결하여 사용하는 것을 권장합니다.**

<br>

## Scrollbar Method

|           이름          |            의미           |                          설명                          |
|:-----------------------:|:-------------------------:|:------------------------------------------------------:|
|           set           |        스크롤 부착        |                  위젯에 스크롤바 적용                  |
| set(좌측상단, 우측하단) |        스크롤 부착        | 위젯에 스크롤바의 좌측상단 좌표와 우측하단 좌표에 고정 |
|          get()          | (좌측상단, 우측하단) 반환 |  현재 스크롤바의 좌측상단 좌표와 우측하단 좌표를 반환  |
        
<br>
<br>

## Scrollbar Parameter

### 스크롤바 형태 설정

|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
|      width     |            스크롤바의 너비           |         17        |                    상수                    |
|     relief     |        스크롤바의 테두리 모양        |       flat       | flat, groove, raised, ridge, solid, sunken |
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
|    takefocus |    Tab 키를 이용하여 위젯 이동 허용 여부  | True |  Boolean |
|    command |    스크롤이 active 상태일 때 실행하는 메서드(함수)   | - |  메서드, 함수 |
|    jump  | 스크롤이 동작할때 마다 command callback 호출   | False |  Boolean |
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
