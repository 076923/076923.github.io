---
bg: "python.jpg"
layout: post
comments: true
title: "Python tkinter 강좌 : 제 17강 – Scale"
crawlertitle: "Python tkinter 강좌 : 제 17강 - Scale"
summary: "Python tkinter Scale"
date: 2018-05-27
categories: posts
tags: ['Python-tkinter']
author: 윤대희
star: true
---

### Scale (수치 조정 바) ###
----------

`Scale`을 이용하여 `값`을 `설정`하기 위한 `수치 조정 바`를 생성할 수 있습니다.

<br>
<br>
### Scale 사용 ###
----------
{% highlight Python %}

import tkinter

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

def select(self):
    value="값 : "+str(scale.get())
    label.config(text=value)

var=tkinter.IntVar()

scale=tkinter.Scale(window, variable=var, command=select, orient="horizontal", showvalue=False, tickinterval=50, to=500, length=300)
scale.pack()

label=tkinter.Label(window, text="값 : 0")
label.pack()

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

def select(self):
    value="값 : "+str(scale.get())
    label.config(text=value)

var=tkinter.IntVar()

scale=tkinter.Scale(window, variable=var, command=select, orient="horizontal", showvalue=False, tickinterval=50, to=500, length=300)
scale.pack()

label=tkinter.Label(window, text="값 : 0")
label.pack()

{% endhighlight %}


`tkinter. Scrollbar(윈도우 창, 파라미터1, 파라미터2, 파라미터3, ...)`을 사용하여 해당 `윈도우 창`에 표시할 `수치 조정 바의 속성`을 설정할 수 있습니다.

`파라미터`를 사용하여 `수치 조정 바의 속성`을 설정합니다.

`수치 조정 바`를 생성 후,  `command` 파라미터를 `적용할 함수`와 연결합니다.

<br>

* Tip : `scale.get()`는 현재 `수치 조정 바`에 표시된 값을 가져오며, `var.get()`은 `var`에 저장된 값을 가져옵니다.

<br>

<br>
<br>
### Scale Method###
----------

|           이름          |            의미           |                          설명                          |
|:-----------------------:|:-------------------------:|:------------------------------------------------------:|
|           set()           |        값 변경        |                  수치 조정 바의 값을 변경                   |
|          get()          | 값 반환 |  수치 조정 바의 값을 반환  |
        
<br>
<br>

### Scale Parameter ###
----------

##### 수치 조정 바 텍스트 설정 #####


|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
|      label     |            수치 조정 바에 표시되는 문자        |         -         |                    문자                    |
|     showvalue  |        수치 조정 바에 값 표시 유/무        |       True       | Boolean |


<br>
<br>


##### 수치 조정 바 형태 설정 #####


|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
| width | 수치 조정 바의 슬라이드 너비 | 15 | 상수 |
| length | 수치 조정 바의 길이 | 100 | 상수 |
| relief | 수치 조정 바의 테두리 모양 | flat | flat, groove, raised, ridge, solid, sunken |
| borderwidth=bd | 수치 조정 바의 테두리 두께 | 0 | 상수 |
| background=bg | 수치 조정 바의 배경 색상 | SystemButtonFace | color |
| troughcolor | 수치 조정 바의 내부 배경 색상  | SystemScrollbar | color |
| orient | 수치 조정 바의 표시 방향   | vertical | vertical, horizontal |
| sliderlength | 수치 조정 바의 슬라이더 길이 | 30 | 상수 |
| sliderrelief  | 수치 조정 바의 슬라이더 테두리 모양 | raised | flat, groove, raised, ridge, solid, sunken |
| tickinterval | [수치 조정 바의 수치 값 간격](#reference-1) | 0 | 상수 |


<br>
<br>

##### 수치 조정 바 형식 설정 #####


|   이름   |                           의미                          |     기본값    |                                          속성                                          |
|:--------:|:-------------------------------------------------------:|:-------------:|:--------------------------------------------------------------------------------------:|
|  font   |       수치 조정 바의 문자열 글꼴 설정              |    TkDefaultFont    |      font        |
|  cursor  |      수치 조정 바의 마우스 커서 모양                 |       -       |                                    [커서 속성](#reference-2)                                   |
|  digits |     수치 조정 바의 숫자 값을 문자열로 변활 할 때 사용할 숫자의 수   |    0    |      상수        |
|  from_  |      수치 조정 바의 최솟값   |    0    |      상수        |
|  to |     수치 조정 바의 최댓값   |    100    |      상수        |
|  resolution |   수치 조정 바의 간격   |    1    |      상수        |



<br>
<br>

##### 수치 조정 바 상태 설정 #####


|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    state  |    상태 설정	 | normal  | normal, active, disabled  |
|    activebackground |    active 상태일 때 수치 조정 바의 슬라이더 색상	 | SystemButtonFace |  color  |


<br>
<br>

##### 수치 조정 바 하이라이트 설정 #####


|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    highlightcolor   |    수치 조정 바가 선택되었을 때 색상   | SystemWindowFrame |  color  |
| highlightbackground | 수치 조정 바가 선택되지 않았을 때 색상 |  SystemButtonFace |  color  |
|  highlightthickness |    수치 조정 바가 선택되었을 때 두께 [(두께 설정)](#reference-3)     |         2         | 상수 |

<br>
<br>

##### 수치 조정 바 동작 설정 #####


|         이름        |              의미              |         기본값        | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    takefocus |    Tab 키를 이용하여 위젯 이동 허용 여부  | False |  Boolean |
|    command |    수치 조정 바가 active 상태일 때 실행하는 메소드(함수)   | - |  메소드, 함수 |
|    variable | 수치 조정 바의 상태를 저장할 제어 변수  | - |  tkinter.IntVar(), tkinter.StringVar() |
|    bigincrement | [수치 조정 바가 Tab된 상태에서 Ctrl 키를 이용하여 한 번에 이동할 양](#reference-4)  | 0 |  상수 |
| repeatdelay | [버튼이 눌러진 상태에서 command 실행까지의 대기 시간](#reference-5)   |  300 |  상수(ms) |
|  repeatinterval |    [버튼이 눌러진 상태에서 command 실행의 반복 시간](#reference-6)    |         100         | 상수(ms) |


<br>
<br>

##### 참고 #####
----------

<a id="reference-1"></a>

* `tickinterval`를 설정하였을 경우, 수치 조정 바에 `tickinterval` 값 마다 수치가 표시됨


<a id="reference-2"></a>

* cursor 파라미터

    - arrow, based_arrow_down, based_arrow_up, boat, bogosity, bottom_left_corner, bottom_right_corner, bottom_side, bottom_tee, box_spiral, center_ptr, circle, clock,	coffee_mug, cross, cross_reverse, crosshair, diamond_cross, dot, dotbox, double_arrow, draft_large, draft_small, draped_box, exchange, fleur, gobbler, gumby, hand1, hand2, heart, icon, iron_cross, left_ptr, left_side, left_tee, leftbutton, ll_angle, lr_angle, man, middlebutton, mouse, pencil, pirate, plus, question_arrow, right_ptr, right_side, right_tee, rightbutton, rtl_logo, sailboat, sb_down_arrow, sb_h_double_arrow, sb_left_arrow, sb_right_arrow, sb_up_arrow, sb_v_double_arrow, shuttle, sizing, spider, spraycan, star, target, tcross, top_left_arrow, top_left_corner, top_right_corner, top_side, top_tee, trek, ul_angle, umbrella, ur_angle, watch, wait, xterm, X_cursor


<a id="reference-3"></a>

* `highlightbackground`를 설정하였을 경우, 수치 조정 바가 선택되지 않았을 때에도 두께가 표시됨


<a id="reference-4"></a>


* `takefocus`가 `True`로 설정한 뒤, `Tab`이 된 상태에서 `Ctrl`키와 `좌우 방향키`를 동시에 눌러 `bigincrement` 의 값 만큼 한 번에 이동됨


<a id="reference-5"></a>

* `repeatdelay=100` 일 경우, **누르고 있기 시작한 0.1초 후**에 `command`가 실행됨


<a id="reference-6"></a>

* `repeatdelay=1000`, `repeatinterval=100` 일 경우, **1초 후에 command가 실행되며 0.1초마다 버튼을 뗄 때까지** `command`가 계속 실행됨




