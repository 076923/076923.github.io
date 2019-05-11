---
bg: "python.jpg"
layout: post
comments: true
title: "Python tkinter 강좌 : 제 27강 – Progressbar"
crawlertitle: "Python tkinter 강좌 : 제 27강 - Progressbar"
summary: "Python tkinter Progressbar"
date: 2018-06-08
categories: posts
tags: ['Python-tkinter']
author: 윤대희
star: true
---

### Progressbar (프로그래스바) ###
----------
`Progressbar`을 이용하여 **현재 진행 상황**을 표시하는 `프로그래스바`를 생성할 수 있습니다.

<br>
<br>
### Progressbar 사용 ###
----------
{% highlight Python %}

import tkinter
import tkinter.ttk

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

progressbar=tkinter.ttk.Progressbar(window, maximum=100, mode="indeterminate")
progressbar.pack()

progressbar.start(50)

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

import tkinter.ttk

{% endhighlight %}

<br>

상단에 `import tkinter.ttk`를 사용하여 `ttk 모듈`을 포함시킵니다. tkinter.ttk 함수의 사용방법은 `tkinter.ttk.*`를 이용하여 사용이 가능합니다.

<br>

{% highlight Python %}

progressbar=tkinter.ttk.Progressbar(window, maximum=100, mode="indeterminate")
progressbar.pack()

progressbar.start(50)

{% endhighlight %}


`tkinter.ttk.Progressbar(윈도우 창, 파라미터1, 파라미터2, 파라미터3, ...)`을 사용하여 `프로그래스바의 속성`을 설정할 수 있습니다.

`파라미터`를 사용하여 `프로그래스바의 속성`을 설정합니다.

<br>
<br>
### Progressbar Method ###
----------

##### 프로그래스바 메소드 #####

|              이름              |       의미       |                       설명                      |
|:------------------------------:|:----------------:|:-----------------------------------------------:|
|     start(ms)    |   시작  |      프로그래스바가 `밀리 초`마다 움직임       |
| step(value) |    값 증가   | 현재 표시되는 값에서 `value`만큼 증가 |
| stop() |    종료  | 프로그래스바 작동 종료 |

<br>
<br>

### Progressbar Parameter ###
----------


##### 프로그래스바 형태 설정 #####

|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
| length | 프로그래스바의 너비 | 100 | 상수 |
| orient | 프로그래스바의 표시 방향 | vertical | vertical, horizontal |
| mode | 프로그래스바의 표시 스타일 | determinate | [determinate , indeterminate](#reference-1)    |

<br>
<br>

##### 프로그래스바 형식 설정 #####

|   이름   |                           의미                          |     기본값    |                                          속성                                          |
|:--------:|:-------------------------------------------------------:|:-------------:|:--------------------------------------------------------------------------------------:|
|  cursor  |      프로그래스바의 마우스 커서 모양                 |       -       |          [커서 속성](#reference-2)             |
|  class_  |      클래스 설정                 |       -       |      -    |  
|   maximum |          프로그래스바의 최댓값 설정 | 100 |   상수 |
|   value |  프로그래스바의 현재값을 설정 | 0 |    상수        |
|   variable |  프로그래스바의 현재값을 가져올 변수 | - |    -        |
|   phase |  프로그래스바의 고유값을 설정 | 0 |    상수        |

<br>
<br>

##### 프로그래스바 동작 설정 #####


|         이름        |              의미              |         기본값        | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    takefocus |    Tab 키를 이용하여 위젯 이동 허용 여부  | False |  Boolean |

<br>
<br>

##### 참고 #####
----------

<a id="reference-1"></a>

* mode 파라미터

    - `determinate` : 표시기가 처음부터 끝까지 채워짐
    - `indeterminate` : 표시기가 처음부터 끝까지 반복 이동


<a id="reference-2"></a>

* cursor 파라미터

    - arrow, based_arrow_down, based_arrow_up, boat, bogosity, bottom_left_corner, bottom_right_corner, bottom_side, bottom_tee, box_spiral, center_ptr, circle, clock,	coffee_mug, cross, cross_reverse, crosshair, diamond_cross, dot, dotbox, double_arrow, draft_large, draft_small, draped_box, exchange, fleur, gobbler, gumby, hand1, hand2, heart, icon, iron_cross, left_ptr, left_side, left_tee, leftbutton, ll_angle, lr_angle, man, middlebutton, mouse, pencil, pirate, plus, question_arrow, right_ptr, right_side, right_tee, rightbutton, rtl_logo, sailboat, sb_down_arrow, sb_h_double_arrow, sb_left_arrow, sb_right_arrow, sb_up_arrow, sb_v_double_arrow, shuttle, sizing, spider, spraycan, star, target, tcross, top_left_arrow, top_left_corner, top_right_corner, top_side, top_tee, trek, ul_angle, umbrella, ur_angle, watch, wait, xterm, X_cursor



