---
layout: post
title: "Python tkinter 강좌 : 제 3강 – Button"
tagline: "Python tkinter Button"
image: /assets/images/tkinter.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-Tkinter']
keywords: Python, Python tkinter, tkinter Button
ref: Python-Tkinter
category: posts
permalink: /posts/Python-tkinter-3/
comments: true
---

## Button(버튼) ##
----------

`Button`을 이용하여 `메소드` 또는 `함수` 등을 실행시키기 위한 `단추`를 생성할 수 있습니다.

<br>
<br>

## Button 사용 ##
----------

{% highlight Python %}

import tkinter

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

count=0

def countUP():
    global count
    count +=1
    label.config(text=str(count))

label = tkinter.Label(window, text="0")
label.pack()

button = tkinter.Button(window, overrelief="solid", width=15, command=countUP, repeatdelay=1000, repeatinterval=100)
button.pack()

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

count=0

def countUP():
    global count
    count +=1
    label.config(text=str(count))

label = tkinter.Label(window, text="0")
label.pack()

button = tkinter.Button(window, overrelief="solid", width=15, command=countUP, repeatdelay=1000, repeatinterval=100)
button.pack()

{% endhighlight %}

`tkinter.Button(윈도우 창, 파라미터1, 파라미터2, 파라미터3, ...)`을 사용하여 해당 `윈도우 창`에 표시할 `버튼의 속성`을 설정할 수 있습니다.

`파라미터`를 사용하여 `버튼의 속성`을 설정합니다.

파라미터 중 `command`를 이용하여 `사용자 정의 함수 : countUP`을 실행시킬 수 있습니다.

<br>
<br>

## Button Method ##
----------

|    이름    |      의미      |
|:----------:|:--------------:|
|  invoke()  | 버튼 실행 |
|   flash()  |     깜빡임     |

<br>

* Tip : `invoke()` : 버튼을 클릭했을 때와 동일한 실행
* Tip : `flash()` : `normal` 상태 배경 색상과 `active` 상태 배경 색상 사이에서 깜빡임

<br>
<br>

## Button Parameter ##
----------

## 버튼 문자열 설정 ##


|     이름     |                    의미                   | 기본값 |                 속성                |
|:------------:|:-----------------------------------------:|:------:|:-----------------------------------:|
|     text     |            버튼에 표시할 문자열           |    -   |                  -                  |
| textvariable |     버튼에 표시할 문자열을 가져올 변수    |    -   |                  -                  |
|    anchor    |     버튼안의 문자열 또는 이미지의 위치    | center | n, ne, e, se, s, sw, w, nw, center  |
|    justify   | 버튼의 문자열이 여러 줄 일 경우 정렬 방법 | center |         center, left, right         |
|  wraplength  |           자동 줄내림 설정 너비           |    0   |                 상수                |

<br>
<br>

## 버튼 형태 설정 ##


|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
|      width     |            버튼의 너비           |         0        |                    상수                    |
|     height     |            버튼의 높이           |         0        |                    상수                    |
|     relief     |        버튼의 테두리 모양        |       flat       | flat, groove, raised, ridge, solid, sunken |
|     overrelief      |        버튼에 마우스를 올렸을 때 버튼의 테두리 모양        |       raised       | flat, groove, raised, ridge, solid, sunken |
| borderwidth=bd |        버튼의 테두리 두께        |         2        |                    상수                    |
|  background=bg |           버튼의 배경 색상        | SystemButtonFace |                    color                 |
|  foreground=fg |          버튼의 문자열 색상         | SystemButtonFace |                    color                   |
|      padx      | 버튼의 테두리와 내용의 가로 여백 |         1        |                    상수                    |
|      pady      | 버튼의 테두리와 내용의 세로 여백 |         1        |                    상수                    |

<br>
<br>

## 버튼 형식 설정 ##


|   이름   |                           의미                          |     기본값    |                                          속성                                          |
|:--------:|:-------------------------------------------------------:|:-------------:|:--------------------------------------------------------------------------------------:|
|  bitmap  |                버튼에 포함할 기본 이미지                |       -       | info, warning, error, question,   questhead, hourglass, gray12, gray25, gray50, gray75 |
|   image  |                버튼에 포함할 임의 이미지                |       -       |                                            -                                           |
| compound | 버튼에 문자열과 이미지를 동시에 표시할 때 이미지의 위치 |      none     |                         bottom, center, left, none, right, top                         |
|   font   |                버튼의 문자열 글꼴 설정               | TkDefaultFont |                                          font                                          |
|  cursor  |                 버튼의 마우스 커서 모양                 |       -       |                                    [커서 속성](#reference-1)                                   |

<br>
<br>

## 버튼 상태 설정 ##


|        이름        |                   의미                   |       기본값       |           속성           |
|:------------------:|:----------------------------------------:|:------------------:|:------------------------:|
|        state       |                 상태 설정                 |       normal       | [normal](#reference-2), active, disabled |
|  activebackground  |   active 상태일 때 버튼의 배경 색상   |  SystemButtonFace  |           color          |
|  activeforeground  |  active 상태일 때 버튼의 문자열 색상  |  SystemButtonText  |           color          |
| disabledforeground | disabeld 상태일 때 버튼의 문자열 색상 | SystemDisabledText |           color          |

<br>
<br>

## 버튼 하이라이트 설정 ##


|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    highlightcolor   |    버튼이 선택되었을 때 색상   | SystemWindowFrame |  color  |
| highlightbackground | 버튼이 선택되지 않았을 때 색상 |  SystemButtonFace |  color  |
|  highlightthickness |    버튼이 선택되었을 때 두께 [(두께 설정)](#reference-3)     |         0         | 상수 |

<br>
<br>

## 버튼 동작 설정 ##


|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    takefocus |    Tab 키를 이용하여 위젯 이동 허용 여부  | True |  Boolean |
|    command |    버튼이 active 상태일 때 실행하는 메소드(함수)   | - |  메소드, 함수 |
| repeatdelay | [버튼이 눌러진 상태에서 command 실행까지의 대기 시간](#reference-4)   |  0 |  상수(ms) |
|  repeatinterval |    [버튼이 눌러진 상태에서 command 실행의 반복 시간](#reference-5)    |         0         | 상수(ms) |

<br>
<br>

## 참고 ##
----------
<a id="reference-1"></a>

* cursor 파라미터

    - arrow, based_arrow_down, based_arrow_up, boat, bogosity, bottom_left_corner, bottom_right_corner, bottom_side, bottom_tee, box_spiral, center_ptr, circle, clock,	coffee_mug, cross, cross_reverse, crosshair, diamond_cross, dot, dotbox, double_arrow, draft_large, draft_small, draped_box, exchange, fleur, gobbler, gumby, hand1, hand2, heart, icon, iron_cross, left_ptr, left_side, left_tee, leftbutton, ll_angle, lr_angle, man, middlebutton, mouse, pencil, pirate, plus, question_arrow, right_ptr, right_side, right_tee, rightbutton, rtl_logo, sailboat, sb_down_arrow, sb_h_double_arrow, sb_left_arrow, sb_right_arrow, sb_up_arrow, sb_v_double_arrow, shuttle, sizing, spider, spraycan, star, target, tcross, top_left_arrow, top_left_corner, top_right_corner, top_side, top_tee, trek, ul_angle, umbrella, ur_angle, watch, wait, xterm, X_cursor


<a id="reference-2"></a>

* 기본 설정은 `normal` 상태의 설정을 의미함 (`bg`, `fg` 등의 설정)

<a id="reference-3"></a>

* `highlightbackground`를 설정하였을 경우, 버튼이 선택되지 않았을 때에도 두께가 표시됨

<a id="reference-4"></a>

* `repeatdelay=100` 일 경우, **누르고 있기 시작한 0.1초 후**에 `command`가 실행됨

<a id="reference-5"></a>

* `repeatdelay=1000`, `repeatinterval=100` 일 경우, **1초 후에 command가 실행되며 0.1초마다 버튼을 뗄 때까지** `command`가 계속 실행됨

* `repeatinterval` 파라미터는 `repeatdelay` 파라미터와 **같이** 사용해야함


