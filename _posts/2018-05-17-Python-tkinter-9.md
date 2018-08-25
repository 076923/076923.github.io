---
bg: "python.jpg"
layout: post
comments: true
title: "Python tkinter 강좌 : 제 9강 – Menubutton"
crawlertitle: "Python tkinter 강좌 : 제 9강 - Menubutton"
summary: "Python tkinter Menubutton"
date: 2018-05-17
categories: posts
tags: ['Python-tkinter']
author: 윤대희
star: true
---

### Menubutton (메뉴버튼) ###
----------
`Menubutton`을 이용하여 `메뉴`기능을 가진 `단추`를 생성할 수 있습니다.

<br>
<br>
### Menubutton 사용 ###
----------
{% highlight Python %}
import tkinter

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

menubutton=tkinter.Menubutton(window,text="메뉴 메뉴버튼", relief="raised", direction="right")
menubutton.pack()

menu=tkinter.Menu(menubutton, tearoff=0)
menu.add_command(label="하위메뉴-1")
menu.add_separator()
menu.add_command(label="하위메뉴-2")
menu.add_command(label="하위메뉴-3")

menubutton["menu"]=menu

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

menubutton=tkinter.Menubutton(window,text="메뉴 메뉴버튼", relief="raised", direction="right")
menubutton.pack()

menu=tkinter.Menu(menubutton, tearoff=0)
menu.add_command(label="하위메뉴-1")
menu.add_separator()
menu.add_command(label="하위메뉴-2")
menu.add_command(label="하위메뉴-3")

menubutton["menu"]=menu

{% endhighlight %}


`tkinter.Menubutton(윈도우 창, 파라미터1, 파라미터2, 파라미터3, ...)`을 사용하여 해당 `윈도우 창`에 표시할 `메뉴버튼의 속성`을 설정할 수 있습니다.

`파라미터`를 사용하여 `메뉴버튼의 속성`을 설정합니다.

이 후, `tkinter.Menu(메뉴버튼 이름, 파라미터1, 파라미터2, 파라미터3, ...)`을 사용하여 `메뉴의 속성`을 설정할 수 있습니다.

마지막으로 메뉴버튼의 파라미터 중 `menu`를 마지막에 사용하여 `메뉴버튼과 메뉴`를 연결합니다.

<br>
<br>
### Menubutton Parameter ###
----------

##### 메뉴버튼 문자열 설정 #####

|     이름     |                    의미                   | 기본값 |                 속성                |
|:------------:|:-----------------------------------------:|:------:|:-----------------------------------:|
|     text     |            메뉴버튼에 표시할 문자열           |    -   |                  -                  |
| textvariable |     메뉴버튼에 표시할 문자열을 가져올 변수    |    -   |                  -                  |
|    anchor    |     메뉴버튼안의 문자열 또는 이미지의 위치    | center | n, ne, e, se, s, sw, w, nw, center  |
|    justify   | 메뉴버튼의 문자열이 여러 줄 일 경우 정렬 방법 | center |         center, left, right         |
|  wraplength  |           자동 줄내림 설정 너비           |    0   |                 상수                |

<br>
<br>

##### 메뉴버튼 형태 설정 #####


|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
|      width     |            메뉴버튼의 너비           |         0        |                    상수                    |
|     height     |            메뉴버튼의 높이           |         0        |                    상수                    |
|     relief     |        메뉴버튼의 테두리 모양        |       flat       | flat, groove, raised, ridge, solid, sunken |
|  background=bg |           메뉴버튼의 배경 색상        | SystemButtonFace |                    color                 |
|  foreground=fg |          메뉴버튼의 문자열 색상         | SystemButtonFace |                    color                   |
|      padx      | 메뉴버튼의 테두리와 내용의 가로 여백 |         5        |                    상수                    |
|      pady      | 메뉴버튼의 테두리와 내용의 세로 여백 |         4        |                    상수                    |

<br>
<br>

##### 메뉴버튼 형식 설정 #####


|   이름   |                           의미                          |     기본값    |                                          속성                                          |
|:--------:|:-------------------------------------------------------:|:-------------:|:--------------------------------------------------------------------------------------:|
|  bitmap  |                메뉴버튼에 포함할 기본 이미지                |       -       | info, warring, error, question,   questhead, hourglass, gray12, gray25, gray50, gray75 |
|   image  |                메뉴버튼에 포함할 임의 이미지                |       -       |                                            -                                           |
| compound | 메뉴버튼에 문자열과 이미지를 동시에 표시할 때 이미지의 위치 |      none     |                         bottom, center, left, none, right, top                         |
|   font   |                메뉴버튼의 문자열 글꼴 설정               | TkDefaultFont |                                          font                                          |
|  cursor  |                 메뉴버튼의 마우스 커서 모양                 |       -       |                                    [커서 속성](#reference-1)                                   |

<br>
<br>

##### 메뉴버튼 상태 설정 #####


|        이름        |                   의미                   |       기본값       |           속성           |
|:------------------:|:----------------------------------------:|:------------------:|:------------------------:|
|        state       |                 상태 설정                 |       normal       | [normal](#reference-2), active, disabled |
|  activebackground  |   active 상태일 때 메뉴버튼의 배경 색상   |  SystemButtonFace  |           color          |
|  activeforeground  |  active 상태일 때 메뉴버튼의 문자열 색상  |  SystemButtonText  |           color          |
| disabledforeground | disabeld 상태일 때 메뉴버튼의 문자열 색상 | SystemDisabledText |           color          |

<br>
<br>

##### 메뉴버튼 하이라이트 설정 #####


|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    highlightcolor   |    메뉴버튼이 선택되었을 때 색상   | SystemWindowFrame |  color  |
| highlightbackground | 메뉴버튼이 선택되지 않았을 때 색상 |  SystemButtonFace |  color  |
|  highlightthickness |    메뉴버튼이 선택되었을 때 두께 [(두께 설정)](#reference-3)     |         0         | 상수 |

<br>
<br>

##### 메뉴버튼 동작 설정 #####


|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    takefocus |    Tab 키를 이용하여 위젯 이동 허용 여부  | True |  Boolean |
|    menu  |   메뉴버튼이 선택되었을 때 나타나는 메뉴 위젯  | -  |  tktiner.Menu() |
|    direction  |   메뉴버튼이 선택되었을 때 나타나는 메뉴 위젯의 방향  | below |  above, below, left, right, flush |
|    indicatoron |    메뉴버튼의 위젯 일치화 여부 | False |  [Boolean](#reference-4) |

<br>
<br>

##### 참고 #####
----------
<a id="reference-1"></a>

* cursor 파라미터

    - arrow, based_arrow_down, based_arrow_up, boat, bogosity, bottom_left_corner, bottom_right_corner, bottom_side, bottom_tee, box_spiral, center_ptr, circle, clock,	coffee_mug, cross, cross_reverse, crosshair, diamond_cross, dot, dotbox, double_arrow, draft_large, draft_small, draped_box, exchange, fleur, gobbler, gumby, hand1, hand2, heart, icon, iron_cross, left_ptr, left_side, left_tee, leftbutton, ll_angle, lr_angle, man, middlebutton, mouse, pencil, pirate, plus, question_arrow, right_ptr, right_side, right_tee, rightbutton, rtl_logo, sailboat, sb_down_arrow, sb_h_double_arrow, sb_left_arrow, sb_right_arrow, sb_up_arrow, sb_v_double_arrow, shuttle, sizing, spider, spraycan, star, target, tcross, top_left_arrow, top_left_corner, top_right_corner, top_side, top_tee, trek, ul_angle, umbrella, ur_angle, watch, wait, xterm, X_cursor


<a id="reference-2"></a>

* 기본 설정은 `normal` 상태의 설정을 의미함 (`bg`, `fg` 등의 설정)


<a id="reference-3"></a>

* `highlightbackground`를 설정하였을 경우, 메뉴버튼이 선택되지 않았을 때에도 두께가 표시됨

<a id="reference-4"></a>

* `indicatoron`을 True로 설정할 경우, 메뉴버튼에 메뉴 기호가 나타남


