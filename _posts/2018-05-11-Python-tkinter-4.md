---
bg: "python.jpg"
layout: post
comments: true
title: "Python tkinter 강좌 : 제 4강 – Entry"
crawlertitle: "Python tkinter 강좌 : 제 4강 - Entry"
summary: "Python tkinter Entry"
date: 2018-05-11
categories: posts
tags: ['Python-tkinter']
author: 윤대희
star: true
---

### Entry (기입창) ###
----------
`Entry`을 이용하여 텍스트를 `입력`받거나 `출력`하기 위한 `기입창`을 생성할 수 있습니다

<br>
<br>
### Entry 사용 ###
----------
{% highlight Python %}

import tkinter
from math import*

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x480+100+100")
window.resizable(False, False)

def calc(event):
    label.config(text="결과="+str(eval(entry.get())))

entry=tkinter.Entry(window)
entry.bind("<Return>", calc)
entry.pack()

label=tkinter.Label(window)
label.pack()

window.mainloop()
{% endhighlight %}

<br>

{% highlight Python %}

def calc(event):
    label.config(text="결과="+str(eval(entry.get())))

entry=tkinter.Entry(window)
entry.bind("<Return>", calc)
entry.pack()

label=tkinter.Label(window)
label.pack()

{% endhighlight %}


`tkinter.Entry(윈도우 창, 파라미터1, 파라미터2, 파라미터3, ...)`을 사용하여 해당 `윈도우 창`에 표시할 `기입창의 속성`을 설정할 수 있습니다.

`파라미터`를 사용하여 `기입창의 속성`을 설정합니다.

`Entry.bind()`를 이용하여 `key`나 `mouse` 등의 `event`를 처리하여 `메소드`나 `함수`를 실행시킬 수 있습니다.

`기입창`에 간단한 `수학함수` 등을 작성 후 `Enter 키`를 입력시, `라벨`에 결과가 표시됩니다.

<br>

* Tip : `4+4*cos(0.5)`을 입력시 `결과=7.51033...`이 반환됩니다.


<br>
<br>
### Entry Method###
----------

|                  이름                  |                              의미                             |
|:--------------------------------------:|:-------------------------------------------------------------:|
|        insert(index, "문자열")       |                  `index` 위치에 `문자열` 추가                 |
|     delete(start_index, end_index)     |        `start_index`부터 `end_index`까지의 문자열 삭제        |
|                  get()                 |                기입창의 텍스트를 문자열로 반환                |
|              index(index)              |                  `index`에 대응하는 위치 획득                 |
|             icursor(index)             |                 `index` 앞에 키보드 커서 설정                 |
|          select_adjust(index)          |              `index` 위치까지의 문자열을 블록처리             |
| select_range(start_index,   end_index) |           `start_index`부터 `end_index`까지 블록처리          |
|            select_to(index)            |              키보드 커서부터 `index`까지 블록처리             |
|           select_from(index)           | 키보드 커서의 색인 위치를 `index` 위치에 문자로 설정하고 선택 |
|            select_present()            |        블록처리 되어있는 경우 `True`, 아닐 경우 `False`       |
|             select_clear()             |                         블록처리 해제                         |
|                 xview()                |                        가로스크롤 연결                       |
|         xview_scroll(num, str)         |                       가로스크롤의 속성 설정              |

<br>

* xview_scroll

    - `num`
        - num > 0 : 왼쪽에서 오른쪽으로 스크롤
        - num < 0 : 오른쪽에서 왼쪽으로 스크롤
        
    - `str`
        - `units` : 문자 너비로 스크롤
        - `pages` : 위젯 너비로 스크롤
        <br>
        
<br>
<br>

### Entry Parameter ###
----------

##### 기입창 문자열 설정 #####

|     이름     |                    의미                   | 기본값 |                 속성                |
|:------------:|:-----------------------------------------:|:------:|:-----------------------------------:|
|     show     |            기입창에 표시되는 문자           |    -   |                  [문자](#reference-1)                  |
| textvariable |     기입창에 표시할 문자열을 가져올 변수    |    -   |                  -                  |
|    justify   | 기입창의 문자열이 여러 줄 일 경우 정렬 방법 | center |         center, left, right         |

<br>
<br>

##### 기입창 형태 설정 #####


|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
|      width     | 기입창의 너비 | 0 | 상수 |
|     relief     | 기입창의 테두리 모양 | flat | flat, groove, raised, ridge, solid, sunken |
| borderwidth=bd | 기입창의 테두리 두께 | 2 | 상수 |
|  background=bg | 기입창의 배경 색상 | SystemButtonFace | color |
|  foreground=fg | 기입창의 문자열 색상 | SystemButtonFace | color |
| insertwidth | 기입창의 키보드 커서 너비 | 2 | 상수 |
| insertborderwidth | 기입창의 키보드 커서 테두리 두께 | 0 | 상수 |
| insertbackground | 기입창의 키보드 커서 색상 | SystemWindowText | color |
| selectborderwidth | 기입창의 문자열 블록처리 테두리 두께 | 0 | 상수 |
| selectbackground | 기입창의 문자열 블록처리 배경 색상 | SystemHighlight | color |
| selectforeground | 기입창의 문자열 블록처리 문자열 색상 | SystemHighlight | color |


<br>
<br>

##### 기입창 형식 설정 #####


|   이름   |                           의미                          |     기본값    |                                          속성                                          |
|:--------:|:-------------------------------------------------------:|:-------------:|:--------------------------------------------------------------------------------------:|
|   font   | 기입창의 문자열 글꼴 설정 | TkDefaultFont | font |
|  cursor  | 기입창의 마우스 커서 모양 | - | [커서 속성](#reference-2) |
|   xscrollcommand   | 기입창의 가로스크롤 위젯 적용 | - | Scrollbar위젯.set |

<br>
<br>

##### 기입창 상태 설정 #####


|        이름        |                   의미                   |       기본값       |           속성           |
|:------------------:|:----------------------------------------:|:------------------:|:------------------------:|
|        state       |                 상태 설정                 |       normal       | [normal](#reference-3), readonly, disabled |
|  readonlybackground |   readonly 상태일 때 기입창의 배경 색상   |  SystemButtonFace  |           color          |
|  disabledbackground |  disabeld 상태일 때 기입창의 배경 색상  |  SystemButtonFace  |           color          |
| disabledforeground | disabeld 상태일 때 기입창의 문자열 색상 | SystemDisabledText |           color          |

<br>
<br>

##### 기입창 하이라이트 설정 #####


|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    highlightcolor   |    기입창이 선택되었을 때 색상   | SystemWindowFrame |  color  |
| highlightbackground | 기입창이 선택되지 않았을 때 색상 |  SystemButtonFace |  color  |
|  highlightthickness |    기입창이 선택되었을 때 두께 [(두께 설정)](#reference-4)     |         0         | 상수 |

<br>
<br>

##### 기입창 동작 설정 #####


|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    takefocus |    Tab 키를 이용하여 위젯 이동 허용 여부  | True |  Boolean |
|    insertontime |    기입창의 키보드 커서 깜빡임이 보이는 시간   | 600  |  상수(ms) |
| insertofftime | 기입창의 키보드 커서 깜빡임이 보이지 않는 시간  |  300 |  상수(ms) |


<br>
<br>

##### 참고 #####
----------
<a id="reference-1"></a>

* `*` 입력 시, 입력되는 모든 문자는 `*` 처리되어 표시됨


<a id="reference-2"></a>

* cursor 파라미터

    - arrow, based_arrow_down, based_arrow_up, boat, bogosity, bottom_left_corner, bottom_right_corner, bottom_side, bottom_tee, box_spiral, center_ptr, circle, clock,	coffee_mug, cross, cross_reverse, crosshair, diamond_cross, dot, dotbox, double_arrow, draft_large, draft_small, draped_box, exchange, fleur, gobbler, gumby, hand1, hand2, heart, icon, iron_cross, left_ptr, left_side, left_tee, leftbutton, ll_angle, lr_angle, man, middlebutton, mouse, pencil, pirate, plus, question_arrow, right_ptr, right_side, right_tee, rightbutton, rtl_logo, sailboat, sb_down_arrow, sb_h_double_arrow, sb_left_arrow, sb_right_arrow, sb_up_arrow, sb_v_double_arrow, shuttle, sizing, spider, spraycan, star, target, tcross, top_left_arrow, top_left_corner, top_right_corner, top_side, top_tee, trek, ul_angle, umbrella, ur_angle, watch, wait, xterm, X_cursor


<a id="reference-3"></a>

* 기본 설정은 `normal` 상태의 설정을 의미함 (`bg`, `fg` 등의 설정)


<a id="reference-4"></a>

* `highlightbackground`를 설정하였을 경우, 기입창이 선택되지 않았을 때에도 두께가 표시됨

