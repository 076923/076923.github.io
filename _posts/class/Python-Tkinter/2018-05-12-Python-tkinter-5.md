---
layout: post
title: "Python tkinter 강좌 : 제 5강 – Listbox"
tagline: "Python tkinter Listbox"
image: /assets/images/tkinter.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-Tkinter']
keywords: Python, Python tkinter, tkinter Listbox
ref: Python-Tkinter
category: posts
permalink: /posts/Python-tkinter-5/
comments: true
---

## Listbox(리스트박스) ##
----------

`Listbox`을 이용하여 목록을 불러와 `추가`, `제거` 또는 `선택`하기 위한 `리스트박스`를 생성할 수 있습니다

<br>
<br>

## Listbox 사용 ##
----------

{% highlight Python %}

import tkinter

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x480+100+100")
window.resizable(False, False)

listbox = tkinter.Listbox(window, selectmode='extended', height=0)
listbox.insert(0, "1번")
listbox.insert(1, "2번")
listbox.insert(2, "2번")
listbox.insert(3, "2번")
listbox.insert(4, "3번")
listbox.delete(1, 2)
listbox.pack()

window.mainloop()
{% endhighlight %}

<br>

{% highlight Python %}

listbox = tkinter.Listbox(window, selectmode='extended', height=0)
listbox.insert(0, "1번")
listbox.insert(1, "2번")
listbox.insert(2, "2번")
listbox.insert(3, "2번")
listbox.insert(4, "3번")
listbox.delete(1, 2)
listbox.pack()

{% endhighlight %}


`tkinter.Listbox(윈도우 창, 파라미터1, 파라미터2, 파라미터3, ...)`을 사용하여 해당 `윈도우 창`에 표시할 `리스트박스의 속성`을 설정할 수 있습니다.

`파라미터`를 사용하여 `리스트박스의 속성`을 설정합니다.

`리스트박스.insert(index, "항목")`을 통하여 항목을 `추가`할 수 있습니다.

`리스트박스.delete(start_index, end_index)`를 통하여 항목을 `삭제`할 수 있습니다.

* Tip : `리스트박스.delete(index)`를 통하여 `단일 항목`만 `삭제`할 수 있습니다.

<br>
<br>

## Listbox Method ##
----------

|               이름               |                            의미                            |
|:--------------------------------:|:----------------------------------------------------------:|
|       insert(index, “항목”)      |                  `index` 위치에 항목 추가                  |
| delete(start_index,   end_index) |        `start_index`부터 `end_index`까지의 항목 삭제       |
|              size()              |                       항목 개수 반환                       |
|          activate(index)         |                  `index` 위치에 항목 선택                  |
|          curselection()          |                  선택된 항목을 튜플로 반환                 |
|    get(start_index, end_index)   |   `start_index`부터 `end_index`까지의 항목을 튜플로 반환   |
|           index(index)           |                `index`에 대응하는 위치 획득                |
|            nearest(y)            | 현재 보이는 리스트박스의 항목 중 y에 가장 가까운 값을 반환 |
|            see(index)            |          `index`가 보이도록 리스트박스의 위치 조정         |
|              xview()             |                       가로스크롤 연결                      |
|      xview_scroll(num, str)      |                   가로스크롤의 속성 설정                   |
|         xview_moveto(num)        |                    가로스크롤 이동 (0~1)                   |
|              yview()             |                       세로스크롤 연결                      |
|      yview_scroll(num, str)      |                   세로스크롤의 속성 설정                   |
|         yview_moveto(num)        |                    세로스크롤 이동 (0~1)                   |

<br>

* xview_scroll

    * `num` : 스크롤 이동 횟수
    
    * `str`
    
        * `units` : 문자 너비로 스크롤
        * `pages` : 위젯 너비로 스크롤
        
<br>
        
* yview_scroll

    * `num` : 스크롤 이동 횟수
    
    * `str`
    
      * `units` : 줄 높이로 스크롤
      * `pages` : 위젯 높이로 스크롤
        <br>
<br>
<br>

## Listbox Parameter ##
----------

## 리스트박스 문자열 설정 ##

|     이름     |                    의미                   | 기본값 |                 속성                |
|:------------:|:-----------------------------------------:|:------:|:-----------------------------------:|
| listvariable |     리스트박스에 표시할 문자열을 가져올 변수    |    -   |                  -                  |

<br>
<br>

## 리스트박스 형태 설정 ##


|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
|      width     |            리스트박스의 너비           |         20        |                    [상수](#reference-1)                     |
|      height     |            리스트박스의 높이           |         10        |                    [상수](#reference-1)                    |
|     relief     |        리스트박스의 테두리 모양        |       flat       | flat, groove, raised, ridge, solid, sunken |
| borderwidth=bd |        리스트박스의 테두리 두께        |         2        |                    상수                    |
|  background=bg |           리스트박스의 배경 색상          | SystemButtonFace |                    color                   |
|  foreground=fg |          리스트박스의 문자열 색상         | SystemButtonFace |                    color                   |
|  selectbackground |          리스트박스 항목의 블록처리 배경 색상         | SystemHighlight |                    color                   |
|  selectforeground |          리스트박스 항목의 블록처리 문자열 색상         | SystemHighlight |                    color                   |
|  selectborderwidth |          리스트박스 항목의 블록처리 테두리 두께     | 0 |                    상수                   |

<br>
<br>

## 리스트박스 형식 설정 ##


|   이름   |                           의미                          |     기본값    |                                          속성                                          |
|:--------:|:-------------------------------------------------------:|:-------------:|:--------------------------------------------------------------------------------------:|
|   font   |                리스트박스의 문자열 글꼴 설정               | TkDefaultFont |                                          font                                          |
|  cursor  |                 리스트박스의 마우스 커서 모양                 |       -       |                                    [커서 속성](#reference-2)                                   |
|   xscrollcommand   |                리스트박스의 가로스크롤 위젯 적용               | - |                                          Scrollbar위젯.set                                          |
|   yscrollcommand   |                리스트박스의 세로스크롤 위젯 적용               | - |                                          Scrollbar위젯.set                                          |
|  exportselection |                 리스트박스간의 항목 선택 유지                 |      True        |                       [Boolean](#reference-3)                           |
|   setgrid   |                리스트박스의 격자 크기 조정 설정               | False |                                          Boolean                                        |


<br>
<br>

## 리스트박스 상태 설정 ##


|        이름        |                   의미                   |       기본값       |           속성           |
|:------------------:|:----------------------------------------:|:------------------:|:------------------------:|
|        state       |                 상태 설정                 |       normal       | [normal](#reference-4), readonly, disabled |
| disabledforeground | disabeld 상태일 때 리스트박스의 문자열 색상 | SystemDisabledText |           color          |

<br>
<br>

## 리스트박스 하이라이트 설정 ##


|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    highlightcolor   |    리스트박스가이 선택되었을 때 색상   | SystemWindowFrame |  color  |
| highlightbackground | 리스트박스가이 선택되지 않았을 때 색상 |  SystemButtonFace |  color  |
|  highlightthickness |    리스트박스가이 선택되었을 때 두께 [(두께 설정)](#reference-5)     |         0         | 상수 |

<br>
<br>

## 리스트박스 동작 설정 ##


|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    takefocus |    Tab 키를 이용하여 위젯 이동 허용 여부  | True |  Boolean |
|    activestyle |    리스트박스의 선택된 항목의 표시 형태  | underline |  [underline, none, dotbox](#reference-6) |
| selectmode | 리스트박스의 항목 선택 방법  |  browse |  [browse, single, mulitple, extended](#reference-7) |


<br>
<br>

## 참고 ##
----------

<a id="reference-1"></a>

* `width`, `height`에 `0`을 입력 시, **항목에 맞춰 크기가 설정됨**

<a id="reference-2"></a>

* cursor 파라미터

    - arrow, based_arrow_down, based_arrow_up, boat, bogosity, bottom_left_corner, bottom_right_corner, bottom_side, bottom_tee, box_spiral, center_ptr, circle, clock,	coffee_mug, cross, cross_reverse, crosshair, diamond_cross, dot, dotbox, double_arrow, draft_large, draft_small, draped_box, exchange, fleur, gobbler, gumby, hand1, hand2, heart, icon, iron_cross, left_ptr, left_side, left_tee, leftbutton, ll_angle, lr_angle, man, middlebutton, mouse, pencil, pirate, plus, question_arrow, right_ptr, right_side, right_tee, rightbutton, rtl_logo, sailboat, sb_down_arrow, sb_h_double_arrow, sb_left_arrow, sb_right_arrow, sb_up_arrow, sb_v_double_arrow, shuttle, sizing, spider, spraycan, star, target, tcross, top_left_arrow, top_left_corner, top_right_corner, top_side, top_tee, trek, ul_angle, umbrella, ur_angle, watch, wait, xterm, X_cursor

<a id="reference-3"></a>

* `exportselection`가 `True`일 때 리스트박스가 2개 이상일 경우 1번 리스트 박스에서 항목 선택 후, 2번 리스트 박스에서 항목을 선택하면 1번 **리스트 박스의 항목 선택 상태가 해제된다.** 항목 선택을 유지하고자 할 때 `False`로 사용

<a id="reference-4"></a>

* 기본 설정은 `normal` 상태의 설정을 의미함 (`bg`, `fg` 등의 설정)

<a id="reference-5"></a>

* `highlightbackground`를 설정하였을 경우, 리스트박스가 선택되지 않았을 때에도 두께가 표시됨

<a id="reference-6"></a>

* activestyle 파리미터
    - dotbox : 선택된 항목에 점선 테두리 적용
    - underline : 선택된 항목의 문자열에 밑줄 적용
    - none : 선택된 항목에 블럭처리만 적용

<a id="reference-7"></a>

* selectmode 파라미터
    - browse : 단일 선택, 방향키 이동 시 선택
    - single : 단일 선택, 방향키 이동 후 스페이스바로 선택
    - multiple : 다중 선택, 마우스 클릭이나 방향키 이동 후 스페이스바로 선택
    - extended : 다중 선택, 마우스 드래그나 쉬프트키 + 방향키 이동으로 선택





