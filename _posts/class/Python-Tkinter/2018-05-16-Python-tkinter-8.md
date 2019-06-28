---
layout: post
title: "Python tkinter 강좌 : 제 8강 – Menu"
tagline: "Python tkinter Menu"
image: /assets/images/tkinter.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-Tkinter']
keywords: Python, Python tkinter, tkinter Menu
ref: Python
category: posts
permalink: /posts/Python-tkinter-8/
comments: true
---

## Menu(메뉴) ##
----------

`Menu`을 이용하여 `자주 사용하는 기능` 등을 **다양한 선택사항으로 나누는** `메뉴`을 생성할 수 있습니다.

<br>
<br>

## Menu 사용 ##
----------

{% highlight Python %}

import tkinter

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x480+100+100")
window.resizable(False, False)

def close():
    window.quit()
    window.destroy()

menubar=tkinter.Menu(window)

menu_1=tkinter.Menu(menubar, tearoff=0)
menu_1.add_command(label="하위 메뉴 1-1")
menu_1.add_command(label="하위 메뉴 1-2")
menu_1.add_separator()
menu_1.add_command(label="하위 메뉴 1-3", command=close)
menubar.add_cascade(label="상위 메뉴 1", menu=menu_1)

menu_2=tkinter.Menu(menubar, tearoff=0, selectcolor="red")
menu_2.add_radiobutton(label="하위 메뉴 2-1", state="disable")
menu_2.add_radiobutton(label="하위 메뉴 2-2")
menu_2.add_radiobutton(label="하위 메뉴 2-3")
menubar.add_cascade(label="상위 메뉴 2", menu=menu_2)

menu_3=tkinter.Menu(menubar, tearoff=0)
menu_3.add_checkbutton(label="하위 메뉴 3-1")
menu_3.add_checkbutton(label="하위 메뉴 3-2")
menubar.add_cascade(label="상위 메뉴 3", menu=menu_3)

window.config(menu=menubar)

window.mainloop()

print("Window Close")

{% endhighlight %}

<br>

{% highlight Python %}

def close():
    window.quit()
    window.destroy()

menubar=tkinter.Menu(window)

menu_1=tkinter.Menu(menubar, tearoff=0)
menu_1.add_command(label="하위 메뉴 1-1")
menu_1.add_command(label="하위 메뉴 1-2")
menu_1.add_separator()
menu_1.add_command(label="하위 메뉴 1-3", command=close)
menubar.add_cascade(label="상위 메뉴 1", menu=menu_1)

menu_2=tkinter.Menu(menubar, tearoff=0, selectcolor="red")
menu_2.add_radiobutton(label="하위 메뉴 2-1", state="disable")
menu_2.add_radiobutton(label="하위 메뉴 2-2")
menu_2.add_radiobutton(label="하위 메뉴 2-3")
menubar.add_cascade(label="상위 메뉴 2", menu=menu_2)

menu_3=tkinter.Menu(menubar, tearoff=0)
menu_3.add_checkbutton(label="하위 메뉴 3-1")
menu_3.add_checkbutton(label="하위 메뉴 3-2")
menubar.add_cascade(label="상위 메뉴 3", menu=menu_3)

window.config(menu=menubar)

window.mainloop()

print("Window Close")

{% endhighlight %}


`메뉴 이름=tkinter.Menu(윈도우 창)`을 사용하여 해당 `윈도우 창`에 `메뉴`를 사용할 수 있습니다.

`상위 메뉴 이름=tkinter.Menu(메뉴 이름, 파라미터1, 파라미터2, 파라미터3, ...)`을 사용하여 해당 `메뉴창`에 표시할 `상위 메뉴의 속성`을 설정할 수 있습니다.

`파라미터`를 사용하여 `상위 메뉴의 속성`을 설정합니다.

`상위 메뉴 이름.메소드(파라미터1, 파라미터2, 파라미터3, ...)`를 사용하여 `메소드`에 해당하는 `하위 메뉴`를 추가할 수 있습니다.

`파라미터`를 사용하여 `하위 메뉴의 속성`을 설정합니다.

`윈도우 창.config(menu=메뉴 이름)`을 통하여 해당 `윈도우 창`에 `메뉴`를 등록할 수 있습니다.

`window.quit()`는 위젯이 유지된 채 `window.mainloop()` 이후의 코드를 실행시킵니다.

`window.destroy()`는 위젯을 파괴하고 `window.mainloop()` 이후의 코드를 실행시킵니다.

<br>
<br>

## Menu Method ##
----------

|              이름              |                                       의미                                      |
|:------------------------------:|:-------------------------------------------------------------------------------:|
|      add_command(파라미터)     |                               기본 메뉴 항목 생성                               |
|    add_radiobutton(파라미터)   |                            라디오버튼 메뉴 항목 생성                            |
|    add_checkbutton(파라미터)   |                             체크버튼 메뉴 항목 생성                             |
|      add_cascade(파라미터)     |                           상위 메뉴와 하위 메뉴 연결                            |
|         add_separator()        |                                   구분선 생성                                   |
|       add(유형, 파라미터)      |                           `특정 유형`의 메뉴 항목 생성                          |
| delete(start_index, end_index) |                  `start_index`부터 `end_index`까지의 항목 삭제                  |
|  entryconfig(index, 파라미터)  |                          `index` 위치의 메뉴 항목 수정                          |
|           index(item)          |                       `item` 메뉴 항목의 `index` 위치 반환                      |
|    insert_separator (index)    |                            `index` 위치에 구분선 생성                           |
|          invoke(index)         |                             `index` 위치의 항목 실행                            |
|           type(속성)           | 선택 유형 반환 (command, radiobutton, checkbutton, cascade, separator, tearoff) |

<br>

* Tip : 파라미터 중, : `label=이름`을 이용하여 메뉴의 이름을 설정할 수 있습니다.
* Tip : `메뉴 이름.add_cascade(label="상위 메뉴 이름", menu=연결할 상위 메뉴)`를 이용하여 메뉴를 부착할 수 있습니다.

<br>
<br>

## Menu Parameter ##
----------

## 메뉴 형태 설정 ##


|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
|     relief     |        메뉴의 테두리 모양        |       flat       | flat, groove, raised, ridge, solid, sunken |
|  background=bg |           메뉴의 배경 색상        | SystemButtonFace |                    color                 |
|  foreground=fg |          메뉴의 문자열 색상         | SystemButtonFace |                    color                   |
|  selectcolor |          하위 메뉴의 선택 표시(√) 색상         | SystemWindow |                    color                   |

<br>
<br>

## 메뉴 형식 설정 ##


|   이름   |                           의미                          |     기본값    |                                          속성                                          |
|:--------:|:-------------------------------------------------------:|:-------------:|:--------------------------------------------------------------------------------------:|
|   font   |                메뉴의 문자열 글꼴 설정               | TkDefaultFont |                                          font                                          |
|  cursor  |                 메뉴의 마우스 커서 모양                 |       -       |                                    [커서 속성](#reference-1)                                   |

<br>
<br>

## 메뉴 상태 설정 ##


|        이름        |                   의미                   |       기본값       |           속성           |
|:------------------:|:----------------------------------------:|:------------------:|:------------------------:|
|  activeborderwidth |   active 상태일 때 메뉴의 테두리 두께   |  1   |           상수          |
|  activebackground  |   active 상태일 때 메뉴의 배경 색상   |  SystemHighlight |           color          |
|  activeforeground  |  active 상태일 때 메뉴의 문자열 색상  |  SystemButtonText  |           color          |
| disabledforeground | disabeld 상태일 때 메뉴의 문자열 색상 | SystemDisabledText |           color          |

<br>
<br>

## 메뉴 동작 설정 ##


|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    postcommand |    메뉴가 선택되었을 때 실행하는 메소드(함수)  | - |  메소드, 함수 |
|    tearoff |   하위메뉴의 분리 기능 사용 유/무   | False |  Boolean |
|    title |     하위메뉴의 분리 기능의 제목 | - | 문자열 |
|    tearoffcommand |    메뉴의 위젯 일치화 여부 | - |  메소드, 함수 |

<br>
<br>

## 참고 ##
----------

<a id="reference-1"></a>

* cursor 파라미터

    - arrow, based_arrow_down, based_arrow_up, boat, bogosity, bottom_left_corner, bottom_right_corner, bottom_side, bottom_tee, box_spiral, center_ptr, circle, clock,	coffee_mug, cross, cross_reverse, crosshair, diamond_cross, dot, dotbox, double_arrow, draft_large, draft_small, draped_box, exchange, fleur, gobbler, gumby, hand1, hand2, heart, icon, iron_cross, left_ptr, left_side, left_tee, leftbutton, ll_angle, lr_angle, man, middlebutton, mouse, pencil, pirate, plus, question_arrow, right_ptr, right_side, right_tee, rightbutton, rtl_logo, sailboat, sb_down_arrow, sb_h_double_arrow, sb_left_arrow, sb_right_arrow, sb_up_arrow, sb_v_double_arrow, shuttle, sizing, spider, spraycan, star, target, tcross, top_left_arrow, top_left_corner, top_right_corner, top_side, top_tee, trek, ul_angle, umbrella, ur_angle, watch, wait, xterm, X_cursor



