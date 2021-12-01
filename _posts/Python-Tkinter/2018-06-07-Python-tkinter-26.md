---
layout: post
title: "Python tkinter 강좌 : 제 26강 - Combobox"
tagline: "Python tkinter Combobox"
image: /assets/images/tkinter.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Tkinter']
keywords: Python, Python tkinter, tkinter Combobox
ref: Python-Tkinter
category: Python
permalink: /posts/Python-tkinter-26/
comments: true
toc: true
---

## Combobox(콤보 박스)

`Combobox`을 이용하여 **텍스트와 허용된 값의 드롭다운 목록**을 표시하는 `콤보 박스`를 생성할 수 있습니다.

<br>
<br>

## Combobox 사용

{% highlight Python %}

import tkinter
import tkinter.ttk

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

values=[str(i)+"번" for i in range(1, 101)] 

combobox=tkinter.ttk.Combobox(window, height=15, values=values)
combobox.pack()

combobox.set("목록 선택")

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

import tkinter.ttk

{% endhighlight %}

<br>

상단에 `import tkinter.ttk`를 사용하여 `ttk 모듈`을 포함시킵니다.

tkinter.ttk 함수의 사용방법은 `tkinter.ttk.*`를 이용하여 사용이 가능합니다.

<br>

{% highlight Python %}

values=[str(i)+"번" for i in range(1, 101)] 

combobox=tkinter.ttk.Combobox(window, height=15, values=values)
combobox.pack()

combobox.set("목록 선택")

{% endhighlight %}

`tkinter.ttk.Combobox(윈도우 창, 매개변수1, 매개변수2, 매개변수3, ...)`을 사용하여 `콤보 박스의 속성`을 설정할 수 있습니다.

`매개변수`를 사용하여 `콤보 박스의 속성`을 설정합니다.

<br>
<br>

## Combobox Method

|              이름              |       의미       |                       설명                      |
|:------------------------------:|:----------------:|:-----------------------------------------------:|
|     set("문자열")    |    표시값 변경 |       콤보 박스의 현재 텍스트 변경       |
|     get()    |    표시값 반환 |       콤보 박스의 현재 텍스트 반환       |
| current(index) |    목록 표시   | 해당 `index`의 목록 표시 |

<br>
<br>

## Combobox Parameter

### 콤보 박스 텍스트 설정

|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
| textvariable | 콤보 박스에 표시할 문자열을 가져올 변수 | - | - |
| justify | 콤보 박스의 문자열이 여러 줄 일 경우 정렬 방법 | left | center, left, right |

<br>

### 콤보 박스 형태 설정

|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
| width | 콤보 박스의 너비 | 20 | 상수 |
| height | 콤보 박스의 드롭 다운 목록이 표시할 개수 | 20 | 상수 |

<br>

### 콤보 박스 형식 설정

|   이름   |                           의미                          |     기본값    |                                          속성                                          |
|:--------:|:-------------------------------------------------------:|:-------------:|:--------------------------------------------------------------------------------------:|
|  cursor  |      콤보 박스의 마우스 커서 모양                 |       -       |                                    [커서 속성](#reference-1)                                   |
|  class_  |      클래스 설정                 |       -       |      -    |  
|   xscrollcommand  |          콤보 박스의 가로스크롤 위젯 적용            | - |          Scrollbar위젯.set |
|   values |            콤보 박스의 목록 값            | - |          [list, tuple 등](#reference-2)            |
|  exportselection |     수치 조정 기압창의 선택 항목 여부 설정   |    True    |      Boolean        |

<br>

### 콤보 박스 동작 설정

|         이름        |              의미              |         기본값        | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    takefocus |    Tab 키를 이용하여 위젯 이동 허용 여부  | False |  Boolean |
|    postcommand |    콤보 박스에서 드롭다운 목록을 클릭할 때 실행하는 메서드(함수)   | - |  메서드, 함수 |
|    validate |    콤보 박스의 유효성 검사 실행 조건  | none |  [none, focus, focusin, focusout, key, all](#reference-3) |
|    validatecommand |   유효성 검사 평가 함수  | - |  함수 |
|    invalidcommand |    validateCommand가 False를 반환 할 때 실행할 함수 | - |  함수  |

<br>

<a id="reference-1"></a>

### 참고

<a id="reference-2"></a>

* cursor 매개변수

    - arrow, based_arrow_down, based_arrow_up, boat, bogosity, bottom_left_corner, bottom_right_corner, bottom_side, bottom_tee, box_spiral, center_ptr, circle, clock,	coffee_mug, cross, cross_reverse, crosshair, diamond_cross, dot, dotbox, double_arrow, draft_large, draft_small, draped_box, exchange, fleur, gobbler, gumby, hand1, hand2, heart, icon, iron_cross, left_ptr, left_side, left_tee, leftbutton, ll_angle, lr_angle, man, middlebutton, mouse, pencil, pirate, plus, question_arrow, right_ptr, right_side, right_tee, rightbutton, rtl_logo, sailboat, sb_down_arrow, sb_h_double_arrow, sb_left_arrow, sb_right_arrow, sb_up_arrow, sb_v_double_arrow, shuttle, sizing, spider, spraycan, star, target, tcross, top_left_arrow, top_left_corner, top_right_corner, top_side, top_tee, trek, ul_angle, umbrella, ur_angle, watch, wait, xterm, X_cursor

<a id="reference-3"></a>

* `values`를 `[5, 10, 50, 100]`으로 설정하였을 경우, `목록의 순서`로 드롭다운 목록 표시

* validate 매개변수

    - `none` : 콤보 박스의 유효성 검사 실행하지 않음
    - `focus` : 콤보 박스가 포커스를 받거나 잃을 때 validateCommand 실행
    - `focusin` : 콤보 박스가 포커스를 받을 때 validateCommand 실행
    - `focusout` : 콤보 박스가 포커스를 잃을 때 validateCommand 실행
    - `key` : 콤보 박스가 수정될 경우 validateCommand 실행
    - `all` : 콤보 박스의 모든 validate에 대해 validateCommand 실행
