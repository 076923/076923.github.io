---
bg: "python.jpg"
layout: post
comments: true
title: "Python tkinter 강좌 : 제 25강 – Spinbox"
crawlertitle: "Python tkinter 강좌 : 제 25강 - Spinbox"
summary: "Python tkinter Spinbox"
date: 2018-06-06
categories: posts
tags: ['Python-tkinter']
author: 윤대희
star: true
---

### Spinbox (수치 조정 기입창) ###
----------
[![1]({{ site.images }}/Python/tkinter/ch25/1.png)]({{ site.images }}/Python/tkinter/ch25/1.png)

`Spinbox`을 이용하여 **수치를 조정하고 입력**받는 `수치 조정 기입창`를 생성할 수 있습니다.

<br>
<br>
### Spinbox 사용 ###
----------
{% highlight Python %}

import tkinter

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

label=tkinter.Label(window, text="숫자를 입력하세요.", height=3)
label.pack()

def value_check(self):
    label.config(text="숫자를 입력하세요.")
    valid = False
    if self.isdigit():
        if (int(self) <= 50 and int(self) >= 0):
            valid = True
    elif self == '':
        valid = True
    return valid

def value_error(self):
    label.config(text=str(self) + "를 입력하셨습니다.\n올바른 값을 입력하세요.")

validate_command=(window.register(value_check), '%P')
invalid_command=(window.register(value_error), '%P')

spinbox=tkinter.Spinbox(window, from_ = 0, to = 50, validate = 'all', validatecommand = validate_command, invalidcommand=invalid_command)
spinbox.pack()

{% endhighlight %}

<br>

{% highlight Python %}

label=tkinter.Label(window, text="숫자를 입력하세요.", height=3)
label.pack()

def value_check(self):
    label.config(text="숫자를 입력하세요.")
    valid = False
    if self.isdigit():
        if (int(self) <= 50 and int(self) >= 0):
            valid = True
    elif self == '':
        valid = True
    return valid

def value_error(self):
    label.config(text=str(self) + "를 입력하셨습니다.\n올바른 값을 입력하세요.")

validate_command=(window.register(value_check), '%P')
invalid_command=(window.register(value_error), '%P')

spinbox=tkinter.Spinbox(window, from_ = 0, to = 50, validate = 'all', validatecommand = validate_command, invalidcommand=invalid_command)
spinbox.pack()

{% endhighlight %}


`tkinter.Spinbox(윈도우 창, 파라미터1, 파라미터2, 파라미터3, ...)`을 사용하여 `수치 조정 기입창의 속성`을 설정할 수 있습니다.

`파라미터`를 사용하여 `수치 조정 기입창의 속성`을 설정합니다.

`사용자 정의 함수`에 `래핑`하여 `콜백`을 포함하여야 `유효성 검사`를 실행할 수 있습니다.

`validate`를 이용하여 `유효성 검사`를 `반응 조건`을 설정합니다.

`validatecommand`를 이용하여 유효성을 검사하며, `invalidcommand`를 통하여 올바르지 않을 때의 함수를 실행합니다.

<br>
<br>
### Spinbox Method###
----------

##### 수치 조정 기입창 메소드 #####

|              이름              |       의미       |                       설명                      |
|:------------------------------:|:----------------:|:-----------------------------------------------:|
|     insert(index, "문자열")    |    문자열 삽입   |       해당 `index` 위치에 `문자열`을 삽입       |
| delete(start_index, end_index) |    문자열 삭제   | `start_index`부터 `end_index`까지의 문자열 삭제 |
|   get(start_index, end_index)  |    문자열 반환   | `start_index`부터 `end_index`까지의 문자열 반환 |
|          index(index)          |    인덱스 반환   |      `index`가 음수 일 경우, `1.0`으로 반환     |
|           identify(x, y)         | 요소 식별 |   `x, y` 위치의 위젯 요소 반환 (`entry`, `buttonup`, `buttondown`, `''`)  |
|           invoke(button)         | 버튼 실행 |  `buttonup` 또는 `buttondown` 실행  |

<br>
<br>

##### 수치 조정 기입창 콜백 #####

| 이름 |           의미          |                               설명                               |
|:----:|:-----------------------:|:----------------------------------------------------------------:|
|  %d  |      액션 코드 반환     | `삭제=0`, `삽입=1`, `포커스 인=textvariable`, `포커스 아웃=textvariable` |
|  %i  | 텍스트 수정 인덱스 반환 |    `삽입 또는 삭제=해당 index`, `포커스인 또는 포커스 아웃=-1`   |
|  %P  |      텍스트 부여 값     |               변경이 가능한 경우 텍스트에 부여될 값              |
|  %s  |  텍스트가 변경되기 전 값  |                 텍스트가 변경되기 전의 유효한 값                 |
|  %S  |  텍스트 변경된 후 값  |                  텍스트가 변경된 후의 유효한 값                  |
|  %v  |     validate 현재 값    |          수치 조정 기입창의 validate 파라미터의 현재 값          |
|  %V  |        콜백 확인        |               어떤 호출로 인하여 콜백되었는지 확인               |
|  %W  |       위젯의 이름       |                     호출된 위젯의 이름을 확인                    |


<br>
<br>

### Spinbox Parameter ###
----------

##### 수치 조정 기입창 텍스트 설정 #####

|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
| textvariable | 수치 조정 기입창에 표시할 문자열을 가져올 변수 | - | - |
| justify | 수치 조정 기입창의 문자열이 여러 줄 일 경우 정렬 방법 | left | center, left, right |

<br>
<br>

##### 수치 조정 기입창 형태 설정 #####


|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
| width | 수치 조정 기입창의 너비 | 20 | 상수 |
| relief | 수치 조정 기입창의 테두리 모양 | sunken | flat, groove, raised, ridge, solid, sunken |
| borderwidth=bd | 수치 조정 기입창의 테두리 두께 | 0 | 상수 |
| background=bg | 수치 조정 기입창의 배경 색상 | SystemButtonFace | color |
| foreground=fg | 수치 조정 기입창의 문자열 색상 | SystemButtonFace | color |
| insertwidth | 수치 조정 기입창의 키보드 커서 너비 | 2 | 상수 |
| insertborderwidth | 수치 조정 기입창의 키보드 커서 테두리 두께 | 0 | 상수 |
| insertbackground | 수치 조정 기입창의 키보드 커서 색상 | SystemWindowText | color |
| selectborderwidth | 수치 조정 기입창의 문자열 블록처리 테두리 두께 | 0 | 상수 |
| selectbackground | 수치 조정 기입창의 문자열 블록처리 배경 색상 | SystemHighlightText | color |
| selectforeground | 수치 조정 기입창의 문자열 블록처리 문자열 색상 | SystemHighlightText | color |
| buttonbackground | 수치 조정 기입창의 수치 조정 버튼 배경 색상 | SystemButtonFace | color |


<br>
<br>

##### 수치 조정 기입창 형식 설정 #####


|   이름   |                           의미                          |     기본값    |                                          속성                                          |
|:--------:|:-------------------------------------------------------:|:-------------:|:--------------------------------------------------------------------------------------:|
|  font   |       수치 조정 기입창의 문자열 글꼴 설정              |    TkDefaultFont    |      font        |
|  cursor  |      수치 조정 기입창의 마우스 커서 모양                 |       -       |                                    [커서 속성](#reference-1)                                   |
|  buttoncursor |    수치 조정 기입창의 수치 조정 버튼 마우스 커서 모양              |       -       |                                    [커서 속성](#reference-1)                                   |
|   xscrollcommand  |          수치 조정 기입창의 가로스크롤 위젯 적용            | - |          Scrollbar위젯.set |
|   from_  |           수치 조정 기입창의 최솟값           | 0 |          상수          |
|   to   |           수치 조정 기입창의 최댓값         | 0 |          상수          |
|   increment |            수치 조정 기입창의 수치 간격            | 1 |          상수          |
|   values |            수치 조정 기입창의 사용자 정의 수치 값            | - |          [list, tuple 등](#reference-2)            |
|   format_ |            수치 조정 기입창의 수치 표시 형식            | - |          정밀도 설정          |
|  exportselection |     수치 조정 기압창의 선택 항목 여부 설정   |    True    |      Boolean        |

<br>
<br>


##### 수치 조정 기입창 상태 설정 #####


|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    state  |    상태 설정  | normal  | normal, readonly, disabled  |
|    activebackground |    active 상태일 때 수치 조정 기입창의 배경 색상 | SystemButtonFace |  color  |
|  readonlybackground |   readonly 상태일 때 수치 조정 기입창의 배경 색상   |  SystemButtonFace  |           color          |
|  disabledbackground |  disabeld 상태일 때 수치 조정 기입창의 배경 색상  |  SystemButtonFace  |           color          |
| disabledforeground | disabeld 상태일 때 수치 조정 기입창의 문자열 색상 | SystemDisabledText |           color          |

<br>
<br>

##### 수치 조정 기입창 하이라이트 설정 #####


|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    highlightcolor   |    수치 조정 기입창이가 선택되었을 때 색상   | SystemWindowFrame |  color  |
| highlightbackground | 수치 조정 기입창이가 선택되지 않았을 때 색상 |  SystemButtonFace |  color  |
|  highlightthickness |    수치 조정 기입창이가 선택되었을 때 두께 [(두께 설정)](#reference-3)     |         0         | 상수 |

<br>
<br>

##### 수치 조정 기입창 동작 설정 #####


|         이름        |              의미              |         기본값        | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    takefocus |    Tab 키를 이용하여 위젯 이동 허용 여부  | False |  Boolean |
|    command |    수치 조정 기입창에서 수치가 조정될 때 실행하는 메소드(함수)   | - |  메소드, 함수 |
|    wrap  |  수치 조정 기입창에서 수치가 초과되거나 미만일 경우, 최솟값이나 최댓값으로 변경  | False |  Boolean |
|    insertontime |    기입창의 키보드 커서 깜빡임이 보이는 시간   | 600  |  상수(ms) |
| insertofftime | 기입창의 키보드 커서 깜빡임이 보이지 않는 시간  |  300 |  상수(ms) |
| repeatdelay | [수치 조정 버튼이 눌러진 상태에서 command 실행까지의 대기 시간](#reference-4)   |  400 |  상수(ms) |
|  repeatinterval |    [수치 조정 버튼이 눌러진 상태에서 command 실행의 반복 시간](#reference-5)    |         100         | 상수(ms) |
|    validate |    수치 조정 기입창의 유효성 검사 실행 조건  | none |  [none, focus, focusin, focusout, key, all](#reference-6) |
|    validatecommand |   유효성 검사 평가 함수  | - |  함수 |
|    invalidcommand |    validateCommand가 False를 반환 할 때 실행할 함수 | - |  함수  |

<br>
<br>

##### 참고 #####
----------


<a id="reference-1"></a>

* cursor 파라미터

    - arrow, based_arrow_down, based_arrow_up, boat, bogosity, bottom_left_corner, bottom_right_corner, bottom_side, bottom_tee, box_spiral, center_ptr, circle, clock,	coffee_mug, cross, cross_reverse, crosshair, diamond_cross, dot, dotbox, double_arrow, draft_large, draft_small, draped_box, exchange, fleur, gobbler, gumby, hand1, hand2, heart, icon, iron_cross, left_ptr, left_side, left_tee, leftbutton, ll_angle, lr_angle, man, middlebutton, mouse, pencil, pirate, plus, question_arrow, right_ptr, right_side, right_tee, rightbutton, rtl_logo, sailboat, sb_down_arrow, sb_h_double_arrow, sb_left_arrow, sb_right_arrow, sb_up_arrow, sb_v_double_arrow, shuttle, sizing, spider, spraycan, star, target, tcross, top_left_arrow, top_left_corner, top_right_corner, top_side, top_tee, trek, ul_angle, umbrella, ur_angle, watch, wait, xterm, X_cursor



<a id="reference-2"></a>

* `values`를 `[5, 10, 50, 100]`으로 설정하였을 경우, `목록의 순서`로 증감



<a id="reference-3"></a>

* `highlightbackground`를 설정하였을 경우, 수치 조정 기입창이 선택되지 않았을 때에도 두께가 표시됨


<a id="reference-4"></a>

* `repeatdelay=100` 일 경우, **누르고 있기 시작한 0.1초 후**에 `command`가 실행됨


<a id="reference-5"></a>

* `repeatdelay=1000`, `repeatinterval=100` 일 경우, **1초 후에 command가 실행되며 0.1초마다 버튼을 뗄 때까지** `command`가 계속 실행됨


<a id="reference-6"></a>

* validate 파라미터

    - `none` : 수치 조정 기입창의 유효성 검사 실행하지 않음
    - `focus` : 수치 조정 기입창이 포커스를 받거나 잃을 때 validateCommand 실행
    - `focusin` : 수치 조정 기입창이 포커스를 받을 때 validateCommand 실행
    - `focusout` : 수치 조정 기입창이 포커스를 잃을 때 validateCommand 실행
    - `key` : 수치 조정 기입창이 수정될 경우 validateCommand 실행
    - `all` : 수치 조정 기입창의 모든 validate에 대해 validateCommand 실행
