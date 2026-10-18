---
layout: post
title: "Python tkinter 강좌 : 제 18강 - Text"
tagline: "Python tkinter Text"
image: /assets/images/tkinter.png
tags: ['Tkinter']
keywords: Python, Python tkinter, tkinter Text, tkinter 텍스트 위젯, tkinter Text insert, tkinter Text get, tkinter Text tag, tkinter 텍스트 인덱스
ref: Python-Tkinter
category: Python
permalink: /posts/Python-tkinter-18/
comments: true
toc: true
---

## Text(텍스트)

`Text`를 이용하여 `여러 줄`의 `문자열`을 입력하거나 출력하기 위한 `텍스트`를 생성할 수 있습니다.

<br>
<br>

## Text 사용

{% highlight Python %}
import tkinter

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(True, True)

text=tkinter.Text(window)

text.insert(tkinter.CURRENT, "안녕하세요.\n")
text.insert("current", "반습니다.")
text.insert(2.1, "갑")

text.pack()

text.tag_add("강조", "1.0", "1.6")
text.tag_config("강조", background="yellow") 
text.tag_remove("강조", "1.1", "1.2")

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

text=tkinter.Text(window)

text.insert(tkinter.CURRENT, "안녕하세요.\n")
text.insert("current", "반습니다.")
text.insert(2.1, "갑")

text.pack()

text.tag_add("강조", "1.0", "1.6")
text.tag_config("강조", background="yellow") 
text.tag_remove("강조", "1.1", "1.2")

{% endhighlight %}

`tkinter.Text(윈도우 창, 매개변수1, 매개변수2, 매개변수3, ...)`을 사용하여 해당 `윈도우 창`에 표시할 `텍스트의 속성`을 설정할 수 있습니다.

`매개변수`를 사용하여 `텍스트의 속성`을 설정합니다.

`insert(index, 문자열)`로 문자열을 삽입하며, 예제에서는 `tkinter.CURRENT`(마우스 포인터와 가장 가까운 위치)에 두 문자열을 차례로 삽입한 뒤, `2.1`(두 번째 줄, 두 번째 문자 앞) 위치에 `갑`을 삽입해 `반갑습니다.`를 완성합니다.

`tag_add`로 첫 번째 줄의 `1.0`부터 `1.6`까지 `강조` 태그를 지정하고, `tag_config`로 태그의 배경 색상을 노란색으로 설정한 뒤, `tag_remove`로 `1.1`부터 `1.2` 사이의 글자 하나만 태그를 제거합니다.

- Tip : `index`는 `"1.0"`처럼 **문자열로 전달하는 것을 권장**합니다. `2.1`과 같이 실수로 전달하면 `2.10`이 `2.1`로 해석되는 등 의도와 다른 위치를 가리킬 수 있습니다.

- Tip : 텍스트의 **모든 내용**을 가져오려면 `text.get("1.0", "end-1c")`를 사용합니다. `"end"`까지 가져오면 텍스트 위젯이 자동으로 추가하는 마지막 줄 바꿈 문자까지 포함됩니다.

- Tip : 사용자가 입력하지 못하도록 하려면 `state="disabled"`로 설정합니다. 단, 이 상태에서는 `insert`, `delete`도 동작하지 않으므로, 코드로 내용을 바꿀 때에는 `state="normal"`로 변경한 뒤 다시 `disabled`로 설정합니다.

<br>
<br>

## Text Method

### 텍스트 문자열 메서드

|              이름              |       의미       |                       설명                      |
|:------------------------------:|:----------------:|:-----------------------------------------------:|
|     insert(index, "문자열")    |    문자열 삽입   |       해당 `index` 위치에 `문자열`을 삽입       |
| delete(start_index, end_index) |    문자열 삭제   | `start_index`부터 `end_index`까지의 문자열 삭제 |
|   get(start_index, end_index)  |    문자열 반환   | `start_index`부터 `end_index`까지의 문자열 반환 |
|          index(index)          |    인덱스 반환   |      `index` 표현식을 `y.x` 형식의 문자열로 변환하여 반환 `예) index("end") → "3.0"`     |
|           see(index)           | 문자열 표시 |   `index` 위치의 문자열이 보이도록 텍스트를 스크롤  |
|           search(패턴, index)  |  문자열 검색  |  `index`부터 `패턴`을 검색하여 처음 발견된 위치의 인덱스 반환, 없을 경우 빈 문자열 반환  |

- Tip : `index`는 `y.x`를 사용, `y줄, x번째 문자`를 의미함 `예) 첫 번째 줄, 첫 번째 문자 = 1.0`

- Tip : 줄 번호는 `1`부터, 문자 번호는 `0`부터 시작하므로 `1.0`이 가장 첫 위치가 됩니다.

<br>

### 텍스트 마크 메서드

|            이름            |            의미            |                         설명                         |
|:--------------------------:|:--------------------------:|:----------------------------------------------------:|
|    mark_set(mark, index)   |       마크 위치 설정       |                해당 마크의 위치 재설정               |
|      mark_unset(mark)      |       마크 위치 제거       |                 해당 마크의 표시 제거                |
| mark_gravity(mark,gravity) | 키보드 커서 삽입 위치 변경 | 해당 마크 사용 시, 좌측 삽입 또는 우측 삽입으로 변경 |
|         index(mark)        |       마크 위치 반환       |                  해당 마크 위치 반환                 |
|        mark_names()        |       모든 마크 반환       |          텍스트에서 사용된 모든 마크를 반환          |

- Tip : `gravity`는 `left`(좌측 삽입)와 `right`(우측 삽입)로 사용, 기본값은 `right`를 사용

<br>

### 텍스트 마크

|        이름       |      텍스트 이름      |                   의미                  |
|:-----------------:|:---------------------:|:---------------------------------------:|
|         -         |         `y.x`         |          y 번째 줄 x 번째 문자          |
|         -         |         `1.0`         |         첫 번째 줄 첫 번째 문자         |
|         -         |         `y.0`         |          y 번째 줄 첫 번째 문자         |
|         -         |        `y.end`        |          y 번째 줄 마지막 문자          |
|   tkinter.INSERT  |         insert        |             삽입 커서의 위치            |
|  tkinter.CURRENT  |        current        | 마우스 포인터에 가장 가까운 문자의 위치 |
|    tkinter.END    |          end          |        텍스트의 마지막 문자 다음 위치        |
| tkinter.SEL_FIRST |       sel.first      |       블록처리 되었을 때의 앞 부분      |
|  tkinter.SEL_LAST |        sel.last       |       블록처리 되었을 때의 뒷 부분      |
|         -         | `[인덱스] linestart` |             해당 줄의 시작 위치            |
|         -         |  `[인덱스] lineend`  |             해당 줄의 끝 위치            |
|         -         | `[인덱스] wordstart` |            해당 위치가 포함된 단어의 시작           |
|         -         | `[인덱스] wordend` |            해당 위치가 포함된 단어의 끝           |
|         -         | `[인덱스] + n chars` |            해당 위치에서 `n` 글자 뒤 `예) 1.0 + 3 chars`           |
|         -         | `[인덱스] - n lines` |            해당 위치에서 `n` 줄 앞 `예) end - 1 lines`           |

- Tip : `3.5 wordstart`로 마크를 사용할 경우, `3번째 줄 5 번째에 포함된 단어`의 앞 부분으로 설정됩니다. 

- Tip : `end`는 텍스트의 **마지막 문자 다음** 위치이므로, `text.insert("end", "문자열")`을 사용하면 문자열이 항상 끝에 추가됩니다.

- Tip : 블록(선택)이 없는 상태에서 `sel.first`, `sel.last`를 사용하면 `TclError`가 발생합니다. `text.tag_ranges("sel")`로 선택 여부를 먼저 확인할 수 있습니다.

<br>

### 텍스트 태그 메서드

|                      이름                      |     의미    |                            설명                           |
|:----------------------------------------------:|:-----------:|:---------------------------------------------------------:|
|    tag_add(tagname, start_index, end_index)    |  태그 생성  |    `start_index`부터 `end_index`까지의 `tagname`을 생성   |
|   tag_remove(tagname, start_index, end_index)  | 태그 제거 | `start_index`부터 `end_index`까지의 `tagname`의 설정 제거 |
|               tag_delete(tagname)              |  태그 삭제  |               `tagname`의 설정 및 선언 삭제               |
| tag_config(tagname, 매개변수1, 매개변수2, ...) |  태그 속성 설정  |               `tagname` 범위 만큼 속성 설정 `예) background, foreground, font, underline`               |
| tag_bind(tagname, 이벤트, 함수) |  태그 이벤트 연결  |               `tagname` 범위의 문자열에 마우스 클릭 등의 `이벤트` 연결               |

- Tip : 태그는 범위가 겹칠 수 있으며, 같은 속성이 겹칠 경우 **나중에 생성된 태그**의 속성이 우선 적용됩니다.

<br>
<br>

## Text Parameter ##

### 텍스트 텍스트 설정

|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
|  wrap | [자동 줄 내림 설정 방법](#reference-1)  |        char        |  none, char, word    |
| tabs |    텍스트의 Tab 간격     |       -       | 상수 |
| tabstyle |    텍스트의 Tab 간격 형식     | tabular | tabular, [wordprocessor](#reference-2)|
| startline |    텍스트의 데이터 저장소에 저장될 시작 줄 |       시작 줄       | 상수 |
| endline |    텍스트의 데이터 저장소에 저장될 마지막 줄  |       마지막 줄       | 상수 |
| spacing1 |    텍스트의 상단 수직 간격     |       0       | 상수 |
| spacing2 |    텍스트의 줄과 줄 사이 간격     |       0       | 상수 |
| spacing3 |    텍스트의 하단 수직 간격      |       0       | 상수 |

<br>

### 텍스트 형태 설정

|      이름      |               의미               |      기본값      |                    속성                    |
|:--------------:|:--------------------------------:|:----------------:|:------------------------------------------:|
| width | 텍스트의 너비 | 80 | 상수 |
| height | 텍스트의 높이 | 24 | 상수 |
| relief | 텍스트의 테두리 모양 | sunken | flat, groove, raised, ridge, solid, sunken |
| borderwidth=bd | 텍스트의 테두리 두께 | 1 | 상수 |
| background=bg | 텍스트의 배경 색상 | SystemWindow | color |
| foreground=fg | 텍스트의 문자열 색상 | SystemWindowText | color |
| insertwidth | 텍스트의 키보드 커서 너비 | 2 | 상수 |
| insertborderwidth | 텍스트의 키보드 커서 테두리 두께 | 0 | 상수 |
| insertbackground | 텍스트의 키보드 커서 색상 | SystemWindowText | color |
| selectborderwidth | 텍스트의 문자열 블록처리 테두리 두께 | 0 | 상수 |
| selectbackground | 텍스트의 문자열 블록처리 배경 색상 | SystemHighlight | color |
| selectforeground | 텍스트의 문자열 블록처리 문자열 색상 | SystemHighlightText | color |
| inactiveselectbackground | 텍스트의 문자열 블록처리 중 다른 위젯 선택시 블록처리 배경 색상 | - | color |
| padx | 텍스트의 테두리와 내용의 가로 여백 | 1 | 상수 |
| pady | 텍스트의 테두리와 내용의 세로 여백 | 1 | 상수 |

<br>

### 텍스트 형식 설정

|   이름   |                           의미                          |     기본값    |                                          속성                                          |
|:--------:|:-------------------------------------------------------:|:-------------:|:--------------------------------------------------------------------------------------:|
|  font   |       텍스트의 문자열 글꼴 설정              |    TkFixedFont    |      font        |
|  cursor  |      텍스트의 마우스 커서 모양                 |       -       |                                    [커서 속성](#reference-3)                                   |
|  xscrollcommand |    텍스트의 가로스크롤 객체 적용   |  -    |      Scrollbar위젯.set |
|  yscrollcommand |  텍스트의 세로스크롤 객체 적용   |   -    |      Scrollbar위젯.set |
|  exportselection |     텍스트의 선택 항목 여부 설정   |    True    |      Boolean        |
|  setgrid |   텍스트의 격자 크기 조정 설정 |    False    |      Boolean        |

<br>

### 텍스트 상태 설정

|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    state  |    상태 설정	 | normal  | normal, disabled  |


<br>

### 텍스트 하이라이트 설정

|         이름        |              의미              |       기본값      | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    highlightcolor   |    텍스트가 선택되었을 때 색상   | SystemWindowFrame |  color  |
| highlightbackground | 텍스트가 선택되지 않았을 때 색상 |  SystemButtonFace |  color  |
|  highlightthickness |    텍스트가 선택되었을 때 두께 [(두께 설정)](#reference-4)     |         0         | 상수 |

<br>

### 텍스트 동작 설정

|         이름        |              의미              |         기본값        | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    takefocus |    Tab 키를 이용하여 위젯 이동 허용 여부  | - |  Boolean |
|    blockcursor |  텍스트의 키보드 커서를 블록으로 사용   | False  |  Boolean |
|    undo |  [텍스트의 실행 취소 사용 여부](#reference-5)    | False  |  Boolean |
|    maxundo |  텍스트의 실행 취소 최대 횟수, `0`일 경우 제한 없음   | 0  |  상수 |
|    autoseparators |  [텍스트의 실행 취소 자동 저장 여부](#reference-6)  | True  |  Boolean |

<br>

<a id="reference-1"></a>

### 참고

<a id="reference-2"></a>

* wrap 매개변수

    - `none` : 줄 내림 하지 않음
    - `char` : 글자 단위로 줄 내림
    - `word` : 단어 단위로 줄 내림


<a id="reference-3"></a>

* `tabs`를 설정하지 않으면 글꼴의 문자 `8`개 너비를 Tab 간격으로 사용함

* `wordprocessor`로 사용할 경우, 워드프로세서 기준 `표준 간격`으로 사용함

<a id="reference-4"></a>

* cursor 매개변수

    - arrow, based_arrow_down, based_arrow_up, boat, bogosity, bottom_left_corner, bottom_right_corner, bottom_side, bottom_tee, box_spiral, center_ptr, circle, clock,	coffee_mug, cross, cross_reverse, crosshair, diamond_cross, dot, dotbox, double_arrow, draft_large, draft_small, draped_box, exchange, fleur, gobbler, gumby, hand1, hand2, heart, icon, iron_cross, left_ptr, left_side, left_tee, leftbutton, ll_angle, lr_angle, man, middlebutton, mouse, pencil, pirate, plus, question_arrow, right_ptr, right_side, right_tee, rightbutton, rtl_logo, sailboat, sb_down_arrow, sb_h_double_arrow, sb_left_arrow, sb_right_arrow, sb_up_arrow, sb_v_double_arrow, shuttle, sizing, spider, spraycan, star, target, tcross, top_left_arrow, top_left_corner, top_right_corner, top_side, top_tee, trek, ul_angle, umbrella, ur_angle, watch, wait, xterm, X_cursor


<a id="reference-5"></a>

* `highlightbackground`를 설정하였을 경우, 텍스트가 선택되지 않았을 때에도 두께가 표시됨

<a id="reference-6"></a>

* `undo=True`로 설정한 경우, `Ctrl + Z`를 사용하여 **실행 취소**, `Ctrl + Y`를 사용하여 **다시 실행**

* `autoseparators`를 `True`로 설정하였을 경우, 삽입에서 삭제로 바뀌는 등 **편집 동작이 바뀔 때마다 구분자가 자동으로 추가**되어, 연속으로 입력한 문자열이 한 번의 실행 취소로 함께 지워짐
