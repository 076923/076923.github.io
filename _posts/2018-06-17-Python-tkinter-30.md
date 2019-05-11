---
bg: "python.jpg"
layout: post
comments: true
title: "Python tkinter 강좌 : 제 30강 – Treeview"
crawlertitle: "Python tkinter 강좌 : 제 30강 - Treeview"
summary: "Python tkinter Treeview"
date: 2018-06-17
categories: posts
tags: ['Python-tkinter']
author: 윤대희
star: true
---

### Treeview (표) ###
----------
[![1]({{ site.images }}/Python/tkinter/ch30/1.png)]({{ site.images }}/Python/tkinter/ch30/1.png)
`Treeview`을 이용하여 **행과 열**로 구성된 `표`를 생성할 수 있습니다.

<br>
<br>
### Treeview 사용 ###
----------
{% highlight Python %}

import tkinter
import tkinter.ttk

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

def cc(self):
    treeview.tag_configure("tag2", background="red")

treeview=tkinter.ttk.Treeview(window, columns=["one", "two"], displaycolumns=["two", "one"])
treeview.pack()

treeview.column("#0", width=70)
treeview.heading("#0", text="num")

treeview.column("one", width=100, anchor="center")
treeview.heading("one", text="문자", anchor="e")

treeview.column("#2", width=100, anchor="w")
treeview.heading("two", text="ASCII", anchor="center")

treelist=[("A", 65), ("B", 66), ("C", 67), ("D", 68), ("E", 69)]

for i in range(len(treelist)):
    
    treeview.insert('', 'end', text=i, values=treelist[i], iid=str(i)+"번")

top=treeview.insert('', 'end', text=str(len(treelist)), iid="5번", tags="tag1")
top_mid1=treeview.insert(top, 'end', text="5-2", values=["SOH", 1], iid="5번-1")
top_mid2=treeview.insert(top, 0, text="5-1", values=["NUL", 0], iid="5번-0", tags="tag2")
top_mid3=treeview.insert(top, 'end', text="5-3", values=["STX", 2], iid="5번-2", tags="tag2")

treeview.tag_bind("tag1", sequence="<<TreeviewSelect>>", callback=cc)

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

def cc(self):
    treeview.tag_configure("tag2", background="red")

treeview=tkinter.ttk.Treeview(window, columns=["one", "two"], displaycolumns=["two", "one"])
treeview.pack()

treeview.column("#0", width=70)
treeview.heading("#0", text="num")

treeview.column("one", width=100, anchor="center")
treeview.heading("one", text="문자", anchor="e")

treeview.column("#2", width=100, anchor="w")
treeview.heading("two", text="ASCII", anchor="center")

treelist=[("A", 65), ("B", 66), ("C", 67), ("D", 68), ("E", 69)]

for i in range(len(treelist)):
    
    treeview.insert('', 'end', text=i, values=treelist[i], iid=str(i)+"번")

top=treeview.insert('', 'end', text=str(len(treelist)), iid="5번", tags="tag1")
top_mid1=treeview.insert(top, 'end', text="5-2", values=["SOH", 1], iid="5번-1")
top_mid2=treeview.insert(top, 0, text="5-1", values=["NUL", 0], iid="5번-0", tags="tag2")
top_mid3=treeview.insert(top, 'end', text="5-3", values=["STX", 2], iid="5번-2", tags="tag2")

treeview.tag_bind("tag1", sequence="<<TreeviewSelect>>", callback=cc)

{% endhighlight %}


`tkinter.ttk.Treeview(윈도우 창, 파라미터1, 파라미터2, 파라미터3, ...)`을 사용하여 `표의 속성`을 설정할 수 있습니다.

`표.column()`, `표.heading()`을 이용하여 `행`을 설정할 수 있습니다.

`표.insert()`를 이용하여 `열`을 설정할 수 있습니다.

`변수 이름`과 `iid`는 서로 대체하여 사용할 수 있습니다.

<br>
<br>

### Treeview Method###
----------

##### 표 메소드 (1) #####

|                 이름                 |                                의미                               |
|:------------------------------------:|:-----------------------------------------------------------------:|
|     column(col, column_option)    |                        표 열의 속성을 설정                        |
|     heading(col, heading_option)     |                         표의 행 제목 설정                         |
| insert(상위 항목, 삽입 위치, option) |                    표에 항목을 삽입 위치에 삽입                   |
|           item(iid, option)          |                    iid 값을 가지는 항목을 수정                    |
|    move(iid, 상위 항목, 삽입 위치)   |        iid 값을 가지는 항목을 상위 항목의 삽입 위치로 이동        |
|  reattach(iid, 상위 항목, 삽입 위치) | iid 값을 가지는 항목을 상위 항목의 삽입 위치로 이동 (move와 동일) |
|              delete(iid)             |                     iid 값을 가지는 항목 삭제                     |
|              detach(iid)             |               iid 값을 가지는 항목 연결 해제 및 숨김              |
|              index(iid)              |                iid 값을 가지는 항목의 index 값 반환               |
|              focus(iid)              |                   iid 값을 가지는 항목에 포커스                   |
|                xview()               |                          가로스크롤 연결                          |
|                yview()               |                          세로스크롤 연결                          |

<br>

* Tip : `col` : **#0**은 열의 첫번째 위치를 의미, **#1, #2, #3, ...** 또는 열**의 이름1, 열의 이름2, 열의 이름3, ...**을 이용하여 열을 선택 가능

<br>
<br>

##### 표 메소드 column_option #####

|   이름   |                    의미                    | 기본값 |                속성                |
|:--------:|:------------------------------------------:|:------:|:----------------------------------:|
|  anchor  |         행 제목의 문자열 위치 설정         |    w   | n, e, w, s, ne, nw, se, sw, center |
|   width  |                  열의 너비                 |   200  |                상수                |
| minwidth |               열의 최소 너비               |   20   |                상수                |
|  stretch | 위젯 크기 조정 시 열의 너비 조정 설정 여부 |  TRUE  |               Boolean              |


<br>
<br>

##### 표 메소드 heading_option #####

|   이름  |                    의미                   | 기본값 |                속성                |
|:-------:|:-----------------------------------------:|:------:|:----------------------------------:|
|   text  |               행 제목의 이름              |    -   |               문자열               |
|  image  |              행 제목의 이미지             |    -   |                  -                 |
|  anchor |           열의 문자열 위치 설정           |    w   | n, e, w, s, ne, nw, se, sw, center |
| command | 행 제목을 클릭할 때 실행하는 메소드(함수) |    -   |            메소드, 함수            |

<br>
<br>

##### 표 메소드 option #####

|  이름  |           의미           | 기본값 |   속성  |
|:------:|:------------------------:|:------:|:-------:|
|  text  |  표 항목에 표시할 텍스트 |    -   |  문자열 |
|  image |  표에 포함할 임의 이미지 |    -   |    -    |
| values |    표 행에 포함될 항목   |    -   |    -    |
|  open  | 표의 하위 항목 숨김 여부 |  False | Boolean |
|  tags  |    표의 태그 이름 설정   |    -   |  문자열 |
|   iid  |     표의 고유값 설정     |    -   |    -    |


<br>
<br>

##### 표 메소드 (2) #####

|                     이름                    |                               의미                               |
|:-------------------------------------------:|:----------------------------------------------------------------:|
|            bbox(iid, column=상수)           | iid 값을 가지는 행과 column 값의 (width, height, x, y)   값 반환 |
|                get_children()               |                        표의 하위 항목 반환                       |
|              get_children(iid)              |                  iid 값을 가지는 하위 항목 반환                  |
| set_children(iid, 하위 iid, 하위 iid, ... ) |   iid 값을 가지는 상위 항목에서 하위 iid값을 가지는 항목만 남김  |
|                 exists(iid)                 |             iid 값을 가지는 항목이 있다면 참 값 반환             |
|          identify(component, x, y)          |        표의 x, y 위치에 지정된 component의 세부 사항 반환        |
|               identify_row(y)               |                    표의 y 위치의 iid 값을 반환                   |
|              identify_column(x)             |                   표의 x 위치의 열 식별자 반환                   |
|            identify_element(x, y)           |                   표의 x, y 위치에 요소를 반환                   |
|            identify_region(x, y)            |                  표의 x, y 위치에 세부 사항 반환                 |
|                  next(iid)                  |           iid 값을 가지는 항목의 다음 순서 iid 값 반환           |
|                 parent(iid)                 |               iid 값을 가지는 항목의 상위 항목 반환              |
|                  prev(iid)                  |           iid 값을 가지는 항목의 이전 순서 iid 값 반환           |
|                   see(iid)                  |               iid 값을 가지는 항목이 보이는지 확인               |
|              selection_set(iid)             |                     iid 값을 가지는 항목 선택                    |
|              selection_add(iid)             |                  iid 값을 가지는 항목 다중 선택                  |
|            selection_remove(iid)            |                  iid 값을 가지는 항목 선택 취소                  |
|            selection_toggle(iid)            |                  iid 값을 가지는 항목 반전 선택                  |
|                   set(iid)                  |                iid 값을 가지는 항목의 사전 값 반환               |
|             set(iid, column=col)            |              iid 값을 가지는 항목에서 col의 값 반환              |
|       set(iid, column=col, value=val)       |          iid 값을 가지는 항목에서 col의 값을 val로 변경          |

<br>

* component : element, region, item, column, row

* identify_region(x, y)의 반환값
    - heading : 행 제목
    - separator : 사이 간격 선
    - tree : 열 제목
    - cell : 셀 영역

<br>
<br>

##### 표 태그 메소드 #####

|                           이름                          |                           의미                          |   속성  |
|:-------------------------------------------------------:|:-------------------------------------------------------:|:-------:|
| tag_bind(태그 이름, sequence="이벤트", callback=함수) | 태그 이름의 목록에서 이벤트가 실행되었을 때 함수를 실행 |    -    |
|           tag_configure(태그 이름, tag_option)          |                  태그 이름에 형태 설정                  |    -    |
|                    tag_has(태그 이름)                   |              태그 이름에 할당된 iid 값 반환             |    -    |
|               tag_has(태그 이름, item=iid)              |                 태그 이름과 iid 값 비교                 | Boolean |

<br>
<br>

##### 표 태그 메소드 sequence #####

|        이름       |          의미         |
|:-----------------:|:---------------------:|
| <<TreeviewSelect> | 표의 항목이 선택될 때 |
|  <<TreeviewOpen>> |   표의 항목을 열 때   |
| <<TreeviewClose>> |  표의 항목을 닫을 때  |
    
<br>
<br>

##### 표 태그 메소드 tag_option #####
    
|    이름    |           의미           |      기본값      |  속성 |
|:----------:|:------------------------:|:----------------:|:-----:|
| background |      표의 배경 색상      | SystemButtonFace | color |
| foreground |     표의 문자열 색상     | SystemButtonFace | color |
|    image   |  표에 포함할 임의 이미지 |         -        |   -   |    
|    font    | 표 안의 문자열 글꼴 설정 |   TkDefaultFont  |  font |


<br>
<br>

### Treeview Parameter ###
----------

##### 표 형태 설정 #####

|   이름   |                           의미                          |     기본값    |               속성                    |
|:--------:|:-------------------------------------------------------:|:-------------:|:-------------:|
|  height  |      표의 행 높이                 |       10     |          상수       |
|  padding  |      표의 여백                 |       0       |      상수    |  

<br>
<br>

##### 표 형식 설정 #####

|   이름   |                           의미                          |     기본값    |               속성                    |
|:--------:|:-------------------------------------------------------:|:-------------:|:-------------:|
|  cursor  |      표의 마우스 커서 모양                 |       -       |          [커서 속성](#reference-1)             |
|  class_  |      클래스 설정                 |       -       |      -    |  
|  columns  |      표의 열 이름 설정 |       -       |      리스트    |  
|  displaycolumns |   표의 열 순서 설정  |       -       |      리스트    |  
|  selectmode |    표의 선택 형식 설정  |       extended |      [extended, browse, none](#reference-2) |  
|  show |   표의 제목 표시 여부  |       tree headings |      [tree headings, tree, headings](#reference-3) |  


<br>
<br>

##### 표 동작 설정 #####


|         이름        |              의미              |         기본값        | 속성 |
|:-------------------:|:------------------------------:|:-----------------:|:----:|
|    takefocus |    Tab 키를 이용하여 위젯 이동 허용 여부  | False |  Boolean |

<br>
<br>

##### 참고 #####
----------


<a id="reference-1"></a>

* cursor 파라미터

    - arrow, based_arrow_down, based_arrow_up, boat, bogosity, bottom_left_corner, bottom_right_corner, bottom_side, bottom_tee, box_spiral, center_ptr, circle, clock, coffee_mug, cross, cross_reverse, crosshair, diamond_cross, dot, dotbox, double_arrow, draft_large, draft_small, draped_box, exchange, fleur, gobbler, gumby, hand1, hand2, heart, icon, iron_cross, left_ptr, left_side, left_tee, leftbutton, ll_angle, lr_angle, man, middlebutton, mouse, pencil, pirate, plus, question_arrow, right_ptr, right_side, right_tee, rightbutton, rtl_logo, sailboat, sb_down_arrow, sb_h_double_arrow, sb_left_arrow, sb_right_arrow, sb_up_arrow, sb_v_double_arrow, shuttle, sizing, spider, spraycan, star, target, tcross, top_left_arrow, top_left_corner, top_right_corner, top_side, top_tee, trek, ul_angle, umbrella, ur_angle, watch, wait, xterm, X_cursor


<a id="reference-2"></a>

* selectmode 파라미터

    - `extended` : Ctrl 키 또는 Shift 키를 활용하여 모록을 다중 선택 가능
    - `browse` : 한 번에 하나의 목록만 선택 가능
    - `none` : 목록 선택 불가

<a id="reference-3"></a>

* show 파라미터

    - `tree headings` : 행 제목과 열 제목 표시
    - `tree` : 행 제목 표시하지 않음
    - `headings` : 열 제목 표시하지 않음

