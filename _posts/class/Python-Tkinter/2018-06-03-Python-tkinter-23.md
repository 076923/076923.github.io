---
layout: post
title: "Python tkinter 강좌 : 제 23강 - Bind"
tagline: "Python tkinter CanvaBinds"
image: /assets/images/tkinter.png
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Python-Tkinter']
keywords: Python, Python tkinter, tkinter Bind
ref: Python-Tkinter
category: posts
permalink: /posts/Python-tkinter-23/
comments: true
---

## Bind(이벤트 실행) ##
----------

`Bind`를 이용하여 위젯들의 `이벤트`와 실행할 `함수`를 설정할 수 있습니다.

<br>
<br>

## Bind 사용 ##
----------

{% highlight Python %}

import tkinter

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(True, True)

width=1

def drawing(event):
    if width>0:
        x1=event.x-1
        y1=event.y-1
        x2=event.x+1
        y2=event.y+1
        canvas.create_oval(x1, y1, x2, y2, fill="blue", width=width)

def scroll(event):
    global width
    if event.delta==120:
        width+=1
    if event.delta==-120:
        width-=1
    label.config(text=str(width))

canvas=tkinter.Canvas(window, relief="solid", bd=2)
canvas.pack(expand=True, fill="both")
canvas.bind("<B1-Motion>", drawing)
canvas.bind("<MouseWheel>", scroll)

label=tkinter.Label(window, text=str(width))
label.pack()

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

width=1

def drawing(event):
    if width>0:
        x1=event.x-1
        y1=event.y-1
        x2=event.x+1
        y2=event.y+1
        canvas.create_oval(x1, y1, x2, y2, fill="blue", width=width)

def scroll(event):
    global width
    if event.delta==120:
        width+=1
    if event.delta==-120:
        width-=1
    label.config(text=str(width))

canvas=tkinter.Canvas(window, relief="solid", bd=2)
canvas.pack(expand=True, fill="both")
canvas.bind("<B1-Motion>", drawing)
canvas.bind("<MouseWheel>", scroll)

label=tkinter.Label(window, text=str(width))
label.pack()

window.mainloop()

{% endhighlight %}


`위젯.Bind("이벤트", 함수)`를 사용하여 `위젯`의 `이벤트`가 작동할 때 실행할 `함수`를 설정할 수 있습니다.

`Bind`를 `중복`해서 사용하여도 `모두 적용`됩니다.

`event.delta`를 이용하여 `마우스 휠`의 `delta`값을 확인할 수 있습니다.

`event.delta`의 값이 `120`일 경우 `업 스크롤`이며, `-120`일 경우 `다운 스크롤`입니다.

<br>
<br>

## Bind Event ##
----------

## Button ##

|     이름     |             의미             |
|:------------:|:----------------------------:|
|  \<Button-1>  |  마우스 왼쪽 버튼을 누를 때  |
|  \<Button-2>  |   마우스 휠 버튼을 누를 때   |
|  \<Button-3>  | 마우스 오른쪽 버튼을 누를 때 |
|  \<Button-4>  |           스크롤 업          |
|  \<Button-5>  |          스크롤 다운         |
| \<MouseWheel> |        마우스 휠 이동        |

<br>
<br>

## Motion ##

|     이름    |                   의미                  |
|:-----------:|:---------------------------------------:|
| \<Motion> |  마우스가 움직일 때  |
| \<B1-Motion> |  마우스 왼쪽 버튼을 누르면서 움직일 때  |
| \<B2-Motion> |   마우스 휠 버튼을 누르면서 움직일 때   |
| \<B3-Motion> | 마우스 오른쪽 버튼을 누르면서 움직일 때 |

<br>
<br>

## Release ##

|        이름       |            의미            |
|:-----------------:|:--------------------------:|
| \<ButtonRelease-1> |  마우스 왼쪽 버튼을 뗄 때  |
| \<ButtonRelease-2> |   마우스 휠 버튼을 뗄 때   |
| \<ButtonRelease-3> | 마우스 오른쪽 버튼을 뗄 때 |

<br>
<br>

## Double Click ##

|        이름       |                 의미                |
|:-----------------:|:-----------------------------------:|
| \<Double-Button-1> |  마우스 왼쪽 버튼을 더블 클릭할 때  |
| \<Double-Button-2> |   마우스 휠 버튼을 더블 클릭할 때   |
| \<Double-Button-3> | 마우스 오른쪽 버튼을 더블 클릭할 때 |

<br>
<br>

## Widget Operation ##

|    이름    |                    의미                    |
|:----------:|:------------------------------------------:|
|   \<Enter>  |   위젯 안으로 마우스 포인터가 들어왓을 때  |
|   \<Leave>  |    위젯 밖으로 마우스 포인터가 나갔을 때   |
|  \<FocusIn> |  위젯 안으로 Tab 키를 이용하여 들어왔을 때 |
| \<FocusOut> | 위젯 밖으로 Tab 키를 이용하여 나갔을 때    |
| \<Configure> | 위젯의 모양이 수정되었을 때   |

<br>
<br>

## Key Input ##

|   이름   |             의미             |
|:--------:|:----------------------------:|
|   \<Key>  |    특정 키가 입력되었을 때   |
| \<Return> |   Enter 키가 입력되었을 때   |
|  \<Cancel>  | Break 키가 입력되었을 때 |
|  \<Pause>  | Pause 키가 입력되었을 때 |
|  \<BackSpace>  | 백스페이스 키가 입력되었을 때 |
|  \<Caps_Lock>  | 캡스 락 키가 입력되었을 때 |
|  \<Escape>  | 이스케이프 키가 입력되었을 때 |
|  \<Home>  | Home 키가 입력되었을 때 |
|  \<End>  | End 키가 입력되었을 때 |
|  \<Print> |  Print 키가 입력되었을 때  |
|  \<Insert> |  Insert 키가 입력되었을 때  |
|  \<Delete> |  Delete 키가 입력되었을 때  |
|  \<Prior> |  Page UP 키가 입력되었을 때  |
|  \<Up>  | 윗쪽 방향키가 입력되었을 때 |
|  \<Down>  | 아랫쪽 방향키가 입력되었을 때 |
|  \<Right>  | 오른쪽 방향키가 입력되었을 때 |
|  \<Left>  | 왼쪽 방향키가 입력되었을 때 |

<br>

* Tip : `<Key>` 이벤트 입력 시, `<s>`, `<0>`, `<F1>`, `<F4>` 등 특정 문자가 입력되었을 때도 가능, `공백`은 **제외됩니다.**
* Tip : `<Key>` 이벤트 입력 시, 키 이벤트가 할당된 컨트롤에 `*.focus_set()`을 추가해 **포커스**를 할당합니다.

<br>
<br>

## Assistant Key Input ##

|   이름   |             의미             |
|:--------:|:----------------------------:|
|   <Shift-`Key`>  |   Shift + 특정 키가 입력되었을 때   |
| <Contrl-`Key`> |   Ctrl + 특정 키가 입력되었을 때   |
|  <Alt-`Key`> |  Alt + 특정 키가 입력되었을 때  |


