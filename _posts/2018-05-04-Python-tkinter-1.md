---
bg: "python.jpg"
layout: post
comments: true
title: "Python tkinter 강좌 : 제 1강 - GUI 생성"
crawlertitle: "Python tkinter 강좌 : 제 1강 - GUI 생성"
summary: "Python tkinter"
date: 2018-05-04
categories: posts
tags: ['Python-tkinter']
author: 윤대희
star: true
---

### tkinter ###
----------
`tkinter`는 `GUI`에 대한 `표준 Python 인터페이스`이며 `Window 창`을 생성할 수 있습니다.

<br>
<br>
### tkinter 사용 ###
----------
{% highlight Python %}

import tkinter

{% endhighlight %}

상단에 `import tkinter`를 사용하여 `GUI 모듈`을 포함시킵니다. tkinter 함수의 사용방법은 `tkinter.*`를 이용하여 사용이 가능합니다.

<br>
<br>
{% highlight Python %}

import tkinter

window=tkinter.Tk()

window.mainloop()

{% endhighlight %}

`윈도우이름=tkinter.Tk()`를 이용하여 가장 상위 레벨의 `윈도우 창`을 **생성할 수 있습니다.**

`윈도우이름.mainloop()`를 사용하여 `윈도우이름`의 윈도우 창을 **윈도우가 종료될 때 까지 실행시킵니다.**

`생성` 구문과 `반복` 구문 사이에 `위젯`을 생성하고 적용합니다.

<br>
<br>
[![1]({{ site.images }}/Python/tkinter/ch1/1.png)]({{ site.images }}/Python/tkinter/ch1/1.png)

<br>

`tkinter.Tk()`를 적용할 경우 가장 기본적인 `윈도우 창`이 생성됩니다.

<br>
<br>
### Window 창 설정 ###
----------
{% highlight Python %}

import tkinter

window=tkinter.Tk()

window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

window.mainloop()

{% endhighlight %}

<br>

`윈도우이름.title("제목")`을 이용하여 `윈도우 창`의 `제목`을 설정할 수 있습니다.

`윈도우이름.geometry("너비x높이+x좌표+y좌표")`를 이용하여 `윈도우 창`의 `너비와 높이`, 초기 화면 위치의 `x좌표와 y좌표`를 설정할 수있습니다.

`윈도우이름.resizeable(상하, 좌우)`을 이용하여 `윈도우 창`의 `창 크기 조절 가능 여부`를 설정할 수 있습니다. `True`로 설정할 경우 `윈도우 창`의 크기를 조절할 수 있습니다.

<br>

* Tip : `resizeable()`을 적용할 때, `True=1`, `False=0`을 의미하여 `상수`를 입력해도 적용이 가능합니다.

<br>
<br>
[![2]({{ site.images }}/Python/tkinter/ch1/2.png)]({{ site.images }}/Python/tkinter/ch1/2.png)

<br>

`윈도우 창`의 이름이 `YUN DAE HEE`로 설정되었으며 `크기`와 `초기 화면 위치`, `윈도우 창의 크기 조절 불가`로 설정된 것을 확인할 수 있습니다.

<br>
<br>
### Widget 배치 ###
----------
{% highlight Python %}

import tkinter

window=tkinter.Tk()

window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

label=tkinter.Label(window, text="안녕하세요.")
label.pack()

window.mainloop()

{% endhighlight %}

<br>

`위젯이름=tkinter.Label(윈도우창, text="내용")`을 사용하여 `윈도우 창`에 `Label` 위젯을 설정할 수 있습니다.

`위젯이름.pack()`을 사용하여 **위젯을 배치할 수 있습니다.**

<br>
<br>
[![3]({{ site.images }}/Python/tkinter/ch1/3.png)]({{ site.images }}/Python/tkinter/ch1/3.png)

속성을 설정하지 않아 `기본 속성`으로 설정되어 **가장 최상단에 라벨이 배치된 것을 확인할 수 있습니다.**


<br>


