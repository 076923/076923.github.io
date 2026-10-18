---
layout: post
title: "Python tkinter 강좌 : 제 34강 - 대화 상자"
tagline: "Python tkinter Dialog"
image: /assets/images/tkinter.png
tags: ['Tkinter']
keywords: Python, Python tkinter, tkinter messagebox, tkinter filedialog, tkinter simpledialog, tkinter colorchooser, tkinter askopenfilename, tkinter asksaveasfilename, tkinter askdirectory, tkinter showinfo, tkinter askyesno
ref: Python-Tkinter
category: Python
permalink: /posts/Python-tkinter-34/
comments: true
toc: true
---

## 대화 상자(Dialog)

`대화 상자(Dialog)`는 사용자에게 **알림을 표시하거나 값을 입력받기 위해 잠시 띄우는 작은 창**입니다.

`tkinter`는 운영체제의 기본 대화 상자를 호출하는 모듈을 제공하므로, 직접 `Toplevel`로 창을 만들지 않아도 메시지 알림, 파일 선택, 값 입력, 색상 선택 등을 처리할 수 있습니다.

| 모듈 | 용도 |
|:---:|:---:|
| tkinter.messagebox | 알림, 경고, 오류, 예/아니오 질문 |
| tkinter.filedialog | 파일 열기, 파일 저장, 폴더 선택 |
| tkinter.simpledialog | 문자열, 정수, 실수 입력 |
| tkinter.colorchooser | 색상 선택 |

<br>

- Tip : 대화 상자 모듈은 `import tkinter`만으로는 불러와지지 않으므로, `from tkinter import messagebox`처럼 **직접 가져와야** 합니다.

- Tip : 대화 상자는 **모달(Modal)** 방식으로 동작합니다. 대화 상자가 닫힐 때까지 호출한 함수가 반환되지 않으며, 그동안 메인 창은 조작할 수 없습니다.

<br>
<br>

## 메시지 상자 사용

{% highlight Python %}

import tkinter
from tkinter import messagebox

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

def info():
    messagebox.showinfo("알림", "작업이 완료되었습니다.")

def warning():
    messagebox.showwarning("경고", "저장되지 않은 내용이 있습니다.")

def error():
    messagebox.showerror("오류", "파일을 열 수 없습니다.")

def question():
    result = messagebox.askyesno("확인", "정말로 종료하시겠습니까?")
    if result:
        window.destroy()

button1=tkinter.Button(window, text="알림", width=20, command=info)
button2=tkinter.Button(window, text="경고", width=20, command=warning)
button3=tkinter.Button(window, text="오류", width=20, command=error)
button4=tkinter.Button(window, text="종료", width=20, command=question)

button1.pack(pady=5)
button2.pack(pady=5)
button3.pack(pady=5)
button4.pack(pady=5)

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

from tkinter import messagebox

{% endhighlight %}

`tkinter.messagebox` 모듈을 가져옵니다.

<br>

{% highlight Python %}

messagebox.showinfo("알림", "작업이 완료되었습니다.")

{% endhighlight %}

`messagebox.함수(제목, 내용)`을 이용하여 `메시지 상자`를 표시할 수 있습니다.

`showinfo`, `showwarning`, `showerror`는 아이콘만 다를 뿐 모두 **확인 버튼 하나**만 표시되며, 반환값은 `"ok"`입니다.

<br>

{% highlight Python %}

result = messagebox.askyesno("확인", "정말로 종료하시겠습니까?")

{% endhighlight %}

`ask`로 시작하는 함수는 사용자의 선택을 **반환값**으로 돌려줍니다.

`askyesno`는 `예`를 누르면 `True`, `아니오`를 누르면 `False`를 반환합니다.

<br>
<br>

## 메시지 상자 함수

| 함수 | 버튼 | 반환값 |
|:---:|:---:|:---:|
| showinfo(title, message) | 확인 | "ok" |
| showwarning(title, message) | 확인 | "ok" |
| showerror(title, message) | 확인 | "ok" |
| askokcancel(title, message) | 확인, 취소 | True, False |
| askyesno(title, message) | 예, 아니오 | True, False |
| askretrycancel(title, message) | 다시 시도, 취소 | True, False |
| askquestion(title, message) | 예, 아니오 | "yes", "no" |
| askyesnocancel(title, message) | 예, 아니오, 취소 | True, False, None |

<br>

- Tip : `askquestion`은 `Boolean`이 아니라 **문자열** `"yes"`, `"no"`를 반환하므로 비교할 때 주의합니다.

- Tip : `askyesnocancel`에서 `취소`를 누르거나 창을 닫으면 `None`이 반환되므로, `if result:`가 아닌 `if result is None:`으로 취소 여부를 먼저 구분합니다.

<br>
<br>

## 메시지 상자 Parameter

| 이름 | 의미 | 기본값 | 속성 |
|:---:|:---:|:---:|:---:|
| title | 대화 상자 제목 | - | 문자열 |
| message | 대화 상자 내용 | - | 문자열 |
| detail | 내용 아래에 작은 글씨로 표시되는 상세 설명 | - | 문자열 |
| icon | 표시할 아이콘 | 함수별 기본값 | info, warning, error, question |
| default | 기본으로 선택된 버튼 | 첫 번째 버튼 | ok, cancel, yes, no, retry |
| parent | 대화 상자를 소유할 창 | - | 위젯 |

<br>

- Tip : `parent=window`를 지정하면 대화 상자가 **해당 창의 중앙**에 표시되며, 지정하지 않으면 화면 기준으로 표시될 수 있습니다.

<br>
<br>

## 파일 대화 상자 사용

{% highlight Python %}

import tkinter
from tkinter import filedialog

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

label=tkinter.Label(window, text="선택된 경로가 표시됩니다.", wraplength=600)

def open_file():
    path = filedialog.askopenfilename(
        title="파일 열기",
        initialdir=".",
        filetypes=(("텍스트 파일", "*.txt"), ("Python 파일", "*.py"), ("모든 파일", "*.*"))
    )
    if path:
        label.config(text=path)

def save_file():
    path = filedialog.asksaveasfilename(
        title="파일 저장",
        defaultextension=".txt",
        filetypes=(("텍스트 파일", "*.txt"), ("모든 파일", "*.*"))
    )
    if path:
        with open(path, "w", encoding="utf-8") as file:
            file.write("YUN DAE HEE")
        label.config(text=path)

def open_directory():
    path = filedialog.askdirectory(title="폴더 선택")
    if path:
        label.config(text=path)

button1=tkinter.Button(window, text="파일 열기", width=20, command=open_file)
button2=tkinter.Button(window, text="파일 저장", width=20, command=save_file)
button3=tkinter.Button(window, text="폴더 선택", width=20, command=open_directory)

button1.pack(pady=5)
button2.pack(pady=5)
button3.pack(pady=5)
label.pack(pady=20)

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

path = filedialog.askopenfilename(
    title="파일 열기",
    initialdir=".",
    filetypes=(("텍스트 파일", "*.txt"), ("Python 파일", "*.py"), ("모든 파일", "*.*"))
)

{% endhighlight %}

`filedialog.askopenfilename()`을 이용하여 `파일 열기 대화 상자`를 표시할 수 있습니다.

사용자가 파일을 선택하면 **파일의 전체 경로**가 문자열로 반환되고, 취소하면 **빈 문자열**이 반환됩니다.

`filetypes`는 `(표시 이름, 패턴)` 튜플의 목록으로, 대화 상자의 **파일 형식 필터**를 구성합니다.

<br>

{% highlight Python %}

path = filedialog.asksaveasfilename(
    title="파일 저장",
    defaultextension=".txt",
    filetypes=(("텍스트 파일", "*.txt"), ("모든 파일", "*.*"))
)

{% endhighlight %}

`filedialog.asksaveasfilename()`은 저장할 **경로만 반환**하며, 실제로 파일을 생성하지는 않습니다.

반환된 경로로 직접 `open()`을 호출해 내용을 저장해야 합니다.

`defaultextension`을 설정하면 사용자가 확장자를 입력하지 않았을 때 자동으로 덧붙입니다.

<br>

- Tip : 파일 대화 상자는 취소 시 빈 문자열을 반환하므로, 반환값을 사용하기 전에 `if path:`로 **선택 여부를 확인**해야 합니다.

- Tip : 반환되는 경로의 구분자는 운영체제와 관계없이 `/`로 통일됩니다.

<br>
<br>

## 파일 대화 상자 함수

| 함수 | 의미 | 반환값 |
|:---:|:---:|:---:|
| askopenfilename() | 파일 한 개 열기 | 경로 문자열 |
| askopenfilenames() | 파일 여러 개 열기 | 경로 튜플 |
| asksaveasfilename() | 저장 경로 선택 | 경로 문자열 |
| askdirectory() | 폴더 선택 | 경로 문자열 |
| askopenfile() | 파일 한 개를 열어 파일 객체 반환 | 파일 객체 |
| asksaveasfile() | 저장 파일을 생성해 파일 객체 반환 | 파일 객체 |

<br>

## 파일 대화 상자 Parameter

| 이름 | 의미 | 기본값 | 속성 |
|:---:|:---:|:---:|:---:|
| title | 대화 상자 제목 | - | 문자열 |
| initialdir | 처음 표시할 폴더 | 현재 작업 폴더 | 경로 문자열 |
| initialfile | 처음 입력되어 있는 파일 이름 | - | 문자열 |
| filetypes | 파일 형식 필터 | 모든 파일 | (이름, 패턴) 튜플의 목록 |
| defaultextension | 확장자 미입력 시 덧붙일 확장자 | - | 문자열 |
| multiple | 여러 파일 선택 허용 | False | Boolean |
| mustexist | 존재하는 폴더만 선택 허용 (askdirectory) | False | Boolean |
| parent | 대화 상자를 소유할 창 | - | 위젯 |

<br>
<br>

## 입력 대화 상자 사용

{% highlight Python %}

import tkinter
from tkinter import simpledialog
from tkinter import colorchooser

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("640x400+100+100")
window.resizable(False, False)

label=tkinter.Label(window, text="결과가 표시됩니다.", width=40, height=3)

def ask_string():
    name = simpledialog.askstring("이름 입력", "이름을 입력하세요.", parent=window)
    if name is not None:
        label.config(text="이름 : " + name)

def ask_integer():
    age = simpledialog.askinteger("나이 입력", "나이를 입력하세요.", parent=window, minvalue=0, maxvalue=150)
    if age is not None:
        label.config(text="나이 : " + str(age))

def ask_color():
    color = colorchooser.askcolor(title="색상 선택", parent=window)
    if color[1] is not None:
        label.config(text="색상 : " + color[1], bg=color[1])

button1=tkinter.Button(window, text="문자열 입력", width=20, command=ask_string)
button2=tkinter.Button(window, text="정수 입력", width=20, command=ask_integer)
button3=tkinter.Button(window, text="색상 선택", width=20, command=ask_color)

button1.pack(pady=5)
button2.pack(pady=5)
button3.pack(pady=5)
label.pack(pady=20)

window.mainloop()

{% endhighlight %}

<br>

{% highlight Python %}

name = simpledialog.askstring("이름 입력", "이름을 입력하세요.", parent=window)

{% endhighlight %}

`simpledialog.askstring(제목, 내용)`을 이용하여 `문자열 입력 대화 상자`를 표시할 수 있습니다.

`askinteger`와 `askfloat`는 입력값을 각각 **정수와 실수로 변환**해 반환하며, 숫자가 아닌 값을 입력하면 오류 메시지를 표시하고 다시 입력을 요구합니다.

`minvalue`와 `maxvalue`를 설정하면 입력 가능한 **값의 범위**를 제한할 수 있습니다.

취소하면 `None`이 반환되므로, 빈 문자열 입력과 취소를 구분하려면 `is not None`으로 확인합니다.

<br>

{% highlight Python %}

color = colorchooser.askcolor(title="색상 선택", parent=window)

{% endhighlight %}

`colorchooser.askcolor()`를 이용하여 `색상 선택 대화 상자`를 표시할 수 있습니다.

반환값은 `((R, G, B), "#RRGGBB")` 형태의 튜플이며, 취소하면 `(None, None)`이 반환됩니다.

위젯의 `bg`, `fg` 등 색상 매개변수에는 두 번째 값인 **16진수 문자열**을 그대로 사용할 수 있습니다.

<br>
<br>

## 입력 대화 상자 함수

| 함수 | 의미 | 반환값 |
|:---:|:---:|:---:|
| simpledialog.askstring(title, prompt) | 문자열 입력 | 문자열, None |
| simpledialog.askinteger(title, prompt) | 정수 입력 | 정수, None |
| simpledialog.askfloat(title, prompt) | 실수 입력 | 실수, None |
| colorchooser.askcolor() | 색상 선택 | ((R, G, B), "#RRGGBB"), (None, None) |

<br>

## 입력 대화 상자 Parameter

| 이름 | 의미 | 기본값 | 속성 |
|:---:|:---:|:---:|:---:|
| title | 대화 상자 제목 | - | 문자열 |
| prompt | 입력 안내 문구 | - | 문자열 |
| initialvalue | 처음 입력되어 있는 값 | - | 문자열, 정수, 실수 |
| minvalue | 입력 가능한 최솟값 (askinteger, askfloat) | - | 정수, 실수 |
| maxvalue | 입력 가능한 최댓값 (askinteger, askfloat) | - | 정수, 실수 |
| show | 입력 문자를 대신 표시할 문자 (askstring) | - | 문자열 |
| color / initialcolor | 처음 선택되어 있는 색상 (askcolor) | - | 색상 문자열 |
| parent | 대화 상자를 소유할 창 | - | 위젯 |

<br>

- Tip : `askstring`에 `show="*"`를 설정하면 **비밀번호 입력**처럼 입력 문자를 가릴 수 있습니다.

- Tip : 메인 창을 만들지 않고 대화 상자만 사용하려면 `window = tkinter.Tk()` 뒤에 `window.withdraw()`로 메인 창을 숨긴 뒤 호출합니다.
