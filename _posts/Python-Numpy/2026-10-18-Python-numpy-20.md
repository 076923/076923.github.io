---
layout: post
title: "Python Numpy 강좌 : 제 20강 - 파일 입출력"
tagline: "Python Numpy File I/O"
image: /assets/images/numpy.png
tags: ['Numpy']
keywords: Python, Python Numpy, Numpy save, Numpy load, Numpy savez, Numpy savez_compressed, Numpy savetxt, Numpy loadtxt, Numpy genfromtxt, npy, npz, csv
ref: Python-Numpy
category: Python
permalink: /posts/Python-numpy-20/
comments: true
toc: true
---

## 배열 저장 및 불러오기(Save & Load)

{% highlight Python %}

import numpy as np

array = np.arange(6).reshape(2, 3)

np.save("array.npy", array)
load = np.load("array.npy")

print(load)
print(load.dtype, load.shape)

{% endhighlight %}

**결과**
:    
[[0 1 2]<br>
&nbsp;[3 4 5]]<br>
int64 (2, 3)<br>
<br>

`저장 함수(np.save)`는 배열 하나를 numpy 전용 **바이너리 파일(.npy)**로 저장합니다.

`np.save(파일 경로, 배열)`로 배열을 저장하며, 파일 경로에 확장자가 없다면 `.npy`가 자동으로 추가됩니다.

`불러오기 함수(np.load)`는 저장된 파일을 다시 numpy 배열로 불러옵니다.

`.npy` 파일은 배열의 값뿐만 아니라 **자료형(dtype)**과 **형태(shape)**, **메모리 레이아웃**까지 함께 저장합니다.

그러므로 불러온 배열은 저장하기 전의 배열과 **완전히 같은 배열**이 되며, 형태를 다시 지정하거나 자료형을 변환할 필요가 없습니다.

- Tip : `.npy` 파일은 값을 텍스트로 변환하지 않고 메모리의 내용을 그대로 기록하므로, 텍스트 파일보다 **저장과 불러오기가 빠르고** 실수의 정밀도도 손실되지 않습니다.

<br>
<br>

## 여러 배열 저장(Savez)

{% highlight Python %}

import numpy as np

a = np.array([1, 2, 3])
b = np.array([
    [1.5, 2.5],
    [3.5, 4.5]
])

np.savez("arrays.npz", first=a, second=b)

with np.load("arrays.npz") as data:
    print(data.files)
    print(data["first"])
    print(data["second"])

np.savez("noname.npz", a, b)

with np.load("noname.npz") as data:
    print(data.files)

{% endhighlight %}

**결과**
:    
['first', 'second']<br>
[1 2 3]<br>
[[1.5 2.5]<br>
&nbsp;[3.5 4.5]]<br>
['arr_0', 'arr_1']<br>
<br>

`다중 저장 함수(np.savez)`는 **여러 개의 배열**을 하나의 `.npz` 파일로 저장합니다.

`np.savez(파일 경로, 이름1=배열1, 이름2=배열2, ...)`로 배열을 저장합니다.

`.npz` 파일을 불러오면 배열이 아닌 `NpzFile` 객체가 반환되며, **사전(dict)**처럼 저장할 때 지정한 이름으로 각 배열을 조회합니다.

`files` 속성은 파일에 저장된 **배열의 이름 목록**을 반환합니다.

이름을 지정하지 않고 배열만 입력하면, 입력한 순서대로 `arr_0`, `arr_1`, ...의 이름이 자동으로 부여됩니다.

`NpzFile` 객체는 배열을 조회할 때 파일에서 읽어 오므로, 사용이 끝나면 파일을 닫아야 합니다. `with` 구문을 사용하면 블록이 끝날 때 **자동으로 파일이 닫힙니다.**

<br>
<br>

## 압축 저장(Savez Compressed)

{% highlight Python %}

import os
import numpy as np

big = np.zeros((1000, 1000))

np.savez("plain.npz", big=big)
np.savez_compressed("compressed.npz", big=big)

print(os.path.getsize("plain.npz"))
print(os.path.getsize("compressed.npz"))

{% endhighlight %}

**결과**
:    
8000260<br>
7999<br>
<br>

`압축 저장 함수(np.savez_compressed)`는 `np.savez`와 사용 방법이 같지만, 배열을 **압축**해서 저장합니다.

예제의 배열은 `float64` 자료형의 원소 100만 개로 구성되어 있으므로, 압축하지 않으면 약 `8MB`의 파일이 생성됩니다.

같은 배열을 압축해서 저장하면 약 `8KB`로 **천 배 가까이** 크기가 줄어듭니다.

단, 압축률은 데이터의 특성에 따라 크게 달라집니다. 예제처럼 같은 값이 반복되는 배열은 압축이 잘 되지만, 무작위 실수처럼 규칙이 없는 데이터는 크기가 거의 줄어들지 않습니다.

압축된 파일은 저장하고 불러올 때 압축과 해제 과정이 추가되므로 **시간이 더 오래 걸립니다.** 저장 공간과 속도 중 무엇이 중요한지에 따라 선택합니다.

<br>
<br>

## 텍스트 파일 저장(Savetxt)

{% highlight Python %}

import numpy as np

array = np.array([
    [1.5, 2.0, 3.25],
    [4.0, 5.5, 6.0]
])

np.savetxt("array.txt", array)
np.savetxt("array.csv", array, delimiter=",", fmt="%.2f", header="x,y,z", comments="")

{% endhighlight %}

**array.txt**
:    
1.500000000000000000e+00 2.000000000000000000e+00 3.250000000000000000e+00<br>
4.000000000000000000e+00 5.500000000000000000e+00 6.000000000000000000e+00<br>
<br>

**array.csv**
:    
x,y,z<br>
1.50,2.00,3.25<br>
4.00,5.50,6.00<br>
<br>

`텍스트 저장 함수(np.savetxt)`는 배열을 사람이 읽을 수 있는 **텍스트 파일**로 저장합니다.

`np.savetxt(파일 경로, 배열, fmt=형식, delimiter=구분자, header=머리글, comments=주석 기호)`로 배열을 저장합니다.

`형식(fmt)`의 기본값은 `'%.18e'`이며, 모든 값을 **지수 표기법**으로 소수점 아래 18자리까지 기록합니다. `fmt='%.2f'`처럼 지정하면 원하는 자릿수로 저장할 수 있으며, 정수는 `'%d'`를 사용합니다.

`구분자(delimiter)`의 기본값은 공백입니다. `delimiter=","`로 지정하면 엑셀 등에서 열 수 있는 **CSV 파일**이 됩니다.

`머리글(header)`은 파일의 첫 줄에 기록되며, 기본적으로 앞에 `'# '`이 붙어 **주석**으로 저장됩니다. `comments=""`로 지정하면 주석 기호 없이 머리글을 그대로 기록합니다.

`np.savetxt`는 **1차원 또는 2차원 배열**만 저장할 수 있습니다. 3차원 이상의 배열은 `reshape`로 차원을 줄이거나 `np.save`를 사용합니다.

<br>
<br>

## 텍스트 파일 불러오기(Loadtxt)

{% highlight Python %}

import numpy as np

load = np.loadtxt("array.csv", delimiter=",", skiprows=1)
print(load)

column = np.loadtxt("array.csv", delimiter=",", skiprows=1, usecols=(0, 2))
print(column)

x, z = np.loadtxt("array.csv", delimiter=",", skiprows=1, usecols=(0, 2), unpack=True)
print(x, z)

{% endhighlight %}

**결과**
:    
[[1.5&nbsp;&nbsp;2.&nbsp;&nbsp;&nbsp;3.25]<br>
&nbsp;[4.&nbsp;&nbsp;&nbsp;5.5&nbsp;&nbsp;6.&nbsp;&nbsp;]]<br>
[[1.5&nbsp;&nbsp;3.25]<br>
&nbsp;[4.&nbsp;&nbsp;&nbsp;6.&nbsp;&nbsp;]]<br>
[1.5 4. ] [3.25 6.&nbsp;&nbsp;]<br>
<br>

`텍스트 불러오기 함수(np.loadtxt)`는 텍스트 파일을 읽어 numpy 배열로 반환합니다.

`np.loadtxt(파일 경로, dtype=float, delimiter=구분자, skiprows=건너뛸 줄 수, usecols=열 번호, unpack=False)`로 파일을 불러옵니다.

`skiprows=1`은 파일의 첫 줄인 **머리글을 건너뜁니다.** `#`으로 시작하는 주석 줄은 지정하지 않아도 자동으로 무시됩니다.

`usecols`에 열 번호를 튜플로 입력하면 **필요한 열만** 불러옵니다.

`unpack=True`로 지정하면 결과를 **전치**해서 반환하므로, 각 열을 서로 다른 변수에 나누어 할당할 수 있습니다.

자료형(dtype)의 기본값은 `float`이므로, 파일에 정수만 있어도 실수 배열로 반환됩니다. 정수 배열이 필요하다면 `dtype=int`를 지정합니다.

<br>
<br>

## 결측값이 있는 파일(Genfromtxt)

{% highlight Python %}

import numpy as np

with open("missing.csv", "w") as file:
    file.write("x,y,z\n1,2,3\n4,,6\n7,8,\n")

try:
    np.loadtxt("missing.csv", delimiter=",", skiprows=1)
except ValueError as e:
    print(e)

data = np.genfromtxt("missing.csv", delimiter=",", skip_header=1)
print(data)
print(np.isnan(data).sum())

fill = np.genfromtxt("missing.csv", delimiter=",", skip_header=1, filling_values=0)
print(fill)

{% endhighlight %}

**결과**
:    
could not convert string '' to float64 at row 1, column 2.<br>
[[ 1.&nbsp;&nbsp;2.&nbsp;&nbsp;3.]<br>
&nbsp;[ 4. nan&nbsp;&nbsp;6.]<br>
&nbsp;[ 7.&nbsp;&nbsp;8. nan]]<br>
2<br>
[[1. 2. 3.]<br>
&nbsp;[4. 0. 6.]<br>
&nbsp;[7. 8. 0.]]<br>
<br>

실제 데이터에는 값이 비어 있는 **결측값(Missing Value)**이 자주 포함됩니다.

`np.loadtxt`는 빈 값을 숫자로 변환할 수 없으므로 `ValueError`가 발생합니다.

`np.genfromtxt`는 결측값을 처리할 수 있는 텍스트 불러오기 함수이며, 비어 있는 값을 **`NaN`으로 채워** 반환합니다.

`np.genfromtxt`에서는 머리글을 건너뛰는 매개변수의 이름이 `skip_header`입니다. `np.loadtxt`의 `skiprows`와 이름이 다르므로 주의합니다.

`np.isnan(배열).sum()`으로 결측값의 개수를 확인할 수 있습니다. `True`는 `1`, `False`는 `0`으로 계산되기 때문입니다.

`filling_values`를 지정하면 결측값을 `NaN` 대신 **지정한 값**으로 채웁니다.

- Tip : `NaN`이 포함된 배열에 `np.mean`, `np.sum` 등을 사용하면 결과도 `NaN`이 됩니다. 결측값을 제외하고 계산하려면 `np.nanmean`, `np.nansum`처럼 이름 앞에 `nan`이 붙은 함수를 사용합니다.

<br>
<br>

## 머리글을 이름으로 사용하기

{% highlight Python %}

import numpy as np

data = np.genfromtxt("missing.csv", delimiter=",", names=True)

print(data.dtype.names)
print(data["y"])

{% endhighlight %}

**결과**
:    
('x', 'y', 'z')<br>
[ 2. nan&nbsp;&nbsp;8.]<br>
<br>

`names=True`로 지정하면 파일의 첫 줄을 **열의 이름**으로 사용합니다.

반환되는 배열은 각 열에 이름이 붙은 `구조화 배열(Structured Array)`이며, `data["y"]`처럼 **열 이름으로** 값을 조회할 수 있습니다.

머리글을 이름으로 사용하므로 `skip_header`를 따로 지정할 필요가 없습니다.

- Tip : 열마다 자료형이 다르거나 열 이름으로 다루는 작업이 많다면 `pandas` 라이브러리의 `pd.read_csv`가 더 편리합니다. numpy의 텍스트 함수는 **숫자로 구성된 단순한 표**에 적합합니다.

<br>
<br>

## 객체 배열과 보안

{% highlight Python %}

import numpy as np

array = np.array([{"a": 1}, [1, 2]], dtype=object)
np.save("object.npy", array)

try:
    np.load("object.npy")
except ValueError as e:
    print(e)

print(np.load("object.npy", allow_pickle=True))

{% endhighlight %}

**결과**
:    
Object arrays cannot be loaded when allow_pickle=False<br>
[{'a': 1} list([1, 2])]<br>
<br>

사전이나 리스트처럼 파이썬 객체를 담은 `객체 배열(dtype=object)`은 파이썬의 `pickle` 형식으로 저장됩니다.

`pickle`은 파일을 불러오는 과정에서 **임의의 코드를 실행**할 수 있으므로, 신뢰할 수 없는 파일을 불러오면 보안 문제가 발생할 수 있습니다.

그러므로 `np.load`의 `allow_pickle` 매개변수 기본값은 `False`이며, 객체 배열을 불러오면 `ValueError`가 발생합니다.

**직접 저장한 파일처럼 출처를 신뢰할 수 있는 경우에만** `allow_pickle=True`를 지정합니다.

<br>
<br>

## 문자열과 인코딩

{% highlight Python %}

import numpy as np

text = np.array(["사과", "바나나"])

np.savetxt("text.txt", text, fmt="%s", encoding="utf-8")
print(np.loadtxt("text.txt", dtype=str, encoding="utf-8"))

{% endhighlight %}

**결과**
:    
['사과' '바나나']<br>
<br>

문자열 배열을 텍스트 파일로 저장하려면 `fmt="%s"`를 지정합니다. 기본 형식인 `'%.18e'`는 숫자에만 사용할 수 있습니다.

한글처럼 영문이 아닌 문자가 포함되어 있다면 `encoding="utf-8"`을 **저장과 불러오기에 모두** 지정합니다.

인코딩을 지정하지 않으면 운영체제의 기본 인코딩이 사용될 수 있어, 다른 환경에서 파일을 열었을 때 글자가 깨질 수 있습니다.

<br>
<br>

## 파일 입출력 함수 정리

| 함수 | 파일 형식 | 설명 |
|:---:|:---:|:---|
| np.save | .npy | 배열 하나를 바이너리로 저장 |
| np.savez | .npz | 여러 배열을 이름과 함께 저장 |
| np.savez_compressed | .npz | 여러 배열을 압축해 저장 |
| np.load | .npy, .npz | 바이너리 파일 불러오기 |
| np.savetxt | 텍스트, .csv | 1차원 또는 2차원 배열을 텍스트로 저장 |
| np.loadtxt | 텍스트, .csv | 결측값이 없는 텍스트 파일 불러오기 |
| np.genfromtxt | 텍스트, .csv | 결측값과 열 이름을 처리하며 불러오기 |

<br>

- Tip : numpy 프로그램에서만 사용할 데이터라면 `.npy`나 `.npz`를, 다른 프로그램이나 사람과 공유할 데이터라면 `.csv` 같은 텍스트 형식을 사용합니다.
