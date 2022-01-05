---
layout: post
title: "C# Tesseract 강좌 : 제 2강 - 영문자 판독"
tagline: "C# Tesseract English OCR"
image: /assets/images/tesseract.PNG
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Tesseract']
keywords: C#, Visual Studio, tesseract, ocr, tesseract english
ref: C#-Tesseract
category: C#
permalink: /posts/C-tesseract-2/
comments: true
toc: true
---

## Tesseract English OCR ##

![1]({{ site.images }}/assets/posts/C-Sharp/Tesseract/lecture-2/1.webp){:class="lazyload" width="100%" height="100%"}

`Tesseract - OCR`를 이용하여 `Bitmap`으로된 이미지 파일에서 영문자를 인식하여 `string`형식으로 반환하여 인식합니다.

<br>
<br>

## Tesseract 준비 ##

![2]({{ site.images }}/assets/posts/C-Sharp/Tesseract/lecture-2/2.webp){:class="lazyload" width="100%" height="100%"}

1강에서 설치한 `Tesseract 언어 데이터 파일`을 `프로젝트/bin/Debug`에 저장합니다.

- `Tesseract 설치하기` : [1강 바로가기][1강] 

<br>

![3]({{ site.images }}/assets/posts/C-Sharp/Tesseract/lecture-2/3.webp){:class="lazyload" width="100%" height="100%"}

`tessdata` 폴더에 위와 같은 `eng.*` 파일이 정상적으로 저장되어있는지 확인합니다.

<br>
<br>

## 프로젝트 구성

![4]({{ site.images }}/assets/posts/C-Sharp/Tesseract/lecture-2/4.webp){:class="lazyload" width="100%" height="100%"}

`Form`창에 위와 같이 `pictureBox`와 `button`을 배치합니다. `pictureBox`에 이미지를 등록합니다.

<br>
<br>

## 전체 코드

{% highlight C# %}

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Tesseract;

namespace tesseract
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Bitmap img = new Bitmap(pictureBox1.Image);
            var ocr = new TesseractEngine("./tessdata", "eng", EngineMode.TesseractAndCube);
            var texts = ocr.Process(img);
            MessageBox.Show(texts.GetText());
        }
    }
}

{% endhighlight %}

<br>

### 세부 코드

{% highlight C# %}

using Tesseract;

{% endhighlight %}

`namespace`에 `Tesseract`를 사용할 수 있도록 선언합니다.

<br>

{% highlight C# %}

Bitmap img = new Bitmap(pictureBox1.Image);

{% endhighlight %}

`pictureBox1`의 이미지를 Bitmap으로 변환하여 `img` 변수에 저장합니다.

<br>

{% highlight C# %}

var ocr = new TesseractEngine("./tessdata", "eng", EngineMode.TesseractAndCube);

{% endhighlight %}

`ocr` 변수에 `TesseractEngine()`을 이용하여 `언어 데이터 파일`을 사용하여 판독합니다.

`TesseractEngine(언어 데이터 파일 경로, 언어, 엔진모드)`입니다. 영어의 경우 `eng`입니다.

<br>

* `EngineMode.*`

    * `EngineMode.Default` : 기본값으로 판독
    * `EngineMode.CubeOnly` : 큐브 방식으로 정확도는 높아지지만, 속도가 느림
    * `EngineMode.TesseractOnly` : Tesseract 방식만 실행하며, 속도가 가장 빠름
    * `EngineMode.TesseractAndCube` : 큐브와 Tesseract 방식의 결합, 가장 높은 정확도

<br>

{% highlight C# %}

var texts = ocr.Process(img);
MessageBox.Show(texts.GetText());

{% endhighlight %}

`texts`에 `ocr`에서 셋팅된 방법으로 `img`를 이용해 판독할 `문자들`을 저장합니다.

`texts.GetText()`를 이용하여 `string`형태로 불러올 수 있습니다.

<br>
<br>

## 출력 결과

![5]({{ site.images }}/assets/posts/C-Sharp/Tesseract/lecture-2/5.webp){:class="lazyload" width="100%" height="100%"}

[1강]: https://076923.github.io/posts/C-tesseract-1/
