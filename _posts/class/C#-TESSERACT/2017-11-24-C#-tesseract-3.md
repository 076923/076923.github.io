---
layout: post
title: "C# Tesseract 강좌 : 제 3강 - 한글 판독"
tagline: "C# Tesseract Korea OCR"
image: /assets/images/tesseract.PNG
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['C#-Tesseract']
keywords: C#, Visual Studio, tesseract, ocr, tesseract korea
ref: C#-Tesseract
category: posts
permalink: /posts/C-tesseract-3/
comments: true
---

## Tesseract Korea OCR ##
----------

![1]({{ site.images }}/assets/images/C/tesseract/ch3/1.png)
`Tesseract - OCR`를 이용하여 `Bitmap`으로된 이미지 파일에서 한글을 인식하여 `string`형식으로 반환하여 인식합니다.

<br>
<br>

### Tesseract 준비 ###
----------

![2]({{ site.images }}/assets/images/C/tesseract/ch3/2.png)
1강에서 설치한 `Tesseract 언어 데이터 파일`을 `프로젝트/bin/Debug`에 저장합니다.

<br>

`Tesseract 설치하기` : [1강 바로가기][1강] 

<br>
<br>

![3]({{ site.images }}/assets/images/C/tesseract/ch3/3.png)
`tessdata` 폴더에 위와 같은 `kor.traineddata` 파일이 정상적으로 저장되어있는지 확인합니다.

<br>
<br>

## 프로젝트 구성 ##
----------

![4]({{ site.images }}/assets/images/C/tesseract/ch3/4.png)
`Form`창에 위와 같이 `pictureBox`와 `button`을 배치합니다. `pictureBox`에 이미지를 등록합니다.

<br>
<br>

## Main ##
----------

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
            var ocr = new TesseractEngine("./tessdata", "kor", EngineMode.Default);
            var texts = ocr.Process(img);
            MessageBox.Show(texts.GetText());
        }
    }
}

{% endhighlight %}

<br>
<br>

## Main Code ##
----------

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

var ocr = new TesseractEngine("./tessdata", "kor", EngineMode.Default);

{% endhighlight %}

`ocr` 변수에 `TesseractEngine()`을 이용하여 `언어 데이터 파일`을 사용하여 판독합니다. `TesseractEngine(언어 데이터 파일 경로, 언어, 엔진모드)`입니다. 한글의 경우 `kor`입니다.

<br>

* Tip : 한글은 `EngineMode.Default`만 지원합니다.

<br>

{% highlight C# %}

var texts = ocr.Process(img);
MessageBox.Show(texts.GetText());

{% endhighlight %}

`texts`에 `ocr`에서 셋팅된 방법으로 `img`를 이용해 판독할 `문자들`을 저장합니다. `texts.GetText()`를 이용하여 `string`형태로 불러올 수 있습니다.

<br>
<br>

## Result ##
----------

![5]({{ site.images }}/assets/images/C/tesseract/ch3/5.png)

[1강]: https://076923.github.io/posts/C-tesseract-1/

