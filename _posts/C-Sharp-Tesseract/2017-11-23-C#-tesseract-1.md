---
layout: post
title: "C# Tesseract 강좌 : 제 1강 - Tesseract 설치"
tagline: "C# Tesseract 3.0.2 install"
image: /assets/images/tesseract.PNG
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['Tesseract']
keywords: C#, Visual Studio, tesseract, ocr, tesseract install
ref: C#-Tesseract
category: C#
permalink: /posts/C-tesseract-1/
comments: true
toc: true
---

## Tesseract

![0]({{ site.images }}/assets/posts/C-Sharp/Tesseract/lecture-1/0.jpg)

`Tesseract - OCR`은 **문자를 판독**해주는 오픈 소스 라이브러리입니다.

약 60여개 국가의 이미지로된 언어를 판독하여 `text`형식으로 반환해줍니다.

`자동차 번호판 인식`, `명함의 문자 인식`, `문서의 문자 인식` 등에 이용할 수 있습니다.

<br>
<br>

## Tesseract 설치

![1]({{ site.images }}/assets/posts/C-Sharp/Tesseract/lecture-1/1.png)

`프로젝트` → `NuGet 패키지 관리(N)...`을 통하여 Tesseract를 쉽게 설치할 수 있습니다.

<br>

![2]({{ site.images }}/assets/posts/C-Sharp/Tesseract/lecture-1/2.png)

위와 같은 화면이 나오게 되고 `찾아보기`를 눌러 `검색창`에 `tesseract`를 검색합니다.

현재 NuGet 패키지를 통해 설치할 수 있는 최신 버전은 `3.0.2`버전입니다.

이후, 상단의 `▶시작` 버튼을 눌러 `Tesseract`를 프로젝트에 등록합니다.

NuGet 패키지를 통하여 `3.0.2` 버전을 설치를 완료하셨다면 `언어 데이터 파일`을 설치하셔야합니다.

아래의 `다운로드`링크를 통하면 별도의 추가 설치 없이 사용 가능합니다.

`한국어 / 영어 언어 데이터 파일` : [다운로드][download]

<br>
<br>

## Tesseract 다른 버전 / 언어 데이터 파일

`3.0.2 버전보다 높거나 낮은 버전을 설치 시` : [Tesseract alpha 버전 / 구 버전 설치][tesseract_install] 

`3.0.2 버전과 다른 경우 또는 다른 언어 사용 시` : [언어 데이터 파일][tesseract_data]

[download]: https://github.com/076923/076923.github.io/raw/master/download/tesseract-ocr/tessdata.zip
[tesseract_install]: https://www.nuget.org/packages/Tesseract/
[tesseract_data]: https://github.com/tesseract-ocr/tesseract/wiki/Data-Files#data-files-for-version-302/
