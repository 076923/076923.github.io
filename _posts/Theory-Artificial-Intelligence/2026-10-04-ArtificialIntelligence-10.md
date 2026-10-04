---
layout: post
title: "Artificial Intelligence Theory : 최적화 알고리즘(Optimizer)"
tagline: "Optimizer"
image: /assets/images/ai.jpg
header:
  image: /assets/patterns/asanoha-400px.png
tags: ['AI']
keywords: Artificial Intelligence, Deep Learning, Optimizer, Gradient Descent, Stochastic Gradient Descent, SGD, Mini-batch, Momentum, Nesterov, AdaGrad, RMSProp, Adam, AdamW, Learning Rate, Learning Rate Scheduler, Warmup, Cosine Annealing
ref: Theory-AI
category: Theory
permalink: /posts/AI-10/
comments: true
toc: true
plotly: true
---

## 최적화(Optimization)

[순전파(Forward Propagation) & 역전파(Back Propagation)][AI-5강]에서 역전파는 손실 함수를 각 가중치로 미분한 `기울기(Gradient)`를 구하는 과정이라고 설명했습니다. 그러나 기울기는 **어느 방향으로 가중치를 바꾸면 손실이 줄어드는지**를 알려 줄 뿐, 실제로 가중치를 얼마나 어떻게 바꿀지는 정해 주지 않습니다.

기울기를 받아 가중치를 **실제로 갱신하는 규칙**이 `최적화 알고리즘(Optimizer)`입니다. 같은 모델, 같은 데이터, 같은 기울기라도 최적화 알고리즘에 따라 학습 속도와 최종 성능이 크게 달라지므로, 역전파가 지도를 그리는 일이라면 최적화 알고리즘은 그 지도를 보고 **어떻게 걸을지 정하는 일**에 해당합니다.

손실 함수를 가중치 공간 위에 펼친 지형으로 상상하면, 최적화는 이 지형에서 **가장 낮은 골짜기를 찾아 내려가는 과정**입니다. 다만 수백만 개의 가중치가 만드는 지형은 한눈에 볼 수 없고, 현재 서 있는 지점의 경사만 알 수 있습니다. 그러므로 모든 최적화 알고리즘은 **발밑의 경사만 보고 다음 발걸음을 정하는** 방식으로 동작합니다.

<br>
<br>

## 경사 하강법(Gradient Descent)

$$ w \leftarrow w - \eta \frac{\partial \mathcal{L}}{\partial w} $$

<br>

가장 기본이 되는 규칙은 `경사 하강법(Gradient Descent)`입니다. 기울기는 손실이 **가장 가파르게 증가하는 방향**을 가리키므로, 그 반대 방향으로 가중치를 조금 옮기면 손실이 줄어듭니다. 이 과정을 반복해 골짜기로 내려갑니다.

여기서 $$ \eta $$는 한 번에 얼마나 움직일지를 정하는 `학습률(Learning Rate)`로, 최적화에서 가장 중요한 하이퍼파라미터입니다. 학습률이 너무 작으면 골짜기에 도달하기까지 지나치게 많은 반복이 필요하고, 너무 크면 골짜기를 건너뛰어 반대편 비탈로 올라가 버리며 손실이 오히려 커지거나 발산합니다.

경사 하강법은 기울기를 계산할 때 **몇 개의 데이터를 사용하느냐**에 따라 세 가지로 나뉩니다.

- `배치 경사 하강법(Batch Gradient Descent)` : 전체 훈련 데이터로 기울기를 구한 뒤 한 번 갱신합니다. 기울기가 정확하지만 데이터가 많으면 한 걸음에 드는 비용이 너무 큽니다.
- `확률적 경사 하강법(Stochastic Gradient Descent, SGD)` : 데이터 **하나**로 기울기를 구해 갱신합니다. 매우 빠르지만 데이터 하나의 기울기는 전체의 기울기와 크게 다를 수 있어 경로가 심하게 요동칩니다.
- `미니배치 경사 하강법(Mini-batch Gradient Descent)` : 데이터를 수십에서 수백 개 단위의 `미니배치(Mini-batch)`로 묶어 기울기를 구합니다. 두 방법의 절충안이며, GPU의 병렬 연산과도 잘 맞아 **사실상 표준**으로 사용됩니다.

미니배치의 기울기는 전체 기울기의 **추정치**이므로 매 걸음마다 약간의 잡음이 섞입니다. 이 잡음은 단점만은 아닙니다. 지형의 얕은 웅덩이에 갇혔을 때 잡음 덕분에 빠져나올 수 있으며, [과대적합과 정칙화(Overfitting & Regularization)][AI-9강]에서 다룬 것처럼 부수적인 정칙화 효과도 냅니다.

- Tip : 오늘날 딥러닝에서 **SGD**라 부르는 것은 거의 항상 미니배치 경사 하강법을 뜻합니다. PyTorch의 `optim.SGD` 역시 미니배치 단위로 동작하며, 배치 크기는 `DataLoader`의 `batch_size`로 정합니다.

<br>
<br>

## 경사 하강법의 한계

<center>
<div id="OptimizerPathPlot" style="width:100%;max-width:700px"></div>
<script>
(function(){
  var gx=function(x,y){return 2*x/100;};
  var gy=function(x,y){return 2*y;};
  function run(kind,steps){
    var x=-9,y=1.0,vx=0,vy=0,sx=0,sy=0,mx=0,my=0,b1=0.9,b2=0.999;
    var X=[x],Y=[y];
    for(var t=1;t<=steps;t++){
      var dx=gx(x,y),dy=gy(x,y);
      if(kind==="sgd"){x-=0.9*dx;y-=0.9*dy;}
      else if(kind==="momentum"){vx=0.9*vx-0.9*dx;vy=0.9*vy-0.9*dy;x+=vx;y+=vy;}
      else if(kind==="adam"){
        mx=b1*mx+(1-b1)*dx;my=b1*my+(1-b1)*dy;
        sx=b2*sx+(1-b2)*dx*dx;sy=b2*sy+(1-b2)*dy*dy;
        var mhx=mx/(1-Math.pow(b1,t)),mhy=my/(1-Math.pow(b1,t));
        var shx=sx/(1-Math.pow(b2,t)),shy=sy/(1-Math.pow(b2,t));
        x-=0.3*mhx/(Math.sqrt(shx)+1e-8);y-=0.3*mhy/(Math.sqrt(shy)+1e-8);
      }
      X.push(x);Y.push(y);
    }
    return {x:X,y:Y};
  }
  var cx=[],cy=[],cz=[];
  for(var i=-10;i<=2;i+=0.25){cx.push(i);}
  for(var j=-1.5;j<=1.5;j+=0.1){cy.push(j);}
  for(var a=0;a<cy.length;a++){var row=[];for(var b=0;b<cx.length;b++){row.push(cx[b]*cx[b]/100+cy[a]*cy[a]);}cz.push(row);}
  var s=run("sgd",40),m=run("momentum",40),ad=run("adam",40);
  Plotly.newPlot("OptimizerPathPlot",
    [{x:cx,y:cy,z:cz,type:"contour",showscale:false,contours:{coloring:"lines"},line:{color:"lightgray"},hoverinfo:"skip"},
     {x:s.x,y:s.y,mode:"lines+markers",name:"SGD",marker:{size:4}},
     {x:m.x,y:m.y,mode:"lines+markers",name:"Momentum",marker:{size:4}},
     {x:ad.x,y:ad.y,mode:"lines+markers",name:"Adam",marker:{size:4}}],
    {xaxis:{title:"w₁",range:[-10,2]},yaxis:{title:"w₂",range:[-1.5,1.5]},legend:{x:0.75,y:1}});
})();
</script>
</center>

위 그래프는 가로 방향으로는 완만하고 세로 방향으로는 가파른 **길쭉한 골짜기** 모양의 손실 지형입니다. 등고선이 좁게 모인 세로 방향은 조금만 움직여도 손실이 크게 변하고, 등고선이 넓게 퍼진 가로 방향은 많이 움직여도 손실이 조금밖에 변하지 않습니다.

이러한 지형에서 SGD는 **기울기가 큰 세로 방향으로 심하게 진동**하면서 정작 골짜기 바닥을 향하는 가로 방향으로는 조금씩밖에 전진하지 못합니다. 세로 방향의 진동을 줄이려고 학습률을 낮추면 가로 방향의 진행은 더 느려지므로, 학습률 하나로는 두 방향을 동시에 만족시킬 수 없습니다.

실제 신경망의 손실 지형은 이보다 훨씬 복잡합니다. 기울기가 거의 0이어서 움직임이 멈추는 `안장점(Saddle Point)`과 `평원(Plateau)`이 곳곳에 있고, 가중치마다 기울기의 크기도 수백 배씩 차이가 납니다. 이후의 최적화 알고리즘은 모두 이 문제들을 **두 가지 방향**에서 개선한 것입니다. 하나는 **이전 걸음의 방향을 기억**하는 것이고, 다른 하나는 **가중치마다 학습률을 다르게** 주는 것입니다.

<br>
<br>

## 모멘텀(Momentum)

$$ v \leftarrow \gamma v - \eta \frac{\partial \mathcal{L}}{\partial w}, \qquad w \leftarrow w + v $$

<br>

`모멘텀(Momentum)`은 공이 비탈을 굴러 내려가는 모습에서 착안한 방법입니다. 현재 기울기만으로 움직이지 않고, **이전까지의 이동 방향을 속도 $$ v $$로 누적**해 거기에 현재 기울기를 더합니다. $$ \gamma $$는 이전 속도를 얼마나 유지할지 정하는 값으로 보통 0.9를 사용합니다.

길쭉한 골짜기에서 세로 방향의 기울기는 매 걸음 부호가 바뀌므로 누적하면 서로 상쇄되어 진동이 줄어듭니다. 반면 가로 방향의 기울기는 항상 같은 부호이므로 누적되어 **점점 빨라집니다.** 위 그래프에서 Momentum의 경로가 SGD보다 진동이 작고 골짜기 바닥에 먼저 도달하는 이유입니다.

속도가 쌓이면 기울기가 거의 0인 평원이나 안장점도 관성으로 지나칠 수 있습니다. 다만 같은 이유로 골짜기 바닥을 지나쳐 반대편으로 올라갔다 되돌아오는 현상도 생기며, 이를 완화한 것이 `네스테로프 모멘텀(Nesterov Momentum)`입니다. 현재 위치가 아니라 **관성대로 움직였을 때 도착할 위치**에서 기울기를 계산해, 지나치기 전에 미리 속도를 줄입니다.

- Tip : PyTorch에서는 `optim.SGD(params, lr=0.01, momentum=0.9, nesterov=True)`처럼 SGD의 인자로 적용합니다. 이미지 분류처럼 잘 알려진 과제에서는 적절히 조정된 모멘텀 SGD가 Adam보다 **최종 성능이 높은 경우**도 많습니다.

<br>
<br>

## 적응적 학습률(Adaptive Learning Rate)

모멘텀이 방향을 다듬는 방법이라면, 적응적 학습률은 **가중치마다 보폭을 다르게** 주는 방법입니다. 기울기가 계속 큰 가중치는 보폭을 줄이고, 기울기가 작은 가중치는 보폭을 키워, 모든 가중치가 비슷한 속도로 골짜기에 접근하도록 만듭니다.

<br>

### AdaGrad

$$ s \leftarrow s + \left( \frac{\partial \mathcal{L}}{\partial w} \right)^{2}, \qquad w \leftarrow w - \frac{\eta}{\sqrt{s} + \epsilon} \frac{\partial \mathcal{L}}{\partial w} $$

<br>

`AdaGrad(Adaptive Gradient)`는 가중치마다 **지금까지의 기울기 제곱을 모두 누적**한 $$ s $$를 두고, 학습률을 $$ \sqrt{s} $$로 나눕니다. 기울기가 컸던 가중치일수록 $$ s $$가 커져 보폭이 작아지고, 기울기가 작았던 가중치는 상대적으로 큰 보폭을 유지합니다. $$ \epsilon $$은 0으로 나누는 것을 막기 위한 아주 작은 값입니다.

드물게 등장하는 특성에 연결된 가중치는 기울기가 작게 누적되므로 큰 보폭으로 빠르게 학습되며, 이 성질 덕분에 자연어 처리처럼 **희소한 입력**을 다루는 과제에서 효과적입니다. 그러나 $$ s $$는 줄어들지 않고 계속 커지기만 하므로, 학습이 길어지면 모든 가중치의 보폭이 0에 가까워져 **골짜기에 도달하기도 전에 학습이 멈추는** 문제가 있습니다.

<br>

### RMSProp

$$ s \leftarrow \beta s + (1 - \beta) \left( \frac{\partial \mathcal{L}}{\partial w} \right)^{2}, \qquad w \leftarrow w - \frac{\eta}{\sqrt{s} + \epsilon} \frac{\partial \mathcal{L}}{\partial w} $$

<br>

`RMSProp(Root Mean Square Propagation)`은 AdaGrad의 누적 방식을 바꿔 이 문제를 해결합니다. 기울기 제곱을 그대로 더하는 대신 `지수 이동 평균(Exponential Moving Average)`으로 누적해, **오래된 기울기의 영향은 점차 잊고 최근 기울기를 더 반영**합니다. $$ \beta $$는 보통 0.9 또는 0.99를 사용합니다.

$$ s $$가 최근 기울기의 평균 크기를 유지하므로 보폭이 끝없이 줄어들지 않으며, 지형의 가파름이 바뀌어도 거기에 맞춰 보폭을 다시 조정합니다. 오래 학습해도 멈추지 않는다는 점에서 AdaGrad를 대체했으며, 다음에 설명할 Adam의 절반을 이룹니다.

<br>

### Adam

$$ m \leftarrow \beta_{1} m + (1 - \beta_{1}) \frac{\partial \mathcal{L}}{\partial w}, \qquad s \leftarrow \beta_{2} s + (1 - \beta_{2}) \left( \frac{\partial \mathcal{L}}{\partial w} \right)^{2} $$

$$ \hat{m} = \frac{m}{1 - \beta_{1}^{t}}, \qquad \hat{s} = \frac{s}{1 - \beta_{2}^{t}}, \qquad w \leftarrow w - \frac{\eta}{\sqrt{\hat{s}} + \epsilon} \hat{m} $$

<br>

`Adam(Adaptive Moment Estimation)`은 **모멘텀과 RMSProp을 결합**한 알고리즘입니다. 기울기의 지수 이동 평균 $$ m $$으로 방향을 다듬고, 기울기 제곱의 지수 이동 평균 $$ s $$로 가중치마다 보폭을 조절합니다. $$ \beta_{1} $$은 0.9, $$ \beta_{2} $$는 0.999가 기본값입니다.

$$ m $$과 $$ s $$는 0에서 출발하므로 학습 초반에는 실제보다 작게 추정됩니다. Adam은 이를 $$ 1 - \beta^{t} $$로 나누어 보정하며, 이를 `편향 보정(Bias Correction)`이라 합니다. 반복 횟수 $$ t $$가 커질수록 분모가 1에 가까워져 보정의 영향은 사라집니다.

Adam은 학습률에 덜 민감하고 대부분의 과제에서 별다른 조정 없이 안정적으로 동작하므로, 새로운 모델을 시도할 때 **가장 먼저 선택하는 기본값**이 되었습니다. 위 그래프에서도 Adam은 진동 없이 가장 곧은 경로로 골짜기에 도달합니다. 다만 매 걸음의 크기가 학습률에 의해 사실상 상한이 정해지므로, 지형이 단순한 경우에는 모멘텀 SGD보다 느릴 수도 있습니다.

<br>

### AdamW

[과대적합과 정칙화(Overfitting & Regularization)][AI-9강]에서 L2 정칙화를 가중치 감쇠와 같은 것으로 설명했습니다. 이는 SGD에서는 성립하지만 **Adam에서는 성립하지 않습니다.** L2 항을 손실에 더하면 그 기울기도 $$ s $$로 나누어지므로, 기울기가 큰 가중치일수록 정칙화가 약해지는 의도치 않은 결과가 생깁니다.

`AdamW`는 가중치 감쇠를 기울기에 섞지 않고, Adam의 갱신이 끝난 뒤 **가중치에서 직접 일정 비율을 빼는** 방식으로 분리했습니다. 이 작은 수정만으로 일반화 성능이 눈에 띄게 개선되어, 현재 트랜스포머 계열 모델을 비롯한 대부분의 대규모 학습에서 Adam 대신 사용됩니다.

- Tip : PyTorch에서 `optim.Adam(params, weight_decay=0.01)`은 L2 정칙화 방식으로, `optim.AdamW(params, weight_decay=0.01)`은 분리된 가중치 감쇠 방식으로 동작합니다. 이름이 비슷하지만 **동작이 다르므로** 구분해서 사용해야 합니다.

<br>
<br>

## 학습률 스케줄링(Learning Rate Scheduling)

최적화 알고리즘이 보폭을 가중치마다 조절한다면, `학습률 스케줄러(Learning Rate Scheduler)`는 **학습이 진행됨에 따라** 전체 학습률을 바꿉니다. 학습 초반에는 골짜기에서 멀리 있으므로 큰 보폭으로 빠르게 접근하고, 후반에는 바닥을 지나치지 않도록 보폭을 줄이는 것이 기본 전략입니다.

- `단계 감쇠(Step Decay)` : 정해진 에폭마다 학습률을 일정 비율로 줄입니다. 단순하고 예측하기 쉬워 오랫동안 사용되었습니다.
- `코사인 감쇠(Cosine Annealing)` : 학습률을 코사인 곡선을 따라 부드럽게 0까지 줄입니다. 급격한 변화 없이 후반에 충분히 작은 보폭을 확보할 수 있습니다.
- `웜업(Warmup)` : 학습 **초반 일정 기간 동안 학습률을 0에서 목표값까지 서서히 올립니다.** 가중치가 무작위 상태인 초반에 큰 보폭으로 움직이면 학습이 불안정해지기 쉬우며, Adam의 $$ s $$가 아직 충분히 추정되지 않아 보폭이 지나치게 커지는 것을 막는 효과도 있습니다.
- `정체 시 감쇠(Reduce on Plateau)` : 검증 오차가 일정 기간 개선되지 않으면 학습률을 줄입니다. 일정에 따르지 않고 학습 상태에 반응한다는 점이 다릅니다.

현재 가장 널리 쓰이는 조합은 **웜업 뒤 코사인 감쇠**입니다. 수천 걸음 동안 학습률을 올린 뒤 남은 기간 동안 코사인 곡선으로 줄이는 방식으로, 대규모 모델 학습의 사실상 표준이 되었습니다.

- Tip : PyTorch의 스케줄러는 `optim.lr_scheduler`에 모여 있으며, 매 에폭 또는 매 걸음마다 `scheduler.step()`을 호출해야 학습률이 갱신됩니다. 호출 시점을 에폭 단위로 할지 걸음 단위로 할지는 스케줄러마다 다르므로 문서를 확인해야 합니다.

<br>
<br>

## 최적화 알고리즘 정리

| 알고리즘 | 방향 보정 | 보폭 조절 | 특징 |
|---|---|---|---|
| SGD | 없음 | 없음 | 단순하지만 진동이 크고 학습률에 민감 |
| Momentum | 속도 누적 | 없음 | 진동을 줄이고 평원을 관성으로 통과 |
| AdaGrad | 없음 | 기울기 제곱 누적 | 희소한 입력에 강하지만 보폭이 0으로 수렴 |
| RMSProp | 없음 | 지수 이동 평균 | AdaGrad의 보폭 소멸 문제를 해결 |
| Adam | 지수 이동 평균 | 지수 이동 평균 | 조정 없이 안정적인 기본 선택 |
| AdamW | 지수 이동 평균 | 지수 이동 평균 | Adam에서 가중치 감쇠를 올바르게 분리 |

새로운 과제를 시작할 때는 **AdamW와 웜업 후 코사인 감쇠**를 기본값으로 두고, 학습 곡선을 확인하며 학습률부터 조정하는 것이 일반적입니다. 학습이 안정적으로 진행된다면 모멘텀 SGD로 바꿔 최종 성능을 더 끌어올릴 수 있는지 시도해 볼 수 있습니다.

어떤 알고리즘을 선택하든 **학습률이 가장 큰 영향**을 미칩니다. 알고리즘을 바꾸기 전에 학습률을 10배 단위로 바꿔 가며 손실이 발산하지 않으면서 가장 빠르게 줄어드는 값을 먼저 찾는 것이, 다른 어떤 하이퍼파라미터를 조정하는 것보다 효과가 큽니다.

- Tip : 손실이 `NaN`으로 바뀌며 학습이 무너진다면 대부분 **학습률이 너무 큰 것**이 원인입니다. 학습률을 낮추거나 웜업을 추가하고, 그래도 해결되지 않으면 기울기의 크기에 상한을 두는 `기울기 클리핑(Gradient Clipping)`을 적용합니다.

<br>
<br>

* Writer by : 윤대희

[AI-5강]: https://076923.github.io/posts/AI-5/
[AI-9강]: https://076923.github.io/posts/AI-9/
