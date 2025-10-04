--- 
title: "python system path 정리하기"
description: datasets 패키지와 huggingface_hub 패키지 버전이 맞지 않아, 서로 싸우고 있다. 아니 같이 설치했는데 왜 버전이 안 맞는거야? 
author: cylanokim
date: 2025-10-04 12:00:00 +0800
categories: [python]
tags: [python, sys.path]
pin: true
math: true
mermaid: true
---

huggingface의 datasets 라이브러리를 이용하여 데이터를 받으려고 하는데, 아래와 같은 에러가 발생하였습니다.

```python
from datasets import load_dataset
```

```bash
ModuleNotFoundError: No module named 'huggingface_hub.errors'
```

원인은 `datasets` 라이브러리 내부에서 `huggingface_hub.errors` 모듈을 찾지 못해서 발생하는 문제로 `datasets` 버전과 `huggingface_hub` 버전이 서로 맞지 않을 때 생기는 문제라고 합니다. (feat. chatGPT) GPT 왈, `huggingface_hub`는 0.24.0 이상, `datasets`도 2.20.0 이상이면 정상 작동한다고 하여, 제 가상 환경에 설치된 `huggingface_hub` 버전을 확인해 보았습니다. 

```powershell
pip show huggingface_hub
```

확인 결과 버전은 0.19.4로 `datsets`와 궁합(?)이 안 맞는 문제라 판단하였습니다. 이럴땐 다시 깔면 되죠. 그래서 `datasets`,`huggingface_hub`를 지우고 다시 설치하였습니다.  

```powershell
pip uninstall datasets huggingface_hub
pip install datasets huggingface_hub
```

그리고 다시 `huggingface_hub` 버전을 확인하였는데.... 여전히 0.19.4 이었습니다. 이후 다시 지우고 다시 설치하고를 서너번 반복하고, 가상 환경을 다시 만들어서 재설치를 해보기도 하였지만 `huggingface_hub`의 버전은 여전히 0.19.4 였습니다. 항상 느끼는 것인데, 프로그래밍 보다 환경 세팅하는 과정이 훨씬 더 어렵고 스트레스 받는 과정인 것 같습니다. 다행인 것은 이런 과정에서 받는 스트레스를 이젠 어느정도 참을 만큼 인내심이 생겼다는 것인데, 이번 포스팅은 이 과정을 해결해 나가는 과정을 짧게 정리한 글입니다. 

## 1. 문제의 원인 
길게 심호흡을 하고 에러 메시지를 찬찬히 읽어 보았습니다. 예전에는 에러 나면 일단 분노가 치밀었는데, 이 생활도 오래하다 보니 어느정도 이런 상황에 익숙해지네요. 역시 도를 닦는 과정인 것 같습니다.

```bash
Installing collected packages: huggingface_hub Attempting uninstall: huggingface_hub Found existing installation: huggingface-hub 0.19.4 Not uninstalling huggingface-hub at c:\python_0719\lib\site-packages, outside environment C:\Users\USER-PC\Desktop\xxx\xxx\env_transformer Can't uninstall 'huggingface-hub'. No files were found to uninstall.
```

GPT와 함께 현 상황의 문제를 요약하면 아래와 같습니다.
    - `huggingface-hub` 0.19.4 가 base python (c:\python_0719\lib\site-packages) 에 있음
    - 새로운 환경(env_transformer)에는 huggingface-hub 0.23.0 이 설치됨.
    - 즉, 가상 환경과 Base Python이 섞인 상황

정리하면, pip로 `huggingface_hub`를 아무리 재 설치를 시도하여도, pip는 base python에 이미 존재하는 `huggingface_hub` 예전 버전을 인식하고 새로운 버전을 설치 못하는 것이었습니다.  

## 2. 시스템 패스 (sys.path)의 문제점
그런데 이해가 안되던 것은 분명 가상 환경 안에서 pip를 이용하여 라이브러리를 설치한 것인데, 왜 pip가 base python에 있는 `huggingface_hub`를 확인하냐는 것이었습니다. 가상 환경은 철저히 격리된 공간이어야 할텐데 말입니다. 그래서 가상 환경 내에서 **시스템 패스(sys.path)**를 확인해보기로 했습니다. `system path`는 python에서 모듈을 찾는 경로 목록을 말하는데, import 문을 실행할 때 python이 어떤 폴더를 탐색해야 할지를 담고 있는 리스트입니다. `system path` 확인 방법에는 아래와 같이 두 가지 방법이 있습니다.

```python
import sys 
print(sys.path)
```

```powershell
python -m site
```

위 코드를 실행하면 리스트 형태의 `system path` 를 아래와 같이 확인이 가능합니다.

```python
sys.path = [ 
    'C:\\Users\\USER-PC\\Desktop\\xxx\\xxxx', 
    'c:\\python_0719\\lib\\site-packages',  # <- 문제가 된....
    'C:\\Users\\USER-PC\\AppData\\Local\\Programs\\Python\\Python313\\python313.zip', 
    'C:\\Users\\USER-PC\\AppData\\Local\\Programs\\Python\\Python313\\DLLs', 
    'C:\\Users\\USER-PC\\AppData\\Local\\Programs\\Python\\Python313\\Lib', 
    'C:\\Users\\USER-PC\\AppData\\Local\\Programs\\Python\\Python313', 
    'C:\\Users\\USER-PC\\Desktop\\xxx\\xxx\\.venv', 
    'C:\\Users\\USER-PC\\Desktop\\xxx\\xxx\\.venv\\Lib\\site-packages', ]
```

여기서 원인이 밝혀지네요. 두 번째 'c:\\python_0719\\lib\\site-packages' 가 `system path` 에 등록된 상태였던 것입니다. 이렇게 `system path` 에 다른 python 이 있어서 독립된 가상 환경에서도 저 곳을 바라보니까 새로운 라이브러리 설치가 안되던 것이었습니다. 왜 저 위치가 `system path`에 등록된 것인지 알기 위해서는 `PYTHONPATH` 라는 환경 변수를 알아야합니다. 만약 윈도우에서 `PYTHONPATH`가 설정되어 있으면, 어떤 가상 환경에서도 그 경로가 무조건 `system path`에 추가 된다는 것입니다 (feat. GPT). 이를 위해서 `PYTHONPATH`를 확인해보겠습니다. 

입력
```powershell
echo $env:PYTHONPATH
```
결과
```
'c:\\python_0719\\lib\\site-packages'
```

찾았다 이놈! PYTHONPATH가 설정되어 있었습니다! 다만 분명 내가 했을텐데..... 아무튼 원인을 찾았고 윈도우의 환경 변수에서 `PYTHONPATH`를 삭제하였습니다. 

<p align="left">
  <img src="/assets/img/python_environment.PNG" alt="EmbeddingBag" width="300">
</p>

그리고 다시 가상 환경을 만든 후 `system path`을 확인한 결과! 항상 낄끼빠빠 못하던 `PYTHONPATH`가 없어졌습니다.

```python
sys.path = [ 
    'C:\\Users\\USER-PC\\Desktop\\xxx\\xxxx', 
    'C:\\Users\\USER-PC\\AppData\\Local\\Programs\\Python\\Python313\\python313.zip', 
    'C:\\Users\\USER-PC\\AppData\\Local\\Programs\\Python\\Python313\\DLLs', 
    'C:\\Users\\USER-PC\\AppData\\Local\\Programs\\Python\\Python313\\Lib', 
    'C:\\Users\\USER-PC\\AppData\\Local\\Programs\\Python\\Python313', 
    'C:\\Users\\USER-PC\\Desktop\\xxx\\xxx\\.venv', 
    'C:\\Users\\USER-PC\\Desktop\\xxx\\xxx\\.venv\\Lib\\site-packages', ]
```

## 3. 정리
이 후 가상 환경을 다시 만들고 datasets을 불러오니 정상적으로 불러와지네요 ㅠㅠ 어떻게 보면 `PYTHONPATH`를 설정한 제 업보겠지만 이를 통해 python에 대하여 좀더 깊게 이해할 수 있는 시간이었던 것 같습니다. 하지만 앞으로 이런 문제가 발생하지 않았으면하는 마음으로.... 이번 포스팅을 마치겠습니다! 


