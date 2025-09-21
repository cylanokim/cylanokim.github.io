---
title: "깃허브 블로그 Build & Deploy 문제 해결 과정!"
description: 로컬 서버에서 잘 되던 블로그! 너 왜 웹에서는 에러가 발생하는거야? 
author: cylanokim
date: 2025-09-21 14:00:00 +0800
categories: [blog]
tags: [blog]
pin: true
math: true
mermaid: true
---
깃허브 블로그는 자신의 로컬 서버에서 아래 명령어로 띄운 후, 실제 웹에서 어떻게 보일지 확인할 수 있는 장점이 있습니다. 
```bash
bundle exec jekyll serve
```
<br>
로컬 서버에서 아무 문제가 없어서 수정한 내용을 깃허브 저장소에 옮긴 후, 블로그에 접속하였는데 아래와 같은 화면만 나옵니다.

```
--- layout: home # Index page ---
``` 
이번 포스팅은 이 문제를 해결했던 과정을 간단하게 정리해보겠습니다. 

# GitHub 블로그 빌드 & 배포 과정
깃허브 블로그가 동작하는 과정은 **코드를 올리고 → GitHub Actions가 빌드 → GitHub Pages가 배포 → 사용자가 접속**의 흐름입니다. 

---

## 🔄 순서 정리 

1. **로컬에서 작성**  
   - 블로그 파일(소스코드, 마쿠다운으로 만들어진 포스팅) 작성

2. **GitHub Repository에 Push**  
   - `git add .` → `git commit -m "..."` → `git push`  
   - 이때 소스 코드가 GitHub 저장소에 올라감.  

3. **GitHub Actions 트리거**  
   - 저장소에 Push 이벤트가 발생하면, `.github/workflows/` 안에 있는 CI/CD 설정 파일이 실행됨.  
   - 여기서 Ruby, Jekyll 등 필요한 환경을 설치하고, `jekyll build` 같은 명령으로 정적 파일(`_site/`)을 생성.  

4. **빌드 결과물을 GitHub Pages 브랜치로 배포**  
   - 보통 `gh-pages` 브랜치나 `docs/` 폴더, 또는 GitHub Pages 전용 아티팩트로 빌드 결과가 저장됨.  
   - GitHub Pages 서비스가 이 정적 HTML/CSS/JS 파일을 호스팅할 준비를 함.  

5. **GitHub Pages 배포 완료**  
   - 설정한 GitHub Pages URL (예: `https://username.github.io/`) 로 접속 가능.  

---

이 과정을 해당 깃 저장소의 Actions 탭에서 확인이 가능합니다. 그런데 이 Build 하는 과정에서 에러가 발생하였더라구요. 
![build_error](/assets/build_error.png)

GPT와 씨름한 끝에, 위 에러는 Jekyll에서 SCCSS 파일을 빌드할 때, import 대상 stylesheet을 찾지 못해서 발생하는 전형적인 오류라는 것을 확인했습니다. 이 과정이 상당히 복잡하고, 설명하기 어려운데 일단 해결책은 아래와 같이 2개의 작업을 진행하는 것이었습니다.

# ✅ 해결 방법
1. 깃 저장소의 Settings > Pages > Build and deployment source를 반드시 **GitHub Actions**로 수정 

2. Gemfile에 아래와 같이 테마 설치 명령어 직접 추가
```ruby
gem "jekyll-remote-theme"
```
로컬 실행 시
```bash
bundle install
bundle exec jekyll serve
```

생각보다 간단한 해결 방법이었지만, 그 과정은 길고 힘든 시간이었습니다. 결국 블로그가 정상 빌드가 되는거 보고 감동했네요 ㅠ 편안한 길 놔두고 깃허브 블로그로 글 쓰시는 모든 선배님들에게 리스팩을 보내며, 오늘 포스팅은 여기에서 줄이겠습니다 :D 


