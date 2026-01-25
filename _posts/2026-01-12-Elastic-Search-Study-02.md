--- 
title: "Elastic Search 스터디 02"
description: Spring Boot에 ES 연결 하기. 역 인덱스(Inverted Index)를 이용하여 검색해보기. 오랜만이구나 spring. 
author: cylanokim
date: 2026-01-12 12:00:00 +0800
categories: [BACK_END]
tags: [DB, ElasticSearch, ES, Inverted_Index]
pin: true
math: true
mermaid: true
---

## ✅ 1. Spring Boot 띄우기
- appilcation.properties → .yaml로 바꾸기
- localhost:8080 spring boot project 띄우기 
- UserDocument → UserController → UserDocumentRepository 객체 만들기
- UserCreateRequestDto, UserUpdateRequestDto 만들기

## ✅ 2. API 만들기
### 참고) 왜 user를 생성하는데 setter가 아닌 getter일까?
```Java
    @PostMapping
    public UserDocument createUser(@RequestBody UserCreateRequestDto requestDto) {
        UserDocument user = new UserDocument(
                requestDto.getId(), // DTO에서 값을 꺼냄!!
                requestDto.getName(),
                requestDto.getAge(),
                requestDto.getIsActive()
        );
        return userDocumentRepository.save(user);
    }
```
- 데이터가 저장되는 흐름은 아래와 같다.
- 1) HTTP에 바디(Json)를 통해 요청 → UserCreateRequestDto → 값을 꺼냄(get) → UserDocument 생성 → 저장
- 2) `@RequstBody`가 JSON을 자동으로 Mapping하여 내부 필드 값을 매치하여 UserCreateRequestDto 생성
- 3) 내부 필드 값에서 값을 꺼냄 (get)
- 4) UserDocument 객체를 생성하여 저장. 

## ✅ 3. 단어의 순서가 바뀌어도 검색이 가능?
### 1. 사전 데이터 준비 (간단하게 데이터 테이블 정의 후 ES 검색)

```
```Elastic Search
# index(테이블) 정의 & mapping(columns) 정의
PUT /products
{
    "mappings": {
        "properties": {
            "name": {
                "type": "text"
            }
        }
    }
}

# document(데이터) 삽입
POST /products/_create/1
{
    "name": "Apple 2025 맥북 에어 13"
}

POST /products/_create/2
{
    "name": "Apple 2024 맥북 미니 12"
}

POST /products/_create/3
{
    "name": "Apple 2023 아이패드 프로 pro"
}

# 전체 검색
GET /products/_search

# 키워드 매칭 검색
GET /products/_search 
{
    "query": {
        "match": {
            "name": "Apple 2023"
        }
    }
}
```
### 2. ES의 검색 원리
- 역인덱스(Inverted Index): 단어를 기준으로 문서를 찾기 위한 구조
- 일치하는 토큰의 개수가 많은 문서가 먼저 검색
- 단, text type을 써야 ES의 유연한 검색 기능 가능

일반 인덱스
```
문서 1 → [AI, 모델, 학습]
문서 2 → [AI, 데이터, 처리]
문서 3 → [모델, 추론]
```

역인덱스
```
AI     → [문서 1, 문서 2]
모델   → [문서 1, 문서 3]
학습   → [문서 1]
데이터 → [문서 2]
처리   → [문서 2]
추론   → [문서 3]
```
### 2. 문서의 스코어 계산 로직
#### 1) Term Frequency (TF)
문서 내에서 검색어가 얼마나 자주 등장하냐? → 많이 등장할수록 점수 증가

#### 2) InverseDocument Frequency (IDF)
검색어가 전체 문서 중 얼마나 희귀하냐? → 흔한 단어일수록 점수 감소

#### 3) Field Length Normalization
문서(필드)가 짧을수록 점수 증가


## ✅ 출처
- 본 포스트는 인프런 지식 공유자 박재성님의 "실전에서 바로 써먹는 ES 입문 강의"를 기반으로 작성되었습니다. 
