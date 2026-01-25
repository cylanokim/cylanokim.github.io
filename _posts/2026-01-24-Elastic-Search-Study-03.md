--- 
title: "Elastic Search 스터디 03"
description: ES만의 특징인 Analyzer를 사용하여 다양한 검색 기능을 활용해 보자!
author: cylanokim
date: 2026-01-24 12:00:00 +0800
categories: [BACK_END]
tags: [DB, ElasticSearch, ES, Analyzer]
pin: true
math: true
mermaid: true
---

## ✅ Analyzer란?
- 입력된 문자열(text)를 토큰으로 변환 시켜주는 장치

### 1. Character Filter (문자 전처리)
  - HTML 태그 제거, 특수 문자 치환 
  - Mapping 필터, Pattern Replace 필터

  ### 2. Tokenizer

| Tokenizer  | 설명             |
| ---------- | -------------- |
| standard   | 공백, 문장부호 기준 분리 |
| whitespace | 공백 기준          |
| keyword    | 전체를 하나의 토큰으로   |
| ngram      | 부분 문자열로 쪼갬     |
| edge_ngram | 접두어 자동완성용      |


### 3. Token Filter
  - 생성된 토큰을 변형 / 정규화

| Filter       | 역할                    |
| ------------ | --------------------- |
| lowercase    | 소문자 변환                |
| stop         | 불용어 제거 (the, is 등)    |
| stemmer      | 어간 추출 (running → run) |
| synonym      | 동의어 처리                |
| asciifolding | 악센트 제거 (café → cafe)  |

## ✅ Analyzer API란?
- 특정 Analyzer가 텍스트를 어떻게 토큰화하는지 확인하는 API

```
# 1) 특정 analyzer 사용
POST /_analyze
{
  "analyzer": "standard",
  "text": "Elasticsearch Is AWESOME!"
}

# 2) 전체 파이프 라인
POST /_analyze
{
  "char_filter": ["html_strip"],
  "tokenizer": "standard",
  "filter": ["lowercase"],
  "text": "<b>Hello</b> WORLD"
}
```

```
{
  "tokens": [
    {
      "token": "elasticsearch",
      "start_offset": 0,
      "end_offset": 13,
      "type": "<ALPHANUM>",
      "position": 0
    },
    {
      "token": "is",
      "start_offset": 14,
      "end_offset": 16,
      "type": "<ALPHANUM>",
      "position": 1
    },
    {
      "token": "awesome",
      "start_offset": 17,
      "end_offset": 24,
      "type": "<ALPHANUM>",
      "position": 2
    }
  ]
}
```

## ✅ Custom Analyzer 설정하여 간단한 인덱스(=Table) 만들기

char_filter, tokenizer, filter를 직접 적용한 인덱스(=Table)를 만들어보자. ~~아직도 인덱스 = 테이블이라는 개념이 좀 낯설긴 하지만 익숙해지겠지....~~

### 1. products라는 인덱스 (=테이블)을 만들고, name이라는 properties(=column, schema)를 정의하자.
```
PUT /products 
{
    "settings":{
        "analysis": {
            "analyzer": {
                "products_name_analyzer":{
                    "char_filter": [],
                    "tokenizer": "standard",
                    "filter": ["lowercase"]
                }
            }
        }

    },
    "mappings":{
        "properties": {
            "name": {
                "type": "text",
                "analyzer": "products_name_analyzer"
            }
        }
    }
}
```

### 2. 간단한 문장을 넣고
```
POST /products/_create/1
{
    "name": "The last few years have been exhilarating"
}
```

### 3. 검색해 보자!
```
GET products/_search
{
    "query": {
        "match": {
          "name": "the"
        }
    }
}
```

### 4. Custom으로 정의한 토크나이저 필터 사용해보기!
```
GET /products/_analyze
{
    "field": "name",
    "text": "The last few years have been exhilartating"
}
```

## ✅ html_strip, 불용어(a, the....)제거 필터 적용하기

### 1. 인덱스 만들기
```
PUT /boards 
{
    "settings": {
        "analysis": {
            "analyzer": {
                "boards_content_analyzer":{
                    "char_filter": ["html_strip"],
                    "tokenizer": "standard",
                    "filter": ["lowercase", "stop"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "content": {
                "type": "text",
                "analyzer": "boards_content_analyzer"
            }
        }
    }
}
```

### 2. document 넣기
```
POST /boards/_doc 
{
    "content": "<h1>Crowds gathered in the vicinity of Trafalgar Square.</h1>"
}
```

### 3. 검색 및 어떻게 토크나이징 되었나 확인하기 (Analyze API 사용)
```
GET /boards/_search
{
    "query": {
        "match": {
            "content": "runnig"
        }
    }
}

GET /boards/_analyze
{
    "field": "content",
    "text": "<h1>Crowds gathered in the vicinity of Trafalgar Square.</h1>"
}
```

## ✅ Synonym 필터 설정
ES는 다양한 동의어를 갖는 경우에도 검색이 가능할 수 있게 `synonym` 필터를 정의하고 적용할 수 있다. 나만 기억하기 위해서는 analyzer와 동일한 rank로 정의하고 filter list에 넣는다.

### 1. 인덱스 정의
```
PUT /products 
{
    "settings":{
        "analysis": {
            "filter": {
                "products_synonym_filter": {
                    "type": "synonym",
                    "synonyms": [
                        "notebook, 노트북, 랩탑, laptop",
                        "samsung, 삼성"
                    ]
                }
            },
            "analyzer": {
                "products_name_analyzer":{
                    "char_filter": [],
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "products_synonym_filter"
                    ]
                }
            }
        }

    },
    "mappings":{
        "properties": {
            "name": {
                "type": "text",
                "analyzer": "products_name_analyzer"
            }
        }
    }
}
```

### 2. 데이터 넣기
```
POST /products/_doc 
{
    "name": "SAMSUNG NOTEBOOK"
}
```

### 3. 검색
```
GET /products/_search
{
    "query": {
        "match": {
            "name": "노트북"
        }
    }
}
```

## ✅ 출처
- 본 포스트는 인프런 지식 공유자 박재성님의 "실전에서 바로 써먹는 ES 입문 강의"를 기반으로 작성되었습니다. 
