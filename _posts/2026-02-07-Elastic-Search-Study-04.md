--- 
title: "Elastic Search 스터디 04"
description: ES에서 한글 데이터를 저장하는 방법과 매핑에 대하여 알아보자 :)
author: cylanokim
date: 2026-02-07 12:00:00 +0800
categories: [BACK_END]
tags: [DB, ElasticSearch, ES, Analyzer, Nori-Analyzer, mappings]
pin: true
math: true
mermaid: true
---

## ✅ Nori Analyzer?
- 한글을 형태소 단위로 분해하여 역 인덱스에 저장한다.
- 플러그인 Nori Analyzer 설치 필요
- 설치 후 Docker 컨네이너를 다시 띄어야함

## ✅ Nori Analzyer 설치 

### 1. 폴더 구성
`elastic-nori/` 폴더를 만들고 아래 2개의 파일을 생성한다.
- Dockerfile
- docker-compose.yml

### 1. Dockerfile 
```dockerfile
FROM docker.elastic.co/elasticsearch/elasticsearch:8.13.4

RUN elasticsearch-plugin install --batch analysis-nori
```

### 2. docker-compose.yml 설치 
```dockerfile
services:
  elasticsearch:
    build: .
    container_name: es-nori
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - ES_JAVA_OPTS=-Xms1g -Xmx1g
    ports:
      - "9200:9200"
      - "9300:9300"
    volumes:
      - esdata:/usr/share/elasticsearch/data

  kibana:
    image: docker.elastic.co/kibana/kibana:8.13.4
    container_name: kibana
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch

volumes:
  esdata:
```

### 3. 이미지 Build & 컨테이너 백그라운드로(-d: detached) 띄우기(up)
```
docker compose up -d --build
```

## ✅ 간단한 Nori Analyzer를 사용한 인덱스 만들고 검색해보기
### 1. 인덱스 정의
```
PUT /boards 
{
    "settings":{
        "analysis": {
            "analyzer": {
                "boards_content_analyzer": {
                    "char_filter": [],
                    "tokenizer": "nori_tokenizer",
                    "filter": ["nori_part_of_speech", "nori_readingform", "lowercase", "stop","stemmer"]
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
### 2. 인덱스에 데이터 넣기
```
POST /boards/_doc 
{
  "content": "오늘 나온 dish는 정말 elegant하고 amazing하여 chef의 의도를 understanding할 수 있었어요"
}
```

### 3. 검색해보기
```
GET /boards/_search 
{
  "query": {
    "match": {
      "content": "의도"
    }
  }
}

```

## ✅ 매핑이란?
매핑이란 document의 각 필드가 어떤 데이터 타입을 가지고 있는 지 정의하는 설정


## ✅ 매핑의 특이한 특징과 이를 이용하여 인덱스 만들어 보기

### 1. ES의 데이터 타입

| 데이터 타입    | 설명                                        |
|-----------|-------------------------------------------|
| `integer` | 10억 이하의 정수                                |
| `long`    | 10억 이상의 정수                                |
| `double`  | 실수                                        |
| `text`    | 토크나이징하는 문자열 → 유연한 검색                      |
| `keyword` | 토크나이징 X → 정확한 검색 (ex: 휴대폰 번호, 이메일, 주문 번호) |
| `date`    | 날짜                                        |
| `boolean` | true, false                        |

### 2. null 허용
- 필드에 항상 nullable함 → 필수 필드 강제는 ES가 아니라 애플리케이션에서 해야한다. 
- 이는 스키마의 강제력이 약한 검색 엔진이기 때문
- 모든 필드는 존재할 수도, 안 할 수도 있음

### 3. 배열 허용
- 배열 형태의 데이터 삽입 가능

```
"mappings": {
  "properties": {
      "hashtags": {"type": "text"}
  }
}
```

아래와 같은 데이터 삽입도 허용. 각각의 원소로 검색이 가능하다. 

```
{
  "hashtags": ["spring", "python", "AI"]
}
```

### 4. 매핑을 정의하여 index를 만들어 보자
- 인덱스 생성 시, `settings`, `mappings`를 정의한다!!! (아래 예제에서는 mappings만 정의하자)
- 데이터의 일부만 입력해도 검색을 하는 데이터는 `text`로 type을 지정하여 토크나이징을 진행하여 역 인덱스로 저장한다.

```
PUT /product_revies
{
  "mappings" : {
    "properties": {
        "review_id": {
          "type": "long"
          },
        "user_name": {
          "type": "text"
          },
        "categories": {
          "type": "keyword"
          },
        "rating": {
          "type": "double"
          },
        "is_verified_purchase": {
          "type": "boolean"
          },
        "date": {
          "type": "date"
          }
      }
  }
}
```

## ✅ 출처
- 본 포스트는 인프런 지식 공유자 박재성님의 "실전에서 바로 써먹는 ES 입문 강의"를 기반으로 작성되었습니다. 
