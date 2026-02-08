--- 
title: "Elastic Search 스터디 04"
description: ES에서 한글 데이터를 저장하고 검색해보자 :)
author: cylanokim
date: 2026-02-07 12:00:00 +0800
categories: [BACK_END]
tags: [DB, ElasticSearch, ES, Analyzer, Nori-Analyzer]
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

## ✅ 출처
- 본 포스트는 인프런 지식 공유자 박재성님의 "실전에서 바로 써먹는 ES 입문 강의"를 기반으로 작성되었습니다. 
