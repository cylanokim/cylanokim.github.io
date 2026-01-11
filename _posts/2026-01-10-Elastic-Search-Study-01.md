--- 
title: "Elastic Search 스터디 01"
description: Elastic Search는 뭘까?
author: cylanokim
date: 2026-01-10 12:00:00 +0800
categories: [BACK_END]
tags: [DB, ElasticSearch, ES]
pin: true
math: true
mermaid: true
---

## ✅ Elastic Search(이하 ES) 간단 개요 
- Elastic Search는 `검색과 데이터 분석에 최적화된 데이터 베이스` 이다.
- ES는 웹 표준 기술인 HTTP, REST API를 사용하여 9200 포트의 프로세스와 통신.
- 확인 방법 : wsl에서 curl http://localhost:9200
- MySQL은 SQL문으로 요청을 보내나, ES는 REST API 방식으로 통신한다. 
- ES의 대표적인 GUI 툴은 **Kibana** 이다. (5601 포트에서 실행)
- ES와 MySQL 용어 비교

| MySQL        | ES       |
|--------------|----------|
| table        | index    |
| column       | field    |
| record, row  | document |
| schema | mapping  |

- MySQL의 인덱스와 ES의 인덱스는 별개의 개념이다.
- ES에서 INDEX를 생성하는 것은 Table을 생성하는 것과 같다.

## ✅ 윈도우 환경에서 ES 실행
```
# 1. powersheel 접속
# 2. `wsl` 입력하여 wsl 접속
# 3. 윈도우에서 `Docker Desktop` 실행 (시간 걸림)
# 4. `docker start` 명령어로 ES, Kibana 실행 
# 5. `localhost:5601`로 kibana 접속
```

## ✅ Elastic Search 기초 명령어 모음
#### 1. index (table) 생성 및 조회, 삭제
```
PUT /users
GET /users
DELETE /boards
```

#### 2. index의 maapings (column, schema) 정의 
```
PUT /users/_mappings
{
    "properties": {
        "name":{"type": "keyword"},
        "age":{"type": "integer"},
        "is_active":{"type": "boolean"}
    }
}
```

#### 3. 데이터 (Documents) 삽입 
```
POST /users/_doc
{
    "name": "Alice",
    "age":28,
    "is_active":true
}

```

#### 4. 데이터 확인
```
GET /users/_search
```
  - 검색 결과는 `hits` 안에 있음
  - `_source` 안에 저장한 데이터가 있음

#### 5. id 지정해서 POST
```
POST /users/_create/1
{
    "name": "Max",
    "age":33,
    "is_active":false
}
```
  - id에 random 값이 아닌 지정한 값이 부여되며, 유일한 값으로 저장된다. 

#### 6. UPSERT (Update + Insert)
```
PUT /users/_doc/2
{
    "name": "Jason",
    "age":54,
    "is_active":false
}

GET /users/_search

PUT /users/_doc/2
{
    "name": "Jason2",
    "age":54,
    "is_active":false
}
```

#### 7. id 값으로 특정 Document (데이터) 검색
```
GET /users/_doc/1
GET /users/_doc/2
```

#### 8. Document Update
```
PUT /users/_doc/1
{
    "name": "cylanokim"
}

POST /users/_update/2
{
    "doc": {
        "name": "Ross",
        "is_active": false
    }
}
```

#### 9. 특정 document 삭제
```
DELETE /users/_doc/2
```

## ✅ 출처

- 본 포스트는 인프런 지식 공유자 박재성님의 "실전에서 바로 써먹는 ES 입문 강의"를 기반으로 작성되었습니다. 
- 문케이크 Blog (https://mooncake1.tistory.com/318)
