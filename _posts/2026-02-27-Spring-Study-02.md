--- 
title: "Spring 다시 부르기 02"
description: Spring Boot에서 DB에 접근하는 기술인 JDBC와 JPA를 이해해보자!
author: cylanokim
date: 2026-02-27 12:00:00 +0800
categories: [BACK_END]
tags: [Spring Boot, MVC, DB, JDBC]
pin: true
math: true
mermaid: true
---
자 이제 DB에 직접 연결하여, CRUD를 해보자!! (미완성)


## ✅ JDBC에 대하여

### 1. JDBC(Java DataBase Connectivity)란? 
- Java에서 데이터 베이스에 접속하고 SQL을 실행하기 위한 표준 API
- Java 프로그램 ↔ 데이터베이스(MySQL, Oracle, PostgreSQL 등)를 연결해주는 다리

### 2. JDBC가 필요한 이유
- Java는 기본적으로 DB와 통신 불가
- 그래서 JDBC API + 각 DB에 맞는 Driver가 필요 

```
Java Application
       ↓
JDBC API
       ↓
JDBC Driver (DB별)
       ↓
Database
```

### 3. Spring Boot와의 관계
- Spring은 내부적으로만 JDBC를 사용하고 JDBC 코드를 거의 사용하지 않음
- 대신 `JdbcTemplate`, `JPA`, `Hibernate` 같은 기술을 사용
- 아래 코드에서 service 객체는 @Autowired를 통해 MemberRepository라는 이름을 갖는 bean 객체를 주입받는데, 이때 `SpringConfig.java` 파일에서 MemberRepository를 무엇으로 정의하냐에 따라 DB를 쉽게 변경할 수 있다.

```java
public class MemberService {
    private final MemberRepository memberRepository;

    @Autowired
    public MemberService(MemberRepository memberRepository) {
        this.memberRepository = memberRepository;
    }
```

```java
@Configuration
public class SpringConfig {

    @Bean
    public MemberService memberService() {
        return new MemberService(memberRepository());
    }

    @Bean
    public MemberRepository memberRepository() {
        return new MemoryMemberRepository();
    }
}
```

- 주의: 순수 JDBC는 DB가 바뀌어도 매우 반복되는 코드가 많고 복잡하여 유지 보수 난이도가 매우 높다. 

## ✅ JPA
- JPA는 반복 코드는 물론이고, 기본적인 SQL도 JPA가 직접 만들어서 실행해준다.
- JPA를 사용하면, SQL과 데이터 중심의 설게에서 객체 중심의 설계로 패러다임 전환이 가능 → 생산성 향상

### 1. 필수 dependency 설정 (build.gradle)

| dependency                                               | 설명                                                |
|----------------------------------------------------------|---------------------------------------------------|
| 'org.springframework.boot:spring-boot-starter-webmvc'    | HTTP → Controller → 응답(json/html) 흐름을 처리해주는 기능 제공 |
| 'org.springframework.boot:spring-boot-starter-data-jpa'  | JPA!. Entity 중심으로 DB를 다루게 해주는 라이브러리               |
| 'com.mysql:mysql-connector-j' | MySQL JDBC 드라이버                                |

### 2. application.properties 
- 어플리케이션의 환경 설정을 담당하는 중앙 설정 파일
- 대표적인 설정 예시
  #### 1) 서버 포트 변경
  ```properties
  server.port = 8081
  ```
  
  #### 2) DB 연결 설정 (MySQL)
  ```properties
  spring.datasource.url=jdbc:mysql://localhost:3306/testdb
  spring.datasource.username=root
  spring.datasource.password=1234
  spring.datasource.driver-class-name=com.mysql.cj.jdbc.Driver
  ```
  
  #### 3) JPA 설정
  ```properties
  spring.jpa.hibernate.ddl-auto=update
  spring.jpa.show-sql=true
  spring.jpa.properties.hibernate.format_sql=true
  ```
  
  #### 4) 환경 분리
  ```properties
  application-dev.properties
  application-prod.properties
  ```

### 3. Domain
- DB 테이블과 매핑을 위하여 JPA 관리 대상
```JAVA
@Entity
@Table(name = "member")
@Getter
@NoArgsConstructor
public class Member {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String name;
}
```

| 어노테이션             | 역할         |
| ----------------- | ---------- |
| `@Entity`         | JPA 엔티티 선언 |
| `@Table`          | 테이블 이름 지정  |
| `@Id`             | PK         |
| `@GeneratedValue` | 자동 증가      |
| `@Column`         | 컬럼 설정      |

### 4. SERVICE
- 비지니스 로직, 트랜잭션 관리, Repository 호출
```JAVA 
@Service
@RequiredArgsConstructor
@Transactional
public class MemberService {

    private final MemberRepository memberRepository;

    public List<Member> findAll() {
        return memberRepository.findAll();
    }
}
```

| 어노테이션            | 역할                                                  |
| ---------------- |-----------------------------------------------------|
| `@Service`       | 스프링 Bean 등록 (SpringConfig.java 파일에서 Bean으로 관리해도 된다. |
| `@Transactional` | 트랜잭션 처리                                             |






## ✅ 출처
- 본 포스트는 인프런 지식 김영한님의 "스프링 입문 - 코드로 배우는 스프링 부트, 웹 MVC, DB 접근 기술" 강의를 기반으로 작성되었습니다. 
