--- 
title: "Spring 다시 부르기 01"
description: Spring Boot 기본을 다시 점검하자 (미완성!)
author: cylanokim
date: 2026-02-20 12:00:00 +0800
categories: [BACK_END]
tags: [Spring Boot, MVC, DB]
pin: true
math: true
mermaid: true
---
조금 급하게 Spring Boot에 대한 대략적인 이해가 필요하다. 김영한 선생님의 강의를 통해 기본적인 java와 spring boot를 이용한 BE 를 점검해보자.


## ✅ JAVA 설치
- java 설치는 아래 링크를 참고하자
- https://mozzi-devlog.tistory.com/44#google_vignette

```
java -version
```

## ✅ Spring ViewResolver를 이용하여 html 연결해보기
- ViewResolver는 @Controller가 붙은 Controller가 반환한 "뷰 이름"을 실제 html 파일 경로로 변환해주는 객체!
- 여러가지 annotation이 있는데 정리하면....
- 간단한 정리

| 상황                            | ViewResolver 동작 | 결과         |
| ----------------------------- | --------------- | ---------- |
| `@Controller` + String return | ✅ 동작            | HTML 연결    |
| `@RestController`             | ❌ 안함            | 문자열 그대로 출력 |
| `@ResponseBody`               | ❌ 안함            | 문자열 그대로 출력 |

### 1. Controller 객체
```java
@Controller
public class HelloController {
    @GetMapping("hello")
    public String hello(Model model) {
        model.addAttribute("data", "hello@@!!!");
        return "hello";
    }
}
```

### 2. 내부 동작 흐름

#### 1) 브라우저 요청
- 브라우저의 HTTP 요청
```
http://localhost:8080/hello
```

#### 2) Tomcat이 요청을 받음
- Tomcat은 TCP 연결 수락, HTTP 요청 파싱, 웹 어플리케이션으로 전달
- Tomcat은 웹 서버/서블릿 컨테이너로서, 요청을 받아서 서블릿에게 넘겨준다. 

#### 3) Tomcat → DispatcherServlet으로 전달 
- Spring Boot는 모든 요청을 DispatcherServlet이 받도록 매핑
- 어떤 컨트롤러/메서드가 이 요청을 처리해야하지? 를 결정하는 입구

#### 4) DispatcherServlet이 handler(컨트롤러 메서드)를 찾음
- DispatcherServlet은 내부에서 Handler Mapping에게 /hello 요청을 처리할 메서드가 어디야?

#### 5) Contoller 메서드 실행 + Model에 데이터 저장
- `Model`은 뷰(템플릿)에게 넘겨주는 데이터 모음?

#### 6) Controller가 Sting을 반환
- @Controler 어노테이션의 경우 spring MVC에서 기본적으로 뷰 이름을 반환 
- 그러나 `@RestContoller`, `@ResponseBody`면 문자열 데이터로 응답

#### 7) ViewResolver가 등장
- hclasspath:/templates/hello.html 라는 템플릿 파일을 찾는다. 

#### 8) Thymeleaf가 HTML을 렌더링한다. 
- hello.html 파일을 읽고, Model에 있는 데이터를 대입하여 완성된 "HTML 문자열"을 만든다.

<div style="text-align: center;">
  <img src="/assets/img/what_tomcat_does.PNG" alt="what_tomcat_does" width="600">
</div>

## ✅ Spring이 외부 입력을 처리하여 내보내는 방식 3가지

### 1. 정적 컨텐츠 (static contents)
- 가공되지 않은 파일을 그냥 그대로 내려준다!!
- hello-static 컨트롤러가 spring 컨테이너에 없다 → resoutces/static/hello-static.html로 연결

### 2. MVC와 템플릿 엔진
- 처리된 html을 내려준다!!
- thymeleaf(템플릿 엔진) 사용
- 요청에 매핑되는 Controller 확인 → viewResolver는 Controller 가 return 하는 텍스트와 같은 template를 연결

### 3. API
- 객체(데이터)를 반환한다!! 
- 클라이언트(엡, 뒙)나 다른 서버에 구조화된 데이터만을 전달하기 위해

```java
@GetMapping("hello-string")
    @ResponseBody
    public Hello helloApi(@RequestParam("name") String name) {
        Hello hello = new Hello();
        hello.setName(name);
        return hello;
    }
```

- `@ResponseBody`를 사용하면.... 
  - HTTP 통신 프로토콜의 Body부에 이 데이터를 직접 넣겠다. → 패아지 소스 보면 html이 아님
  - `viewResolver` 대신 `HttpMessageConverter`가 동작하여 객체를 json 등으로 변환하여 body에 담음
  - 기본 객체 처리: `MappingJackson2HttpMessageConverter`

<div style="text-align: center;">
  <img src="/assets/img/what_tomcat_does_api.PNG" alt="what_tomcat_does" width="600">
</div>

## ✅ 회원 관리 예제 - back end 개발

### 1. 웹 에플리케이션 계층 구조
- 컨트롤러: 웹 mvc 컨트롤러 역할
- 서비스: 핵심 비지니스 로직 구현
- 리포지토리: DB 접근, 도메인 객체를 DB에 저장하고 관리. DB 종류가 바뀌기에 interface로 추상화
- 도메인: 데이터 베이스에 저장하고 관리됨
- 

<div style="text-align: center;">
  <img src="/assets/img/web_appication_hierarchy_structure.PNG" alt="what_tomcat_does" width="600">
</div>


### 2. 구현된 코드, test


### 3. Component Scan과 자동 의존 관계 설정
- "Spring이 자동으로 Bean을 찾아서 IoC 컨테이너에 등록해주는 기능"

#### 1) Component Scan이 필요한 이유
- Spring은 객체를 직접 생성하지 않는다. 가령 `new MyService()`
- 바로 Component Scan이 **해당 클래스를 찾아서** Bean에 등록하기 때문

#### 2) 어떤 것들을 스캔?

| 어노테이션             | 의미        |
| ----------------- | --------- |
| `@Component`      | 기본 컴포넌트   |
| `@Service`        | 서비스 계층    |
| `@Repository`     | DB 접근 계층  |
| `@Controller`     | MVC 컨트롤러  |
| `@RestController` | REST 컨트롤러 |

#### 3) @Autowired 의 역할 이해하기
Spring이 객체를 주입하는 과정은 아래와 같음

```
1. Component Scan이 클래스를 찾는다
2. Bean으로 등록한다 (IoC 컨테이너에 저장)
3. @Autowired가 Bean을 찾아서 주입한다
```
즉, Component Scan이 등록, @Autowired가 연결

#### 4) 예제로 이해하기
1) Repository
```java
@Repository
public class DataRepository {
}
```
`@Repository` → Component Scan 대상 → Spring이 Bean으로 등록

2) Service
```java
@Service
public class dataService {
    @Autowired
    private DataRepository repository;
}
```
`@Autowired` → Spring 컨테이너에 등록된 Bean 중, DataRepository 타입의 Bean을 찾아서 자동으로 넣어줘

<div style="text-align: center;">
  <img src="/assets/img/spring_bean_dependency.PNG" alt="what_tomcat_does" width="600">
</div>


## ✅ 출처
- 본 포스트는 인프런 지식 김영한님의 "스프링 입문 - 코드로 배우는 스프링 부트, 웹 MVC, DB 접근 기술" 강의를 기반으로 작성되었습니다. 
