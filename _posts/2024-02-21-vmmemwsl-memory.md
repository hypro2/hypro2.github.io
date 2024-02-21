---
layout: post
title: VmmemWSL 메모리 부족과 초기화 하는 방법
---

WSL로 프로그램을 실행하다보면 메모리가 가끔 부족할 때가 있고, 쓰지 않을 때도 메모리를 점유하고 있을 때가 있다. 작업관리자에 들어가면 VmmemWSL이라는 프로세스가 많은 메모리를 사용하고 있고, 이것 조차 가끔 부족할 때가 있다. PC의 메모리를 어느 정도까지 쓸건지 .wslconfig를 생성 해주는 것으로 직접 어느 정도의 메모리를 사용할 지 알려줄 수 있다.

그리고 사용하지 않을 때는 wsl를 shutdown 시켜주므로써 메모리를 반환 받을 수 있는 방법에 대해서 기록 할려고한다.


#### WSL의 점유 상태
<img width="769" alt="image" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbgsCSA%2FbtsE9LWqGAa%2FfKkXeKWR8CchoOWkxOYzKk%2Fimg.png">

#### .wslconfig 파일 생성

<img width="369" alt="image" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb4mv8v%2FbtsE6CTL2k3%2Fz9Tyo5eLkqIHT1oNzhoi70%2Fimg.png">
<img width="369" alt="image" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fcki4xo%2FbtsE6T107Ik%2FPN8g4GvJ4zsD4gx6ucOm4K%2Fimg.png">
%USERPROFILE% 위치에 .wslconfig를 메모장 아래 내용을 입력해주고 만들어주면된다.

#### wslconfig 예시

```
[wsl2]  
memory=10GB  
processors=8  
swap=4GB  
localhostForwarding=true  
```

#### wsl 메모리 초기화

```
wsl --shutdown
```

자료 출처:
https://learn.microsoft.com/ko-kr/windows/wsl/wsl-config#configuration-options
