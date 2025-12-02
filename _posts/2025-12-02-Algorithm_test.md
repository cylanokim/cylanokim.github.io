--- 
title: "주요 알고리즘"
description: 주요 알고리즘
author: cylanokim
date: 2025-12-02 12:00:00 +0800
categories: [Book Review]
tags: [Book, review]
pin: true
math: true
mermaid: true
---

## 1. Backtracking 알고리즘_1
```python

arr = [1,2,3]
subset = []

def backtrack(idx):
    # 1) 종료 조건
    if idx == len(arr):
      print(arr)
      return 
    
    # 2) idx 원소 선택 O
    subset.append(arr[idx])
    backtrack(idx+1)

    # 3) idx 원소 선택 X
    subset.pop()
    backtrack(idx+1)

backtrack(0)
```

## 2. Backtracking 알고리즘_2
```python
N = int(input())
visited = [False] * N 
arr = []

def backtrack():
  if len(arr) == N:
      print(arr)
      return 
  
  for i in range(N):
      if visited[i]:
          continue 
      
      visited[i] = True
      arr.append(i+1)

      backtrack()

      visited[i] = False
      arr.pop()

backtrack()
```
