--- 
title: "주요 알고리즘"
description: 주요 알고리즘
author: cylanokim
date: 2025-11-05 12:00:00 +0800
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

## 3. 치즈 BFS
```python
from collections import deque 

N, M = int(input().split())
A = [list(map(int, input().split())) for _ in range(N)]

def bfs():
    q = deque([(0,0)])
    visited = [[False]*M for _ in range(N)]
    melt = []

    visited[0][0] = True
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr = r + dr
            nc = c + dc 
            if 0 <= nr < N and 0 <= nc < M and not visited[nr][nc]:
                visited[nr][nc]
                if A[nr][nc] == 0:
                    q.append((nr, nc))
                elif A[nr][nc] == 1:
                    melt.append((nr, nc))
    # 치즈 녹이기
    for r, c in melt:
        A[r][c] = 0
    return len(melt)
    
time = 0
while True:
  melted_cheese = bfs()
  if len(melted_cheese) == 0:
      break
  time += 1

print(time)
```

## 4. 맛있는 음식 DFS
```python

N = int(input().split())

A = []
for _ in range(N):
    s, b = map(int, input().split())
    A.append((s,b))

ans = float('inf')

def dfs(idx, sour, bitter, count):
    global ans 

    if idx == N:
        if count >= 1:
            diff = abs(sour, bitter)
            if diff < ans:
                ans = diff
        return 
    
    s, b = A[idx]

    # 1) idx 번째 재료를 사용 O
    dfs(idx+1, sour*s, bitter+b, count+1)

    # 2) idx 번째 재료를 사용 X
    dfs(idx+1, sour, bitter, count)

dfs(0, 1, 0, 0)
```


## 5. 조합
