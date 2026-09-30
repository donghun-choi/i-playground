# i-playground 작업 규칙

## 0. 파이썬 기반
- 모든 코드는 Python(3.11+)으로 작성한다.
- 가능하면 표준 라이브러리만 쓴다. 외부 패키지가 필요하면 `requirements.txt`에 적는다.

## 1. 항상 웹에서 실시간 시각화
- 모든 실험/기능은 결과를 localhost 웹 페이지에서 **실시간으로** 볼 수 있어야 한다.
- 기본 도구는 이 저장소의 `livevis` 패키지다 (표준 라이브러리만 사용, 설치 불필요).

  ```python
  from livevis import LiveVis

  viz = LiveVis("실험 이름").start()      # http://localhost:8000 에 서버가 뜬다
  for step in range(1000):
      viz.log(step, loss=loss, acc=acc)   # 지표마다 차트가 하나씩 생기고 바로 갱신된다
  viz.wait()                              # Ctrl+C 전까지 페이지를 유지
  ```

- 포트는 `LiveVis(port=...)` 또는 환경변수 `LIVEVIS_PORT`로 바꾼다.
- 스칼라 지표가 아닌 시각화가 필요하면 `livevis`를 확장하거나, 같은 방식(Python 서버 + SSE + 브라우저)으로 만든다.
- 새 기능을 끝냈다고 말하기 전에 서버를 띄워 실제 페이지가 갱신되는지 확인한다.

## 2. 커밋 규칙
- 커밋 작성자(author/committer) 이름은 항상 `donghun-choi`.
- 커밋 메시지, 트레일러, PR 본문 어디에도 `claude`를 쓰지 않는다.
  (`Co-Authored-By: Claude ...`, `Claude-Session: ...`, "Generated with Claude Code" 등 전부 금지)
- 머지 커밋도 마찬가지다. `claude/...` 브랜치를 머지할 때는 `git merge -m "..."`로 메시지를 직접 쓴다.
- 강제 장치:
  - `.claude/hooks/session-start.sh`: 세션 시작 시 git 작성자를 `donghun-choi`로 맞추고 `core.hooksPath`를 `.githooks`로 설정한다.
  - `.githooks/commit-msg`: attribution 트레일러를 지우고, 작성자가 `donghun-choi`가 아니거나 메시지에 `claude`가 남아 있으면 커밋을 거부한다.
  - `.claude/settings.json`: 자동 attribution(커밋/PR)을 끈다.

## 명령어
- 데모 실행: `python -m livevis` → http://localhost:8000
- 테스트: `python -m unittest -v`
