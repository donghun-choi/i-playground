# i-playground

파이썬 실험 놀이터. 모든 실험은 localhost 웹 페이지에서 실시간으로 볼 수 있게 만든다.

## 빠른 시작

```bash
python -m livevis        # 데모 → http://localhost:8000
python -m unittest -v    # 테스트
```

설치할 것은 없다 (Python 3.11+ 표준 라이브러리만 사용).

## 내 실험에 붙이기

```python
from livevis import LiveVis

viz = LiveVis("실험 이름").start()       # http://localhost:8000
for step in range(1000):
    loss, acc = train_one_step()
    viz.log(step, loss=loss, acc=acc)    # 지표마다 차트가 생기고 바로 갱신된다
viz.wait()                               # Ctrl+C 전까지 페이지 유지
```

- 포트 변경: `LiveVis(port=9000)` 또는 `LIVEVIS_PORT=9000`
- 브라우저를 늦게 열어도 지금까지의 기록이 모두 보인다 (지표당 최근 `max_points=5000`개).
- 페이지의 "표 보기"로 지표별 최신값/최소/최대를 표로 볼 수 있다.

작업 규칙은 [CLAUDE.md](CLAUDE.md)에 있다.
