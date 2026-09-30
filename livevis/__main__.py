"""데모: 가짜 학습 곡선을 실시간으로 그린다.

    python -m livevis   →   http://localhost:8000
"""

import math
import random
import time

from livevis import LiveVis


def main() -> None:
    viz = LiveVis("demo: 가짜 학습 곡선").start()
    walk = 0.0
    step = 0
    try:
        while True:
            walk += random.gauss(0, 1)
            viz.log(
                step,
                loss=2.5 * math.exp(-step / 300) + 0.1 + random.gauss(0, 0.05),
                accuracy=1 - 0.9 * math.exp(-step / 250) + random.gauss(0, 0.01),
                lr=1e-3 * 0.5 * (1 + math.cos(math.pi * (step % 1000) / 1000)),
                random_walk=walk,
            )
            step += 1
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        viz.stop()


if __name__ == "__main__":
    main()
