# comma video compression challenge

<https://github.com/commaai/comma_video_compression_challenge>

1분짜리 대시캠 영상(`videos/0.mkv`, 37.5 MB, 1164x874, 1200프레임)을 최대한 작게 만들되,
두 신경망이 원본과 같은 출력을 내도록 복원해야 한다. 점수는 낮을수록 좋다.

```
score = 100 * segnet_dist + sqrt(10 * posenet_dist) + 25 * rate
```

| 항 | 의미 |
| --- | --- |
| `segnet_dist` | 각 프레임 쌍의 마지막 프레임을 512x384로 줄여 SegNet(5클래스)을 돌렸을 때 argmax가 원본과 다른 픽셀 비율 |
| `posenet_dist` | 연속 두 프레임(YUV6, 512x384)을 PoseNet에 넣은 6차원 pose 출력의 MSE |
| `rate` | `archive.zip` 크기 / 원본 크기 |

- 평가 제한: inflate + 평가 합쳐 30분 (CPU 4코어/16GB 또는 T4 GPU).
- 압축 쪽에는 원본 영상, 두 모델 등 무엇이든 써도 된다. inflate에 신경망 가중치가 필요하면 그건 archive 크기에 포함된다.
- 리더보드: 베이스라인 4.39 → 상위권 ~0.15 (2026-10 기준).

## 쓰는 법

```bash
bash comma_vcc/setup.sh                      # 챌린지 클론 + LFS + venv (한 번만)
bash comma_vcc/run.sh baseline_fast          # 압축 → inflate → 실시간 평가, http://localhost:8000
bash comma_vcc/run.sh my_idea --recompress   # compress.sh 를 다시 돌림
```

- 우리 제출물은 `comma_vcc/submissions/<이름>/` 에 `compress.sh`, `inflate.sh` (+ `inflate.py`) 를 두고
  `setup.sh` 를 다시 돌리면 챌린지의 `submissions/` 에 링크된다. 형식은 `challenge/submissions/baseline_fast` 참고.
- `live_eval.py` 는 공식 `evaluate.py` 와 같은 데이터로더/모델/수식을 쓴다. 페이지에는 누적 점수와 항별 기여,
  배치별 왜곡, 그리고 배치에서 SegNet이 가장 많이 틀린 샘플의 원본/복원/불일치 맵이 실시간으로 뜬다.
- 결과는 `<제출물>/live_report.txt` 에 남는다.

## 주의: 챌린지의 LLM 사용 정책

챌린지는 코딩 에이전트가 **제출 코드 전부를 쓰는 것**과 **PR 설명/공개 코멘트를 쓰는 것**을 금지한다.
도구·분석·리뷰·일부 코드 작성은 허용. 실제로 제출할 거라면 핵심 압축 코드는 직접 쓰고 이해해야 한다.

## 기록

| 제출물 | segnet | posenet | rate | score | 메모 |
| --- | --- | --- | --- | --- | --- |
| baseline_fast (재현) | 0.00947 | 0.380 | 0.0598 | 4.39 | x265 ultrafast crf30, 45% 축소, 평가 227s (CPU 4코어) |
