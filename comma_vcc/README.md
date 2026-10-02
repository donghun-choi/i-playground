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

## 우리 접근: semantic_cpu (CPU 만으로)

영상을 복원하지 않는다. 평가 네트워크 두 개가 원본과 같은 출력을 내는 프레임을 만든다.

```
archive = [seg 맵 600장 (무손실)] + [렌더러] + [pose carrier]
홀수 프레임 = 렌더러(seg 맵)                     → SegNet 이 같은 맵을 내도록 학습
짝수 프레임 = 127.5 + bicubic(Σ c[i,k] · B_k)    → PoseNet 이 같은 pose 를 내도록 학습 (기저 B 공유, 계수 c 쌍마다)
```

### 분석에서 나온 사실 (lab/analyze.py 등)
- 평가 네트워크는 1164x874 를 512x384 로 bilinear 축소해서 본다. 축소가 읽는 원본 픽셀은 768행 x 1024열뿐이고 겹치지 않는다.
  → 512x384 이미지만 설계하면 되고, **각 칸의 2x2 원본 픽셀을 서로 다른 정수로 채우면 정수 사이 값까지 표현된다** (model.expand_fine).
  정수 반올림 대비 축소 오차 0.22 → 0.0035, GT 를 넣었을 때 pose 항 0.034 → 0.0013.
- 영상은 야간 고속도로. SegNet 5클래스 비율: 도로 23%, 차선 0.6%, 배경 50%, 차량 1.2%, 보닛 25%.
  평균색만 칠한 렌더는 SegNet 불일치 4.5% → 학습된 렌더러가 필요.
- pose 0번 차원(전진) 평균 31.3, 표준편차 1.26. 짝수=홀수(정지)면 pose 항 40.
- bf16 은 pose 31 근처에서 해상도 0.125 라 carrier 미세조정은 fp32 로 해야 한다.
- 이 CPU 는 AMX 가 있어 SegNet forward+backward 가 bf16 에서 이미지당 0.13s (fp32 의 2.7배 빠름).
- 코어 수보다 스레드를 많이 띄우면 학습이 4~7배 느려진다.

### seg 맵 코덱 (segcodec.py)
- 계층 순서: stride 32 격자 → 레벨마다 (A) 칸 중심, (B) 변 중점. 한 패스 안의 픽셀은 서로 독립이라 디코드가 벡터화된다.
- 확률: 작은 CNN 이 [부분 디코드된 현재 맵, 이전 2프레임 맵 클래스 비율, 레벨/패스] 를 보고 예측.
- 평가 머신에서 비트 단위로 같은 확률을 내도록 **정수 CNN**: 정수 가중치/활성값, 부분합 < 2^24 라 float32 합성곱이 정확.
  시작할 때 float64 와 대조하는 자체 검사, 어긋나면 float64 로.
- 문맥 모델별 크기: 카운트 문맥 530 B/frame → CNN 16ch 285 → CNN 24ch5층 ~245.

### 실험 기록 (pose carrier, 64쌍)
| 짝수 프레임 구성 | posenet_dist |
| --- | --- |
| 홀수 + 회색 학습 기저 96x128 x12 | 0.00059 (기저가 147K 값이라 너무 큼) |
| 홀수 + 회색 학습 기저 24x32 x12 | 0.021 |
| 원근 변환(warp) 8 파라미터 | 42 |
| 고정 DCT 기저 24개 | 38 |
| 내용 적응형 CNN 섭동 | 0.25 (100 step) |
| PoseNet Jacobian 방향 (J^T a) | 연속값 0.05, 반올림 시 붕괴 |
| **회색 127.5 + RGB 학습 기저 24x32 x12** (PR #130 구조) | 0.0021 (반올림 포함) |

## 주의: 챌린지의 LLM 사용 정책

챌린지는 코딩 에이전트가 **제출 코드 전부를 쓰는 것**과 **PR 설명/공개 코멘트를 쓰는 것**을 금지한다.
도구·분석·리뷰·일부 코드 작성은 허용. 실제로 제출할 거라면 핵심 압축 코드는 직접 쓰고 이해해야 한다.

## 기록

| 제출물 | segnet | posenet | rate | score | 메모 |
| --- | --- | --- | --- | --- | --- |
| baseline_fast (재현) | 0.00947 | 0.380 | 0.0598 | 4.39 | x265 ultrafast crf30, 45% 축소, 평가 227s (CPU 4코어) |
| semantic_cpu v1 | 0.00153 | 0.00067 | 0.00650 | **0.397** | seg맵 142KB + 문맥모델 23KB + 렌더러 42KB + carrier 36KB, inflate 740s + 평가 251s (4코어) |
