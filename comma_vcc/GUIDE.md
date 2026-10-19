# semantic_cpu 다시 짜기 가이드

지금 코드 (v6.26, 로컬 점수 **0.1196**, archive 147,611 B) 를 직접 다시 짜서 제출할 때 보는 문서.
챌린지 LLM 정책상 **제출 코드 대부분과 PR 설명은 직접 써야 한다** (9절). 여기 있는 코드는 참조 구현이자 검증 기준이다.

목차
1. 한 장 요약
2. 코드 지도
3. archive 형식 (바이트 단위)
4. 디코더 단계별 명세
5. 다시 짜는 순서와 단계별 검증 (golden)
6. 학습 레시피 (압축 쪽)
7. 함정
8. 시간·환경
9. 제출 체크리스트
10. 다음에 해볼 것

---

## 1. 한 장 요약

영상을 복원하지 않는다. 평가 네트워크 두 개 (SegNet, PoseNet) 가 원본과 같은 출력을 내는 프레임을 만든다.

```
score 0.1196 = seg 0.0188 (100 × 불일치 0.000188) + pose 0.0026 (√(10 × 6.7e-7)) + rate 0.0983 (25 × 147,611 / 37,545,489)

archive.zip (147,611 B) = zip(stored) 안의 파일 'p' (147,511 B)
  ctxe  14,318 B  정수 문맥 CNN (23→24 채널 6층 3x3, dilation 1,2,4,8,2,1 + 1x1 출력, 가중치 31,008개)
  segs 106,056 B  seg 맵 601장 무손실 스트림 (177 B/장)
  rend  16,439 B  렌더러 U-Net 24·32·40 (가중치 81,899개, 채널별 학습 비트, 살아 있는 출력 채널 213/251)
  pos4  10,662 B  pose: 회색 기저 8개 (24x32, 5비트) + 쌍마다 계수 8개·아핀 6개 (8비트 격자)
```

평가가 하는 일 (challenge/evaluate.py): 1200프레임을 (짝수, 홀수) 600쌍으로 나눠
- SegNet: 쌍의 **홀수 프레임**을 512x384 로 줄여 argmax → 원본과 다른 픽셀 비율
- PoseNet: 쌍 두 장 (YUV6, 512x384) → 6차원 pose 의 MSE
- 둘 다 1164x874 프레임을 512x384 로 **bilinear 축소한 뒤** 본다.

그래서 프레임을 이렇게 만든다:

```
M_i (i = 0..599)  = 원본 홀수 프레임 i 의 SegNet argmax 맵 (384x512, 5클래스)
M_pre             = 원본 짝수 프레임 0 의 SegNet argmax 맵 (첫 쌍의 '이전 프레임' 용)
홀수 프레임 i     = 렌더러(M_i)                                       → SegNet 이 M_i 를 내도록 학습
짝수 프레임 i     = clamp(아핀_i(홀수 프레임 i-1) + bicubic(Σ_k c[i,k]·B_k))   (홀수 프레임 -1 = 렌더러(M_pre))
                                                                     → PoseNet 이 원본 pose 를 내도록 피팅
1164x874 기록     = 512x384 float 이미지를 2x2 서브픽셀 정수로 펼침 (평가의 bilinear 축소가 거의 그대로 되돌림)
```

seg 맵 601장 (M_pre, M_0..M_599) 은 계층 순서 + 정수 문맥 CNN + range coder 로 무손실 부호화한다.
렌더러·문맥 CNN 가중치도 archive 에 들어가므로 크기 손실을 넣어 학습했다 (self-compression).

## 2. 코드 지도

| 파일 | 줄 | 내용 |
| --- | --- | --- |
| `submissions/semantic_cpu/inflate.py` (+ `inflate.sh`) | 85 | 디코더 진입점: 섹션 풀기 → seg 복호화 → 렌더 → 짝수 합성 → 기록 |
| `submissions/semantic_cpu/segcodec.py` | 281 | 패스 마스크, 23채널 입력, 정수 CNN (QNet), 확률표, stride-32 카운트 모델, encode/decode |
| `submissions/semantic_cpu/wcodec.py` | 198 | 정수 가중치 엔트로피 부호 (행별 이산 라플라스 + range coder) |
| `submissions/semantic_cpu/model.py` | 186 | 렌더러, 가중치 형식 (pack_state), 아핀 + 기저 합성, 서브픽셀 확장 |
| `submissions/semantic_cpu/archive.py` | 120 | 'p' 컨테이너, rend/pos4 섹션, Rice 부호 |
| `lab/common.py` | 169 | 경로, 평가 네트워크, GT 캐시, 평가 경로 근사 `fine_q`, 렌더러 불러오기 |
| `lab/build_cache.py` | 51 | 원본을 한 번 디코드해서 GT seg 맵 / pose 캐시 |
| `lab/ctxmodel.py` | 276 | 문맥 CNN 학습 (float, QAT, self-compression) |
| `lab/renderer.py` | 262 | 렌더러 학습 (flip 손실, QAT, self-compression) |
| `lab/selfcomp.py` | 116 | self-compression: 채널별 비트·스케일 학습 + 크기 손실 |
| `lab/pose_fit.py`, `lab/pose_refine.py` | 163, 235 | pose 처음 피팅, 다듬기 (기저 학습, 격자 양자화, 탐욕 탐색) |
| `lab/posefmt.py` | 76 | pose 중간 파일 (pose2 blob) ↔ pos4 |
| `lab/build_archive.py` | 139 | 문맥 CNN 정수화 + seg 부호화 + archive.zip 조립 |
| `lab/golden.py` | 173 | 단계별 기준값 저장 / 비교 (5절) |
| `lab/test_codec.py` | 152 | 형식 단위 테스트 (캐시 없이 몇 초) |

- 제출물 (`submissions/semantic_cpu`) 은 디코더와 그 형식의 짝 (encode, pack_*) 만 둔다. 학습·정수화는 lab.
- 실패한 실험 스크립트 (carrier, 분석, vjp, swa 등) 는 지웠다. 커밋 `4c31838` 에 남아 있다:
  `git show 4c31838:comma_vcc/lab/seg_errors.py`. README 의 '실험 기록' 이 그 결과다.
- `colab/` 은 GPU 렌더러 사이클용 (README 'GPU (Colab)').

## 3. archive 형식 (바이트 단위)

모든 정수는 little-endian. `xz(...)` 는 Python `lzma.compress(data, format=FORMAT_RAW, filters=[{"id": FILTER_LZMA2, "preset": 9 | PRESET_EXTREME}])`.
`RC(...)` 는 constriction `stream.queue.RangeEncoder()` (u32 단어, 기본 설정) + `stream.model.Categorical(perfect=False)` 로 부호화한 단어열.

### 3.1 컨테이너
- `archive.zip`: 파일 `p` 하나, 압축 없음 (ZIP_STORED).
- `p` = `b"SCP1"` + 섹션 나열. 섹션 = `[이름 4바이트 ASCII][u32 길이][데이터]`. 순서: `ctxe`, `segs`, `rend`, `pos4`.

### 3.2 ctxe: 문맥 CNN
```
[u32 L] + xz(raw) (L 바이트) + RC(정수 가중치)
raw = [u8 층 수 n]
      + 층마다: [u16 o][u16 i][u8 k][u8 dil][u8 relu | 플래그] [u8 qmax] [r × o 바이트] [B int32 × o] [M int32 × o]
```
- 플래그: 첫 층의 relu 바이트 비트 1 (float32 재양자화) 과 비트 2 (빈도표 최대 2^20) — 둘 다 켜져 있어야 한다 (값 7).
- 가중치 W (o, i, k, k) 는 RC 로: 층 순서, 출력 채널 (행) 순서로 행마다 i·k·k 개를 C 순서로. `qmax = max|W|` (층마다), 0 이면 그 층은 RC 에 없음.
  기호 = q + qmax (0..2·qmax), 확률은 그 행의 r 로 만든 이산 라플라스 (3.6).
- RC 를 풀어 W 를 채우면 `QNet.to_bytes` 형식이 된다: `[u8 n]` + 층마다 `[7바이트 헤더][W int8 (o,i,k,k)][B int32][M int32]`.

### 3.3 segs: seg 맵 스트림
```
[u32 n = 601] + RC(기호들)
```
기호 순서 (디코드 순서와 같다): 맵 t = 0..600 마다
1. stride-32 격자 192칸 (행 우선, 픽셀 (32a, 32b)): 확률 = `CoarseModel` (4.3)
2. 레벨 s = 32, 16, 8, 4, 2 마다, 패스 A1, A2, B 마다: 그 패스의 보낼 칸들 (stride-h 격자에서 행 우선, h = s/2), 확률 = 문맥 CNN (4.2)

맵 0 = M_pre, 맵 t ≥ 1 = M_{t-1}.

### 3.4 rend: 렌더러
```
[u8 13][u16 c1][u16 c2][u16 c3] + ec
ec = [u16 텐서 수 T][u8 0xEC][u32 L] + xz(raw) (L 바이트) + RC(정수 가중치)
raw = [u8 bits] + 텐서마다 (state_dict 순서): [u8 kind]
        kind 0 (1차원, bias): fp16 값 × numel
        kind 1 (2차원 이상):   [u8 qmax][fp16 스케일 × o][r × o 바이트]
RC: kind 1 텐서마다, 출력 채널마다 numel/o 개 (C 순서), 기호 = q + qmax, 확률 = 그 행의 r 로 만든 라플라스
```
- 풀면 `pack_state` 형식: `[u16 T][u8 bits]` + 텐서마다 `[u8 kind]` + (kind 1: fp16 스케일 × o + int8 q (텐서 모양), kind 0: fp16 값).
- 가중치 = q × 스케일 (float32), bias = fp16 → float32. bits 칸은 0 (채널별 학습 비트라 의미 없음).
- state_dict 순서 (c1,c2,c3 = 24,32,40, 18개):
  `e1.0.{weight (c1,7,3,3), bias}`, `e1.2 (c1,c1,3,3)`, `e2.0 (c2,c1,3,3)`, `e2.2 (c2,c2,3,3)`, `e3.0 (c3,c2,3,3)`,
  `e3.2 (c3,c3,3,3) dilation 2`, `d2.0 (c2,c3+c2,3,3)`, `d1.0 (c1,c2+c1,3,3)`, `out (3,c1,1,1)` — 각각 weight 다음 bias.

### 3.5 pos4: pose
```
[u16 n=600][u16 k=8][u8 C=1][u16 bh=24][u16 bw=32]
+ f32 c_step × k + f32 a_step × 6 + f32 B_scale × k
+ [u8 qmax][r × (k·C) 바이트][u32 단어 수 W][u32 단어 × W]          ← 기저 B_q (k·C 행 × bh·bw, 행별 라플라스 RC)
+ Rice: [i32 평균 m][u8 kk] × (k+6) 행 + 비트열                       ← 계수 정수 (k+6) 행 × n: c 의 k 행, 그다음 a 의 6 행
```
- Rice 비트열: 행 순서로 값마다 v - m 을 zigzag (z = 2v if v ≥ 0 else -2v-1) → `(z >> kk)` 개의 1, 0 하나, z 의 아래 kk 비트 (MSB 먼저).
  전체를 MSB 먼저 바이트로 묶고 마지막 바이트는 0 으로 채운다 (`np.packbits`).
- 값: `B = B_q (float32) × B_scale[k]`, `c = float32(c_int × float64(c_step))`, `a = float32(a_int × float64(a_step))`.

### 3.6 라플라스 행 분포 (wcodec)
```
f(0) = 2^24,  f(j+1) = max(1, (f(j) · r) >> 8)   (정수),  r = 1..255
P(q) = f(|q|) / Σ_{q'=-qmax..qmax} f(|q'|)    (float64 나눗셈)
```
r 은 인코더가 행마다 교차 엔트로피가 가장 작은 것을 고른다 (디코더는 받기만).

## 4. 디코더 단계별 명세

### 4.1 패스와 입력 (segcodec.masks, Inputs)
stride-h 격자 (gh, gw) = (384/h, 512/h), 격자점 (i, j) = 픽셀 (i·h, j·h). 레벨 s 에서 h = s/2.
```
even = i, j 둘 다 짝수   (= 이전 레벨 (stride s) 격자, 이미 앎)
odd  = i, j 둘 다 홀수
A1: 보낼 칸 = odd & (i mod 4 == j mod 4)      아는 칸 = even
A2: 보낼 칸 = odd & ~A1                        아는 칸 = even | A1
B : 보낼 칸 = ~(even | odd)                     아는 칸 = even | odd
```
입력 23채널 (값 0..64 의 정수, float32 로 들고 다님):
```
0-4   현재 맵 one-hot (격자점 값) × 아는 칸 × 64
5     아는 칸 × 64
6-10  이전 맵 (t-1) 클래스 비율: floor(h×h 블록 안 클래스 c 픽셀 수 × 64 / h²), 블록 = 격자점이 왼쪽 위인 h×h (없으면 0)
11-15 전전 맵 (t-2) 클래스 비율 (같은 방식)
16    64 if 패스가 A1/A2 else 0
17-21 레벨 one-hot × 64 (LEVELS = 32,16,8,4,2 순서)
22    64 if 패스 == A2 else 0
```

### 4.2 정수 CNN 과 확률 (segcodec.QNet, probs)
```
층마다: acc = conv2d(x, W, padding = dil·(k//2), dilation = dil)     (0 패딩, 정확한 정수)
       y   = floor( f32( f32(acc + B) × f32(M · 2^-16) ) )            (덧셈·곱셈 각각 float32 로 반올림, FMA 금지)
       relu 층: y = clamp(y, 0, 255)
logit = 마지막 층 y (5채널, 1/16 nat 단위)
d_c   = min(max_c' logit_c' - logit_c, 255)
freq_c = FREQ[d_c],  FREQ[d] = max(1, round(2^20 · exp(-d/16)))
p_c   = freq_c / Σ freq   (float64)
```
- 합성곱이 정확하려면 모든 부분합이 2^24 미만이어야 한다: 입력 최댓값 (64, 이후 255) × Σ|W| + |B| < 2^24 (QNet 생성 때 확인).
- `self_check`: 레벨별 격자 크기마다 무작위 입력으로 float32 합성곱 결과가 float64 합성곱과 같은지 보고, 다르면 float64 합성곱으로 (재양자화는 그대로 float32).

### 4.3 stride-32 격자 (segcodec.CoarseModel)
- 카운트 표 c[6][5], 처음 전부 1. 문맥 = 이전 맵 (t-1) 의 같은 픽셀 클래스, 맵 0 은 5.
- 확률 = c[문맥] / Σ (float64). 맵 하나의 192칸을 **같은 표로** 한 번에 부호화한 뒤, 칸마다 c[문맥][기호] += 16.

### 4.4 복호화 루프 (segcodec.decode)
```
for t in 0..600:
    격자 192칸 복호화 → seg[t] 에 기록, 카운트 갱신
    for s in 32,16,8,4,2:  for 패스 in A1,A2,B:
        x = 입력 (4.1, 현재 맵은 지금까지 복호화한 값), logit = QNet(x)
        보낼 칸들의 확률 (4.2) 로 한 번에 복호화 → seg[t] 의 그 격자점들에 기록
```
s = 2 (h = 1) 의 B 패스가 끝나면 모든 픽셀이 채워진다.

### 4.5 렌더러 (model.Renderer)
```
x  = [one-hot(M) 5채널, y, x 좌표 (linspace(-1,1,384), linspace(-1,1,512))]        (7, 384, 512)
f1 = relu(conv(relu(conv(x))))                       c1, 3x3
f2 = relu(conv(relu(conv(avgpool2(f1)))))            c2
f3 = relu(conv_dil2(relu(conv(avgpool2(f2)))))       c3 (두 번째 3x3 은 dilation 2, padding 2)
u2 = relu(conv(cat[bilinear×2(f3), f2]))             c2      (interpolate scale_factor=2, align_corners=False)
u1 = relu(conv(cat[bilinear×2(u2), f1]))             c1
RGB = sigmoid(conv1x1(u1)) × 255                     float, 0..255
```
inflate 는 홀수 프레임을 8장씩 (맵 1..600), M_pre 는 따로 1장 렌더한다.

### 4.6 짝수 프레임 (model.even_frames)
```
a' = a ⊙ [0.01, 0.01, 2/512, 0.01, 0.01, 2/384]
θ  = [[1+a'0, a'1, a'2], [a'3, 1+a'4, a'5]]
아핀 = grid_sample(이전 홀수 프레임, affine_grid(θ), bilinear, padding border, align_corners=False)
기저 = interpolate(Σ_k c_k B_k  (1, 24, 32), size (384, 512), bicubic, align_corners=False)   (회색: RGB 에 같은 값)
짝수 = clamp(아핀 + 기저, 0, 255)
```

### 4.7 기록 (model.expand_fine)
평가의 bilinear 축소 (874x1164 → 384x512, align_corners=False) 에서 작은 이미지 한 칸은 원본 2x2 칸의 가중합이고, 칸들이 겹치지 않는다.
- 칸마다 base = clamp(floor(x), 1, 253), 2x2 에 base + b (b ∈ {-1,0,1,2}^4, 256 조합) 중 가중합이 x 에 가장 가까운 것을 쓴다.
- 축소가 읽지 않는 원본 행/열은 가까운 읽히는 행/열을 복사.
- 출력: 쌍마다 (짝수, 홀수) 순서로 874x1164x3 uint8 (HWC, RGB), 1200장 = 3,662,409,600 B.

## 5. 다시 짜는 순서와 단계별 검증 (golden)

먼저 기준값을 만든다 (약 5분):
```bash
cd comma_vcc/lab
../.venv/bin/python golden.py dump        # → cache/golden/ (sections.json 에 단계별 sha256)
../.venv/bin/python -m unittest test_codec -v
```
내 구현의 중간 결과를 같은 이름의 `.npy`/`.npz` 로 저장하고 `golden.py diff 내파일.npy` 로 비교한다.

기준 sha256 (v6.26): archive.zip `1cb48d5f…f8fe7b`, 'p' `2ff16e1e…fd95`, seg 맵 601장 `04156f76…b2af`, 전체 .raw `27cc17b9…8b5f`.

| 순서 | 만들 것 | 확인 (golden) | 같아야 하는 정도 |
| --- | --- | --- | --- |
| 1 | 'p' 컨테이너 풀기 | `sections.json` 의 섹션 길이·sha256 | 정확히 |
| 2 | ctxe → QNet 바이트 (xz + 라플라스 RC) | `qnet_bytes_sha256`, `qnet_layers.npz` | 정확히 |
| 3 | QNet 산술 (4.2) | `qnet_probe_in.npy` → `qnet_probe_out.npy` | 정확히 |
| 4 | 패스 마스크·23채널 입력 (4.1) | `inputs_t5_s4_A2.npy`, `logits_t5_s4_A2.npy` | 정확히 |
| 5 | 확률표 (4.2) | `probs_t5_s4_A2.npy` | 정확히 (float64 비트까지) |
| 6 | 격자 카운트 + 복호화 루프 (4.3, 4.4) | `seg.npz` (`seg_sha256`) | 정확히 |
| 7 | rend → 렌더러 가중치 | `renderer.npz` | 정확히 |
| 8 | 렌더러 forward (4.5) | `odd_0_3.npy`, `frame_sums.npy` | ~1e-4 (합성곱 구현 차이) |
| 9 | pos4 → B, c, a (Rice + RC) | `pose.npz` | 정확히 |
| 10 | 짝수 프레임 (4.6) | `even_0_3.npy` | ~1e-4 |
| 11 | 서브픽셀 확장 (4.7) | `raw_pair0.npy`, `raw.sha256` | 8·10 이 정확히 같을 때만 정확히 |
| 12 | 전체 | `bash run.sh <이름>` → score 0.1196 | 0.1196 |

- 6 이 하나라도 틀리면 그 뒤 range decoder 가 어긋나서 쓰레기가 나온다. 2~5 를 먼저 맞추면 6 은 거의 저절로 맞는다.
- 8·10 은 float 연산이라 순서가 다르면 1e-4 레벨로 다를 수 있다. 점수에는 영향 없다 (AVX2 전용 커널로 돌려도 0.1198 그대로였다).
  **가중치·pose 값 자체 (7, 9) 가 다르면 pose 항이 크게 무너진다** (7절).
- 형식을 바꾸고 싶으면 (예: 섹션을 합치거나 헤더를 줄이기) 인코더 (lab) 도 같이 바꿔야 한다. 그때는 바뀐 단계 앞까지만 golden 이 유효하다.
- 인코더를 다시 짤 때는 `build_archive.py` 가 만든 archive.zip 이 지금 것과 같은 바이트인지 (sha256 `1cb48d5f…`) 로 확인할 수 있다
  (같은 체크포인트를 쓰면 문맥 CNN 정수화·seg 부호화·가중치 부호화까지 모두 결정적이다).

추천 순서: **디코더 (제출되는 부분) → 확인 → 인코더/학습 (원하면)**.
디코더만 다시 짜고 학습 결과 (cache 의 체크포인트) 는 지금 것을 써도 되지만, 압축 스크립트를 PR 에 넣을 거라면 그것도 직접 짜는 게 정책 취지에 맞다.

## 6. 학습 레시피 (압축 쪽)

실제로 돈 순서 (v1 → v6.26, 약 2주) 는 README '재현' 에 그대로 있다 (정리 전 옵션 이름이라 일부는 지금 코드와 다르다:
`--split-a` 는 없어짐 (항상 23채널), `--ctx-f32 --wide-probs --wec` 는 build_archive 기본 동작, `--film`, `--arch v2`, `--round`, `--carrier` 는 삭제).
처음부터 다시 한다면 아래 순서를 권한다. **실제 계보를 줄인 것이라 같은 숫자가 나온다는 보장은 없다.** 시간은 CPU 4코어 기준.

모든 명령은 `cd comma_vcc/lab` 에서, 파이썬은 `../.venv/bin/python`. 학습 스크립트는 livevis 로 실시간 차트를 띄운다 (`--port`).

### 6.0 캐시 (3분)
```bash
python build_cache.py          # cache/gt_small.npy, gt_seg.npy (600 홀수 맵), gt_pose.npy, seg_pre.npy (M_pre)
```

### 6.1 문맥 CNN (seg 스트림)
```bash
C="--ch 24 --layers 6 --dils 1,2,4,8,2,1 --threads 4"
python ctxmodel.py $C --steps 20000 --lr 3e-3 --out ../cache/ctx_a.pt                               # float, 약 40~60분
python ctxmodel.py $C --steps 16000 --lr 1e-3 --init ../cache/ctx_a.pt --out ../cache/ctx_b.pt
python ctxmodel.py $C --steps 6000 --lr 1e-3 --qat-bits 5 --init ../cache/ctx_b.pt --out ../cache/ctx_q5.pt
S="--self-compress 1.0 --sc-rate laplace --sc-init-bits 5 --steps 20000"
python ctxmodel.py $C $S --lr 5e-4 --init ../cache/ctx_q5.pt --out ../cache/ctx_s1.pt                # 라운드마다 lr 을 낮춰 (3e-4, 2e-4, 1.5e-4)
python ctxmodel.py $C $S --lr 2e-4 --init ../cache/ctx_s1.pt --out ../cache/ctx_s2.pt                #  이득이 200 B 아래로 떨어질 때까지
python ctxmodel.py $C $S --lr 5e-5 --sc-lr 0 --init ../cache/ctx_s2.pt --out ../cache/ctx_p1.pt      # 다듬기 두 번 (5e-5, 2e-5)
python build_archive.py --ctx ../cache/ctx_p1.pt --segs-only                                        # → cache/segs_pre.bin, 크기 확인
```
- 평가 지표: 학습 로그의 `B/frame (coarse 제외)` (10장) 와 `--segs-only` 의 실제 스트림 크기 + 문맥 모델 크기.
- 실제 계보에서는 5층 22채널 → 23채널 (입력 채널 0 으로 붙임) → 6층 (항등 층 끼움, `--deepen-from`) 순으로 키웠다.
- `--sc-lr 0` 이어도 OneCycleLR 이 모든 파라미터 그룹에 같은 max_lr 를 줘서 비트·스케일이 조금 움직인다 (그래도 효과 있었음).

### 6.2 렌더러 (seg 항)
```bash
W="--widths 24,32,40 --threads 4"
python renderer.py $W --epochs 15 --lr 1.5e-3 --out ../cache/r_a.pt                                  # CE + margin hinge 로 시작
python renderer.py $W --resume ../cache/r_a.pt --epochs 6 --lr 4e-4 --cosine --fp32 --bits 6 --qat --loss flip --full-eval --out ../cache/r_b.pt
python renderer.py $W --resume ../cache/r_b.pt --epochs 10 --lr 3e-4 --cosine --fp32 --bits 4 --qat --loss flip --full-eval --out ../cache/r_c.pt
#   ↑ 4비트 flip 10에폭 사이클을 600장 불일치가 줄어드는 동안 반복 (GPU 권장: colab/vcc_gpu.ipynb, 사이클당 10분, CPU 는 약 1시간)
SC="--epochs 10 --lr 3e-4 --bs 4 --cosine --fp32 --bits 4 --sc-rate laplace --loss flip --full-eval"
python renderer.py $W $SC --self-compress 1.0 --resume ../cache/r_c.pt --out ../cache/r_s1.pt          # self-compression 사이클 (×1 두 번 → ×2 세 번)
python renderer.py $W $SC --self-compress 2.0 --resume ../cache/r_s1.pt --out ../cache/r_s2.pt          #  600장 불일치 + 크기로 가장 좋은 것을 고른다
P="--epochs 5 --lr 3e-5 --sc-lr 0 --bs 4 --cosine --fp32 --bits 4 --self-compress 1.0 --sc-rate laplace --loss flip --full-eval"
python renderer.py $W $P --resume ../cache/r_s2.pt --out ../cache/r_p1.pt                               # 격자 고정 다듬기
python renderer.py $W $P --flip-start 0 --tau 0.06,0.02 --seed 4 --resume ../cache/r_p1.pt --out ../cache/r_p2.pt   # 날카로운 τ
python renderer.py $W $P --flip-start 0 --tau 0.04,0.01 --seed 5 --resume ../cache/r_p2.pt --out ../cache/r_p3.pt
python renderer.py $W $P --flip-start 0 --tau 0.03,0.01 --seed 6 --resume ../cache/r_p3.pt --out ../cache/r_p4.pt   # 한 번 더 (시드 7)
python renderer.py $W ${P/--epochs 5 --lr 3e-5/--epochs 10 --lr 1e-5} --flip-start 0 --tau 0.03,0.01 --seed 8 --resume ../cache/r_p4.pt --out ../cache/r_p5.pt
python renderer.py $W ${P/--epochs 5 --lr 3e-5/--epochs 10 --lr 5e-6} --flip-start 0 --tau 0.03,0.01 --seed 9 --resume ../cache/r_p5.pt --out ../cache/r_final.pt
```
- self-compression 결과는 `r.pt` (inflate 와 같은 float 가중치), `r.pt.q` (archive 에 들어갈 바이트), `r.pt.sc.pt` (이어서 학습용) 세 개.
  `--self-compress` 와 `--resume r.pt` 를 같이 주면 `r.pt.sc.pt` 에서 float 가중치와 비트·스케일을 이어받는다.
- flip 손실: margin = 정답 logit - 나머지 최대. 앞부분 softplus(-margin/0.2)·0.2, `--flip-start` 이후 sigmoid(-margin/τ) (τ 는 `--tau 시작,끝` 선형).
- 판단 기준은 `--full-eval` 의 600장 불일치 (24장 검증은 사이클 간 비교에 부족했다).

### 6.3 pose (pose 항)
렌더러가 **조금이라도** 바뀌면 다시 맞춰야 한다 (7절). 렌더러를 끝낸 뒤에 한다.
```bash
R="--renderer ../cache/r_final.pt --renderer-cfg 24,32,40 --rbits 0"
python pose_fit.py $R --k 12 --cbits 10 --bbits 6 --out ../cache/pz_a.bin                           # 처음 피팅
python pose_refine.py --pose2 ../cache/pz_a.bin $R --cbits 10 --epochs 150 --lr 0.01 --train-b 0.02 --dimw 1.0 --plain-epochs 30 --q-epochs 10 --greedy-rounds 1 --out ../cache/pz_b.bin
python pose_refine.py --pose2 ../cache/pz_b.bin $R --cbits 8 --keep-k 8 --epochs 150 --lr 0.01 --train-b 0.01 --bbits 5 --b-qat --dimw 1.0 --plain-epochs 30 --q-epochs 10 --greedy-rounds 3 --qf --out ../cache/pz_c.bin
python pose_refine.py --pose2 ../cache/pz_c.bin $R --cbits 8 --b-gray --epochs 150 --lr 0.01 --train-b 0.01 --bbits 5 --b-qat --dimw 1.0 --plain-epochs 30 --q-epochs 10 --greedy-rounds 3 --qf --out ../cache/pz_d.bin
F="--cbits 8 --epochs 150 --lr 0.01 --train-b 0.01 --bbits 5 --b-qat --dimw 1.0 --plain-epochs 30 --q-epochs 10 --greedy-rounds 3 --qf"
python pose_refine.py --pose2 ../cache/pz_d.bin $R $F --out ../cache/pz_e.bin                        # 이어 학습 (약 3시간씩), d 가 안 줄 때까지 반복
```
- 마지막 줄의 레시피가 v6.17 이후 모든 pose 재피팅에 쓴 것이다 (이전 pose 결과의 기저에서 이어서).
- `--qf`: 학습과 탐욕 탐색을 평가 경로 (서브픽셀 정수화 후 축소, `common.fine_q`) 근사로. 없으면 float 경로에 과적합한다 (d 3.7배 차이).
- 출력 로그의 `평가 경로 posenet_dist` 와 `pose 관련 점수` (pose 항 + pos4 rate) 로 비교한다.

### 6.4 조립과 평가 (약 15분)
```bash
python build_archive.py --ctx ../cache/ctx_p1.pt --renderer ../cache/r_final.pt --pose2 ../cache/pz_e.bin   # seg 부호화 + 왕복 확인
cd .. && bash run.sh semantic_cpu            # 압축 풀기 → inflate → 평가, http://localhost:8000 에 실시간
```
v6.26 그대로는 인자 없이 `python build_archive.py` (기본값이 v6.26 산출물).

## 7. 함정

**결정성 (seg 스트림)**
- 인코더와 디코더의 확률이 float64 비트까지 같아야 한다. 그래서 문맥 CNN 이 정수 연산이고, 빈도표가 정수, 나눗셈이 한 번이다.
- 재양자화는 `f32(acc + B)` 다음 `f32(× M·2^-16)` 다음 floor. 정수로 `(acc + B)·M >> 16` 을 계산하면 **다른 값**이 나온다 (예전 float64 형식이 그것).
  numpy 로 짤 때 float32 배열끼리 따로 더하고 곱한다 (FMA 로 합치지 않게).
- constriction 은 확률 배열의 stride 를 무시하고 메모리를 그대로 읽는다 → 반드시 C 연속 float64.
- range coder 는 queue (FIFO) 라 부호화 순서 = 복호화 순서. stride-32 격자 192칸은 맵마다 한 번에, 같은 카운트 표로.
- 입력의 현재 맵 one-hot 은 '아는 칸' 으로 가린다. 디코더에서 아직 안 푼 칸 값 (0) 이 새면 인코더와 달라진다.
- 맵 0 은 M_pre (첫 쌍의 이전 프레임용). 홀수 프레임 i 의 맵은 스트림의 i+1 번.

**렌더러·pose**
- pose 는 렌더러 가중치 변화에 극도로 민감하다: 렌더러를 조금만 바꿔도 기존 pose 로 d 7.6e-7 → 0.159. 렌더러를 바꾸면 pose 를 두 번쯤 다시 맞춘다 (약 6시간).
  반대로 같은 가중치에서 합성곱 구현 차이 (1e-4 레벨) 에는 흔들리지 않는다.
- pose 값은 `c_int × float64(c_step)` 를 float32 로. float32 끼리 곱하면 마지막 비트가 다를 수 있다 (영향은 작지만 golden 이 다르게 나온다).
- 렌더러 다듬기에서 self-compression 격자 (채널별 비트·스케일) 를 움직이면 크게 나빠졌다 → `--sc-lr 0`.
- pose 학습은 fp32 로 (bf16 은 pose 31 근처 해상도가 0.125 라 미세 조정이 안 된다).
- 짝수 프레임은 이전 쌍의 **렌더된** 홀수 프레임에서 만든다 (원본이 아니라). 첫 쌍은 렌더러(M_pre).

**환경**
- 렌더러 학습 bs 8 하나에 메모리 약 8GB. 두 개를 같이 돌리면 OOM (16GB 머신).
- 코어 수보다 스레드를 많이 띄우면 4~7배 느려진다. 학습 여러 개를 같이 돌릴 때 `--threads` 합을 코어 수 이하로.
- 서브픽셀 확장 표 (`_table`) 는 float32 행렬곱 + 정렬이라, 같은 합이 둘 이상인 칸에서 고르는 2x2 값이 구현마다 다를 수 있다
  (축소 결과는 같아서 점수는 같지만 `raw.sha256` 은 다를 수 있다).

## 8. 시간·환경

- 로컬 (4코어 AVX-512): inflate 294s (그중 seg 복호화 약 143s) + 평가 235s.
  AVX2 커널만 쓰게 강제하면 (`ATEN_CPU_CAPABILITY=avx2 ONEDNN_MAX_CPU_ISA=AVX2`) seg 복호화 143s → 185s, 점수 그대로.
- 공식 평가: GitHub Actions `ubuntu-latest` (4 vCPU, 16GB), **작업 전체 30분** (apt, LFS, `uv sync`, inflate, 평가 포함).
- 러너의 파이썬 환경은 챌린지 `pyproject.toml` (`uv sync --group cpu`): torch 2.10 CPU, numpy 2.3 (2026-09 클론의 uv.lock 기준).
  우리는 torch 2.14 / numpy 2.4 에서 확인했다. constriction 은 pyproject 에 있지만 그 uv.lock 에는 없었다 → 제출 전에 master 를 다시 확인.
- inflate 가 쓰는 패키지: numpy, torch, constriction, 표준 라이브러리 (lzma, struct). GPU 불필요.

## 9. 제출 체크리스트

**LLM 정책** (challenge/README.md 'coding agents and LLMs policy', rust 정책 참고)
- 허용: 코드 일부 작성·다듬기·검토·제안, 문서화, 정리, 질문 답변, 내부용 분석.
- 금지: **코드 전부를 에이전트가 쓰는 것**, **PR 설명과 공개 코멘트를 에이전트가 쓰는 것**. 위반하면 PR 이 닫히고 반복되면 금지.
- 이 저장소 코드는 에이전트가 썼다. 그대로 내면 안 된다. 최소한 디코더 (inflate 경로) 는 직접 쓰고, 이해한 상태로 낸다.
  PR 설명도 직접 쓴다 (이 문서를 붙여 넣지 말 것).

**형식** (challenge README 'submission format and rules', `.github/pull_request_template.md`)
- [ ] `submissions/<이름>/inflate.sh` — `inflate.sh <archive_dir> <output_dir> <video_names_file>`, archive_dir 에는 압축 푼 `p` 가 있다.
- [ ] 이름이 master 의 `submissions/` 에 없는 것 (평가 워크플로가 확인한다).
- [ ] `archive.zip` 은 저장소에 넣지 않는다. PR 에 업로드 (드래그 앤 드롭) 하고 `curl -L` 로 받아지는지 확인.
- [ ] 로컬 공식 평가: `cd comma_vcc/challenge && bash evaluate.sh --submission-dir ./submissions/<이름> --device cpu` → `report.txt` 를 PR 에 붙인다.
- [ ] PR 템플릿 항목: 제출물 이름, archive.zip 링크, report.txt, GPU 필요 여부 (no), 압축 스크립트 포함 여부, 경쟁력/새로움 설명, 기타.
- [ ] 압축 스크립트를 넣는다면 (선택) 그것도 직접 쓴 코드로.

**시간** (가장 큰 위험)
- [ ] 러너 속도 확인: 챌린지 저장소를 포크해서 Actions 의 `eval` 워크플로를 `workflow_dispatch` 로 돌린다
      (입력: submission_name, submission_url = archive.zip 링크, runner = ubuntu-latest). 30분 안에 끝나는지, 점수가 같은지.
      포크에서는 PR 대신 master 를 체크아웃하므로 제출물 디렉터리를 포크의 master 에 넣어 둬야 한다 (LFS 대역폭 제한에 걸리면 실패할 수 있다).
- [ ] 느리면: seg 복호화 (문맥 CNN) 가 대부분이다. 줄일 곳은 스레드 수 확인 (`INFLATE_THREADS`), 레벨 2 패스의 배치, 기록 단계.

## 10. 다음에 해볼 것

점수 0.1196 = seg 0.0188 + pose 0.0026 + rate 0.0983 (그중 seg 스트림 0.0706, 렌더러 0.0109, 문맥 CNN 0.0095, pose 0.0071).

- **seg 스트림 (가장 큼)**: 1 KB = 점수 0.00067. 시도해서 안 된 것: 양방향 (B 프레임) 순서, B 패스 분할, A1 재분할, 7층, c32, 움직임 보상 문맥 (README).
  남은 방향: 문맥에 이전 맵을 블록 비율이 아니라 원해상도 근방 패턴으로 넣기, 레벨별로 다른 모델 크기, 맵 자체를 SegNet 이 같은 출력을 내는 범위에서 단순화 (손실 seg 맵, 경계 RDO).
- **렌더러 (seg 항 0.0188)**: CPU 다듬기는 수렴 (0.000188). 남은 오류의 48% 는 SegNet margin 이 -0.1 보다 큰 경계 픽셀. GPU 로 긴 사이클 (colab) 이나 더 큰 배치·다른 손실.
- **pose**: 섹션 10.7 KB + 항 0.0026 — 작아서 우선순위 낮음. 기저 4비트, 계수 7비트는 손해였다.
- 리더보드 (2026-10-18): 1위 joint_hops 0.136, 2위 semantic_blocks 0.148. 로컬 0.1196 은 공식 러너 확인 전 숫자.
