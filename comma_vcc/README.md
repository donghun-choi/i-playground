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

- 평가 제한: GitHub Actions 작업 전체 30분 (환경 준비 + inflate + 평가, CPU 4 vCPU/16GB 또는 T4 GPU).
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

영상을 복원하지 않는다. 평가 네트워크 두 개가 원본과 같은 출력을 내는 프레임을 만든다. (현재 v6.10, 점수 0.1390)

```
archive (171KB) = 정수 문맥 CNN (13KB, 채널별 학습 비트) + seg 맵 601장 무손실 스트림 (120KB) + 렌더러 (20KB, 폭 24·32·40, 채널별 학습 비트, 218/251 채널) + pose (17KB, 기저 8개 5비트, 계수 8비트 Rice 부호)
               (문맥 모델·렌더러 정수 가중치는 채널별 이산 라플라스 range coder, wcodec.py)

seg 맵 M_i      = 원본 홀수 프레임의 SegNet argmax (+ 맨 앞에 원본 짝수 프레임 0 의 맵 1장)
홀수 프레임 i   = 렌더러(M_i)                                         → SegNet 이 M_i 를 내도록 학습
짝수 프레임 i   = 아핀_i(홀수 프레임 i-1) + bicubic(Σ_k c[i,k] · B_k)    → PoseNet 이 원본 pose 를 내도록 피팅
                  (아핀 6개 + 계수 8개는 쌍마다, 기저 B 8x3x24x32 는 공유)
1164x874 기록    = 512x384 float 이미지를 2x2 서브픽셀 정수로 펼침 (평가의 bilinear 축소가 그대로 되돌림)
```

### 파일
| 경로 | 내용 |
| --- | --- |
| `submissions/semantic_cpu/segcodec.py` | 계층 순서 + 정수 문맥 CNN + range coder (인코드/디코드) |
| `submissions/semantic_cpu/model.py` | 렌더러, 아핀/carrier, 가중치 저장 형식, 서브픽셀 확장 |
| `submissions/semantic_cpu/archive.py`, `inflate.py`, `inflate.sh` | archive 포맷과 복원 |
| `lab/build_cache.py` | 원본을 한 번 디코드해서 네트워크 입출력 캐시 |
| `lab/ctxmodel.py` | 문맥 CNN 학습 (float) |
| `lab/renderer.py` | 렌더러 학습 |
| `lab/pose_fit.py`, `lab/pose_refine.py` | pose 피팅 (기저+계수), 쌍별 다듬기 |
| `lab/build_archive.py` | 정수화 + 인코드 + archive.zip 조립 |
| 나머지 `lab/*_exp.py`, `analyze.py`, `seg_errors.py`, `bits_breakdown.py` | 실험/분석 |

### v6 재현 (CPU 4코어 + Colab T4, 대략 35시간)
```bash
cd comma_vcc/lab
../.venv/bin/python build_cache.py                                                   # 3분
../.venv/bin/python ctxmodel.py --ch 24 --layers 5 --dils 1,2,4,2,1 --steps 20000 --out ../cache/ctxnet_c24d.pt
../.venv/bin/python ctxmodel.py --ch 24 --layers 5 --dils 1,2,4,2,1 --steps 16000 --lr 1e-3 --init ../cache/ctxnet_c24d.pt --out ../cache/ctxnet_c24d2.pt
../.venv/bin/python renderer.py --epochs 5  --out ../cache/renderer.pt --round          # v1 은 처음에 정수 반올림으로 학습했다
../.venv/bin/python renderer.py --epochs 10 --lr 1.5e-3 --resume ../cache/renderer.pt --out ../cache/renderer_v1.pt --round
../.venv/bin/python renderer.py --epochs 8 --lr 4e-4 --cosine --fp32 --bits 6 --resume ../cache/renderer_v1.pt --out ../cache/renderer_v1ft.pt
../.venv/bin/python pose_fit.py --renderer ../cache/renderer_v1.pt --rbits 6 --out ../cache/pose2_v1r.bin
../.venv/bin/python pose_refine.py --pose2 ../cache/pose2_v1r.bin --renderer ../cache/renderer_v1ft.pt --cbits 10 --out ../cache/pose2_v1ft.bin
../.venv/bin/python pose_refine.py --pose2 ../cache/pose2_v1ft.bin --renderer ../cache/renderer_v1ft.pt --cbits 10 --epochs 40 --lr 0.002 --q-epochs 10 --greedy-rounds 1 --out ../cache/pose2_v1ft2.bin
# (여기까지가 v3) v4: 렌더러를 넓혀서 이어 학습하고 pose 를 다시 맞춘다
../.venv/bin/python renderer.py --widths 24,32,40 --widen-from ../cache/renderer_v1ft.pt --epochs 10 --lr 1e-3 --cosine --bits 6 --out ../cache/renderer_w243240.pt
../.venv/bin/python renderer.py --widths 24,32,40 --resume ../cache/renderer_w243240.pt --epochs 6 --lr 4e-4 --cosine --fp32 --bits 6 --out ../cache/renderer_w243240ft.pt
../.venv/bin/python pose_refine.py --pose2 ../cache/pose2_v1ft2.bin --renderer ../cache/renderer_w243240ft.pt --renderer-cfg 24,32,40 --cbits 10 --epochs 100 --lr 0.02 --q-epochs 10 --greedy-rounds 1 --out ../cache/pose2_w24.bin
# v4b: 회전 차원을 위해 공유 기저 B 도 차원 가중 손실로 다시 학습한 뒤, B 고정 일반 MSE 로 전진 차원을 회복
../.venv/bin/python pose_refine.py --pose2 ../cache/pose2_w24.bin --renderer ../cache/renderer_w243240ft.pt --renderer-cfg 24,32,40 --cbits 10 --epochs 40 --lr 0.002 --train-b 0.02 --dimw 0.5 --plain-epochs 20 --q-epochs 10 --greedy-rounds 1 --out ../cache/pose2_w24b.bin
# v5a: 완전 정규화 손실(1/분산) + 높은 학습률 + B 학습 150에폭 → B 6비트 고정 일반 MSE 30에폭 (78분, 10에폭마다 체크포인트 --resume)
../.venv/bin/python pose_refine.py --pose2 ../cache/pose2_w24b.bin --renderer ../cache/renderer_w243240ft.pt --renderer-cfg 24,32,40 --cbits 10 --epochs 150 --lr 0.01 --train-b 0.02 --dimw 1.0 --plain-epochs 30 --q-epochs 10 --greedy-rounds 1 --out ../cache/pose2_w24c.bin
../.venv/bin/python pose_refine.py --pose2 ../cache/pose2_w24c.bin --renderer ../cache/renderer_w243240ft.pt --renderer-cfg 24,32,40 --cbits 10 --epochs 40 --lr 0.002 --q-epochs 10 --greedy-rounds 1 --out ../cache/pose2_w24d.bin
# (여기까지가 v5a) v5b: 렌더러를 'argmax 가 뒤집힐 확률' 손실 + 양자화 인지 학습(QAT) 으로 다듬고 pose 를 다시 맞춘다
../.venv/bin/python renderer.py --widths 24,32,40 --resume ../cache/renderer_w243240ft.pt --epochs 6 --lr 4e-4 --cosine --fp32 --bits 6 --qat --loss flip --full-eval --out ../cache/renderer_w24flip.pt
../.venv/bin/python renderer.py --widths 24,32,40 --resume ../cache/renderer_w24flip.pt --epochs 6 --lr 4e-4 --cosine --fp32 --bits 5 --qat --loss flip --full-eval --out ../cache/renderer_w24flip5.pt
../.venv/bin/python pose_refine.py --pose2 ../cache/pose2_w24d.bin --renderer ../cache/renderer_w24flip5.pt --renderer-cfg 24,32,40 --rbits 5 --cbits 10 --epochs 60 --lr 0.005 --train-b 0.01 --dimw 1.0 --plain-epochs 30 --q-epochs 10 --greedy-rounds 1 --out ../cache/pose2_v5b.bin
../.venv/bin/python pose_refine.py --pose2 ../cache/pose2_v5b.bin --renderer ../cache/renderer_w24flip5.pt --renderer-cfg 24,32,40 --rbits 5 --cbits 10 --epochs 30 --lr 0.001 --q-epochs 10 --greedy-rounds 1 --out ../cache/pose2_v5b2.bin
# (여기까지가 v5b) v5c: 4비트 QAT 로 flip 학습을 더 하고 (10 + 12에폭), pose 를 v5a 처럼 길게 다시 맞춘다
../.venv/bin/python renderer.py --widths 24,32,40 --resume ../cache/renderer_w24flip5.pt --epochs 10 --lr 4e-4 --cosine --fp32 --bits 4 --qat --loss flip --full-eval --out ../cache/renderer_w24flip4.pt
../.venv/bin/python renderer.py --widths 24,32,40 --resume ../cache/renderer_w24flip4.pt --epochs 12 --lr 3e-4 --cosine --fp32 --bits 4 --qat --loss flip --full-eval --out ../cache/renderer_w24flip4b.pt
../.venv/bin/python pose_refine.py --pose2 ../cache/pose2_v5b2.bin --renderer ../cache/renderer_w24flip4b.pt --renderer-cfg 24,32,40 --rbits 4 --cbits 10 --epochs 150 --lr 0.01 --train-b 0.02 --dimw 1.0 --plain-epochs 30 --q-epochs 10 --greedy-rounds 1 --out ../cache/pose2_v5c.bin
# (여기까지가 v5c) v5d: 렌더러 10에폭 더 (FiLM 없이), pose 200에폭 (v5c 다듬기 결과에서 이어서)
../.venv/bin/python pose_refine.py --pose2 ../cache/pose2_v5c.bin --renderer ../cache/renderer_w24flip4b.pt --renderer-cfg 24,32,40 --rbits 4 --cbits 10 --epochs 40 --lr 0.001 --q-epochs 10 --greedy-rounds 1 --out ../cache/pose2_v5c2.bin
../.venv/bin/python renderer.py --widths 24,32,40 --resume ../cache/renderer_w24flip4b.pt --epochs 10 --lr 3e-4 --cosine --fp32 --bits 4 --qat --loss flip --full-eval --out ../cache/renderer_w24flip4c.pt
../.venv/bin/python pose_refine.py --pose2 ../cache/pose2_v5c2.bin --renderer ../cache/renderer_w24flip4c.pt --renderer-cfg 24,32,40 --rbits 4 --cbits 10 --epochs 200 --lr 0.01 --train-b 0.02 --dimw 1.0 --plain-epochs 30 --q-epochs 10 --greedy-rounds 1 --out ../cache/pose2_v5d.bin
../.venv/bin/python build_archive.py --ctx ../cache/ctxnet_c24d2.pt --cbits 6 --renderer ../cache/renderer_w24flip4c.pt --renderer-cfg 24,32,40 --rbits 4 --pose2 ../cache/pose2_v5d.bin
# (여기까지가 v5d) v6: 렌더러는 Colab GPU (colab/vcc_gpu.ipynb, 4비트 flip 10에폭 × 6사이클, 사이클마다 600장 평가로 최고를 이어서)
#   → colab-results 브랜치의 comma_vcc/colab/results/renderer_best.pt 를 ../cache/renderer_colab_c6.pt 로
# 문맥 모델: 5비트 QAT → self-compression (채널별 비트/스케일 학습, 크기 손실 ×4)
../.venv/bin/python ctxmodel.py --init ../cache/ctxnet_c24d2.pt --ch 24 --layers 5 --dils 1,2,4,2,1 --qat-bits 5 --steps 6000 --lr 1e-3 --threads 4 --out ../cache/ctxnet_c24q5.pt
../.venv/bin/python ctxmodel.py --init ../cache/ctxnet_c24q5.pt --ch 24 --layers 5 --dils 1,2,4,2,1 --self-compress 4.0 --sc-init-bits 5 --steps 6000 --lr 5e-4 --threads 4 --out ../cache/ctxnet_c24sc4.pt
# pose: 계수 격자 9비트 (10비트와 같은 pose 항에 -1.35KB)
../.venv/bin/python pose_refine.py --pose2 ../cache/pose2_v5d2.bin --renderer ../cache/renderer_colab_c6.pt --renderer-cfg 24,32,40 --rbits 4 --cbits 9 --epochs 200 --lr 0.01 --train-b 0.02 --dimw 1.0 --plain-epochs 30 --q-epochs 10 --greedy-rounds 1 --out ../cache/pose2_v6.bin
# v6.1: 기저 B 를 5비트 격자로 가짜 양자화하며 다시 학습 (--bbits 5 --b-qat) → pose 항도 좋아지고 -3.6KB
../.venv/bin/python pose_refine.py --pose2 ../cache/pose2_v6.bin --renderer ../cache/renderer_colab_c6.pt --renderer-cfg 24,32,40 --rbits 4 --cbits 9 --epochs 150 --lr 0.005 --train-b 0.01 --bbits 5 --b-qat --dimw 1.0 --plain-epochs 30 --q-epochs 10 --greedy-rounds 1 --out ../cache/pose2_v6_b5q.bin
# v6.2: 렌더러 Colab 사이클 8 (renderer_best.pt → ../cache/renderer_colab_c8.pt) + pose 를 평가 경로(--qf)로 재피팅
#   렌더러가 바뀌면 출력 색이 흘러가서 (c6 vs c8 평균 6.6레벨) pose 를 처음부터 다시 맞춰야 한다
../.venv/bin/python pose_refine.py --pose2 ../cache/pose2_v6_qf.bin --renderer ../cache/renderer_colab_c8.pt --renderer-cfg 24,32,40 --rbits 4 --cbits 9 --epochs 150 --lr 0.01 --train-b 0.01 --bbits 5 --b-qat --dimw 1.0 --plain-epochs 30 --q-epochs 10 --greedy-rounds 2 --qf --out ../cache/pose2_c8.bin
#   (pose2_v6_qf.bin = pose2_v6_b5q.bin 에서 --epochs 0 --q-epochs 0 --keep-grid --qf --greedy-rounds 3, 렌더러 c6 기준 pose 항 0.0050 → 0.0025)
# v6.3: pose 바이트 줄이기 (평가 경로 탐욕 탐색이 격자 손해를 메운다) - 계수 격자 8비트, 기저 12 → 8개
../.venv/bin/python pose_refine.py --pose2 ../cache/pose2_c8.bin --renderer ../cache/renderer_colab_c8.pt --renderer-cfg 24,32,40 --rbits 4 --cbits 8 --bbits 5 --epochs 0 --lr 0.01 --q-epochs 10 --greedy-rounds 3 --qf --out ../cache/pose2_c8_cb8.bin
../.venv/bin/python pose_refine.py --pose2 ../cache/pose2_c8_cb8.bin --renderer ../cache/renderer_colab_c8.pt --renderer-cfg 24,32,40 --rbits 4 --cbits 8 --keep-k 8 --epochs 150 --lr 0.01 --train-b 0.01 --bbits 5 --b-qat --dimw 1.0 --plain-epochs 30 --q-epochs 10 --greedy-rounds 3 --qf --out ../cache/pose2_c8_k8.bin
# v6.4: 정수 가중치 엔트로피 부호 (--wec)
../.venv/bin/python build_archive.py --ctx ../cache/ctxnet_c24sc4.pt --cbits 5 --renderer ../cache/renderer_colab_c8.pt --renderer-cfg 24,32,40 --rbits 4 --pose2 ../cache/pose2_c8_k8.bin --wec
# v6.5: 렌더러 self-compression 사이클 (CPU 약 1시간, flip 10에폭 + 채널별 비트 학습) → pose 재피팅 (렌더러가 바뀌었으니)
../.venv/bin/python renderer.py --widths 24,32,40 --resume ../cache/renderer_colab_c8.pt --epochs 10 --lr 3e-4 --bs 4 --cosine --fp32 --bits 4 --self-compress 1.0 --loss flip --full-eval --out ../cache/renderer_c8_sc.pt
../.venv/bin/python pose_refine.py --pose2 ../cache/pose2_c8_k8.bin --renderer ../cache/renderer_c8_sc.pt --renderer-cfg 24,32,40 --rbits 0 --cbits 8 --epochs 150 --lr 0.01 --train-b 0.01 --bbits 5 --b-qat --dimw 1.0 --plain-epochs 30 --q-epochs 10 --greedy-rounds 3 --qf --out ../cache/pose2_sc_k8.bin
../.venv/bin/python build_archive.py --ctx ../cache/ctxnet_c24sc4.pt --cbits 5 --renderer ../cache/renderer_c8_sc.pt --renderer-cfg 24,32,40 --rbits 0 --pose2 ../cache/pose2_sc_k8.bin --wec
# v6.6: 렌더러·문맥 모델 self-compression 을 엔트로피 부호 크기 손실 (--sc-rate laplace) 로 한 사이클 더
../.venv/bin/python renderer.py --widths 24,32,40 --resume ../cache/renderer_c8_sc.pt --epochs 10 --lr 3e-4 --bs 4 --cosine --fp32 --bits 4 --self-compress 1.0 --sc-rate laplace --loss flip --full-eval --out ../cache/renderer_c8_sc2.pt
../.venv/bin/python ctxmodel.py --init ../cache/ctxnet_c24sc4.pt --ch 24 --layers 5 --dils 1,2,4,2,1 --self-compress 1.0 --sc-rate laplace --sc-init-bits 5 --steps 4000 --lr 5e-4 --out ../cache/ctxnet_c24scl.pt
../.venv/bin/python build_archive.py --ctx ../cache/ctxnet_c24scl.pt --cbits 5 --pose2 ../cache/pose2_sc_k8.bin --segs-only --wec   # → segs_pre.bin (이 문맥 모델과 짝)
../.venv/bin/python pose_refine.py --pose2 ../cache/pose2_sc_k8.bin --renderer ../cache/renderer_c8_sc2.pt --renderer-cfg 24,32,40 --rbits 0 --cbits 8 --epochs 150 --lr 0.01 --train-b 0.01 --bbits 5 --b-qat --dimw 1.0 --plain-epochs 30 --q-epochs 10 --greedy-rounds 3 --qf --out ../cache/pose2_sc2_k8.bin
../.venv/bin/python build_archive.py --ctx ../cache/ctxnet_c24scl.pt --cbits 5 --renderer ../cache/renderer_c8_sc2.pt --renderer-cfg 24,32,40 --rbits 0 --pose2 ../cache/pose2_sc2_k8.bin --segs ../cache/segs_pre_scl.bin --wec
# v6.7: 같은 렌더러 사이클 한 번 더 (renderer_c8_sc2 → sc3) + pose 재피팅 (pose2_sc2_k8 → sc3_k8, 같은 레시피)
../.venv/bin/python build_archive.py --ctx ../cache/ctxnet_c24scl.pt --cbits 5 --renderer ../cache/renderer_c8_sc3.pt --renderer-cfg 24,32,40 --rbits 0 --pose2 ../cache/pose2_sc3_k8.bin --segs ../cache/segs_pre_scl.bin --wec
# v6.8: 렌더러 사이클 4 (sc3 → sc4) + pose 재피팅 (sc3_k8 → sc4_k8), pose 섹션 pos4 (--wec)
../.venv/bin/python build_archive.py --ctx ../cache/ctxnet_c24scl.pt --cbits 5 --renderer ../cache/renderer_c8_sc4.pt --renderer-cfg 24,32,40 --rbits 0 --pose2 ../cache/pose2_sc4_k8.bin --segs ../cache/segs_pre_scl.bin --wec
# v6.9: 문맥 모델을 낮은 학습률로 더 길게 (20000스텝, 약 40분) → 스트림 재부호화
../.venv/bin/python ctxmodel.py --init ../cache/ctxnet_c24scl.pt --ch 24 --layers 5 --dils 1,2,4,2,1 --self-compress 1.0 --sc-rate laplace --sc-init-bits 5 --steps 20000 --lr 3e-4 --threads 4 --out ../cache/ctxnet_c24scl2.pt
../.venv/bin/python build_archive.py --ctx ../cache/ctxnet_c24scl2.pt --cbits 5 --pose2 ../cache/pose2_sc4_k8.bin --segs-only --wec   # → segs_pre.bin (= segs_pre_scl2.bin)
../.venv/bin/python build_archive.py --ctx ../cache/ctxnet_c24scl2.pt --cbits 5 --renderer ../cache/renderer_c8_sc4.pt --renderer-cfg 24,32,40 --rbits 0 --pose2 ../cache/pose2_sc4_k8.bin --segs ../cache/segs_pre_scl2.bin --wec
# v6.10: 문맥 모델 한 라운드 더 (scl2 → scl3; 컨테이너 재시작으로 12000스텝에서 끊겨 그 체크포인트에서 10000스텝 lr 2e-4 로 이어감)
../.venv/bin/python build_archive.py --ctx ../cache/ctxnet_c24scl3.pt --cbits 5 --renderer ../cache/renderer_c8_sc4.pt --renderer-cfg 24,32,40 --rbits 0 --pose2 ../cache/pose2_sc4_k8.bin --segs ../cache/segs_pre_scl3.bin --wec
cd .. && bash run.sh semantic_cpu
```

### GPU (Colab) 로 길게 학습하기
CPU 로는 렌더러 flip 학습 1에폭이 약 8분이라, 같은 레시피를 Colab GPU 에서 길게 돌린다.
`colab/vcc_gpu.ipynb` 를 Colab 에서 열고 위에서부터 실행하면 된다 (GPU 런타임, 보안 비밀 `GH_TOKEN` = 이 저장소 Contents 쓰기 토큰).

- `colab/setup_colab.sh`: 챌린지 코드 + 평가 네트워크 가중치(해시 확인) + 경량 캐시 (`colab/seed/`: GT seg 맵 xz 512KB, pose, 시작 렌더러/pose). 원본 영상은 필요 없다.
- `colab/run_jobs.py`: 렌더러 사이클(flip 10에폭 + 600장 평가, 가장 좋은 것을 이어서) → 최고 렌더러에 pose 재피팅.
  사이클마다 `colab/results/` 를 `colab-results` 브랜치로 push, 끊기면 다시 실행해서 이어 한다. `--smoke` 는 CPU 에서 흐름만 점검 (앞 8장).
- 실시간 시각화: livevis 를 Colab 포트 프록시로 연다 (렌더러 8020, pose 8014).
- 학습 스크립트는 `--device cuda` 를 받는다 (GPU 에서는 TF32 를 끈다: flip 손실이 margin 근처 정밀도에 민감).
- 결과(renderer_best.pt, pose2_best.bin)는 CPU 쪽에서 `build_archive.py` 로 묶고 공식과 같은 CPU 평가로 확인한다.

### 시간 제한에 대해
우리 머신 (4코어 AVX-512, fp32 행렬곱 632 GFLOPS) 에서 v4 는 inflate 378s + 평가 242s = 10.4분.
공식 러너는 GitHub `ubuntu-latest` (4 vCPU) 이고 30분에는 환경 준비(apt, uv sync, LFS)도 포함된다.
러너가 2~3배 느리면 빠듯할 수 있다 — 실제 제출 전에는 러너에서 시간을 확인해야 한다.

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

### pose 쪽에서 배운 것 (v1 이후)
- pose 6차원 = (전진, 좌우, 상하, roll, pitch, yaw) 로 보인다. 전진만 평균 31 이고 나머지는 표준편차 0.007~0.036.
- v1 carrier (회색 + 학습 기저) 의 잔차는 거의 전부 좌우/상하/yaw 에 있었다 (잔차 RMS ≈ 원래 표준편차 → 전혀 제어 못 함).
  학습 기저가 분산이 1000배 큰 전진 차원만 배운다. 차원별 가중 손실로도 해결 안 됨.
- PoseNet 의 회전/좌우 출력은 '정지' 나 'carrier' 상태에서는 작은 변형에 거의 반응하지 않는다 (Jacobian 특이값 [1.19, 0.02, 0.015, ...]).
  그래서 Jacobian 방향 보정(J^T a) 도 거대한 스텝이 필요해 발산.
- 짝수 프레임을 **이전 쌍의 렌더 프레임**으로 두면 PoseNet 이 자연스러운 운동 상태(전진 ≈ 33)에 들어가고, 거기서는 아핀 변형이 회전 차원도 움직인다.
  64쌍 400 step: 회색 carrier 0.0064 → 홀수+이동 0.00049 → **이전 렌더 + 아핀 + carrier 0.000249**.
- (v5a) v4b 의 차원별 잔차 RMS / 표준편차 = [0.002, 0.70, 0.77, 0.90, 0.86, 0.61] — 전진만 맞고 나머지는 사실상 평균을 내고 있었다.
  24쌍 진단 (`lab/pose_diag.py`, 300 step): 쌍마다 자유 48x64 이미지 8.7e-5, 지금 구조 + B 학습 1.8e-5, 지금 구조 + B 고정은 2.6e-3 (발산).
  → 쌍별 계수만으로는 회전 차원을 못 움직이고 공유 기저가 같이 적응해야 한다.
  600쌍에서 손실을 완전 정규화(1/분산, dimw 1.0)하고 학습률을 5배(0.01) 올려 150에폭 돌리니 1~5차원 RMS 가 1/5~1/10 로 줄었다
  (전진 차원은 가중치가 1e-4 라 0.04 까지 흐트러지지만 B 고정 일반 MSE 단계에서 회복). 결과 posenet 2.6e-4 → **1.6e-5** (항 0.051 → 0.013).
  일반 MSE 단계를 높은 학습률로 시작하면 전진 차원이 한 번 크게 튀므로 마지막 다듬기는 낮은 학습률(0.002)이 낫다.
- (v5b) 렌더러 손실을 바꾼 것이 가장 컸다. CE + margin 2 hinge 는 이미 맞는 픽셀에도 힘을 쓰는데, 앞 절반 softplus(-margin/0.2)·0.2,
  뒤 절반 sigmoid(-margin/τ) (τ 0.15→0.05, '뒤집힐 확률' 의 매끈한 근사) 로 바꾸니 같은 렌더러가 6에폭 만에 불일치 0.00121 → 0.00060.
  그대로 5비트 QAT (STE 가짜 양자화, inflate 와 같은 격자) 로 6에폭 더 → 0.000506 이면서 렌더러 49.8KB → 40.2KB.
  렌더러가 바뀌면 이전 렌더 기반 짝수 프레임도 바뀌어 pose 를 다시 맞춰야 한다 (60에폭으로는 v5a 수준까지 못 감: 1.6e-5 → 4.5e-5).
- (v5c) 비트를 낮춰도 flip 손실로 계속 학습하면 오히려 좋아졌다: 5비트 0.000505 (40KB) → 4비트 10에폭 0.000455 → 12에폭 더 0.000440 (30.5KB).
  렌더러를 넓히는 것(32·40·48)은 4비트 24·32·40 대비 +17~33KB 라 손익분기(불일치 0.00024~0.00035)가 비현실적이라 그만뒀다.
  pose 는 이미 맞춘 기저(v5b)에서 다시 150에폭 → 1~5차원 RMS 가 v5a 의 절반 (0.0009, 0.0023, 0.0014, 0.0007, 0.0034), posenet **6.6e-6** (항 0.0081).
- 프레임별 FiLM (프레임마다 8차원 코드로 병목/디코더 특징 변조, `--film 8`): 같은 조건 10에폭 대조군과 비교하면 600장 불일치
  0.000374 vs 0.000378 로 거의 같은데 렌더러가 +5.1KB → 손해. 개선은 FiLM 이 아니라 추가 학습에서 나왔다.
- 문맥 모델 가중치를 6 → 5비트로 (학습 후 양자화): 모델 16.6KB → 13.4KB 지만 스트림 124.4KB → 141.6KB 로 손해.
- (v6) Colab T4 에서 같은 렌더러 레시피(4비트 flip 10에폭)를 사이클로 반복: 600장 불일치 0.000378 → 0.000348 → 0.000325 → 0.000314 → 0.000305 → 0.000291 → 0.000274
  (사이클당 9.7분, CPU 의 약 9배). GPU 에서 잰 불일치와 CPU 에서 잰 값이 같았다 (0.000274).
- 문맥 모델: 가중치 5비트 QAT (`--qat-bits 5`) → 모델 16.6KB → 13.4KB 인데 스트림도 124.4KB → 123.7KB.
  이어서 self-compression (Cséfalvay 2023: 채널별 비트/스케일 학습 + 크기 손실) ×4 → 13.6KB + 123.1KB = 136.7KB (원래 141.1KB).
  점수 공식 기준(×1) 크기 손실로는 비트가 거의 안 줄었다 — 작은 모델이라 줄일 채널이 별로 없다.
- pose 계수 격자 10 → 9비트: 탐욕 ±1 탐색이 양자화 손해를 메워서 pose 항은 같고 -1.35KB.
  기저 B 를 학습 없이 5비트로 깎으면 pose 항 0.0038 → 0.054 (기저는 함께 학습해야 함).
- (v6.1) pose 기저 B 5비트: 학습 없이 깎으면 pose 항 0.054 로 망가지지만, 기저를 5비트 격자로 가짜 양자화(STE)하며 150에폭 다시 학습하면
  pose 항 0.0057 → 0.0050 으로 오히려 좋아지고 pose 섹션 -3.6KB. 기저 해상도를 12×16 으로 낮추는 것은 회전 차원 제어가 크게 무너져서 실패
  (50에폭에서 1·2·5차원 0.018/0.012/0.009, 24×32 는 0.005/0.0085/0.0083).
- (v6.2) pose 를 평가 경로로 최적화 (`--qf`): 기록 단계(512x384 float → 2x2 서브픽셀 정수 → 평가의 bilinear 축소)가 값을 1/50 레벨 정도 바꾸는데,
  탐욕 ±1 탐색이 float 경로에 과적합해서 학습 경로 d 6.8e-7 vs 실제 2.55e-6 (3.7배). 이 왕복을 원해상도를 만들지 않고 바로 계산하는 `fine_q`
  (위치별 256 조합 가중합 표에서 가장 가까운 값, 평가와 1e-4 이내)를 학습(STE)과 탐욕 탐색에 넣으니 같은 계수 격자에서 2.55e-6 → 6e-7 (pose 항 0.0050 → 0.0025).
  렌더러 c8 로 처음부터 다시 맞춘 것은 1.1e-6 (0.0034) 이지만 seg 항 -0.0024 가 더 커서 c8 을 쓴다.
- (v6.3) pose 가 작아지자 (항 0.0034, d 1.1e-6) 바이트를 줄이는 쪽이 이득: 1바이트 = 점수 6.7e-7 이고, 이 근처에서 d 의 한계 비용은
  √10/(2√d) ≈ 1500 이라 1KB 를 아끼면 d 가 40% 나빠져도 본전. pose 항 + pos3 rate 로 비교:
  계수 격자 9비트 0.01996 → 8비트 0.01906 (d 그대로, -1.36KB) → 7비트 0.01898 (d 1.7e-6, -1.33KB 더; 8비트와 사실상 같아서 8비트).
  기저 12 → 8개 (계수 에너지 × 기저 크기로 고르고 150에폭 다시): 회전 차원은 3배 나빠지지만 (d 2.9e-6, 항 0.0054) pos3 23.6KB → 17.0KB 라 0.0167.
- (v6.4) 정수 가중치를 xz 대신 엔트로피 부호로 (`wcodec.py`): 4비트 렌더러 가중치의 0차 엔트로피는 2.82 비트인데 xz 는 약 3 비트.
  출력 채널마다 이산 라플라스 P(q) ∝ r^|q| (r = 1..255/256, 1바이트) 를 골라 range coder 로 (빈도는 정수 연산만 → 기계와 무관).
  렌더러 31,178 → 29,607 B, 문맥 모델 13,617 → 13,203 B. 텐서별 경험 분포 + 표 (29.0KB) 나 가우시안 (28.7KB + α) 보다 채널별 라플라스가 낫다.
- (v6.5) 렌더러 self-compression (채널별 비트 b·스케일 e 학습, 크기 손실 = 가중치당 평균 비트 × 점수 공식): 2에폭 만에 31 채널이 0비트(제거),
  평균 2.71 비트. 600장 불일치 0.000250 → 0.000240 (같은 flip 사이클을 4비트 고정으로 돌린 것보다 오히려 낫다), 렌더러 29.6KB → 24.8KB (엔트로피 부호 기준).
  다만 크기 손실의 '비트 수' 는 정수 범위라서 실제 부호 크기와 잘 안 맞는다 (첫 에폭 3.37 비트인데 xz 는 -2.4% 뿐) → 다음엔 엔트로피 추정 손실 (`--sc-rate laplace`).
  렌더러가 바뀌면 pose 는 다시 맞춰야 하는데, 이번엔 기저 8개 재피팅이 오히려 더 잘 맞았다 (pose 항 0.0054 → 0.0038, 이어서 학습한 기저 덕).
- (v6.6) self-compression 크기 손실을 실제 부호 크기 추정으로 (`--sc-rate laplace`): 채널마다 m = mean|q| 에서 이산 라플라스 ρ = m/(√(1+m²)+1),
  H = -log2((1-ρ)/(1+ρ)) - m·log2 ρ (실제 wcodec 부호와 0.5% 이내). 가중치 1비트 = 점수 공식의 1비트로 그대로.
  렌더러 한 사이클: 24.8KB → 21.3KB, 600장 불일치 0.000241 → 0.000231 (같이 좋아진다). 문맥 모델은 거의 안 줄어서 (+스트림) -0.5KB.
  (처음엔 ρ = (√(1+m²)-1)/m 로 써서 잘린 채널(m→0)에서 float32 상쇄로 inf 가 났다.)
- 문맥 모델 c24 → c32 (함수 보존 확장 16000스텝 + self-compression): 스트림 122,744 → 121,572 B (-1.2KB) 지만 모델 13,048 → 15,454 B (+2.4KB),
  합 +1.2KB 손해이고 디코드도 1.8배 느려져서 c24 유지. 렌더러 self-compression 사이클 4 는 seg -0.0002, -129 B 로 수렴 (pose 재피팅 보류).
- pose 기저를 QAT 로 학습할 때 에폭별 평가가 float 기저를 써서 0번 차원이 엉뚱하게 크게 보였다 (실제 학습은 정상). 저장 격자 기저로 재도록 고침.
- 효과 없던 것: 패스별 확률 온도 보정 (-0.04%, 문맥 모델이 이미 잘 보정됨), 순수 엔트로피 부호 (xz 가 이미 0차 엔트로피 근처, 전체 -1.4KB).
- pose 계수는 시간 상관이 없다 (차분 분산이 값 분산의 2배) → 차분 + xz 대신 (값 - 평균) 을 Rice 부호로: 14.7KB → 12.4KB (`pos3` 섹션).

### v2 에서 바꾼 것과 근거
| 항목 | v1 | v2 | 근거 |
| --- | --- | --- | --- |
| 문맥 모델 | 24ch 5층, 8비트 | 24ch 5층 dilation(1,2,4,2,1), **6비트** | 227 → 200 B/frame (A2 패스 108 → 67). 6비트는 스트림 +1.5%, 모델 -27% |
| seg 디코드 | numpy int64 | torch float32 입력 + float64 정확 재양자화 | 4코어 490s → 147s, 비트 단위 동일 |
| 렌더러 | v1, 8비트 (41.5KB) | v1, **6비트 (30.3KB)** | 6비트에서 seg 손실 없음. 전해상도 FiLM 렌더러(v2, PR #130 구조 참고)는 CPU 20에폭으로 0.0017 에 정체 → 채택 안 함 |
| pose | 회색 + 학습 기저 | **이전 쌍 렌더 + 쌍별 아핀 + 학습 기저** (기저 6비트) | 회전 차원 제어 |
| 첫 쌍 | - | GT 짝수 프레임 0 의 seg 맵 1장을 seg 스트림 앞에 추가 | 첫 쌍은 '이전 프레임' 이 없다 |

시도했지만 버린 것: 경계 사전보정 (불일치 -12% 에 프레임당 284픽셀 변경), inflate 시점 SegNet 그래디언트 보정 (3스텝 -27%, 600장에 스텝당 2.6분이라 시간 제한 위험), 움직임 보상 문맥 (전역 확대 와핑으로 이전 맵 예측 불일치 1.23% → 1.20% 뿐), 손실 seg 맵 단순화 (10bit 넘는 픽셀이 프레임당 2.7개, 이득 ~0.5KB),
노면 평면 움직임 보상 (바뀐 픽셀 22% 맞히고 15% 새로 틀림), 기저 24개 (-11% 에 저장 2배).

pose 기저를 다시 학습할 때: B 만 학습시키면 (일반 MSE) 회전 차원이 그대로였다 (0.0317 → 0.0308).
회전 차원 그래디언트가 약해서다. 차원별 가중 손실 (1/분산)^0.5 을 넣자 10에폭 만에 0.0317 → 0.0282 로 움직였다.

렌더러 오류의 99% 는 경계 1픽셀, 44% 는 '차선 → 도로' (점선 끝, 멀리 있는 1~2픽셀 점선).
렌더러는 야간 원본과 달리 낮처럼 밝은 그림을 그린다 (SegNet 이 더 확신하는 쪽으로 학습됨).

## 주의: 챌린지의 LLM 사용 정책

챌린지는 코딩 에이전트가 **제출 코드 전부를 쓰는 것**과 **PR 설명/공개 코멘트를 쓰는 것**을 금지한다.
도구·분석·리뷰·일부 코드 작성은 허용. 실제로 제출할 거라면 핵심 압축 코드는 직접 쓰고 이해해야 한다.

## 기록

| 제출물 | segnet | posenet | rate | score | 메모 |
| --- | --- | --- | --- | --- | --- |
| baseline_fast (재현) | 0.00947 | 0.380 | 0.0598 | 4.39 | x265 ultrafast crf30, 45% 축소, 평가 227s (CPU 4코어) |
| semantic_cpu v1 | 0.00153 | 0.00067 | 0.00650 | **0.397** | seg맵 142KB + 문맥모델 23KB + 렌더러 42KB + carrier 36KB, inflate 740s + 평가 251s (4코어) |
| semantic_cpu v2 | 0.00153 | 0.00030 | 0.00561 | **0.3485** | seg맵 129KB(dilation 6비트 문맥모델 17KB) + 렌더러 6비트 30KB + pose(이전 렌더+아핀+carrier) 35KB, inflate 337s + 평가 224s |
| semantic_cpu v3 | 0.00144 | 0.00037 | 0.00543 | **0.3403** | 렌더러 fp32 미세조정, 문맥모델 추가 학습(193 B/frame), pose 재다듬기(계수 10비트). inflate 334s + 평가 231s |
| semantic_cpu v4 | 0.00121 | 0.00036 | 0.00595 | **0.3298** | 렌더러 폭 16·24·32 → 24·32·40 (함수 보존 확장 + bf16 10에폭 + fp32 6에폭, 6비트 49.8KB), pose 재다듬기. inflate 378s + 평가 242s |
| semantic_cpu v4b | 0.00121 | 0.00026 | 0.00595 | **0.3207** | v4 + pose 기저 B 재학습 (차원 가중 0.5 + B 학습 40에폭 → B 고정 일반 MSE 20에폭). inflate 311s + 평가 197s |
| semantic_cpu v5a | 0.00121 | 0.0000162 | 0.00590 | **0.2810** | pose: 완전 정규화 손실 + B 학습 150에폭 → 일반 MSE 다듬기 (회전/좌우 차원을 처음으로 맞춤), pose 계수 Rice 부호 (-2.2KB). inflate 392s + 평가 238s |
| semantic_cpu v5b | 0.000506 | 0.0000453 | 0.00564 | **0.2128** | 렌더러 flip 손실(뒤집힐 확률) + 5비트 QAT (seg 항 0.121 → 0.051, 렌더러 -9.6KB), pose 재피팅. inflate 371s + 평가 226s |
| semantic_cpu v5c | 0.000440 | 0.0000066 | 0.00538 | **0.1865** | 렌더러 4비트 QAT + flip 22에폭 더 (30.5KB), pose 150에폭 다시 (기저 이어서). inflate 367s + 평가 231s |
| semantic_cpu v5d | 0.000378 | 0.0000035 | 0.00538 | **0.1780** | 렌더러 4비트 flip 10에폭 더, pose 200에폭 (기저를 계속 이어 학습할수록 회전 차원이 더 맞음). inflate 369s + 평가 228s |
| semantic_cpu v5d2 | 0.000378 | 0.0000032 | 0.00538 | **0.1778** | v5d + pose 낮은 학습률 다듬기 40에폭 (`--epochs 40 --lr 0.001`, B 고정) |
| semantic_cpu v6 | 0.000274 | 0.0000032 | 0.00523 | **0.1638** | 렌더러 Colab GPU 6사이클 (seg 0.0378 → 0.0274), 문맥 모델 5비트 QAT + self-compression (-4.3KB), pose 계수 9비트 (-1.35KB). inflate 402s + 평가 248s |
| semantic_cpu v6.1 | 0.000274 | 0.0000026 | 0.00514 | **0.1608** | v6 + pose 기저 5비트 QAT 재학습 (pose 항 0.0057 → 0.0050, -3.6KB). inflate 430s |
| semantic_cpu v6.2 | 0.000250 | 0.0000011 | 0.00514 | **0.1569** | 렌더러 Colab 사이클 8 (seg 0.0274 → 0.0250) + pose 를 평가 경로(서브픽셀 왕복, `--qf`)로 재피팅. inflate 419s |
| semantic_cpu v6.3 | 0.000250 | 0.0000029 | 0.00493 | **0.1536** | pose 계수 8비트 + 기저 12 → 8개 (pos3 25.0KB → 17.0KB, pose 항 0.0034 → 0.0054). inflate 403s |
| semantic_cpu v6.4 | 0.000250 | 0.0000029 | 0.00488 | **0.1523** | 문맥 모델·렌더러 정수 가중치 엔트로피 부호 (wcodec, -2.0KB, 무손실) |
| semantic_cpu v6.5 | 0.000241 | 0.0000015 | 0.00475 | **0.1466** | 렌더러 self-compression 사이클 (CPU, 29.6KB → 24.8KB, seg 0.0250 → 0.0241) + pose 재피팅 (항 0.0054 → 0.0038). inflate 400s |
| semantic_cpu v6.6 | 0.000231 | 0.0000013 | 0.00464 | **0.1427** | self-compression 크기 손실을 엔트로피 부호 추정으로: 렌더러 21.3KB (seg 0.0231), 문맥 모델+스트림 -0.5KB, pose 재피팅. inflate 444s |
| semantic_cpu v6.7 | 0.000222 | 0.0000012 | 0.00462 | **0.1412** | 렌더러 self-compression 사이클 3 (20.6KB, seg 0.0222) + pose 재피팅 (항 0.0034). CPU 로 사이클 + 재피팅 약 4시간에 -0.0015. inflate 429s |
| semantic_cpu v6.7b | 0.000222 | 0.0000012 | 0.00462 | **0.1410** | pose 섹션 pos4: 기저 B 를 (기저, 채널) 행별 라플라스 range coder 로 (17,008 → 16,734 B, 무손실) |
| semantic_cpu v6.8 | 0.000220 | 0.0000010 | 0.00461 | **0.1405** | 렌더러 self-compression 사이클 4 (seg 0.0220, 20.5KB) + pose 재피팅 (항 0.0032, 지금까지 최저). inflate 391s |
| semantic_cpu v6.9 | 0.000220 | 0.0000010 | 0.00458 | **0.1397** | 문맥 모델 20000스텝 더 (lr 3e-4): 모델+스트림 135,792 → 134,539 B. inflate 396s |
| semantic_cpu v6.10 | 0.000220 | 0.0000010 | 0.00455 | **0.1390** | 문맥 모델 한 라운드 더: 모델+스트림 134,539 → 133,478 B (스트림 200 B/frame). inflate 406s |
