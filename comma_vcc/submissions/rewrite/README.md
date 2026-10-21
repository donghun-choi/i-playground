# rewrite — 디코더 다시 짜기 뼈대

함수 이름·입출력 모양과 명세만 있고 본문은 전부 `NotImplementedError` 다. 위에서부터 채우면서 단계마다 검사한다.
명세는 `comma_vcc/GUIDE.md` 3·4절, 참조 구현은 `submissions/semantic_cpu/` (보고 베끼지 말고, 막히면 확인용으로).

| 순서 | 파일 · 함수 | GUIDE | 검사 단계 |
| --- | --- | --- | --- |
| 1 | `container.read_sections` | 3.1 | `container` |
| 2 | `entropy.unxz`, `entropy.laplace_probs`, `entropy.Decoder` | 3.2, 3.6 | (아래 단계들로) |
| 3 | `ctxnet.unpack_ctxe`, `ctxnet.parse_qnet` | 3.2 | `ctxe`, `parse_qnet` |
| 4 | `ctxnet.IntCNN` (`run`, `self_check`) | 4.2 | `intcnn` |
| 5 | `segdec.masks`, `segdec.build_input` | 4.1 | `masks`, `inputs` |
| 6 | `ctxnet.probs` | 4.2 | `probs` |
| 7 | `segdec.CoarseCounts`, `segdec.decode_maps` | 3.3, 4.3, 4.4 | `decode` (`--full` 이면 601장) |
| 8 | `render.unpack_renderer` | 3.4 | `renderer_weights` |
| 9 | `render.Renderer`, `render.load_renderer`, `render.render_all` | 4.5 | `renderer_forward` |
| 10 | `pose.rice_decode`, `pose.parse_pose4` | 3.5 | `pose4` |
| 11 | `pose.affine`, `pose.even_frames` | 4.6 | `even` |
| 12 | `upscale.expand_fine` | 4.7 | `expand` |
| 13 | `inflate.py` 흐름 (이미 써 둔 것을 원하면 다시) | 1 | `bash run.sh rewrite` → 0.1196 |

```bash
cd comma_vcc/lab
../.venv/bin/python golden.py dump                      # 기준값 (한 번, 약 6분)
../.venv/bin/python check_rewrite.py                    # 모든 단계: PASS / FAIL / TODO
../.venv/bin/python check_rewrite.py intcnn inputs      # 고른 단계만
../.venv/bin/python check_rewrite.py --full decode      # seg 맵 601장 전부 (약 3분)
bash ../setup.sh && cd .. && bash run.sh rewrite         # 끝까지 다 채운 뒤: 압축 풀기 → inflate → 평가
```

- 단계마다 입력은 **golden 기준값**을 넣는다 (앞 단계가 틀려도 다음 단계를 따로 확인할 수 있게).
- `decode` 를 빼면 정확히 같아야 하는 단계와 float 오차 (1e-3) 를 허용하는 단계 (`renderer_forward`, `even`) 가 있다. 검사기가 알아서 판정한다.
- 함수 이름·인자는 검사기 (`lab/check_rewrite.py`) 와 맞춰 둔 것이다. 바꾸면 검사기도 같이 고친다.
- 파일 나누기·자료형 (numpy / torch) 은 바꿔도 된다. 다만 frame 은 torch (n,3,384,512) float32, seg 맵은 numpy uint8 로 주고받는 것을 검사기가 가정한다.
- 제출할 때 디렉터리 이름은 master 의 `submissions/` 에 없는 것으로 바꾼다 (GUIDE 9절).
