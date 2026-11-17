# 거리영상 기반 보도 유효 폭 추정

학사학위논문 《거리영상 기반 보도 유효 폭 추정에서 보도 윤곽과 주변 시각 정보의 기여》의 실험 코드와 결과이다. 서울의 네 조사 지점에서 한 지점씩 학습에서 제외하고, 보도 윤곽과 내부 RGB 특징에 주변 RGB 특징을 더했을 때 폭 추정 오차가 달라지는지 비교했다. 개발 사진과 빈 마스크를 제외한 377장을 평가했다.

## 입력 조건과 결과

| 조건 | Ridge 회귀기에 제공한 특징 |
| --- | --- |
| A | 보도 윤곽 |
| B | 보도 윤곽과 내부 RGB에서 얻은 특징 차이 |
| C | 보도 윤곽과 외부 RGB에서 얻은 특징 차이 |
| D | 보도 윤곽과 내부·외부 RGB에서 얻은 두 특징 차이 |

네 조건은 같은 224×224 영상과 동일한 분할 마스크에서 출발한다. 보도 내부 특징은 내부 RGB만 남긴 영상의 특징에서 윤곽 특징을 뺀 값이다. 외부 특징은 외부 RGB만 남긴 영상의 특징에서 반전 윤곽 특징을 뺀 값이다. 분기별 특징을 연결하고, 회귀기에 제공하지 않는 분기에는 0 벡터를 넣는다. 학습에서 제외한 지점의 사진을 예측하며, Ridge의 정규화 계수는 1.0으로 고정했다.

| 지표 | DINOv2 | ResNet-18 |
| --- | ---: | ---: |
| B 평균 절대오차 | 0.541 m | 0.615 m |
| B에서 D로 바꿨을 때 오차 개선량 | −3.50 cm | −0.27 cm |
| 조건부 95% 구간 | [−9.15, 2.53] cm | [−3.31, 2.81] cm |

개선량은 B의 절대오차에서 D의 절대오차를 뺀 값이다. 조건부 구간은 서울 네 조사 지점 안에서 100 m 공간 블록을 재추출하여 계산했다. 두 모델에서 모두 구간에 0이 포함됐다. 침식과 팽창을 적용한 민감도 실험에서도 주변 특징으로 오차가 일관되게 줄어들지 않았다.

지점별로 약 20%씩 총 77장을 확인하는 모의실험에서, 1.5 m 미만 사진 55장 가운데 가로띠별 보도 점유율 회귀가 40장, DINOv2의 B 조건이 29장을 찾았다. 2.0 m 미만 사진 138장 가운데 두 방법은 각각 55장을 찾았다. 결과의 적용 범위는 해당 서울 표본에서의 사진 단위 현장 실측 후보 선정이다.

## 저장된 예측으로 결과 검증

Python 3.12 환경에서 저장소 루트를 기준으로 실행한다. 원본 영상이나 GPU는 필요 없다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/check_results.py
```

검증 스크립트는 홀드아웃 분할과 평가 표본을 확인한다. 두 모델의 전체·지점별 MAE, 편향, 조건 간 오차 차이, 공간 블록 재추출 구간, 고정 예산 선별 수를 다시 계산한다. 마스크 경계 민감도 실험의 예측도 같은 방식으로 검사한다.

- [동결한 실험 조건](protocols/seoul_v1.json)
- [본실험 집계](results/main/metrics.json)와 [분할표](results/main/folds.csv)
- [마스크 경계 민감도 프로토콜](protocols/mask_sensitivity.md)과 [집계](results/mask_sensitivity/metrics.json)
- [지점별 좁은 보도 선별 결과](results/site_review/site_results.csv)

결과 CSV에는 원자료와 대응하는 파일명, 실측 폭, 사진별 예측값, 지점과 블록 ID가 있다. 블록 ID는 좌표를 제거하고 지점별 원래 정렬 순서대로 다시 부여했다. 원본 사진, 좌표표, 모델 가중치, 특징 캐시, 논문 편집 파일은 로컬 자료로 관리한다.

## 영상에서 실험 다시 실행

[원래 실험 환경](results/main/environment.json)은 Python 3.12와 NVIDIA RTX 3080을 사용했다. CUDA 환경에 맞는 [PyTorch 빌드](https://pytorch.org/get-started/previous-versions/)를 설치한다. 아래는 원래 환경과 같은 CUDA 13.0 빌드 명령이다.

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-gpu.txt
mkdir -p local/weights local/mask2former local/archives
git clone https://github.com/ProjectSidewalk/sidewalk-tagger-ai.git local/sidewalk-tagger-ai
git -C local/sidewalk-tagger-ai checkout 72e9b6874d7a0c8c48d9edeca2f020b14552f390
```

- [DINOv2 체크포인트](https://huggingface.co/facebook/dinov2-with-registers-base/tree/a1d738ccfa7ae170945f210395d99dde8adb1805)의 `model.safetensors`를 `local/weights/dinov2.safetensors`로 둔다.
- [ResNet-18 체크포인트](https://download.pytorch.org/models/resnet18-f37072fd.pth)를 `local/weights/resnet18-f37072fd.pth`로 둔다.
- 사진부터 전처리하려면 [Seoul Sidewalk Accessibility Image Dataset v3](https://doi.org/10.5281/zenodo.22908353)의 ZIP 세 개를 `local/archives/`에 둔다. [Mask2Former 체크포인트](https://huggingface.co/facebook/mask2former-swin-large-mapillary-vistas-panoptic/tree/b9a3e9eb06a6477a741c98cf91cfca11e620e59d)의 `config.json`, `preprocessor_config.json`, `model.safetensors`를 `local/mask2former/`에 둔다.

```bash
python src/prepare_images.py --archives local/archives \
  --mask2former local/mask2former --output local/prepared

python src/run_study.py --prepared local/prepared --weights local/weights \
  --upstream local/sidewalk-tagger-ai --output local/main-replay

python src/mask_sensitivity.py --prepared local/prepared --weights local/weights \
  --upstream local/sidewalk-tagger-ai --output local/mask-replay
```

준비 파일은 각 원자료 파일명에 `.npz`를 붙인 형태이다. 각 파일의 `rgb`는 224×224 RGB 배열이고, `region`은 96×128 라벨 배열이다. 라벨 1이 보도에 해당한다. 기존 준비 파일이 있다면 `--prepared`에 해당 폴더를 지정하면 된다.

최초 분석에서는 개발 사진 136장의 기존 마스크와 특징 캐시를 사용했다. 그 준비 파일을 보유한 경우, 전처리에 `--pilot-prepared 경로`를 주면 저장된 해시를 확인한 뒤 재사용한다. 본실험에 `--feature-cache 경로`를 주면 기존 `dino_features.npz`와 `resnet_features.npz`를 사용한다. 같은 환경과 기존 캐시에서 두 예측 CSV와 본실험 집계 JSON이 배포본과 바이트 단위로 일치함을 확인했다. 준비 파일에서 특징을 다시 추출하는 경우 배치 구성과 수치 정밀도에 따라 값이 미세하게 달라질 수 있다.

보관된 ZIP으로부터 전처리를 재실행해 등록 사진 514장의 RGB 배열과 원본 파일 해시가 모두 일치함을 확인했다. 개발 마스크를 재사용했을 때, 보도 픽셀 마스크는 514장 중 491장에서 기존 결과와 일치했다. 재분할 전후에 보도 마스크가 비어 있는 사진은 동일했다. 논문 수치는 위의 저장된 예측 검증 절차로 확인할 수 있다.
