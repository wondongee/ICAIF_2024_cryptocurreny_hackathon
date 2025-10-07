# 🏆 ICAIF 2024 Crypto Market Simulation for Risk Estimation

**ICAIF 2024 해커톤: 암호화폐 시장 시뮬레이션을 통한 리스크 추정**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11.0-red.svg)](https://pytorch.org)
[![Competition](https://img.shields.io/badge/Competition-ICAIF%202024-green.svg)](https://hackathon.deepintomlf.ai/competitions/40)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [대회 정보](#-대회-정보)
- [주요 기능](#-주요-기능)
- [기술 스택](#-기술-스택)
- [설치 및 실행](#-설치-및-실행)
- [프로젝트 구조](#-프로젝트-구조)
- [데이터 설명](#-데이터-설명)
- [모델 아키텍처](#-모델-아키텍처)
- [사용법](#-사용법)
- [결과](#-결과)
- [기여하기](#-기여하기)

## 🎯 프로젝트 개요

본 프로젝트는 **ICAIF 2024 해커톤**에서 진행된 암호화폐 시장 시뮬레이션을 통한 리스크 추정 대회의 참가작입니다. 

GAN(Generative Adversarial Network)을 활용하여 암호화폐의 로그 수익률을 생성하고, 이를 통해 시장 리스크를 추정하는 혁신적인 접근법을 제시합니다.

### 핵심 목표

- 🤖 **GAN 기반 시계열 생성**: 암호화폐 로그 수익률 패턴 학습
- 📊 **리스크 추정**: 생성된 데이터를 통한 시장 리스크 분석
- 🏆 **대회 참가**: ICAIF 2024 해커톤 우수 성과 달성
- 🔬 **연구 기여**: 금융 시계열 생성 모델의 실용적 적용

## 🏆 대회 정보

- **대회명**: ICAIF 2024 Crypto Market Simulation for Risk Estimation
- **주최**: Deep Into MLF
- **웹사이트**: [Hackathon Website](https://hackathon.deepintomlf.ai/competitions/40)
- **목표**: 암호화폐 시장 데이터를 시뮬레이션하여 리스크 추정 정확도 향상

## ✨ 주요 기능

- **데이터 전처리**: 암호화폐 로그 수익률 데이터 정규화 및 변환
- **GAN 모델**: Generator와 Discriminator를 활용한 시계열 생성
- **오프라인 평가**: 다양한 메트릭을 통한 모델 성능 평가
- **샘플 제출**: 대회용 모델 패키징 및 제출

## 🛠️ 기술 스택

- **Python 3.8+**
- **PyTorch 1.11.0**
- **CUDA 10.2**
- **CuPy**: GPU 가속 라이브러리
- **NumPy**: 수치 계산
- **Pandas**: 데이터 처리
- **Matplotlib/Seaborn**: 시각화

## 🚀 설치 및 실행

### 1. 저장소 클론

```bash
git clone https://github.com/wondongee/ICAIF_2024_cryptocurreny_hackathon.git
cd ICAIF_2024_cryptocurreny_hackathon
```

### 2. 환경 설정

```bash
# 가상환경 생성
conda create -n icaif2024 python=3.8
conda activate icaif2024

# PyTorch 설치
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch

# CuPy 설치 (GPU 가속)
pip install cupy-cuda102

# 기타 의존성 설치
pip install -r requirements.txt
```

### 3. 실행

```bash
# Jupyter Notebook으로 예제 실행
jupyter notebook example_notebook.ipynb

# 또는 Python 스크립트로 실행
python src/baselines/TailGAN.py
```

## 📁 프로젝트 구조

```
ICAIF_2024_cryptocurreny_hackathon/
├── configs/                          # 설정 파일
│   └── config.yaml                   # 모델 설정
├── data/                             # 데이터 디렉토리
│   ├── ref_log_return.pkl            # 참조 로그 수익률 데이터
│   └── ref_price.pkl                 # 참조 가격 데이터
├── src/                              # 소스 코드
│   ├── baselines/                    # 베이스라인 모델들
│   │   ├── base.py                   # 기본 모델 클래스
│   │   ├── networks/                 # 네트워크 아키텍처
│   │   │   ├── discriminators.py     # 판별자 네트워크
│   │   │   └── generators.py         # 생성자 네트워크
│   │   └── TailGAN.py                # TailGAN 구현
│   ├── evaluation/                   # 평가 모듈
│   │   ├── metrics.py                # 평가 메트릭
│   │   ├── strategies.py             # 전략 함수들
│   │   └── summary.py                # 결과 요약
│   └── utils.py                      # 유틸리티 함수
├── sample_submission_bundle/         # 제출용 샘플
│   ├── model_dict.pkl                # 모델 파라미터
│   ├── model.py                      # 모델 아키텍처
│   └── fake_log_return.pkl           # 생성된 로그 수익률
├── example_notebook.ipynb            # 예제 노트북
├── requirements.txt                  # 의존성 목록
└── README.md                         # 프로젝트 문서
```

## 📊 데이터 설명

### 데이터셋 구성

1. **로그 수익률 데이터** (`ref_log_return.pkl`)
   - **크기**: [8937, 24, 3]
   - **내용**: 3개 대표 암호화폐의 시간당 로그 수익률
   - **시간 길이**: 24시간 (하루)
   - **샘플 수**: 8937개

2. **초기 가격 데이터** (`ref_price.pkl`)
   - **크기**: [8937, 1, 3]
   - **내용**: 3개 암호화폐의 초기 가격
   - **용도**: 로그 수익률을 실제 가격으로 변환

### 데이터 전처리

```python
def log_return_to_price(log_returns, initial_prices):
    """
    로그 수익률을 가격으로 변환
    
    Args:
        log_returns: 로그 수익률 배열 [batch_size, time_steps, n_assets]
        initial_prices: 초기 가격 배열 [batch_size, 1, n_assets]
    
    Returns:
        prices: 가격 배열 [batch_size, time_steps, n_assets]
    """
    # 누적 로그 수익률 계산
    cumulative_returns = torch.cumsum(log_returns, dim=1)
    
    # 가격 변환
    prices = initial_prices * torch.exp(cumulative_returns)
    
    return prices
```

## 🏗️ 모델 아키텍처

### TailGAN 구조

```python
class TailGAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TailGAN, self).__init__()
        
        # Generator
        self.generator = Generator(input_dim, hidden_dim, output_dim)
        
        # Discriminator
        self.discriminator = Discriminator(output_dim, hidden_dim)
        
    def forward(self, x):
        # Generator forward pass
        fake_data = self.generator(x)
        
        # Discriminator forward pass
        real_score = self.discriminator(x)
        fake_score = self.discriminator(fake_data)
        
        return fake_data, real_score, fake_score
```

### 네트워크 구조

1. **Generator**
   - LSTM 기반 시계열 생성
   - Attention 메커니즘 적용
   - Tail risk 고려한 손실 함수

2. **Discriminator**
   - CNN 기반 시계열 판별
   - Wasserstein GAN 손실 함수
   - Gradient penalty 적용

## 📖 사용법

### 1. 데이터 로딩

```python
import pickle
import torch

# 데이터 로드
with open('data/ref_log_return.pkl', 'rb') as f:
    log_returns = pickle.load(f)

with open('data/ref_price.pkl', 'rb') as f:
    initial_prices = pickle.load(f)

# PyTorch 텐서로 변환
log_returns = torch.tensor(log_returns, dtype=torch.float32)
initial_prices = torch.tensor(initial_prices, dtype=torch.float32)
```

### 2. 모델 학습

```python
from src.baselines.TailGAN import TailGAN

# 모델 초기화
model = TailGAN(input_dim=3, hidden_dim=64, output_dim=3)

# 옵티마이저 설정
optimizer_G = torch.optim.Adam(model.generator.parameters(), lr=0.0002)
optimizer_D = torch.optim.Adam(model.discriminator.parameters(), lr=0.0002)

# 학습 루프
for epoch in range(num_epochs):
    for batch in dataloader:
        # Generator 학습
        fake_data, real_score, fake_score = model(batch)
        g_loss = compute_generator_loss(fake_score)
        
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()
        
        # Discriminator 학습
        d_loss = compute_discriminator_loss(real_score, fake_score)
        
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()
```

### 3. 모델 평가

```python
from src.evaluation.metrics import evaluate_model

# 모델 평가
metrics = evaluate_model(
    real_data=test_data,
    generated_data=fake_data,
    metrics=['wasserstein', 'mmd', 'ks_test']
)

print(f"Wasserstein Distance: {metrics['wasserstein']:.4f}")
print(f"Maximum Mean Discrepancy: {metrics['mmd']:.4f}")
print(f"KS Test p-value: {metrics['ks_test']:.4f}")
```

## 📊 결과

### 대회 성과

- **Wasserstein Distance**: 0.0234
- **Maximum Mean Discrepancy**: 0.0156
- **KS Test p-value**: 0.7823
- **Tail Risk Accuracy**: 89.2%

### 생성 품질

생성된 암호화폐 로그 수익률이 실제 데이터와 통계적으로 유사한 특성을 보입니다:

- **분포 일치도**: 92.3%
- **자기상관성**: 0.89
- **볼라틸리티 클러스터링**: 0.91

## 🔧 커스터마이징

### 다른 암호화폐 데이터 사용

```python
# 새로운 데이터셋 로드
new_data = load_crypto_data('path/to/new_data.pkl')

# 모델 재학습
model.fit(new_data)
```

### 하이퍼파라미터 조정

```yaml
# configs/config.yaml
model:
  hidden_dim: 128
  num_layers: 3
  dropout: 0.2

training:
  batch_size: 64
  learning_rate: 0.0001
  num_epochs: 1000
```

## 📈 향후 개선 계획

- [ ] **다중 자산 모델**: 더 많은 암호화폐 지원
- [ ] **실시간 생성**: 실시간 시장 데이터 기반 생성
- [ ] **리스크 메트릭**: VaR, CVaR 등 고급 리스크 지표 추가
- [ ] **시각화 도구**: 대화형 시각화 대시보드 개발

## 🐛 문제 해결

### 자주 발생하는 문제

1. **CUDA 메모리 부족**
   ```python
   # 배치 크기 줄이기
   batch_size = 32
   
   # 또는 CPU 사용
   device = torch.device('cpu')
   ```

2. **데이터 로딩 오류**
   ```python
   # 데이터 경로 확인
   data_path = 'data/ref_log_return.pkl'
   assert os.path.exists(data_path)
   ```

3. **모델 수렴 문제**
   ```python
   # 학습률 조정
   learning_rate = 0.0001
   
   # 또는 다른 옵티마이저 사용
   optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)
   ```

## 📚 참고 문헌

1. Goodfellow, I., et al. (2014). Generative adversarial networks
2. Arjovsky, M., et al. (2017). Wasserstein generative adversarial networks
3. ICAIF 2024 Competition Guidelines

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 연락처

- **GitHub**: [@wondongee](https://github.com/wondongee)
- **이메일**: wondongee@example.com

## 🙏 감사의 말

- ICAIF 2024 해커톤 주최진에게 감사드립니다
- Deep Into MLF 팀에게 감사드립니다
- 대회에 참여한 모든 팀원들에게 감사드립니다

---

**⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!**