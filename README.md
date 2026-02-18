# NexTrade

KOSPI200 구성종목 대상 주식 매매 시그널 알림 시스템. 기술적 분석, 섹터 모멘텀, ML 예측을 결합하여 매매 시그널을 생성하고 텔레그램으로 알림을 보낸다.

**핵심 기능**

- KOSPI200 구성종목 OHLCV 데이터 자동 수집 (pykrx)
- KOSPI200 지수 벤치마크 대비 상대강도 분석
- 기술적 지표 분석 & 매매 시그널 생성
- 섹터 로테이션 / 업종 모멘텀 분석
- LightGBM 기반 5일 수익률 예측 (분류 + 회귀)
- 종합 스크리닝 & 랭킹
- 텔레그램 봇 알림 (일간 리포트, 섹터 리포트, 긴급 시그널)
- 자동 스케줄링 (데이터 수집, 리포트 발송, 모델 재학습)

## 프로젝트 구조

```
nextrade/
├── main.py                  # CLI 엔트리포인트
├── config/
│   ├── settings.py          # 환경변수, 경로 설정
│   └── watchlist.py         # KOSPI200 동적 워치리스트 (업종별 분류)
├── data/
│   ├── collector.py         # KRX OHLCV 데이터 수집 (주식 + 인덱스)
│   ├── cache.py             # SQLite 기반 캐시
│   └── universe.py          # KRX 주식 유니버스 조회
├── analysis/
│   ├── indicators.py        # 기술적 지표 계산
│   ├── signals.py           # 매매 시그널 생성
│   ├── sector_rotation.py   # 섹터 모멘텀 분석
│   └── relative_strength.py # 상대강도 분석
├── ml/
│   ├── features.py          # 피처 엔지니어링
│   ├── trainer.py           # 모델 학습
│   ├── predictor.py         # 예측 수행
│   └── models/              # 학습된 모델 저장
├── screening/
│   └── screener.py          # 종합 스크리닝 & 랭킹
├── notification/
│   ├── telegram_bot.py      # 텔레그램 봇 핸들러
│   ├── scheduler.py         # APScheduler 크론 잡
│   ├── formatter.py         # 메시지 포맷팅
│   └── chart.py             # 차트 이미지 생성
├── utils/
│   └── logger.py            # 로깅 설정
├── tests/                   # pytest 테스트
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## 시작하기 (Quick Start)

### 1. 텔레그램 봇 생성

1. 텔레그램에서 [@BotFather](https://t.me/BotFather)를 검색하여 대화 시작
2. `/newbot` 입력 → 봇 이름과 username 설정
3. 발급된 **Bot Token**을 복사 (예: `123456789:ABCdefGhIJKlmNoPQRsTUVwxYZ`)

### 2. Chat ID 확인

1. 생성한 봇에게 아무 메시지나 전송
2. 브라우저에서 다음 URL 접속:
   ```
   https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   ```
3. 응답 JSON에서 `"chat":{"id": 123456789}` 부분의 숫자가 Chat ID

### 3. `.env` 설정

```bash
cp .env.example .env
```

`.env` 파일을 열어 위에서 확인한 값을 입력:

```
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

## 실행 방법

### Docker 실행 (권장)

Docker만 설치되어 있으면 Python 환경 세팅 없이 바로 실행 가능하다.

```bash
git clone <repo-url> && cd nextrade
docker-compose up -d
```

초기 데이터 수집 & 모델 학습은 컨테이너 안에서 실행:

```bash
docker exec nextrade python main.py collect
docker exec nextrade python main.py train
```

이후 스케줄러가 자동으로 데이터 수집, 리포트 발송, 모델 재학습을 처리한다.

Docker는 `nextrade.db`, `ml/models/`, `logs/`를 호스트에 볼륨 마운트하여 컨테이너를 재시작해도 데이터가 유지된다. `restart: unless-stopped` 설정으로 서버 재부팅 시에도 자동 재시작된다.

### 로컬 실행

Python 3.12+ 환경에서 직접 실행할 경우:

```bash
git clone <repo-url> && cd nextrade
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python main.py collect  # 초기 데이터 수집
python main.py train    # ML 모델 학습
python main.py serve    # 서버 시작
```

## 데이터 수집 & ML 모델 학습

서비스를 시작하기 전에 초기 데이터 수집과 모델 학습이 필요하다.

`collect`는 KOSPI200 구성종목 및 KOSPI200 지수의 OHLCV 데이터를 KRX에서 가져와 로컬 SQLite에 저장한다. `train`은 수집된 데이터로 LightGBM 분류기와 회귀 모델을 학습하여 `ml/models/`에 저장한다. 최초 1회만 직접 실행하면 이후엔 스케줄러가 자동 처리한다.

## CLI 명령어 레퍼런스

| 명령어 | 설명 | 주요 옵션 |
|--------|------|-----------|
| `collect` | 주식 OHLCV 데이터 수집 | `-t <ticker>`, `-a` (전체 유니버스), `-f` (강제 재수집) |
| `status` | 캐시 상태 & 통계 조회 | `-v` (티커별 상세) |
| `watchlist` | 주식 감시 목록 출력 | — |
| `analyze` | 기술적 분석 & 매매 시그널 생성 | `-t <ticker>` (개별 상세) |
| `train` | ML 모델 학습 | — |
| `predict` | ML 예측 (5일 수익률) | `-t <ticker>` (개별 상세) |
| `sector` | 섹터 로테이션 / 업종 모멘텀 | `-c <category>` (업종별 상세) |
| `ranking` | 종합 주식 랭킹 | `-n <N>` (상위 N개), `-c <category>` |
| `serve` | 텔레그램 봇 + 스케줄러 서버 시작 | — |
| `report` | 수동 텔레그램 리포트 발송 (테스트용) | `-t daily\|sector` |

**사용 예시:**

```bash
python main.py collect                    # KOSPI200 전체 수집
python main.py collect -t 005930          # 삼성전자만 수집
python main.py collect -a                 # KRX 전체 주식 유니버스 수집
python main.py analyze -t 005930          # 삼성전자 상세 시그널
python main.py ranking -n 20             # 종합 랭킹 상위 20
python main.py ranking -c electronics    # 전기전자 업종만 필터
python main.py report -t daily           # 일간 리포트 수동 발송
```

## 텔레그램 봇 명령어

`serve` 실행 중 텔레그램 봇에서 사용 가능한 명령어:

| 명령어 | 설명 |
|--------|------|
| `/today` (또는 `/top10`) | 오늘의 종합 스크리닝 리포트 (상위 10개) |
| `/sector` | 섹터 모멘텀 리포트 |
| `/stock <ticker>` | 개별 주식 상세 분석 + 차트 (예: `/stock 005930`) |

## 자동 스케줄

`serve` 실행 시 다음 크론 잡이 자동으로 등록된다 (KST 기준):

| 시간 | 주기 | 작업 |
|------|------|------|
| 15:40 | 월~금 | 데이터 수집 + 일간 리포트 발송 |
| 15:45 | 월~금 | 모델 성능 체크 (성능 저하 시 자동 재학습) |
| 월요일 08:30 | 매주 | 섹터 모멘텀 리포트 발송 |
| 매월 1일 06:00 | 매월 | ML 모델 정기 재학습 |

## 테스트

```bash
pytest
```
