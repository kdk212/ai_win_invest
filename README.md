# AI Invest Korea

한국 주식시장에서 단기 수익률 극대화를 목표로 후보 종목을 매일 선별하고, 백테스트와 추천 이후 성과를 같이 관리하는 투자 모델입니다.

이 프로젝트는 투자 조언을 자동 확정하는 도구가 아니라, 검증 가능한 선별 로직과 기록 체계를 만드는 도구입니다. 실제 매매 전에는 거래비용, 슬리피지, 공시, 실적 발표, 유동성, 본인 위험 한도를 반드시 확인해야 합니다.

## 핵심 흐름

1. `daily`: 오늘 기준 추천 후보를 생성합니다.
2. `backtest`: 과거 데이터로 주간 리밸런싱 전략 성과를 검증합니다.
3. `track`: 과거 추천 종목의 이후 수익률을 갱신합니다.
4. `web`: 로컬 홈페이지에서 최신 추천과 성과 기록을 확인합니다.
5. `telegram`: 최신 추천과 최근 7일 뉴스 리스크를 텔레그램으로 전송합니다.
6. `review-strategy`: 점수 기준, 손절, 익절 후보를 백테스트로 비교합니다.

## 로컬 실행

```powershell
python main.py daily
python main.py daily --as-of-date 2026-05-29 --with-news --send-telegram
python main.py daily --with-news --send-telegram
python main.py backtest --start 2024-11-01 --top-n 7
python main.py optimize-score --start 2024-11-01 --top-n 7
python main.py optimize-take-profit --start 2024-11-01 --top-n 7
python main.py review-strategy --start 2024-11-01 --top-n 7 --send-telegram
python main.py track
python main.py web
```

웹 화면은 기본적으로 `http://127.0.0.1:8787`에서 열립니다.

## PC가 꺼져 있어도 매일 전송

GitHub Actions 워크플로 `.github/workflows/daily-telegram.yml`이 매일 21:00 KST에 실행됩니다. 로컬 PC가 꺼져 있어도 GitHub 서버에서 실행됩니다.

GitHub 저장소에서 아래 비밀값을 등록해야 합니다.

`Settings` -> `Secrets and variables` -> `Actions` -> `New repository secret`

필요한 secret 이름:

- `OPENAI_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

## 특정일 추천주 수동 전송

GitHub 저장소의 `Actions` 탭에서 `Daily Telegram Recommendations` 워크플로를 선택한 뒤 `Run workflow`를 누르면 됩니다.

- `as_of_date`를 비워두면 최신 가능 일자 기준으로 전송합니다.
- `as_of_date`에 `2026-05-29`처럼 입력하면 해당 기준일 추천주를 텔레그램으로 전송합니다.

## 주간 전략 점검 리포트

GitHub Actions 워크플로 `.github/workflows/weekly-strategy-review.yml`이 매주 일요일 21:00 KST에 실행됩니다.

이 리포트는 아래 후보를 백테스트로 비교해서 텔레그램으로 보냅니다.

- raw 점수 기준 후보
- 손절 배수 후보
- 익절 트리거와 트레일링 폭 후보

`Actions` 탭에서 `Weekly Strategy Review`를 수동 실행하면 `start`, `end`, `top_n` 값을 바꿔 테스트할 수 있습니다. 이 워크플로는 전략 설정을 자동 변경하지 않습니다.

## 추천로직 고도화 원칙

추천로직은 자동으로 코드를 바꾸기보다, 성과를 쌓고 검증한 뒤 승인해서 반영하는 구조가 안전합니다.

권장 흐름:

1. 매일 추천주와 이후 수익률을 기록합니다.
2. 매주 또는 매월 백테스트 기간을 최신일까지 확장합니다.
3. raw 점수 기준, 손절 배수, 익절 트리거, 트레일링 폭을 자동 최적화합니다.
4. 기존 로직 대비 CAGR, MDD, Sharpe, 평균 투자비중, 거래 빈도, 최근 구간 성과를 비교합니다.
5. 새 파라미터가 여러 기간에서 안정적으로 우수할 때만 사람이 승인해 설정에 반영합니다.

자동 코드 수정까지 허용하면 과최적화와 예기치 않은 매매 기준 변경 위험이 커집니다. 따라서 자동화는 검증 리포트 생성까지, 실제 전략 변경은 승인 후 반영하는 방식을 기본으로 합니다.

## 로컬 텔레그램 설정

로컬 PC에서 실행할 때는 `telegram.env.example`을 참고해 `telegram.env`를 만들면 됩니다. `telegram.env`와 `openai.env`는 `.gitignore`에 포함되어 GitHub에 올라가지 않습니다.

```powershell
TELEGRAM_BOT_TOKEN=123456:ABC...
TELEGRAM_CHAT_ID=123456789
```

## 전략 요약

- 한국 유동성 상위 종목에서 모멘텀, 추세, 거래대금, 변동성, 과열도를 합산해 raw 점수를 계산합니다.
- 18개월 백테스트 기준 raw 2.0점 이상, 최대 7개 포트폴리오를 사용합니다.
- 손절은 변동성 기반 배수 2.5, 익절은 +30% 도달 후 고점 대비 -10% 트레일링으로 관리합니다.
- OpenAI가 최근 7일 뉴스를 읽고 `리스크 / 판정 / 이유` 형태로 텔레그램과 대시보드에 표시합니다.

자동 선별 결과이며 실제 매매 전 공시, 실적, 거래량, 손절 기준, 본인 위험 한도를 별도로 확인해야 합니다.
