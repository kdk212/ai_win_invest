# AI Invest Korea

한국 주식시장에서 단기 수익률 극대화를 목표로 후보 종목을 매일 선별하고, 백테스트와 추천 이후 성과를 같이 관리하는 투자 모델입니다.

이 프로젝트는 투자 조언을 자동 확정하는 도구가 아니라, 검증 가능한 선별 로직과 기록 체계를 만드는 도구입니다. 실제 매매 전에는 거래비용, 슬리피지, 공시, 실적 발표, 유동성, 본인 위험 한도를 반드시 확인해야 합니다.

## 핵심 흐름

1. `daily`: 오늘 기준 추천 후보를 생성합니다.
2. `backtest`: 과거 데이터로 주간 리밸런싱 전략 성과를 검증합니다.
3. `track`: 과거 추천 종목의 이후 수익률을 갱신합니다.
4. `web`: 로컬 홈페이지에서 최신 추천과 성과 기록을 확인합니다.
5. `telegram`: 최신 추천과 최근 7일 뉴스 리스크를 텔레그램으로 전송합니다.

## 로컬 실행

```powershell
python main.py daily
python main.py daily --with-news --send-telegram
python main.py backtest --start 2024-11-01 --top-n 7
python main.py optimize-score --start 2024-11-01 --top-n 7
python main.py optimize-take-profit --start 2024-11-01 --top-n 7
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

수동 테스트는 GitHub 저장소의 `Actions` 탭에서 `Daily Telegram Recommendations` 워크플로를 선택한 뒤 `Run workflow`를 누르면 됩니다.

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
