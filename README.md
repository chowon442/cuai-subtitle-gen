# CUAI Subtitle Generator

## 실행 방법

```bash
# 개발 서버 실행 (자동 재시작)
uv run uvicorn app.main:app --reload

# 프로덕션 서버 실행
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

서버 실행 후:
- API: http://127.0.0.1:8000
- API 문서: http://127.0.0.1:8000/docs

## 환경 변수

- `OPENROUTER_API_KEY`: OpenRouter API 키
- `DATABASE_URL`: PostgreSQL 데이터베이스 URL (기본값 설정됨)
