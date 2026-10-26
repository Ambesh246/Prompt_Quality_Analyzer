# React frontend

```bash
cp .env.example .env
npm install
npm run dev
npm run build
npm test
```

Set `VITE_API_BASE_URL` to your FastAPI origin. It is public client configuration, not a place for secrets. The optional LLM rewrite key belongs only on the backend.
