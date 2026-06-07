# Belovely backend (FastAPI)

Real generation + fulfillment behind the Shopify funnel.

## Endpoints
- `POST /generate` — `{ order_id, answers }` → kicks background generation (Claude → ElevenLabs → preview cut). Idempotent per `order_id`.
- `GET /orders/{id}/status` — funnel polls this; `song_ready` returns `preview_url`.
- `GET /files/{id}/{preview.mp3|song.mp3}` — serves audio.
- `POST /webhooks/shopify-paid` — Shopify `orders/paid`; matches `order_id` (cart attribute → order `note_attributes`), emails the full song via Resend, marks delivered. HMAC-verified when `SHOPIFY_WEBHOOK_SECRET` is set. Idempotent.
- `GET /healthz`

## Env vars
| var | purpose |
|---|---|
| `ANTHROPIC_API_KEY` | lyric generation |
| `ELEVENLABS_API_KEY` | song generation |
| `RESEND_API_KEY` | fulfillment email (optional; skips email if unset) |
| `EMAIL_FROM` | e.g. `Belovely <hello@belovelygifts.com>` (needs verified Resend domain) |
| `SHOPIFY_WEBHOOK_SECRET` | verify orders/paid HMAC (optional in dev) |
| `ALLOWED_ORIGINS` | CORS — the storefront origin(s) |
| `PUBLIC_BASE_URL` | this service's public URL (so file links are absolute) |

## Run locally
```
pip install -r requirements.txt          # ffmpeg must be on PATH
uvicorn app:app --port 8000
```

## Deploy (Render, Docker)
1. Push this folder to a GitHub repo.
2. Render → New → Blueprint (uses `render.yaml`) or Web Service (Docker).
3. Set the `sync:false` env vars in the dashboard; set `PUBLIC_BASE_URL` to the service URL after first deploy.
4. Then: register the Shopify `orders/paid` webhook → `<PUBLIC_BASE_URL>/webhooks/shopify-paid`, and flip the theme `config.js` to `useMock:false` + `baseUrl=<PUBLIC_BASE_URL>`.

## Production notes (not yet done)
- Disk + SQLite are **ephemeral** on most hosts → move media to R2/S3 and state to Postgres.
- Move generation to a real queue (Celery/RQ) for retries + concurrency.
- Add bot/rate-limit on `/generate` (pre-payment generation cost).
