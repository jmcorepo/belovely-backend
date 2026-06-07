"""
Belovely backend — FastAPI service.

Wires the funnel to real generation + fulfillment:
  POST /generate                 quiz answers -> background generation (Claude + ElevenLabs)
  GET  /orders/{id}/status       funnel polls this; returns state + preview_url when ready
  GET  /files/{id}/{name}        serves preview/full mp3 (signed-ish by obscurity for now)
  POST /webhooks/shopify-paid    orders/paid -> match order_id (cart attribute) -> email full song
  GET  /healthz                  health check

State in SQLite (data/belovely.db), media on disk (data/orders/<id>/). For production,
swap disk -> R2/S3 and SQLite -> Postgres (see notes). Reuses generate_song.py.
"""

from __future__ import annotations

import hashlib
import hmac
import base64
import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
import httpx
from elevenlabs.client import ElevenLabs
from elevenlabs.types.music_prompt import MusicPrompt
from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import generate_song as gs              # prompt + LLM + ElevenLabs + ffmpeg preview
from generate_previews import chorus_start_s  # preview starts at the chorus

# ---------------------------------------------------------------------------
DATA = Path(os.getenv("BELOVELY_DATA_DIR", Path(__file__).parent / "data"))
ORDERS = DATA / "orders"
DB_PATH = DATA / "belovely.db"
ORDERS.mkdir(parents=True, exist_ok=True)

PUBLIC_BASE = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")  # set to deployed URL in prod
ALLOWED_ORIGINS = [o for o in os.getenv(
    "ALLOWED_ORIGINS",
    "https://belovely-bha00qq3.myshopify.com,https://belovelygifts.com,http://127.0.0.1:9292,http://localhost:9292",
).split(",") if o]

RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", "Belovely <hello@belovelygifts.com>")
SHOPIFY_WEBHOOK_SECRET = os.getenv("SHOPIFY_WEBHOOK_SECRET", "")

_db_lock = threading.Lock()


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as c:
        c.execute("""
          CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            status   TEXT NOT NULL,
            answers  TEXT,
            lyrics   TEXT,
            email    TEXT,
            preview_file TEXT,
            full_file    TEXT,
            error    TEXT,
            paid     INTEGER DEFAULT 0,
            delivered INTEGER DEFAULT 0,
            created_at TEXT,
            updated_at TEXT
          )
        """)
init_db()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert(order_id: str, **fields):
    fields["updated_at"] = now()
    with _db_lock, db() as c:
        row = c.execute("SELECT order_id FROM orders WHERE order_id=?", (order_id,)).fetchone()
        if row:
            sets = ", ".join(f"{k}=?" for k in fields)
            c.execute(f"UPDATE orders SET {sets} WHERE order_id=?", (*fields.values(), order_id))
        else:
            fields.setdefault("created_at", now())
            cols = ", ".join(["order_id", *fields])
            ph = ", ".join(["?"] * (1 + len(fields)))
            c.execute(f"INSERT INTO orders ({cols}) VALUES ({ph})", (order_id, *fields.values()))


def get_order(order_id: str) -> dict | None:
    with db() as c:
        row = c.execute("SELECT * FROM orders WHERE order_id=?", (order_id,)).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
app = FastAPI(title="Belovely")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _merge_tags(text: str, tags: list | None) -> str:
    if not tags:
        return text or ""
    extra = ", ".join(tags)
    return (text + "\n" if text else "") + extra


def answers_to_quiz(a: dict) -> dict:
    return {
        "first_name": a.get("first_name", "their"),
        "nickname": a.get("nickname", ""),
        "pronunciation": a.get("pronunciation", ""),
        "relationship": a.get("relationship", "loved one"),
        "occasion": a.get("occasion", "Just Because"),
        "genre": a.get("genre", "Pop"),
        "voice": a.get("voice", "No preference"),
        "mood": a.get("mood", "Heartfelt"),
        "qualities": _merge_tags(a.get("qualities", ""), a.get("qualities_tags")),
        "memories": _merge_tags(a.get("memories", ""), a.get("memories_tags")),
        "message": a.get("message", ""),
    }


def file_url(order_id: str, name: str) -> str:
    base = PUBLIC_BASE or ""
    return f"{base}/files/{order_id}/{name}"


def generate_job(order_id: str, answers: dict):
    """Runs in a background thread. Real Claude + ElevenLabs generation."""
    try:
        upsert(order_id, status="lyrics_ready")
        anth = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        el = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
        quiz = answers_to_quiz(answers)

        plan_dict = gs.call_llm(anth, quiz)
        gs.validate_plan(plan_dict, quiz)
        plan = MusicPrompt(**plan_dict)

        upsert(order_id, status="generating")
        result = gs.call_elevenlabs(el, plan)
        audio = result.audio if isinstance(result.audio, bytes) else b"".join(result.audio)

        d = ORDERS / order_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "song.mp3").write_bytes(audio)
        gs.cut_preview(d / "song.mp3", d / "preview.mp3", gs.PREVIEW_SECONDS, chorus_start_s(plan_dict))

        upsert(order_id, status="song_ready",
               lyrics=json.dumps(plan_dict),
               preview_file="preview.mp3", full_file="song.mp3", error=None)
    except Exception as e:  # noqa: BLE001
        upsert(order_id, status="manual_review", error=str(e)[:500])


@app.get("/healthz")
def healthz():
    return {"ok": True, "public_base": PUBLIC_BASE or None}


@app.post("/generate")
async def generate(req: Request, bg: BackgroundTasks):
    body = await req.json()
    order_id = (body.get("order_id") or "").strip()
    answers = body.get("answers") or {}
    if not order_id:
        return JSONResponse({"error": "order_id required"}, status_code=400)

    existing = get_order(order_id)
    if existing and existing["status"] in ("generating", "lyrics_ready", "song_ready", "delivered"):
        return {"order_id": order_id, "status": existing["status"]}  # idempotent

    upsert(order_id, status="intake_complete", answers=json.dumps(answers),
           email=answers.get("email", ""))
    bg.add_task(generate_job, order_id, answers)
    return {"order_id": order_id, "status": "intake_complete"}


@app.get("/orders/{order_id}/status")
def status(order_id: str):
    o = get_order(order_id)
    if not o:
        return JSONResponse({"status": "not_found"}, status_code=404)
    out = {"status": o["status"]}
    if o["status"] == "song_ready" or o["status"] == "delivered":
        out["preview_url"] = file_url(order_id, "preview.mp3")
    if o["status"] == "manual_review":
        out["error"] = o["error"]
    return out


@app.get("/files/{order_id}/{name}")
def files(order_id: str, name: str):
    if name not in ("preview.mp3", "song.mp3"):
        return JSONResponse({"error": "not found"}, status_code=404)
    p = ORDERS / order_id / name
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p, media_type="audio/mpeg")


def _order_id_from_shopify(payload: dict) -> str | None:
    # cart attributes arrive as note_attributes on the order
    for na in payload.get("note_attributes", []) or []:
        if na.get("name") == "order_id":
            return na.get("value")
    attrs = payload.get("attributes") or {}
    return attrs.get("order_id")


def send_email(to: str, recipient_name: str, full_url: str) -> bool:
    if not (RESEND_API_KEY and to):
        return False
    html = (
        f"<p>Your Belovely song for <b>{recipient_name}</b> is ready. 💛</p>"
        f'<p><a href="{full_url}">▶ Listen &amp; download the full song</a></p>'
        f"<p>Play it for them — headphones, quiet room, two words: \"Just listen.\"</p>"
    )
    try:
        r = httpx.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
            json={"from": EMAIL_FROM, "to": [to],
                  "subject": f"{recipient_name}'s Belovely song is ready 🎵", "html": html},
            timeout=20,
        )
        return r.status_code in (200, 201)
    except Exception:  # noqa: BLE001
        return False


@app.post("/webhooks/shopify-paid")
async def shopify_paid(req: Request):
    raw = await req.body()
    # verify HMAC
    if SHOPIFY_WEBHOOK_SECRET:
        digest = base64.b64encode(
            hmac.new(SHOPIFY_WEBHOOK_SECRET.encode(), raw, hashlib.sha256).digest()
        ).decode()
        sent = req.headers.get("X-Shopify-Hmac-Sha256", "")
        if not hmac.compare_digest(digest, sent):
            return Response(status_code=401)

    payload = json.loads(raw or b"{}")
    order_id = _order_id_from_shopify(payload)
    if not order_id:
        return {"ok": True, "note": "no order_id attribute"}

    o = get_order(order_id)
    if not o:
        return {"ok": True, "note": "order_id not found in store"}
    if o["delivered"]:
        return {"ok": True, "note": "already delivered"}  # idempotent

    email = (payload.get("email") or o["email"] or "")
    name = "your loved one"
    try:
        ans = json.loads(o["answers"] or "{}")
        name = ans.get("nickname") or ans.get("first_name") or name
    except Exception:  # noqa: BLE001
        pass

    full_url = file_url(order_id, "song.mp3")
    sent = send_email(email, name, full_url)
    upsert(order_id, paid=1, delivered=1, status="delivered")
    return {"ok": True, "emailed": sent, "full_url": full_url}
