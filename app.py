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
from collections import deque
from datetime import datetime, timedelta, timezone
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

# .strip() guards against invisible paste artifacts (e.g. trailing  ) that
# corrupt HTTP headers — these are pasted into the host dashboard by hand.
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "").strip()
EMAIL_FROM = os.getenv("EMAIL_FROM", "Belovely <hello@belovelygifts.com>").strip()
SHOPIFY_WEBHOOK_SECRET = os.getenv("SHOPIFY_WEBHOOK_SECRET", "").strip()
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "").strip()             # ops alerts (failures, paid-but-stuck)
RATE_PER_IP_HOUR = int(os.getenv("RATE_PER_IP_HOUR", "5"))     # /generate is pre-payment = cost surface
RATE_GLOBAL_HOUR = int(os.getenv("RATE_GLOBAL_HOUR", "150"))
KEEPSAKE_VARIANT_ID = int(os.getenv("KEEPSAKE_VARIANT_ID", "47891865338009"))  # $19 lyrics-PDF bump
from html import escape  # noqa: E402  (used in fulfillment_html)

# Abandoned-preview recovery (3-email sequence: ~1h reminder, ~24h $20-off, ~72h last call)
STORE_URL = os.getenv("STORE_URL", "https://belovelygifts.com").rstrip("/")
RECOVERY_DISCOUNT_CODE = os.getenv("RECOVERY_DISCOUNT_CODE", "SONG20")
NUDGE_THRESHOLDS_H = [1.0, 24.0, 72.0]  # hours after song_ready for stages 1, 2, 3
SWEEP_INTERVAL_S = int(os.getenv("SWEEP_INTERVAL_S", "120"))

# Tiered delivery — the song is pre-made (for the preview), but the paid tiers
# promise 30-min / 48-hr delivery, so fulfillment is scheduled, never instant.
TIER_30MIN_VARIANT_ID = int(os.getenv("TIER_30MIN_VARIANT_ID", "47891863273625"))  # $119
TIER_48HR_VARIANT_ID = int(os.getenv("TIER_48HR_VARIANT_ID", "47891863306393"))    # $79
DELIVER_30MIN_MIN = int(os.getenv("DELIVER_30MIN_MIN", "30"))       # minutes
DELIVER_48HR_MIN = int(os.getenv("DELIVER_48HR_MIN", "2880"))       # minutes (48h)
DELIVER_DEFAULT_MIN = int(os.getenv("DELIVER_DEFAULT_MIN", "30"))   # fallback

_db_lock = threading.Lock()
_rate_lock = threading.Lock()
_ip_hits: dict = {}
_global_hits: deque = deque()


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
            song_ready_at TEXT,
            nudge_stage INTEGER DEFAULT 0,
            unsub INTEGER DEFAULT 0,
            deliver_at TEXT,
            keepsake INTEGER DEFAULT 0,
            created_at TEXT,
            updated_at TEXT
          )
        """)
        # migrate DBs created before the recovery/delivery columns existed
        for col, ddl in (("song_ready_at", "TEXT"),
                         ("nudge_stage", "INTEGER DEFAULT 0"),
                         ("unsub", "INTEGER DEFAULT 0"),
                         ("deliver_at", "TEXT"),
                         ("keepsake", "INTEGER DEFAULT 0")):
            try:
                c.execute(f"ALTER TABLE orders ADD COLUMN {col} {ddl}")
            except Exception:  # noqa: BLE001  (column already exists)
                pass
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


def client_ip(req: Request) -> str:
    xff = req.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return req.client.host if req.client else "?"


def rate_ok(ip: str) -> bool:
    """Per-IP + global hourly cap on NEW generations (pre-payment cost guard)."""
    now_t = time.time(); cutoff = now_t - 3600
    with _rate_lock:
        dq = _ip_hits.setdefault(ip, deque())
        while dq and dq[0] < cutoff:
            dq.popleft()
        while _global_hits and _global_hits[0] < cutoff:
            _global_hits.popleft()
        if len(dq) >= RATE_PER_IP_HOUR or len(_global_hits) >= RATE_GLOBAL_HOUR:
            return False
        dq.append(now_t); _global_hits.append(now_t)
        return True


def alert_admin(subject: str, body: str) -> None:
    """Best-effort ops alert via Resend; logs if not configured."""
    if not (RESEND_API_KEY and ADMIN_EMAIL):
        print(f"[alert] {subject} :: {body}", flush=True)
        return
    try:
        httpx.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
            json={"from": EMAIL_FROM, "to": [ADMIN_EMAIL], "subject": subject,
                  "html": f"<pre style='font:14px ui-monospace,monospace'>{body}</pre>"},
            timeout=20,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[alert] failed: {e} :: {subject}", flush=True)


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

        upsert(order_id, status="song_ready", song_ready_at=now(),
               lyrics=json.dumps(plan_dict),
               preview_file="preview.mp3", full_file="song.mp3", error=None)
    except Exception as e:  # noqa: BLE001
        upsert(order_id, status="manual_review", error=str(e)[:500])
        alert_admin(f"⚠️ Belovely generation FAILED — {order_id}",
                    f"Order: {order_id}\nError: {e}\n\nThis order needs a manual song.")


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

    # New generation = a real ElevenLabs+Claude cost → rate-limit per IP + globally.
    if not rate_ok(client_ip(req)):
        return JSONResponse({"error": "rate_limited", "retry_after": 3600}, status_code=429)

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
    allowed = {"preview.mp3": "audio/mpeg", "song.mp3": "audio/mpeg",
               "keepsake.pdf": "application/pdf"}
    if name not in allowed:
        return JSONResponse({"error": "not found"}, status_code=404)
    p = ORDERS / order_id / name
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p, media_type=allowed[name])


def _order_id_from_shopify(payload: dict) -> str | None:
    # cart attributes arrive as note_attributes on the order
    for na in payload.get("note_attributes", []) or []:
        if na.get("name") == "order_id":
            return na.get("value")
    attrs = payload.get("attributes") or {}
    return attrs.get("order_id")


def send_email(to: str, subject: str, html: str, attachments: list | None = None) -> tuple[bool, str]:
    """Generic Resend sender. attachments = [{"filename","content"(base64)}]."""
    if not RESEND_API_KEY:
        return False, "RESEND_API_KEY not set"
    if not to:
        return False, "no recipient email"
    payload = {"from": EMAIL_FROM, "to": [to], "subject": subject, "html": html}
    if attachments:
        payload["attachments"] = attachments
    try:
        r = httpx.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
            json=payload, timeout=30,
        )
        ok = r.status_code in (200, 201)
        detail = f"{r.status_code} {r.text[:300]}"
        if not ok:
            print(f"[resend] send failed: {detail}", flush=True)
        return ok, detail
    except Exception as e:  # noqa: BLE001
        print(f"[resend] exception: {e}", flush=True)
        return False, f"exception: {e}"


def _lyric_sections(plan: dict):
    out = []
    for s in (plan.get("sections") or []):
        lines = [str(ln).strip() for ln in (s.get("lines") or []) if str(ln).strip()]
        if lines:
            out.append((str(s.get("section_name") or "").strip(), lines))
    return out


def fulfillment_html(name: str, full_url: str, plan: dict, keepsake_url: str | None = None) -> str:
    """Premium branded fulfillment email (inline styles for email-client safety)."""
    blocks = ""
    for label, lines in _lyric_sections(plan or {}):
        body = "<br>".join(escape(ln) for ln in lines)
        lab = (f'<div style="font:700 12px/1 Helvetica,Arial,sans-serif;letter-spacing:.14em;'
               f'text-transform:uppercase;color:#C9A24B;margin:0 0 6px">{escape(label)}</div>'
               if label else "")
        blocks += (f'<div style="margin:0 0 18px">{lab}'
                   f'<div style="font:400 16px/1.7 Georgia,\'Times New Roman\',serif;color:#2A241F">{body}</div></div>')
    lyrics_card = (
        f'<tr><td style="padding:6px 32px 8px">'
        f'<div style="background:#FFFFFF;border:1px solid #EEE7DA;border-radius:14px;padding:22px 26px">{blocks}</div>'
        f'</td></tr>' if blocks else "")
    keepsake = ""
    if keepsake_url:
        keepsake = (
            f'<tr><td style="padding:0 32px 10px">'
            f'<div style="background:#FBF3E2;border:1px solid #EBD9B0;border-radius:12px;padding:16px 18px;'
            f'font:400 15px/1.55 Helvetica,Arial,sans-serif;color:#5b4b2e">'
            f'&#128140; <b>Your Keepsake Lyrics</b> are attached as a printable PDF &mdash; '
            f'<a href="{escape(keepsake_url)}" style="color:#9a7b2e;font-weight:700">download it here</a> anytime.'
            f'</div></td></tr>')
    return f"""\
<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body style="margin:0;background:#F3ECE0;padding:24px 0;font-family:Helvetica,Arial,sans-serif">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#F3ECE0">
<tr><td align="center">
<table role="presentation" width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%;background:#FBF7F0;border-radius:18px;overflow:hidden;border:1px solid #E7DECF">
  <tr><td style="padding:26px 32px 6px;text-align:center">
    <div style="font:700 20px/1 Georgia,serif;color:#2A241F;letter-spacing:.02em">&#9834; Belovely</div>
  </td></tr>
  <tr><td style="padding:10px 32px 4px;text-align:center">
    <div style="font:700 26px/1.25 Georgia,serif;color:#2A241F">{escape(name)}'s song is ready &#128155;</div>
    <div style="font:400 15px/1.6 Helvetica,Arial,sans-serif;color:#7c6f5b;margin-top:8px">
      It's finished, and it's one of a kind &mdash; written just for {escape(name)}.</div>
  </td></tr>
  <tr><td style="padding:20px 32px 10px;text-align:center">
    <a href="{escape(full_url)}" style="display:inline-block;background:#C9A24B;color:#1c160d;text-decoration:none;
       font:700 16px/1 Helvetica,Arial,sans-serif;padding:15px 30px;border-radius:999px">&#9654; Listen &amp; download the full song</a>
  </td></tr>
  {keepsake}
  {lyrics_card}
  <tr><td style="padding:8px 32px 4px;text-align:center">
    <div style="font:400 14px/1.6 Helvetica,Arial,sans-serif;color:#7c6f5b">
      Our tip: headphones, a quiet room, two words &mdash; <i>"Just listen."</i></div>
  </td></tr>
  <tr><td style="padding:22px 32px 26px;text-align:center;border-top:1px solid #E7DECF">
    <div style="font:400 12px/1.6 Helvetica,Arial,sans-serif;color:#9a8d77">
      Questions? Just reply, or email <a href="mailto:hello@belovelygifts.com" style="color:#9a8d77">hello@belovelygifts.com</a>.<br>
      Belovely &middot; 288 Grove Street, Braintree, MA 02184</div>
  </td></tr>
</table>
</td></tr></table></body></html>"""


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
    ans = {}
    try:
        ans = json.loads(o["answers"] or "{}")
    except Exception:  # noqa: BLE001
        pass
    name = ans.get("nickname") or ans.get("first_name") or "your loved one"
    genre = ans.get("genre") or "Pop"

    # Safety: never email a broken link. If a paid order's song isn't actually
    # ready (e.g. generation failed → manual_review), alert ops to fulfill by hand.
    song_ready = o["status"] in ("song_ready", "delivered") and (ORDERS / order_id / "song.mp3").exists()
    if not song_ready:
        upsert(order_id, paid=1, status="paid_pending")
        alert_admin(f"⚠️ PAID but song NOT ready — {order_id}",
                    f"Order {order_id} was paid but status={o['status']} with no song file.\n"
                    f"Customer: {email}\nFulfill manually ASAP.")
        return {"ok": True, "note": "paid; song not ready — ops alerted"}

    plan = {}
    try:
        plan = json.loads(o["lyrics"] or "{}")
    except Exception:  # noqa: BLE001
        pass

    # Build the Keepsake PDF now (lyrics are ready); it's attached at delivery time.
    bought_keepsake = any(
        int(li.get("variant_id") or 0) == KEEPSAKE_VARIANT_ID
        for li in (payload.get("line_items") or [])
    )
    if bought_keepsake:
        try:
            from keepsake_pdf import build_lyrics_pdf
            build_lyrics_pdf(ORDERS / order_id / "keepsake.pdf", name, genre, plan, ans.get("message"))
        except Exception as e:  # noqa: BLE001
            bought_keepsake = False  # don't promise a PDF we couldn't build
            alert_admin(f"⚠️ Keepsake PDF FAILED — {order_id}",
                        f"Bump was purchased but PDF generation failed: {e}\n"
                        f"Customer: {email}\nSend the lyrics PDF manually.")

    # Honour the delivery tier. The song is pre-made (for the preview), but the
    # 30-min / 48-hr options must NOT deliver instantly — schedule it. The delivery
    # sweep sends the real song email when deliver_at passes.
    variant_ids = [int(li.get("variant_id") or 0) for li in (payload.get("line_items") or [])]
    if TIER_30MIN_VARIANT_ID in variant_ids:
        delay_min, window = DELIVER_30MIN_MIN, "within 30 minutes"
    elif TIER_48HR_VARIANT_ID in variant_ids:
        delay_min, window = DELIVER_48HR_MIN, "within 48 hours"
    else:
        delay_min, window = DELIVER_DEFAULT_MIN, "shortly"
    deliver_at = (datetime.now(timezone.utc) + timedelta(minutes=delay_min)).isoformat()
    upsert(order_id, paid=1, status="paid_scheduled", deliver_at=deliver_at,
           keepsake=1 if bought_keepsake else 0, email=email)

    # Immediate confirmation so they're not left wondering between purchase + delivery.
    csent, _ = send_email(email, f"We're finishing {name}'s song 🎶", crafting_html(name, window))
    return {"ok": True, "scheduled_for": deliver_at, "window": window,
            "keepsake": bought_keepsake, "confirm_emailed": csent}


# ---------------------------------------------------------------------------
# Abandoned-preview recovery — people who saw their preview but didn't buy.
# A daemon thread sweeps the DB and sends a 3-email sequence (1h / 24h / 72h),
# with a $20-off code from email #2 on. A purchase or unsubscribe stops it.
# ---------------------------------------------------------------------------
def _btn(url: str, label: str) -> str:
    return (f'<a href="{escape(url)}" style="display:inline-block;background:#C9A24B;color:#1c160d;'
            f'text-decoration:none;font:700 16px/1 Helvetica,Arial,sans-serif;padding:15px 30px;'
            f'border-radius:999px">{escape(label)}</a>')


def _shell(headline: str, sub: str, button_html: str = "", unsub_url: str = "", extra: str = "") -> str:
    button_row = (f'<tr><td style="padding:22px 32px 6px;text-align:center">{button_html}</td></tr>'
                  if button_html else "")
    unsub_line = (f'<br><a href="{escape(unsub_url)}" style="color:#b9ad99">Unsubscribe from reminders</a>'
                  if unsub_url else "")
    return f"""\
<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;background:#F3ECE0;padding:24px 0;font-family:Helvetica,Arial,sans-serif">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#F3ECE0"><tr><td align="center">
<table role="presentation" width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%;background:#FBF7F0;border-radius:18px;border:1px solid #E7DECF">
  <tr><td style="padding:26px 32px 4px;text-align:center"><div style="font:700 20px/1 Georgia,serif;color:#2A241F">&#9834; Belovely</div></td></tr>
  <tr><td style="padding:10px 32px 4px;text-align:center">
    <div style="font:700 25px/1.3 Georgia,serif;color:#2A241F">{headline}</div>
    <div style="font:400 15px/1.65 Helvetica,Arial,sans-serif;color:#7c6f5b;margin-top:10px">{sub}</div></td></tr>
  {button_row}
  {extra}
  <tr><td style="padding:22px 32px 26px;text-align:center;border-top:1px solid #E7DECF">
    <div style="font:400 12px/1.6 Helvetica,Arial,sans-serif;color:#9a8d77">
      Questions? Reply or email <a href="mailto:hello@belovelygifts.com" style="color:#9a8d77">hello@belovelygifts.com</a>.<br>
      Belovely &middot; 288 Grove Street, Braintree, MA 02184{unsub_line}</div></td></tr>
</table></td></tr></table></body></html>"""


def crafting_html(name: str, window: str) -> str:
    headline = f"We're finishing {escape(name)}'s song &#127926;"
    sub = (f"Thank you! Our team is putting the final touches on {escape(name)}'s one-of-a-kind song "
           f"&mdash; it'll arrive in your inbox <b>{escape(window)}</b>. Keep an eye out; it's worth "
           f"the wait. &#128155;")
    return _shell(headline, sub)


def recovery_email(stage: int, name: str, cta_url: str, preview_url: str, unsub_url: str):
    play = (f'<tr><td style="padding:2px 32px 0;text-align:center">'
            f'<a href="{escape(preview_url)}" style="font:400 14px/1.5 Helvetica,Arial,sans-serif;color:#9a7b2e">'
            f'&#9654; Hear the preview again</a></td></tr>')
    if stage == 1:
        subject = f"{name}'s song is ready to hear \U0001F3B5"
        headline = f"{escape(name)}'s song is finished &#127925;"
        sub = (f"You started something beautiful &mdash; {escape(name)}'s one-of-a-kind song is done and "
               f"waiting for you. Take a listen, then surprise them.")
        html = _shell(headline, sub, _btn(cta_url, "Hear your song"), unsub_url, extra=play)
    elif stage == 2:
        subject = f"A little something: $10 off {name}'s song \U0001F49B"
        headline = f"Here's $10 off {escape(name)}'s song"
        sub = (f"Still thinking it over? We saved {escape(name)}'s song for you &mdash; and we'd love to take "
               f"<b>$10 off</b> so you can give it. The discount is already on the button below.")
        html = _shell(headline, sub, _btn(cta_url, "Get $10 off your song"), unsub_url, extra=play)
    else:
        subject = f"Last call — don't lose {name}'s song"
        headline = f"Last chance for {escape(name)}'s song"
        sub = (f"We only hold finished songs for a little while, and {escape(name)}'s is about to roll off. "
               f"This is the final reminder &mdash; your <b>$10 off</b> is still good.")
        html = _shell(headline, sub, _btn(cta_url, "Claim the song + $10 off"), unsub_url, extra=play)
    return subject, html


def _hours_since(iso: str) -> float:
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds() / 3600.0
    except Exception:  # noqa: BLE001
        return 0.0


def recovery_sweep():
    from urllib.parse import quote
    with db() as c:
        rows = c.execute(
            "SELECT order_id, email, answers, song_ready_at, nudge_stage FROM orders "
            "WHERE paid=0 AND delivered=0 AND COALESCE(unsub,0)=0 AND status='song_ready' "
            "AND email!='' AND song_ready_at IS NOT NULL AND COALESCE(nudge_stage,0) < 3"
        ).fetchall()
    for r in rows:
        try:
            nxt = (r["nudge_stage"] or 0) + 1
            if _hours_since(r["song_ready_at"]) < NUDGE_THRESHOLDS_H[nxt - 1]:
                continue
            oid = r["order_id"]
            ans = json.loads(r["answers"] or "{}")
            name = ans.get("nickname") or ans.get("first_name") or "your loved one"
            reveal_path = f"/pages/reveal?order_id={oid}"
            if nxt == 1:
                cta = f"{STORE_URL}{reveal_path}"
            else:
                cta = f"{STORE_URL}/discount/{RECOVERY_DISCOUNT_CODE}?redirect={quote(reveal_path)}"
            unsub = f"{(PUBLIC_BASE or STORE_URL)}/unsubscribe?o={oid}"
            subject, html = recovery_email(nxt, name, cta, file_url(oid, "preview.mp3"), unsub)
            sent, detail = send_email(r["email"], subject, html)
            if sent:
                upsert(oid, nudge_stage=nxt)
                print(f"[recovery] stage {nxt} -> {oid}", flush=True)
            else:
                print(f"[recovery] stage {nxt} FAILED {oid}: {detail}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[recovery] error: {e}", flush=True)


def delivery_sweep():
    """Send the real song email when a scheduled order's deliver_at has passed."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with db() as c:
        rows = c.execute(
            "SELECT order_id, email, answers, lyrics, keepsake, deliver_at FROM orders "
            "WHERE paid=1 AND delivered=0 AND status='paid_scheduled' AND deliver_at IS NOT NULL"
        ).fetchall()
    for r in rows:
        try:
            if (r["deliver_at"] or "") > now_iso:
                continue  # not due yet
            oid = r["order_id"]
            ans = json.loads(r["answers"] or "{}")
            name = ans.get("nickname") or ans.get("first_name") or "your loved one"
            plan = json.loads(r["lyrics"] or "{}")
            attachments = None
            keepsake_url = None
            if r["keepsake"]:
                pdf = ORDERS / oid / "keepsake.pdf"
                if pdf.exists():
                    attachments = [{"filename": f"{name}'s Song - Belovely Keepsake.pdf",
                                    "content": base64.b64encode(pdf.read_bytes()).decode()}]
                    keepsake_url = file_url(oid, "keepsake.pdf")
            subject = f"{name}'s Belovely song is ready 🎵"
            html = fulfillment_html(name, file_url(oid, "song.mp3"), plan, keepsake_url)
            sent, detail = send_email(r["email"], subject, html, attachments=attachments)
            if sent:
                upsert(oid, delivered=1, status="delivered")
                print(f"[deliver] {oid} sent", flush=True)
            else:
                print(f"[deliver] {oid} FAILED: {detail}", flush=True)
                alert_admin(f"⚠️ Scheduled delivery email FAILED — {oid}", f"Resend: {detail}")
        except Exception as e:  # noqa: BLE001
            print(f"[deliver] error: {e}", flush=True)


def _sweeper_loop():
    import time
    while True:
        try:
            recovery_sweep()
        except Exception as e:  # noqa: BLE001
            print(f"[recovery] loop error: {e}", flush=True)
        try:
            delivery_sweep()
        except Exception as e:  # noqa: BLE001
            print(f"[deliver] loop error: {e}", flush=True)
        time.sleep(SWEEP_INTERVAL_S)


threading.Thread(target=_sweeper_loop, daemon=True).start()


@app.get("/unsubscribe")
def unsubscribe(o: str = ""):
    if o:
        upsert(o, unsub=1)
    return Response(
        content=(
            "<div style='font:16px/1.6 -apple-system,Helvetica,Arial,sans-serif;max-width:480px;"
            "margin:80px auto;text-align:center;color:#2A241F'>"
            "<h2 style='font-family:Georgia,serif'>You're unsubscribed</h2>"
            "<p style='color:#7c6f5b'>You won't get any more song reminders. Your song is still safe &mdash; "
            "email hello@belovelygifts.com anytime.</p></div>"
        ),
        media_type="text/html",
    )
