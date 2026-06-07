"""
Belovely Step 1 prototype (v2) — detailed lyric prompt + order linking.

Pipeline per order:
  order_id generated up front
    -> contact captured first (email, phone, language)
    -> quiz answers (relationship, name/nickname/pronunciation, occasion, genre,
       voice, mood, qualities, memories, message)
    -> Claude lyric step (very detailed prompt -> ElevenLabs composition_plan JSON)
    -> ElevenLabs compose_detailed
    -> ffmpeg preview slice
    -> everything written, LINKED BY order_id, to orders/<order_id>/

The orders/<order_id>/order.json record is the single artifact that threads
contact <-> quiz <-> generated song. It maps 1:1 to the future Postgres `orders`
row + object-storage keys (S4/S5). The `order_id` is our `quizId` / Shopify
cart-attribute stitch.

Run:
  python generate_song.py                 # runs all built-in test profiles
  python generate_song.py --only 0        # run a single profile by index
  python generate_song.py --copyright-test
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import shutil
import string
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# override=True so a stray empty ANTHROPIC_API_KEY= in the shell doesn't block us.
load_dotenv(Path.home() / ".env.belovely", override=True)

import os

import anthropic
from elevenlabs.client import ElevenLabs
from elevenlabs.types.music_prompt import MusicPrompt

HERE = Path(__file__).parent
ORDERS_DIR = HERE / "orders"

LLM_MODEL = "claude-sonnet-4-6"
PREVIEW_SECONDS = 75

# 8-section skeleton. Durations sum to 195000 ms (~3:15). Intro + Outro are
# instrumental (empty lines) so the song is slow to start and breathes at the end.
SECTION_SKELETON = [
    {"section_name": "Intro",        "duration_ms": 18000, "instrumental": True},
    {"section_name": "Verse 1",      "duration_ms": 28000, "instrumental": False},
    {"section_name": "Chorus",       "duration_ms": 26000, "instrumental": False},
    {"section_name": "Verse 2",      "duration_ms": 28000, "instrumental": False},
    {"section_name": "Chorus",       "duration_ms": 26000, "instrumental": False},
    {"section_name": "Bridge",       "duration_ms": 24000, "instrumental": False},
    {"section_name": "Final Chorus", "duration_ms": 30000, "instrumental": False},
    {"section_name": "Outro",        "duration_ms": 15000, "instrumental": True},
]
TARGET_TOTAL_MS = sum(s["duration_ms"] for s in SECTION_SKELETON)  # 195000

# ---------------------------------------------------------------------------
# Test profiles — these stand in for what real customers will submit.
# Contact is captured FIRST (per the funnel design); the quiz follows.
# Three genres so we can hear whether per-genre differentiation lands.
# ---------------------------------------------------------------------------
PROFILES = [
    {
        "contact": {"email": "john@example.com", "phone": "+1 555 0101", "language": "English"},
        "quiz": {
            "relationship": "wife",
            "first_name": "Sarah",
            "nickname": "",
            "pronunciation": "",
            "occasion": "Anniversary",
            "genre": "Pop",
            "voice": "Female",
            "mood": "Heartfelt",
            "qualities": (
                "Patient with our kids even on no sleep. Laughs at her own jokes "
                "before the punchline. Notices when I'm quiet and asks why."
            ),
            "memories": (
                "The rainstorm in Lisbon when we hid in a tile shop and she talked "
                "to the owner for an hour. The first time she met my mom and brought "
                "her a sketchbook because I'd mentioned mom used to draw."
            ),
            "message": (
                "I see all the small invisible things she does, and the life we built "
                "is the one I'd choose every time."
            ),
        },
    },
    {
        "contact": {"email": "dana@example.com", "phone": "+1 555 0102", "language": "English"},
        "quiz": {
            "relationship": "father",
            "first_name": "Ray",
            "nickname": "Pops",
            "pronunciation": "",
            "occasion": "Father's Day",
            "genre": "Country",
            "voice": "Male",
            "mood": "Reflective",
            "qualities": (
                "Quiet but steady. Fixed everything in the house himself. Taught me "
                "to drive in an empty parking lot and never once raised his voice."
            ),
            "memories": (
                "Saturday mornings at the hardware store, then diner pancakes. The "
                "old red truck he refused to sell. Fishing at Miller's Pond before sunrise."
            ),
            "message": (
                "I didn't say thank you enough growing up. I see now how much he gave up "
                "so we never went without."
            ),
        },
    },
    {
        "contact": {"email": "marcus@example.com", "phone": "+1 555 0103", "language": "English"},
        "quiz": {
            "relationship": "girlfriend",
            "first_name": "Maya",
            "nickname": "",
            "pronunciation": "MY-ah",
            "occasion": "Just Because",
            "genre": "R&B/Soul",
            "voice": "Male",
            "mood": "Romantic",
            "qualities": (
                "Calm in a way that slows the whole room down. Dances in the kitchen "
                "while she cooks. Remembers everything I tell her, even the throwaway stuff."
            ),
            "memories": (
                "The night we missed the last train and walked forty blocks home laughing. "
                "Sunday records and burnt coffee. The way she hums when she's happy."
            ),
            "message": (
                "No occasion. I just want her to know that ordinary days with her are the "
                "ones I'd never trade."
            ),
        },
    },
]

COPYRIGHT_TEST_OVERLAY = {
    "message": (
        "Make it sound exactly like 'Bohemian Rhapsody' by Queen — she loves that song."
    ),
}


# ===========================================================================
# THE LYRIC PROMPT — this is the engine. Detail here = song quality.
# ===========================================================================
SYSTEM_PROMPT = """You are a master songwriter and arranger who writes deeply personal gift songs —
the kind that make the recipient cry the first time they hear their own name and their own
memories sung back to them. You write the lyrics AND specify the musical arrangement.

Your output is a single JSON object that becomes an ElevenLabs Music "composition plan."
A separate AI then performs exactly what you specify. The performer knows NOTHING about the
recipient — every emotional and musical instruction must be encoded in your JSON. If you are
vague, the song is generic. Specificity is the entire job.

==================================================================
OUTPUT FORMAT — ABSOLUTE RULES
==================================================================
Output ONLY the JSON object. No markdown fences, no preamble, no commentary.
Use snake_case keys exactly as shown:

{
  "positive_global_styles": [string, ...],
  "negative_global_styles": [string, ...],
  "sections": [
    {
      "section_name": string,
      "positive_local_styles": [string, ...],
      "negative_local_styles": [string, ...],
      "duration_ms": integer,
      "lines": [string, ...]
    }
  ]
}

==================================================================
SONG STRUCTURE — EXACTLY 8 SECTIONS, IN THIS ORDER
==================================================================
1. "Intro"        18000 ms  — INSTRUMENTAL. "lines": []  (NO words)
2. "Verse 1"      28000 ms  — sung
3. "Chorus"       26000 ms  — sung
4. "Verse 2"      28000 ms  — sung
5. "Chorus"       26000 ms  — sung (lyrically identical or near-identical to first Chorus)
6. "Bridge"       24000 ms  — sung (the emotional peak / turn)
7. "Final Chorus" 30000 ms  — sung (the Chorus, often with one added or lifted line)
8. "Outro"        15000 ms  — INSTRUMENTAL. "lines": []  (NO words, let it resolve)

The duration_ms values MUST sum to exactly 195000.
Intro and Outro MUST have "lines": [] and describe the instrumental in their local styles
(e.g. "solo fingerpicked guitar", "soft piano with rising strings", "no vocals", "let the
last chord ring"). This is what makes the song "slow to start" and gives it a real ending.

==================================================================
PACING & TEMPO — THE SONG MUST FEEL SLOW AND SPACIOUS
==================================================================
- This is a BALLAD. Tempo 65–75 BPM. State the BPM in positive_global_styles.
- Leave room to breathe. Phrases should be UNHURRIED. Do NOT cram syllables.
- Target line lengths: Verses 6–9 words per line, 4–6 lines. Chorus 5–8 words per line,
  4 lines. Bridge 4 lines. A listener should be able to FEEL each line land before the next.
- The arrangement builds: sparse Intro → intimate Verse 1 → fuller Chorus → bigger Final
  Chorus → resolving Outro. Encode this build in the local styles per section.

==================================================================
LYRIC CRAFT — THIS IS WHERE SONGS WIN OR LOSE
==================================================================
- USE THE SPECIFIC DETAILS. Every concrete noun the customer gave you — names of places,
  objects, habits, moments — should appear. The Lisbon tile shop, the red truck, the burnt
  coffee. Specifics are what make them think "how did they know that?"
- SHOW, DON'T TELL. Instead of "you are kind," show the kindness in an image
  ("you leave notes inside the lunchbox / so they're not alone at noon").
- ONE clear emotional through-line. Pick the single truest feeling and build everything toward it.
- Natural, singable English. Real grammar. Lines that a person would actually say.
- BANNED CLICHÉS (never use these or close variants): "you mean the world to me", "you
  complete me", "words can't describe", "light up my life", "stars in the sky", "you're my
  everything", "through thick and thin", "by my side", "meant to be", "perfect storm".
- Rhyme is welcome but never force it — a near-rhyme or no rhyme beats a clumsy one.
- The chorus is the heart. It must contain the recipient's name and the central feeling,
  and be simple enough to remember after one listen.

==================================================================
THE RECIPIENT'S NAME
==================================================================
- If a nickname is provided, prefer the NICKNAME in the sung lyrics (that's what they're
  actually called). Use the first name at least once for clarity. If no nickname, use the
  first name.
- The name (nickname or first) MUST appear in: every Chorus, the Final Chorus, and ideally
  once in Verse 1. Place it where it's emphasized — start or end of a line.
- A pronunciation hint may be provided; use it to choose the spelling/placement so the
  performer sings it naturally. Do not put phonetic spelling in the actual lyric lines.

==================================================================
OCCASION → EMOTIONAL ARC (let the occasion shape the whole song)
==================================================================
- Anniversary: the long arc of a shared life; "then vs now"; chose-you-again.
- Wedding: vow-like, forward-looking, beginning of forever.
- Birthday: celebrate THIS person existing; the world is better with them in it.
- Mother's Day: the invisible labor of love; childhood seen through grown eyes.
- Father's Day: steadiness, sacrifice, lessons understood late; gratitude.
- Prayer: a blessing lifted over them; faith, hope, comfort, and gratitude; reverent and uplifting (leans Worship/acoustic).
- Memorial: tender grief; presence-in-absence; carrying them forward; NEVER maudlin.
- Graduation: pride; how far they've come; the road ahead.
- Just Because: the beauty of ordinary days; no occasion needed to love them.
- Thank You: specific gratitude for specific things they did.
- Missing You: distance and longing; the ache of the gap; holding on.
- Christmas: warmth, home, togetherness, the gift of them.

==================================================================
MOOD → TONE
==================================================================
- Heartfelt: sincere, tender, emotionally direct, warm.
- Upbeat: hopeful and bright while staying a ballad (lift, not speed).
- Romantic: intimate, sensual restraint, devotion.
- Reflective: contemplative, looking back, quiet wisdom.
- Fun: playful, light, affectionate humor — still musical, never cheesy.

==================================================================
GENRE → INSTRUMENTATION (put these kinds of tags in the styles)
==================================================================
- Pop: warm contemporary pop ballad, piano + soft synth pads, light percussion, lush vocals.
- Acoustic: solo or duo acoustic guitar, intimate, organic, minimal production.
- Country: acoustic guitar, pedal steel, brushed drums, fiddle, storytelling vocal, warm twang.
- R&B/Soul: smooth soul ballad, electric piano (Rhodes), soft bass, finger snaps, gospel-tinged
  background vocals, melismatic lead.
- Rock: power ballad, clean electric guitar building to fuller band, emotional swell.
- Jazz: jazz ballad, brushed drums, upright bass, piano, intimate crooner vocal.
- Worship: anthemic worship ballad, piano + pads building to swelling strings, congregational lift.
- Folk: fingerpicked acoustic, gentle, earthy, intimate, light strings.
- Rap/Hip-Hop: melodic hip-hop ballad, mellow beat, sung hook with spoken-word-leaning verses,
  soulful and slow — keep it emotional, not aggressive.

==================================================================
VOICE
==================================================================
- Female: state "female lead vocal" and a tone (warm, breathy, soulful, clear).
- Male: state "male lead vocal" and a tone.
- Duet: "male and female duet, trading lines and harmonizing on the chorus".
- No preference: choose what best fits the genre/mood and state it.

==================================================================
GLOBAL STYLE TAGS
==================================================================
positive_global_styles: 7–10 tags covering genre + sub-genre, tempo/BPM, the lead vocal and
tone, key instruments, the emotional mood, and the production feel
(e.g. "intimate and spacious", "emotional build", "studio quality").
negative_global_styles: rule out anything that breaks a tender gift song
(e.g. "no aggressive distortion", "no screamed vocals", "no EDM drop", "not fast",
"no harsh autotune", "no spoken-word only").

==================================================================
QUALITY-BAR EXAMPLE (shows the BAR and SHAPE — do NOT copy its content)
==================================================================
For a Heartfelt Pop anniversary song for a wife named "Ellie", a strong Chorus reads:
  "Ellie, it's the small unspoken kindness"
  "The light you leave on so I find my way"
  "If I had every life to live over"
  "I'd walk back to you, to this, to today"
Notice: name up front, a concrete image (the light left on), a turn ("every life"), and a
resolved last line. Your lyrics must hit this level of specificity and restraint — using the
ACTUAL details from the customer below, not these.

==================================================================
FINAL CHECKLIST BEFORE YOU OUTPUT
==================================================================
[ ] Exactly 8 sections, correct names, durations sum to 195000.
[ ] Intro and Outro have "lines": [] and describe the instrumental.
[ ] Name (nickname if given) in every Chorus + Final Chorus + once in Verse 1.
[ ] Real specific details from the customer woven throughout.
[ ] Zero banned clichés. Slow, spacious, singable lines.
[ ] Occasion arc and mood tone reflected. Genre instrumentation in the styles.
[ ] Output is ONLY the JSON object.
"""


def build_user_prompt(quiz: dict[str, str]) -> str:
    nickname = quiz.get("nickname") or "(none — use first name)"
    pronunciation = quiz.get("pronunciation") or "(none given)"
    return f"""Write the song now using these real details.

RECIPIENT
  First name: {quiz['first_name']}
  Nickname (prefer in lyrics if present): {nickname}
  Pronunciation hint: {pronunciation}
  Relationship to the giver: {quiz['relationship']}

SONG SETTINGS
  Occasion: {quiz['occasion']}
  Genre: {quiz['genre']}
  Voice: {quiz['voice']}
  Mood: {quiz['mood']}

WHAT MAKES THEM SPECIAL (their qualities, the giver's words):
{sanitize(quiz['qualities'])}

FAVORITE MEMORIES / SPECIAL MOMENTS (the giver's words):
{sanitize(quiz['memories'])}

A MESSAGE FROM THE GIVER'S HEART:
{sanitize(quiz['message'])}

Remember: 8 sections summing to 195000 ms, instrumental Intro + Outro, slow ballad,
the name in every chorus, and weave in the SPECIFIC details above. Output ONLY the JSON."""


def sanitize(text: str) -> str:
    """Best-effort strip of quoted song-title patterns. Real protection: S2 + Claude itself."""
    return re.sub(r'["“][^"”]{3,80}["”]', "[a song they love]", text)


# ===========================================================================
# Order identity + linking
# ===========================================================================
_ID_ALPHABET = string.ascii_letters + string.digits


def new_order_id() -> str:
    """quizId-style id, mirrors the competitor's `t-ob5Zk5-jeMWZ44` shape."""
    a = "".join(secrets.choice(_ID_ALPHABET) for _ in range(6))
    b = "".join(secrets.choice(_ID_ALPHABET) for _ in range(6))
    return f"bl-{a}-{b}"


def preview_title(quiz: dict[str, str]) -> str:
    name = quiz.get("nickname") or quiz["first_name"]
    return (
        f"{name}'s Belovely Song · {quiz['genre']} · "
        f"written for {name} (your {quiz['relationship']})"
    )


# ===========================================================================
# LLM + ElevenLabs
# ===========================================================================
def call_llm(client: anthropic.Anthropic, quiz: dict[str, str]) -> dict[str, Any]:
    msg = client.messages.create(
        model=LLM_MODEL,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(quiz)}],
    )
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def validate_plan(plan: dict[str, Any], quiz: dict[str, str]) -> list[str]:
    """Return a list of warnings. Hard-fail only on things that break generation."""
    warnings: list[str] = []
    sections = plan.get("sections", [])
    if not sections:
        raise ValueError("LLM returned no sections")

    total = sum(int(s["duration_ms"]) for s in sections)
    if not (180_000 <= total <= 205_000):
        raise ValueError(f"duration sum {total} out of bounds (180000-205000)")
    if total != TARGET_TOTAL_MS:
        warnings.append(f"duration sum {total} != target {TARGET_TOTAL_MS}")

    if len(sections) != 8:
        warnings.append(f"{len(sections)} sections (expected 8)")

    # Name must appear in a chorus. Token-based so multi-word names/nicknames
    # ("Grandma Rose", "Mary Jane") pass when the lyric uses any part of the name.
    name = (quiz.get("nickname") or quiz["first_name"]).lower()
    tokens = [t for t in re.split(r"\s+", name) if len(t) >= 3]
    chorus_text = " ".join(
        ln for s in sections if "chorus" in s["section_name"].lower() for ln in s["lines"]
    ).lower()
    if tokens and not any(t in chorus_text for t in tokens):
        raise ValueError(f"name '{name}' not found in any chorus")

    # Intro/outro should be instrumental.
    for s in sections:
        if s["section_name"].lower() in ("intro", "outro") and s["lines"]:
            warnings.append(f"{s['section_name']} has lyrics (expected instrumental)")
    return warnings


def extract_suggestion(err: Exception) -> tuple[str | None, MusicPrompt | None]:
    body = getattr(err, "body", None)
    if not isinstance(body, dict):
        return None, None
    detail = body.get("detail") or {}
    status = detail.get("status")
    data = detail.get("data") or {}
    if status == "bad_prompt":
        return data.get("prompt_suggestion"), None
    if status == "bad_composition_plan":
        plan_dict = data.get("composition_plan_suggestion")
        if plan_dict:
            return None, MusicPrompt(**plan_dict)
    return None, None


def call_elevenlabs(el: ElevenLabs, plan: MusicPrompt) -> Any:
    """compose_detailed with one retry on a copyright bounce.

    With a composition_plan, total length comes from per-section duration_ms;
    passing music_length_ms alongside it is rejected (422).
    """
    try:
        return el.music.compose_detailed(composition_plan=plan)
    except Exception as e:
        prompt_sug, plan_sug = extract_suggestion(e)
        if not prompt_sug and not plan_sug:
            raise
        print("  [copyright bounce] retrying with API suggestion...", flush=True)
        if plan_sug is not None:
            return el.music.compose_detailed(composition_plan=plan_sug)
        return el.music.compose_detailed(prompt=prompt_sug, music_length_ms=TARGET_TOTAL_MS)


def cut_preview(src: Path, dst: Path, seconds: int, start: float = 0.0) -> None:
    """Cut a `seconds`-long preview beginning at `start` (s), with fade in/out.
    Start at the chorus (not the instrumental intro) so the preview leads with the hook."""
    fade_out = max(seconds - 1.5, 0)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(start),          # input seek -> begin at the good part
        "-i", str(src),
        "-t", str(seconds),
        "-af", f"afade=t=in:st=0:d=0.8,afade=t=out:st={fade_out}:d=1.5",
        "-c:a", "libmp3lame", "-q:a", "2",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def lyrics_from_plan(plan_dict: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"section": s["section_name"], "lines": s.get("lines", [])}
        for s in plan_dict.get("sections", [])
    ]


# ===========================================================================
# Orchestration for one order
# ===========================================================================
def run_order(
    anth: anthropic.Anthropic,
    el: ElevenLabs,
    contact: dict[str, str],
    quiz: dict[str, str],
) -> dict[str, Any]:
    order_id = new_order_id()
    order_dir = ORDERS_DIR / order_id
    order_dir.mkdir(parents=True, exist_ok=True)
    name = quiz.get("nickname") or quiz["first_name"]
    print(f"\n=== {order_id}  ({name}, {quiz['genre']}, {quiz['occasion']}) ===")

    print("  1/3 LLM lyrics...", flush=True)
    t0 = time.monotonic()
    plan_dict = call_llm(anth, quiz)
    llm_s = time.monotonic() - t0
    warnings = validate_plan(plan_dict, quiz)
    plan = MusicPrompt(**plan_dict)
    total_ms = sum(s.duration_ms for s in plan.sections)
    print(f"      ok {llm_s:.1f}s — {len(plan.sections)} sections, {total_ms}ms"
          + (f"  [warn: {'; '.join(warnings)}]" if warnings else ""))

    print("  2/3 ElevenLabs compose_detailed...", flush=True)
    t1 = time.monotonic()
    result = call_elevenlabs(el, plan)
    eleven_s = time.monotonic() - t1
    audio = result.audio if isinstance(result.audio, bytes) else b"".join(result.audio)
    song_path = order_dir / "song.mp3"
    song_path.write_bytes(audio)
    (order_dir / "elevenlabs.json").write_text(json.dumps(result.json, indent=2, default=str))
    print(f"      ok {eleven_s:.1f}s — {len(audio)/1024:.0f} KB")

    print("  3/3 ffmpeg preview...", flush=True)
    cut_preview(song_path, order_dir / "preview.mp3", PREVIEW_SECONDS)

    total_s = time.monotonic() - t0

    # THE LINKING RECORD — order_id threads contact <-> quiz <-> song.
    # Maps 1:1 to the future Postgres `orders` row + storage keys.
    order_record = {
        "order_id": order_id,
        "status": "song_ready",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contact": contact,                 # captured FIRST in the funnel
        "quiz": quiz,
        "song": {
            "preview_title": preview_title(quiz),
            "genre": quiz["genre"],
            "duration_ms": total_ms,
            "files": {
                "full": "song.mp3",
                "preview": "preview.mp3",
                "elevenlabs_metadata": "elevenlabs.json",
            },
            "lyrics": lyrics_from_plan(plan_dict),
        },
        "composition_plan": plan_dict,
        "timing_s": {"llm": round(llm_s, 1), "eleven": round(eleven_s, 1), "total": round(total_s, 1)},
        "warnings": warnings,
    }
    (order_dir / "order.json").write_text(json.dumps(order_record, indent=2))
    print(f"      linked -> {order_dir}/order.json")
    return order_record


def require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: {name} not set in ~/.env.belovely", file=sys.stderr)
        sys.exit(2)
    return val


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=int, help="run a single profile by index (0-based)")
    parser.add_argument("--copyright-test", action="store_true")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not on PATH. brew install ffmpeg.", file=sys.stderr)
        return 2

    anth = anthropic.Anthropic(api_key=require_env("ANTHROPIC_API_KEY"))
    el = ElevenLabs(api_key=require_env("ELEVENLABS_API_KEY"))

    profiles = PROFILES if args.only is None else [PROFILES[args.only]]
    if args.copyright_test:
        profiles = [dict(PROFILES[0])]
        profiles[0] = {**profiles[0], "quiz": {**profiles[0]["quiz"], **COPYRIGHT_TEST_OVERLAY}}
        print("(copyright-test mode)")

    records = []
    for p in profiles:
        records.append(run_order(anth, el, p["contact"], p["quiz"]))

    print("\n" + "=" * 64)
    print("SUMMARY")
    for r in records:
        t = r["timing_s"]
        print(f"  {r['order_id']}  {r['song']['preview_title']}")
        print(f"      {r['song']['duration_ms']}ms  | llm {t['llm']}s  eleven {t['eleven']}s  total {t['total']}s")
        print(f"      {ORDERS_DIR / r['order_id']}/song.mp3")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
