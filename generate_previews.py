"""
Belovely — batch preview-song generator.

Generates the 18 landing-page preview songs (6 occasions x 3 genre+mood combos,
matching funnel/assets/landing.js). These are UNPERSONALIZED showcase songs, so
each occasion gets an archetypal sample subject (a relatable name + universal
details) — enough material for the lyric engine to write something that feels
personal but applies broadly.

Reuses the validated engine in generate_song.py (same prompt, compose, preview cut).
Logs per-song metrics so we can judge engine coverage across genres/moods we've
never tested (Acoustic, Rock, Folk, Worship; Reflective/Grateful/Proud/etc.).

Run:
  python generate_previews.py --calibration      # 4 songs, one per NEW genre
  python generate_previews.py --only mothers-acoustic-reflective fathers-rock-proud
  python generate_previews.py                     # all 18
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import generate_song as gs  # reuses prompt, call_llm, validate_plan, call_elevenlabs, cut_preview
from elevenlabs.client import ElevenLabs
from elevenlabs.types.music_prompt import MusicPrompt
import anthropic

OUT = Path(__file__).parent / "preview_batch"

# Archetypal subject per occasion — relatable + universal so any viewer connects.
SAMPLE = {
    "mothers": dict(name="Mom", rel="mother", voice="Female",
        qualities="Always put everyone else first; calm in every storm; the one who quietly held it all together.",
        memories="Sunday dinners with the whole family; waiting up no matter how late; the smell of her kitchen in the morning.",
        message="Thank you for every quiet sacrifice we never even saw."),
    "fathers": dict(name="Dad", rel="father", voice="Male",
        qualities="Steady and dependable; taught me everything without many words; never once raised his voice.",
        memories="Early mornings in the old truck; learning to drive in an empty lot; fixing things together in the garage.",
        message="I didn't say thank you enough growing up — I see now everything you carried."),
    "anniversary": dict(name="Sarah", rel="wife", voice="Male",
        qualities="Still my best friend after all these years; patient, funny, impossibly kind.",
        memories="The day we met; the tiny first apartment; building a whole life out of ordinary days.",
        message="If I had every life to live again, I'd choose you every time."),
    "birthday": dict(name="Emma", rel="best friend", voice="Female",
        qualities="Lights up every room; fiercely loyal; the one everyone leans on.",
        memories="Years of laughter and late nights; growing up side by side; always showing up.",
        message="The whole world is better with you in it."),
    "just": dict(name="Maya", rel="partner", voice="Male",
        qualities="Calm in a way that slows the whole room down; warm; makes ordinary days feel like something.",
        memories="Slow coffee mornings; dancing in the kitchen; the long walk home laughing.",
        message="No occasion, no reason — I just love you."),
    "memorial": dict(name="Rose", rel="grandmother", voice="Female",
        qualities="Gentle and faithful; the heart of the whole family; made everyone feel loved.",
        memories="Her garden; her stories told a hundred times; the way she hummed while she cooked.",
        message="We carry you with us, every single day."),
}

# 18 combos — must match funnel/assets/landing.js OCCASIONS.
OCC_COMBOS = {
    "mothers":     [("R&B", "Heartfelt", "Everything You Gave"),
                    ("Acoustic", "Reflective", "The Quiet Years"),
                    ("Pop", "Grateful", "Thank You, Mom")],
    "fathers":     [("Country", "Reflective", "The Things You Carried"),
                    ("Acoustic", "Heartfelt", "My Safe Harbor"),
                    ("Rock", "Proud", "Everything I Know")],
    "anniversary": [("R&B", "Romantic", "Still You, Always"),
                    ("Pop", "Heartfelt", "I'd Choose You"),
                    ("Acoustic", "Tender", "The Small Unspoken")],
    "birthday":    [("Pop", "Joyful", "The World With You In It"),
                    ("R&B", "Upbeat", "Light Up the Room"),
                    ("Acoustic", "Warm", "Glad You're Here")],
    "just":        [("Acoustic", "Tender", "Ordinary Hours"),
                    ("R&B", "Romantic", "No Reason Needed"),
                    ("Folk", "Warm", "Just You, Just This")],
    "memorial":    [("Acoustic", "Reflective", "Carry You With Me"),
                    ("Worship", "Tender", "Until I See You Again"),
                    ("Country", "Heartfelt", "Always Here")],
}
OCC_LABEL = {"mothers": "Mother's Day", "fathers": "Father's Day", "anniversary": "Anniversary",
             "birthday": "Birthday", "just": "Just Because", "memorial": "Memorial"}

# Calibration = one combo per genre we have NOT tested yet (Acoustic, Rock, Folk, Worship).
CALIBRATION = [
    "mothers-acoustic-reflective",
    "fathers-rock-proud",
    "just-folk-warm",
    "memorial-worship-tender",
]


def slug(occ, genre, mood):
    g = genre.lower().replace("&", "").replace("/", "-").replace(" ", "")
    return f"{occ}-{g}-{mood.lower()}"


def all_combos():
    out = []
    for occ, combos in OCC_COMBOS.items():
        for (genre, mood, title) in combos:
            out.append((slug(occ, genre, mood), occ, genre, mood, title))
    return out


def chorus_start_s(plan_dict, lead_in=3.0):
    """Seconds into the song where the first Chorus begins, minus a short lead-in.
    Lets the preview start on the hook instead of the instrumental intro."""
    ms = 0
    for s in plan_dict.get("sections", []):
        if "chorus" in s["section_name"].lower():
            return max(0.0, ms / 1000.0 - lead_in)
        ms += int(s["duration_ms"])
    return 0.0


def build_quiz(occ, genre, mood):
    s = SAMPLE[occ]
    return {
        "first_name": s["name"], "nickname": "", "pronunciation": "",
        "relationship": s["rel"], "occasion": OCC_LABEL[occ],
        "genre": genre, "voice": s["voice"], "mood": mood,
        "qualities": s["qualities"], "memories": s["memories"], "message": s["message"],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--calibration", action="store_true", help="run the 4 new-genre calibration songs")
    p.add_argument("--only", nargs="*", help="run specific combo slugs")
    p.add_argument("--force", action="store_true", help="regenerate even if the .mp3 already exists")
    p.add_argument("--recut", action="store_true", help="re-cut previews from existing full songs (no API calls)")
    args = p.parse_args()

    anth = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    el = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

    combos = all_combos()
    if args.calibration:
        combos = [c for c in combos if c[0] in CALIBRATION]
    elif args.only:
        combos = [c for c in combos if c[0] in set(args.only)]

    OUT.mkdir(parents=True, exist_ok=True)

    # re-cut previews from existing full songs (start at the chorus), no API calls
    if args.recut:
        n = 0
        for (sl, occ, genre, mood, title) in combos:
            full = OUT / f"{sl}.mp3"
            lj = OUT / f"{sl}.lyrics.json"
            if not full.exists():
                continue
            # fall back to 43s (our consistent Intro 18s + Verse1 28s → chorus) when no plan
            start = chorus_start_s(json.load(open(lj))) if lj.exists() else 43.0
            gs.cut_preview(full, OUT / f"{sl}.preview.mp3", gs.PREVIEW_SECONDS, start)
            print(f"recut {sl}  start={start:.0f}s")
            n += 1
        print(f"\nre-cut {n} preview(s) starting at the chorus.")
        return 0

    print(f"Generating {len(combos)} preview song(s) -> {OUT}\n")
    results = []

    for idx, (sl, occ, genre, mood, title) in enumerate(combos, 1):
        print(f"[{idx}/{len(combos)}] {sl}  ({OCC_LABEL[occ]} · {genre} · {mood})")
        if (OUT / f"{sl}.mp3").exists() and not args.force:
            print("     skip (already generated)")
            continue
        quiz = build_quiz(occ, genre, mood)
        rec = {"slug": sl, "occasion": OCC_LABEL[occ], "genre": genre, "mood": mood,
               "title": title, "subject": SAMPLE[occ]["name"]}
        try:
            t0 = time.monotonic()
            plan_dict = gs.call_llm(anth, quiz)
            llm_s = time.monotonic() - t0
            warnings = gs.validate_plan(plan_dict, quiz)
            plan = MusicPrompt(**plan_dict)

            t1 = time.monotonic()
            result = gs.call_elevenlabs(el, plan)
            el_s = time.monotonic() - t1

            audio = result.audio if isinstance(result.audio, bytes) else b"".join(result.audio)
            full = OUT / f"{sl}.mp3"
            full.write_bytes(audio)
            gs.cut_preview(full, OUT / f"{sl}.preview.mp3", gs.PREVIEW_SECONDS, chorus_start_s(plan_dict))
            # dump full lyrics/plan per song for review + future lyric-PDF
            (OUT / f"{sl}.lyrics.json").write_text(json.dumps(plan_dict, indent=2))

            # capture chorus lyrics for review
            chorus = next((s.get("lines", []) for s in plan_dict["sections"]
                           if "chorus" in s["section_name"].lower()), [])
            rec.update(ok=True, llm_s=round(llm_s, 1), eleven_s=round(el_s, 1),
                       warnings=warnings, chorus=chorus,
                       copyright_retry=False)
            print(f"     ok — llm {llm_s:.0f}s, eleven {el_s:.0f}s"
                  + (f"  [warn: {'; '.join(warnings)}]" if warnings else ""))
        except Exception as e:
            rec.update(ok=False, error=str(e)[:300])
            print(f"     FAILED — {str(e)[:160]}")
        results.append(rec)

    (OUT / "_summary.json").write_text(json.dumps(
        {"generated_at": datetime.now(timezone.utc).isoformat(), "results": results}, indent=2))

    ok = [r for r in results if r.get("ok")]
    print("\n" + "=" * 60)
    print(f"DONE: {len(ok)}/{len(results)} succeeded")
    if ok:
        avg = sum(r["eleven_s"] for r in ok) / len(ok)
        print(f"avg ElevenLabs time: {avg:.0f}s")
    fails = [r for r in results if not r.get("ok")]
    if fails:
        print("FAILED:", ", ".join(r["slug"] for r in fails))
    print(f"files + _summary.json in {OUT}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
