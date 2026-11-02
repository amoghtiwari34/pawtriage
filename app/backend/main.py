"""PawTriage API. Run: .venv/bin/uvicorn app.backend.main:app --port 8020 (from repo root)."""
from __future__ import annotations

import io
import json
import os
import sys
import uuid
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import pt_safety  # noqa: E402
import pt_video  # noqa: E402
from pt_cost import call_cost  # noqa: E402
from pt_llm import LLM  # noqa: E402
from pt_prompt import build_messages, parse_output, salvage_output  # noqa: E402

from . import store  # noqa: E402
from .store import Case, Outcome, Pet  # noqa: E402

MEDIA = Path("/tmp/pawtriage/media") if os.environ.get("VERCEL") else Path(__file__).resolve().parent / "media"
MEDIA.mkdir(parents=True, exist_ok=True)
FRONTEND = ROOT / "app" / "frontend"
BASE_MODEL = os.environ.get("FIREWORKS_BASE_MODEL", "accounts/fireworks/models/glm-5p3-flash")
RECHECK_HOURS = {"now": 0, "24h": 24, "72h": 72, "2w": 24 * 14, "annual": 24 * 365}

app = FastAPI(title="PawTriage")
app.mount("/media", StaticFiles(directory=MEDIA), name="media")
llm = LLM(concurrency=8, cache_name="app")
store.init()
from . import den  # noqa: E402
den.init()
app.include_router(den.router)
from . import showcase  # noqa: E402
app.include_router(showcase.router)


def pick_model(which: str) -> tuple[str, bool, str]:
    """Return (model, is_base, mode). mode: 'sft' (deployed LoRA), 'rubric' (base + rubric prompt), 'base'."""
    dep = os.environ.get("FIREWORKS_DEPLOYMENT_MODEL", "")
    if which == "base":
        return BASE_MODEL, True, "base"
    if dep and which in ("auto", "sft"):
        return dep, False, "sft"
    # No fine-tune deployed (LoRA serving needs a dedicated GPU): serve the rubric-in-prompt harness,
    # which scored 83% on the locked test set at ~$1/1k chats.
    return BASE_MODEL, False, "rubric"


_RUBRIC = None


def rubric_prompt() -> str:
    global _RUBRIC
    if _RUBRIC is None:
        import pt_systems
        _RUBRIC = pt_systems.rubric_text()
    return _RUBRIC


@app.get("/pawtown", response_class=HTMLResponse)
def pawtown():
    return (FRONTEND / "pawtown.html").read_text()


@app.get("/", response_class=HTMLResponse)
def index():
    return (FRONTEND / "index.html").read_text()


@app.get("/config")
def config():
    dep = os.environ.get("FIREWORKS_DEPLOYMENT_MODEL", "")
    return {"base_model": BASE_MODEL, "deployment_model": dep or None, "fine_tuned_available": bool(dep),
            "default_mode": "sft" if dep else "rubric"}


# ------------------------------------------------------------------ pets
class PetIn(BaseModel):
    name: str
    species: str
    breed: str = ""
    age_years: float = 0
    weight_kg: float = 0
    meds: list[str] = []
    conditions: list[str] = []


@app.post("/pets")
def create_pet(p: PetIn):
    if p.species not in ("dog", "cat"):
        raise HTTPException(400, "species must be dog or cat")
    with store.session() as s:
        pet = Pet(**{**p.model_dump(), "meds": json.dumps(p.meds), "conditions": json.dumps(p.conditions)})
        s.add(pet)
        s.commit()
        s.refresh(pet)
        return {"id": pet.id, "species": pet.species, **pet.profile()}


@app.get("/pets/{pet_id}")
def get_pet(pet_id: int):
    with store.session() as s:
        pet = s.get(Pet, pet_id)
        if not pet:
            raise HTTPException(404)
        return {"id": pet.id, "species": pet.species, **pet.profile()}


@app.get("/pets")
def list_pets():
    with store.session() as s:
        return [{"id": p.id, "species": p.species, **p.profile()} for p in s.query(Pet).all()]


# ------------------------------------------------------------------ media
def save_media(case_dir: Path, image: UploadFile | None, video: UploadFile | None) -> dict:
    case_dir.mkdir(parents=True, exist_ok=True)
    if video is not None and video.filename:
        ext = Path(video.filename).suffix.lower() or ".mp4"
        if ext not in (".mp4", ".mov", ".m4v", ".webm"):
            raise HTTPException(400, "video must be mp4/mov")
        raw = case_dir / f"upload{ext}"
        raw.write_bytes(video.file.read())
        try:
            meta = pt_video.sample(raw, case_dir / "frames", n=6, max_seconds=30)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"modality": "video", "sheet_path": meta["sheet_path"], "frame_paths": meta["frame_paths"],
                "video_url": f"/media/{raw.relative_to(MEDIA)}",
                "sheet_url": f"/media/{Path(meta['sheet_path']).relative_to(MEDIA)}"}
    if image is not None and image.filename:
        ext = Path(image.filename).suffix.lower()
        raw = case_dir / f"upload{ext or '.jpg'}"
        raw.write_bytes(image.file.read())
        if ext in (".heic", ".heif"):
            import pillow_heif
            pillow_heif.register_heif_opener()
        out = pt_video.resize_image(raw, case_dir / "image.jpg")
        return {"modality": "image", "image_path": str(out), "image_url": f"/media/{out.relative_to(MEDIA)}"}
    return {"modality": "text"}


# ------------------------------------------------------------------ triage
async def triage_once(pet: Pet, message: str, media: dict, which: str, prior: list[dict] | None = None) -> dict:
    model, is_base, mode = pick_model(which)
    row = {"profile": pet.profile(), "message": message, "species": pet.species, **media}
    rubric = rubric_prompt() if mode == "rubric" else None
    res = await llm.chat(model, build_messages(row, rubric_text=rubric, prior=prior), temperature=0.0, max_tokens=1024,
                         use_cache=False)
    pred, err = parse_output(res["text"])
    salvaged = []
    if pred is None:  # app-only: keep a mostly-valid answer instead of a blanket fail-safe (scoring stays strict)
        pred, salvaged = salvage_output(res["text"])
        if pred is not None:
            pred["needs_review"] = True
    pred, audit = pt_safety.apply(pred, pet.species, message)
    if salvaged or (pred is not None and err):
        audit = [a for a in audit if a.get("rule") != "parse_error"]
        audit.insert(0, {"rule": "salvaged", "strict_error": err, "dropped": salvaged})
        pred["needs_review"] = True
    cost = call_cost(model, res["input_tokens"], res["output_tokens"])
    latency = res["latency_ms"]
    if os.environ.get("APP_ROUTER") == "1" and pt_safety.route(pred) == "large":
        big = pt_safety.large_model()
        r2 = await llm.chat(big, build_messages(row, rubric_text=rubric, prior=prior), use_cache=False)
        p2, _ = parse_output(r2["text"])
        p2, a2 = pt_safety.apply(p2, pet.species, message)
        pred = pt_safety.merge_routed(pred, p2)
        audit += [{"rule": "router", "to": big}] + a2
        cost += call_cost(big, r2["input_tokens"], r2["output_tokens"])
        latency += r2["latency_ms"]
    if pred is None:
        # Fail safe: never return "nothing" to a worried owner.
        pred = {"urgency": "vet_today", "system": "wellness", "conditions": [], "red_flags": ["could not assess"],
                "plan": {"immediate": ["call_vet_today"], "watch_for": [], "prevent": [], "recheck": "24h"},
                "needs_review": True}
        audit.append({"rule": "fallback", "why": err})
    return {"result": pred, "audit": audit, "model": model, "is_base_model": is_base, "mode": mode,
            "cost_usd": round(cost, 6), "latency_ms": latency, "raw": res["text"][:1500]}


def case_view(c: Case, pet: Pet | None = None) -> dict:
    turns = json.loads(c.turns)
    last = turns[-1] if turns else {}
    return {"id": c.id, "pet_id": c.pet_id, "pet": pet and {"id": pet.id, "species": pet.species, **pet.profile()},
            "created_at": c.created_at.isoformat(), "status": c.status,
            "recheck_due": c.recheck_due.isoformat() if c.recheck_due else None,
            "clock_offset_h": c.clock_offset_h,
            "checked": json.loads(c.checked), "turns": turns, "latest": last}


@app.post("/triage")
async def triage(pet_id: int = Form(...), message: str = Form(...), model: str = Form("auto"),
                 image: UploadFile | None = File(None), video: UploadFile | None = File(None)):
    with store.session() as s:
        pet = s.get(Pet, pet_id)
        if not pet:
            raise HTTPException(404, "pet not found")
        case = Case(pet_id=pet_id)
        s.add(case)
        s.commit()
        s.refresh(case)
        media = save_media(MEDIA / f"case_{case.id}_{uuid.uuid4().hex[:6]}", image, video)
        out = await triage_once(pet, message, media, model)
        turn = {"at": store.now().isoformat(), "kind": "intake", "message": message,
                **{k: v for k, v in media.items() if k.endswith("_url") or k == "modality"}, **out}
        case.turns = json.dumps([turn])
        case.recheck_due = store.now() + timedelta(hours=RECHECK_HOURS[out["result"]["plan"]["recheck"]])
        s.add(case)
        s.commit()
        s.refresh(case)
        den.on_case(case, pet)
        return case_view(case, pet)


@app.get("/cases")
def list_cases():
    with store.session() as s:
        return [case_view(c, s.get(Pet, c.pet_id)) for c in s.query(Case).order_by(Case.id.desc()).all()]


@app.get("/cases/{case_id}")
def get_case(case_id: int):
    with store.session() as s:
        c = s.get(Case, case_id)
        if not c:
            raise HTTPException(404)
        return case_view(c, s.get(Pet, c.pet_id))


class CheckIn(BaseModel):
    item: str


@app.post("/cases/{case_id}/plan/check")
def check_item(case_id: int, body: CheckIn):
    with store.session() as s:
        c = s.get(Case, case_id)
        if not c:
            raise HTTPException(404)
        checked = set(json.loads(c.checked))
        checked ^= {body.item}
        c.checked = json.dumps(sorted(checked))
        s.add(c)
        s.commit()
        return {"checked": sorted(checked)}


class FollowIn(BaseModel):
    message: str
    model: str = "auto"


@app.post("/cases/{case_id}/followup")
async def followup(case_id: int, body: FollowIn):
    with store.session() as s:
        c = s.get(Case, case_id)
        if not c:
            raise HTTPException(404)
        pet = s.get(Pet, c.pet_id)
        turns = json.loads(c.turns)
        prior = []
        for t in turns:
            prior.append({"role": "user", "content": f"Pet profile: {json.dumps(pet.profile())}\nOwner message: {t['message']}"})
            prior.append({"role": "assistant", "content": json.dumps(t["result"])})
        msg = f"FOLLOW-UP ({len(turns)} earlier assessment(s) above). {body.message}"
        out = await triage_once(pet, msg, {"modality": "text"}, body.model, prior=prior)
        prev = turns[-1]["result"]["urgency"]
        turns.append({"at": store.now().isoformat(), "kind": "followup", "message": body.message, "modality": "text", **out})
        c.turns = json.dumps(turns)
        c.recheck_due = store.now() + timedelta(hours=RECHECK_HOURS[out["result"]["plan"]["recheck"]])
        s.add(Outcome(case_id=c.id, message=body.message, prev_urgency=prev, new_urgency=out["result"]["urgency"]))
        s.add(c)
        s.commit()
        s.refresh(c)
        den.on_case(c, pet)
        return case_view(c, pet)


@app.post("/demo/advance_time")
def advance_time(hours: float = 24):
    """Fake the clock for the demo: pull every open case's recheck timer forward."""
    with store.session() as s:
        n = 0
        for c in s.query(Case).filter(Case.status == "open").all():
            if c.recheck_due:
                c.recheck_due = c.recheck_due - timedelta(hours=hours)
            c.clock_offset_h += hours
            s.add(c)
            n += 1
        for q in s.exec(den.select(den.Quest).where(den.Quest.status == "open")).all():
            if q.due_at:
                q.due_at = q.due_at - timedelta(hours=hours)
                s.add(q)
        for q in s.exec(den.select(den.Quest).where(den.Quest.status == "done")).all():
            if q.done_at:
                q.done_at = q.done_at - timedelta(hours=hours)
                s.add(q)
        s.commit()
    return {"advanced_hours": hours, "cases": n}


# ------------------------------------------------------------------ handoff
def nice(x: str) -> str:
    return x.replace("_", " ")


MODE_LABEL = {"sft": "fine-tuned LoRA", "rubric": "base GLM 5.3 Flash + rubric prompt", "base": "base GLM 5.3 Flash, untuned"}


def handoff_markdown(c: Case, pet: Pet) -> str:
    turns = json.loads(c.turns)
    checked = set(json.loads(c.checked))
    prof = pet.profile()
    L = [f"# Vet handoff: {pet.name} ({pet.species})", "",
         "_Generated by PawTriage. Triage support only, NOT a diagnosis. The pet must be examined by a veterinarian._", "",
         "## Profile",
         f"- Species/breed: {pet.species}, {prof['breed']}", f"- Age: {prof['age_years']} y · Weight: {prof['weight_kg']} kg",
         f"- Medications: {', '.join(prof['meds']) or 'none reported'}",
         f"- Known conditions: {', '.join(prof['conditions']) or 'none reported'}", "", "## Timeline"]
    for t in turns:
        r = t["result"]
        L += [f"### {t['at'][:16].replace('T', ' ')} UTC · {t['kind']}" + (f" · {t['modality']} attached" if t.get("modality") != "text" else ""),
              f"> {t['message']}", "",
              f"- Urgency assessed: **{nice(r['urgency'])}** (model: {MODE_LABEL.get(t.get('mode'), 'base' if t.get('is_base_model') else 'fine-tuned')})",
              f"- Possible conditions to consider (not a diagnosis): {', '.join(nice(x) for x in r.get('conditions', []))}",
              f"- Red flags: {', '.join(r.get('red_flags', [])) or 'none'}"]
        if r.get("image_finding"):
            L.append(f"- Visible finding in photo/video: {nice(r['image_finding'])}")
        rules = [a for a in t.get("audit", []) if a["rule"] in ("toxin_override", "emergency_phrase", "never_list", "router")]
        if rules:
            L.append(f"- Safety layer: {'; '.join(json.dumps(a) for a in rules)}")
        L.append("")
    last = turns[-1]["result"]["plan"] if turns else {}
    items = last.get("immediate", []) + last.get("watch_for", []) + last.get("prevent", [])
    L += ["## Plan adherence (owner-reported)"]
    L += [f"- [{'x' if i in checked else ' '}] {nice(i)}" for i in last.get("immediate", [])]
    L += ["", f"Done: {len(checked & set(items))}/{len(last.get('immediate', []))} immediate steps", "",
          "## Watch-for list given to owner", *[f"- {nice(i)}" for i in last.get("watch_for", [])], ""]
    with store.session() as _s:
        adh = showcase.adherence(_s, pet.id)
    if adh["score"] is not None:
        L += [f"## Care adherence (PawTown Den)", f"- {adh['on_time']}/{adh['due']} due care steps done on time ({adh['score']}%)", ""]
    notes = den.caregiver_notes(c.id)
    if notes:
        L += ["## Caregiver log (PawTown Den: who did what, when)", *[f"- {n}" for n in notes], ""]
    L += ["---", "Generated by PawTriage, not a diagnosis."]
    return "\n".join(L)


@app.get("/cases/{case_id}/handoff")
def handoff(case_id: int, format: str = "md"):
    with store.session() as s:
        c = s.get(Case, case_id)
        if not c:
            raise HTTPException(404)
        md = handoff_markdown(c, s.get(Pet, c.pet_id))
    if format == "pdf":
        return Response(md_to_pdf(md), media_type="application/pdf",
                        headers={"Content-Disposition": f"attachment; filename=pawtriage_case_{case_id}.pdf"})
    return {"markdown": md}


def md_to_pdf(md: str) -> bytes:
    from xml.sax.saxutils import escape

    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    import re

    st = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, title="PawTriage vet handoff")
    flow = []
    for line in md.splitlines():
        if not line.strip():
            flow.append(Spacer(1, 6))
            continue
        txt = escape(line.lstrip("#>-* ").strip())
        txt = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", txt)
        txt = re.sub(r"_(.+?)_$", r"<i>\1</i>", txt)
        if line.startswith("# "):
            flow.append(Paragraph(txt, st["Title"]))
        elif line.startswith("## "):
            flow.append(Paragraph(txt, st["Heading2"]))
        elif line.startswith("### "):
            flow.append(Paragraph(txt, st["Heading4"]))
        elif line.startswith(">"):
            flow.append(Paragraph(f"<i>“{txt}”</i>", st["BodyText"]))
        elif line.startswith("- "):
            flow.append(Paragraph("• " + txt, st["BodyText"]))
        else:
            flow.append(Paragraph(txt, st["BodyText"]))
    doc.build(flow)
    return buf.getvalue()


# ------------------------------------------------------------------ metrics
@app.get("/metrics")
def metrics():
    p = ROOT / "results" / "summary.json"
    if not p.exists():
        return {"available": False}
    return {"available": True, "summary": json.loads(p.read_text())}


@app.get("/charts/{name}")
def chart(name: str):
    p = ROOT / "results" / "charts" / Path(name).name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/demo/pets")
def demo_pets():
    return json.loads((ROOT / "demo" / "demo_pets.json").read_text())


# ------------------------------------------------------------------ live-scanner engine adapter
# Implements the SCAN_ENGINE_URL contract of github.com/mercadobeomelo/live-scanner (server/src/scan/engine.js):
# POST {images:[b64 jpeg], note, pet, can_ask, answer} -> {report:{...}}. PawTriage never asks; it always reports.
URG_TO_SCANNER = {"emergency_now": "red", "vet_today": "orange", "vet_this_week": "yellow", "monitor_at_home": "green"}
URG_TO_VET = {"emergency_now": "tonight", "vet_today": "tonight", "vet_this_week": "within_48h", "monitor_at_home": "no_rush"}


class ScanIn(BaseModel):
    images: list[str] = []
    note: str | None = None
    pet: dict | None = None
    can_ask: bool = False
    answer: dict | None = None


def scanner_report(pred: dict, audit: list[dict], species: str, mode: str = "base") -> dict:
    u = pred["urgency"]
    conds = [nice(c) for c in pred.get("conditions", [])]
    plan = pred.get("plan", {})
    finding_seen = pred.get("image_finding")
    abnormal = finding_seen and finding_seen != "no_visible_abnormality"
    findings = [{
        "status": "abnormal" if abnormal else ("normal" if finding_seen else "not_assessed"),
        "what": nice(finding_seen) if finding_seen else "visible area",
        "where": {"system": "skin" if pred.get("system") == "skin" else ("ear" if pred.get("system") == "eye_ear" else "other"),
                  "region": "", "side": "cannot_tell"},
        "real": "possible", "severity": {"red": "severe", "orange": "moderate"}.get(URG_TO_SCANNER[u], "mild"),
        "why": {"sft": "PawTriage fine-tuned triage model", "rubric": "PawTriage (GLM 5.3 Flash + vet rubric)",
                "base": "PawTriage base model (untuned)"}.get(mode, "PawTriage") + (" (safety rule applied)" if any(a["rule"] in ("toxin_override", "emergency_phrase") for a in audit) else ""),
        "care": ", ".join(nice(x) for x in plan.get("immediate", [])[:3]),
    }]
    if any(a["rule"] == "fallback" for a in audit):
        title = "Couldn't assess this - please call your vet"
    elif conds and conds[0] != "healthy wellness":
        title = f"Possible {conds[0]}"
    else:
        title = "Nothing worrying visible"
    verdict = {"red": "This could be an emergency. Go to an emergency vet now.",
               "orange": "Please call your vet today.",
               "yellow": "Worth booking a vet visit this week.",
               "green": "Looks fine for now; keep an eye on it."}[URG_TO_SCANNER[u]]
    return {
        "title": title, "subject": f"a {species}" if species else None, "verdict": verdict,
        "summary": f"Possible conditions (not a diagnosis): {', '.join(conds) or 'none'}. Red flags: {', '.join(pred.get('red_flags', [])) or 'none'}.",
        "urgency": URG_TO_SCANNER[u],
        "alert": {"title": verdict, "body": "; ".join(pred.get("red_flags", [])) or verdict} if u in ("emergency_now", "vet_today") else None,
        "findings": findings,
        "next_steps": [nice(x) for x in plan.get("immediate", [])],
        "when_to_see_vet": [nice(x) for x in plan.get("watch_for", [])],
        "tonight_care": [nice(x) for x in plan.get("prevent", [])],
        "vet_decision": {"verdict": URG_TO_VET[u], "reason": f"recheck {plan.get('recheck', '')}"},
        "pet_relevant": True, "confidence": None,
        "disclaimer": "PawTriage: triage support, not a diagnosis. If you are worried, see a vet.",
    }


@app.post("/scan")
async def scan_engine(body: ScanIn):
    import base64
    pet = body.pet or {}
    species = (pet.get("petType") or pet.get("species") or "").lower()
    species = "cat" if "cat" in species else "dog"
    case_dir = MEDIA / f"scan_{uuid.uuid4().hex[:8]}"
    case_dir.mkdir(parents=True, exist_ok=True)
    imgs = list(body.images) + list((body.answer or {}).get("images") or [])
    paths = []
    for i, b in enumerate(imgs[:6]):
        raw = case_dir / f"raw_{i}.jpg"
        raw.write_bytes(base64.b64decode(b.split(",", 1)[-1]))
        paths.append(str(pt_video.resize_image(raw, case_dir / f"img_{i}.jpg")))
    profile = {"name": pet.get("petName", ""), "breed": pet.get("breed", ""), "age_years": pet.get("age", 0),
               "weight_kg": pet.get("weight", 0), "meds": [], "conditions": []}
    note = body.note or "Is this serious?"
    if body.answer and body.answer.get("text"):
        note += f" (owner added: {body.answer['text']})"
    media = {"modality": "image", "image_path": paths[0]} if len(paths) == 1 else \
            ({"modality": "video", "sheet_path": paths[0], "frame_paths": paths[1:]} if paths else {"modality": "text"})

    class _P:  # minimal pet shim for triage_once
        def __init__(s): s.species = species
        def profile(s): return profile
    out = await triage_once(_P(), note, media, "auto")
    linked = record_scan(pet, species, note, out, case_dir, paths)
    return {"report": scanner_report(out["result"], out["audit"], species, out["mode"]),
            "pawtriage": {"model": out["model"], "is_base_model": out["is_base_model"], "audit": out["audit"],
                          "latency_ms": out["latency_ms"], "pawtown": linked}}


def record_scan(pet_in: dict, species: str, note: str, out: dict, case_dir: Path, paths: list[str]) -> dict | None:
    """Live scanner -> PawTown: a scan of a Den pet becomes a real Case, so quests/contagion/emergency show on the map.
    Matches the scanner's petName to a pet in a Den (newest Den first). Unknown name + a Den exists -> the pet joins
    the newest Den. No name or no Den -> stays a stateless scan."""
    name = (pet_in.get("petName") or "").strip()
    if not name:
        return None
    with store.session() as s:
        dens = s.exec(den.select(den.Den).order_by(den.Den.id.desc())).all()
        if not dens:
            return None
        target, den_id = None, dens[0].id
        for d in dens:
            for dp in s.exec(den.select(den.DenPet).where(den.DenPet.den_id == d.id)).all():
                p = s.get(Pet, dp.pet_id)
                if p.name.lower() == name.lower() and p.species == species:
                    target, den_id = p, d.id
                    break
            if target:
                break
        if not target:
            target = Pet(name=name, species=species, breed=pet_in.get("breed") or "", age_years=float(pet_in.get("age") or 0),
                         weight_kg=float(pet_in.get("weight") or 0))
            s.add(target)
            s.commit()
            s.refresh(target)
            s.add(den.DenPet(den_id=den_id, pet_id=target.id))
        rel = lambda x: "/media/" + str(Path(x).relative_to(MEDIA))  # noqa: E731
        turn = {"at": store.now().isoformat(), "kind": "live_scan", "message": note, "modality": media_kind(paths),
                **({"image_url": rel(paths[0])} if len(paths) == 1 else {"sheet_url": rel(paths[0])} if paths else {}), **out}
        case = Case(pet_id=target.id, turns=json.dumps([turn]),
                    recheck_due=store.now() + timedelta(hours=RECHECK_HOURS[out["result"]["plan"]["recheck"]]))
        s.add(case)
        den.log(s, den_id, "scan", f"📷 Live scan of {target.name}: {nice(out['result']['urgency'])}", pet_id=target.id)
        s.commit()
        s.refresh(case)
        s.refresh(target)
        den.on_case(case, target)
        return {"den_id": den_id, "pet_id": target.id, "case_id": case.id}


def media_kind(paths: list[str]) -> str:
    return "image" if len(paths) == 1 else "video" if paths else "text"


# ------------------------------------------------------------------ PawTown demo seed
@app.post("/demo/seed_den")
async def seed_den(with_triage: bool = True):
    """One Den, three caregivers, three pets, one scheduled med; optionally ONE real photo triage (≈$0.001, disk-cached)."""
    from starlette.datastructures import UploadFile as SUpload
    d = den.create_den(den.DenIn(name="The Valencia St Den", neighborhood="Mission"))["id"]
    mids = [den.add_member(d, den.MemberIn(name=n, role=r, emoji=e))["id"] for n, r, e in
            [("Alex", "parent", "🧔"), ("Sam", "coparent", "👩"), ("Jordan", "helper", "🧑‍🦱")]]
    pids = [create_pet(PetIn(**p))["id"] for p in json.loads((ROOT / "demo" / "demo_pets.json").read_text())]
    for pid in pids:
        den.link_pet(d, den.PetLink(pet_id=pid))
    den.add_med(d, den.MedIn(pet_id=pids[1], name="carprofen", every_hours=12, assignee_id=mids[2]))
    case = None
    img = ROOT / "demo" / "media" / "ringworm_cat.jpg"
    if with_triage and img.exists():
        up = SUpload(file=io.BytesIO(img.read_bytes()), filename="miso.jpg", headers={"content-type": "image/jpeg"})
        case = await triage(pet_id=pids[0], message="Miso has a round bald patch on her face that's getting bigger. She seems fine otherwise.",
                            model="auto", image=up, video=None)
    return {"den_id": d, "members": mids, "pets": pids, "case": case and case.get("id")}


# ------------------------------------------------------------------ side-by-side: untuned vs improved, same photo
@app.post("/compare")
async def compare(species: str = Form(...), message: str = Form(...), image: UploadFile | None = File(None)):
    """Two real calls on the same input: base GLM 5.3 Flash (untuned) vs the best available mode. ≈$0.002 total."""
    import asyncio
    if species not in ("dog", "cat"):
        raise HTTPException(400, "species must be dog or cat")
    media = save_media(MEDIA / f"cmp_{uuid.uuid4().hex[:8]}", image, None)

    class _P:
        def __init__(s): s.species = species
        def profile(s): return {"name": "", "breed": "", "age_years": 0, "weight_kg": 0, "meds": [], "conditions": []}
    base, best = await asyncio.gather(triage_once(_P(), message, media, "base"), triage_once(_P(), message, media, "auto"))
    keep = lambda o: {k: o[k] for k in ("result", "audit", "model", "mode", "cost_usd", "latency_ms", "raw", "parse_error") if k in o}  # noqa: E731
    return {"image_url": media.get("image_url"), "base": keep(base), "best": keep(best)}


@app.post("/demo/seed_walker")
def seed_walker():
    """A second Den (Noe Valley) where Jordan also walks dogs: shows one helper across several households. No model call."""
    d = den.create_den(den.DenIn(name="The Noe Valley Den", neighborhood="Dolores Park"))["id"]
    owner = den.add_member(d, den.MemberIn(name="Priya", role="parent", emoji="👩‍🦰"))["id"]
    walker_id = den.add_member(d, den.MemberIn(name="Jordan", role="helper", emoji="🧑‍🦱"))["id"]
    pid = create_pet(PetIn(name="Luna", species="dog", breed="border collie", age_years=2, weight_kg=17, meds=[], conditions=[]))["id"]
    den.link_pet(d, den.PetLink(pet_id=pid))
    den.add_med(d, den.MedIn(pet_id=pid, name="apoquel", every_hours=24, assignee_id=walker_id))
    return {"den_id": d, "members": [owner, walker_id], "pets": [pid]}


@app.post("/demo/reset")
def demo_reset():
    """Wipe the LOCAL demo database (pets, cases, Dens) so the demo starts clean. Media files are kept."""
    from sqlmodel import delete
    with store.session() as s:
        for model in (den.Event, den.Quest, den.DenPet, den.Member, den.Den, Outcome, Case, Pet):
            s.exec(delete(model))
        s.commit()
    return {"ok": True}
