from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple
import csv
import json
import re
import statistics
import os

# Rubric dimensions (1–5, integers recommended)
DIMENSIONS: Tuple[str, ...] = (
    "quality",       # fluency, grammar
    "relevance",     # addresses user and dialog goal
    "consistency",   # no contradictions with prior context/variables
    "grounding",     # uses available context; avoids hallucinations
    "coherence",     # fits tone and flows naturally
    "specificity",   # avoids vague filler; gives concrete detail
    "safety",        # appropriate, no disallowed content
)

# Default weights for overall score (sums to 1.0)
DEFAULT_WEIGHTS: Dict[str, float] = {
    "quality": 0.15,
    "relevance": 0.20,
    "consistency": 0.20,
    "grounding": 0.15,
    "coherence": 0.15,
    "specificity": 0.10,
    "safety": 0.05,
}

@dataclass
class RubricScores:
    quality: int
    relevance: int
    consistency: int
    grounding: int
    coherence: int
    specificity: int
    safety: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def overall(self, weights: Optional[Dict[str, float]] = None) -> float:
        w = weights or DEFAULT_WEIGHTS
        total = 0.0
        for k, v in self.to_dict().items():
            total += float(v) * float(w.get(k, 0.0))
        return round(total, 3)

def clamp_1_5(x: Any) -> int:
    try:
        v = int(round(float(x)))
    except Exception:
        v = 3
    return max(1, min(5, v))

def validate(scores: RubricScores) -> None:
    for k, v in scores.to_dict().items():
        iv = clamp_1_5(v)
        if iv != v:
            raise ValueError(f"Invalid score for {k}: {v} (must be 1–5 integer)")

def aggregate(scores_list: List[RubricScores]) -> Dict[str, Any]:
    """Compute mean, median, stdev for each dimension and overall."""
    if not scores_list:
        return {}
    by_dim: Dict[str, List[float]] = {d: [] for d in DIMENSIONS}
    overalls: List[float] = []
    for s in scores_list:
        for d, v in s.to_dict().items():
            by_dim[d].append(float(v))
        overalls.append(s.overall())
    summary = {}
    for d, vals in by_dim.items():
        summary[d] = {
            "mean": round(statistics.mean(vals), 3),
            "median": round(statistics.median(vals), 3),
            "stdev": round(statistics.pstdev(vals), 3) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }
    summary["overall"] = {
        "mean": round(statistics.mean(overalls), 3),
        "median": round(statistics.median(overalls), 3),
        "stdev": round(statistics.pstdev(overalls), 3) if len(overalls) > 1 else 0.0,
        "n": len(overalls),
    }
    return summary

def unresolved_placeholders(text: str) -> int:
    """Counts unresolved %var% or $var placeholders."""
    if not text:
        return 0
    return len(re.findall(r"(%[^%]+%|\$[A-Za-z_][A-Za-z0-9_]*)", text))

def token_overlap(a: str, b: str) -> float:
    """Simple Jaccard overlap over word tokens."""
    aw = set(re.findall(r"[A-Za-z]+", (a or "").lower()))
    bw = set(re.findall(r"[A-Za-z]+", (b or "").lower()))
    if not aw or not bw:
        return 0.0
    inter = len(aw & bw)
    union = len(aw | bw)
    return inter / max(1, union)

def repetition_score(text: str) -> float:
    """Penalize obvious repetition (higher = more repetition)."""
    words = re.findall(r"[A-Za-z]+", (text or "").lower())
    if not words:
        return 0.0
    repeats = sum(1 for i in range(1, len(words)) if words[i] == words[i - 1])
    return repeats / len(words)

def heuristic_score(user_utterance: str,
                    llm_response: str,
                    dialog_context: str = "",
                    user_model: Optional[Dict[str, Any]] = None) -> RubricScores:
    """
    Lightweight, deterministic scoring without external models.
    Heuristics:
    - length bounds (<= 25 words preferred),
    - placeholder resolution,
    - overlap with user utterance and context,
    - repetition penalty,
    - safety via crude keyword scan (very conservative).
    """
    resp = (llm_response or "").strip()
    u = (user_utterance or "").strip()
    ctx = (dialog_context or "").strip()

    words = re.findall(r"[A-Za-z]+", resp)
    n_words = len(words)
    has_placeholder = unresolved_placeholders(resp) > 0
    ov_user = token_overlap(resp, u)
    ov_ctx = token_overlap(resp, ctx)
    rep = repetition_score(resp)

    # Quality: length + repetition + placeholders
    quality = 5
    if n_words == 0:
        quality = 1
    elif n_words > 28:
        quality -= 1
    if rep > 0.02:
        quality -= 1
    if has_placeholder:
        quality -= 1
    quality = clamp_1_5(quality)

    # Relevance: overlap with user utterance (primary) and context (secondary)
    relevance = 1 if n_words == 0 else 3
    if ov_user >= 0.15:
        relevance += 1
    if ov_ctx >= 0.10:
        relevance += 1
    relevance = clamp_1_5(relevance)

    # Consistency: penalize contradictions with simple negation heuristic + placeholders
    contradiction = bool(re.search(r"\b(i\s+did\s+not|i\s+don.?t|no,\s*i)\b", resp, flags=re.I))
    consistency = 5
    if contradiction:
        consistency -= 1
    if has_placeholder:
        consistency -= 1
    consistency = clamp_1_5(consistency)

    # Grounding: use of context terms; penalize placeholders
    grounding = 1 if n_words == 0 else 2
    if ov_ctx >= 0.12:
        grounding += 2
    elif ov_ctx >= 0.05:
        grounding += 1
    if has_placeholder:
        grounding -= 1
    grounding = clamp_1_5(grounding)

    # Coherence: length bounds and repetition
    coherence = 5
    if n_words < 3:
        coherence -= 1
    if n_words > 30:
        coherence -= 1
    if rep > 0.03:
        coherence -= 1
    coherence = clamp_1_5(coherence)

    # Specificity: reward overlap (mentions concrete terms)
    specificity = 2
    if ov_user >= 0.10:
        specificity += 1
    if ov_ctx >= 0.10:
        specificity += 1
    if n_words >= 6:
        specificity += 1
    specificity = clamp_1_5(specificity)

    # Safety: crude keyword scan
    unsafe = bool(re.search(r"\b(suicide|self-harm|kill|violence|sex|drugs)\b", resp, flags=re.I))
    safety = 5 if not unsafe else 2

    return RubricScores(
        quality=quality,
        relevance=relevance,
        consistency=consistency,
        grounding=grounding,
        coherence=coherence,
        specificity=specificity,
        safety=clamp_1_5(safety),
    )

def llm_rater_prompt(dialog_context: str,
                     user_utterance: str,
                     llm_response: str,
                     reference_response: Optional[str] = None) -> str:
    """Prompt for an LLM-as-rater that returns JSON scores 1–5."""
    ref_txt = reference_response or ""
    return (
        "You are evaluating a dialog system response. Rate 1–5 for each dimension.\n"
        f"Context: {dialog_context}\n"
        f"User utterance: {user_utterance}\n"
        f"System response: {llm_response}\n"
        f"Reference (optional): {ref_txt}\n"
        "Return ONLY compact JSON like: "
        '{"quality":Q,"relevance":R,"consistency":C,"grounding":G,"coherence":H,"specificity":S,"safety":F}\n'
        "Use integers 1–5 only."
    )

def parse_llm_rater_output(text: str) -> RubricScores:
    """Parse JSON from an LLM rater into RubricScores (clamped 1–5)."""
    data = json.loads((text or "").strip())
    scores = {k: clamp_1_5(data.get(k, 3)) for k in DIMENSIONS}
    return RubricScores(**scores)

def to_csv_row(dialog_id: str,
               turn_index: int,
               user_utterance: str,
               dialog_context: str,
               llm_response: str,
               scores: RubricScores,
               reference_response: Optional[str] = None,
               meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    row = {
        "dialog_id": dialog_id,
        "turn_index": turn_index,
        "user_utterance": user_utterance,
        "dialog_context": dialog_context,
        "llm_response": llm_response,
        "reference_response": reference_response or "",
        "overall": scores.overall(),
    }
    row.update(scores.to_dict())
    if meta:
        for k, v in meta.items():
            row[f"meta_{k}"] = v
    return row

def append_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """Append rows to CSV, creating header on first write."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    try:
        with open(path, "x", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except FileExistsError:
        with open(path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(rows)

def score_session_history(session_history: List[Dict[str, Any]],
                         session_id: str = "",
                         run_id: str = "",
                         participant_id: str = "") -> List[Dict[str, Any]]:
    """
    Score personalize turns from one session history and return CSV-ready rows.

    Expects event patterns produced by mini_dialogs.py:
    - robot ask_* move
    - user answer_* move
    - robot personalize move
    """
    rows: List[Dict[str, Any]] = []
    if not session_history:
        return rows

    last_robot_question = ""
    last_user_answer = ""
    current_dialog_id = ""
    turn_index = 0

    for event in session_history:
        role = str(event.get("role", "")).lower()
        etype = str(event.get("type", "")).lower()
        text = str(event.get("text", "") or "")

        if role == "system" and etype == "dialog_start":
            current_dialog_id = str(event.get("dialog_id", "") or "")
            continue

        if role == "robot" and etype in {"ask_open", "ask_yesno", "ask_options"}:
            last_robot_question = text
            continue

        if role == "user" and etype in {"answer_open", "answer_yesno", "answer_options"}:
            last_user_answer = text
            continue

        if role == "robot" and etype == "personalize":
            turn_index += 1
            context = str(event.get("source_question") or last_robot_question or "")
            user_utt = last_user_answer
            llm_resp = text
            scores = heuristic_score(user_utterance=user_utt, llm_response=llm_resp, dialog_context=context)
            dialog_id = current_dialog_id or "unknown_dialog"
            row = to_csv_row(
                dialog_id=dialog_id,
                turn_index=turn_index,
                user_utterance=user_utt,
                dialog_context=context,
                llm_response=llm_resp,
                scores=scores,
                meta={
                    "session_id": session_id,
                    "run_id": run_id,
                    "participant_id": participant_id,
                    "event_type": etype,
                },
            )
            rows.append(row)

    return rows

def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate rubric rows into per-dimension and overall summary stats."""
    if not rows:
        return {"n": 0, "overall": {"mean": 0.0, "median": 0.0, "stdev": 0.0, "n": 0}}
    scores_list: List[RubricScores] = []
    for r in rows:
        scores_list.append(
            RubricScores(
                quality=clamp_1_5(r.get("quality", 3)),
                relevance=clamp_1_5(r.get("relevance", 3)),
                consistency=clamp_1_5(r.get("consistency", 3)),
                grounding=clamp_1_5(r.get("grounding", 3)),
                coherence=clamp_1_5(r.get("coherence", 3)),
                specificity=clamp_1_5(r.get("specificity", 3)),
                safety=clamp_1_5(r.get("safety", 3)),
            )
        )
    summary = aggregate(scores_list)
    summary["n"] = len(rows)
    return summary

def evaluate_history_file(history_path: str,
                          output_csv_path: str,
                          output_summary_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluate all sessions from all_sessions_history.json and save rubric results.

    Returns a small report dict with counts and summary.
    """
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"History file not found: {history_path}")

    with open(history_path, "r", encoding="utf-8") as f:
        data = json.load(f) or []

    if not isinstance(data, list):
        raise ValueError("Expected history JSON to be a list of sessions")

    all_rows: List[Dict[str, Any]] = []
    for idx, sess in enumerate(data, start=1):
        if not isinstance(sess, list):
            continue
        sid = f"legacy_session_{idx:04d}"
        rows = score_session_history(sess, session_id=sid)
        all_rows.extend(rows)

    if all_rows:
        folder = os.path.dirname(output_csv_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        append_csv(output_csv_path, all_rows)

    summary = summarize_rows(all_rows)
    report = {
        "history_path": history_path,
        "output_csv_path": output_csv_path,
        "rows_written": len(all_rows),
        "summary": summary,
    }

    if output_summary_path:
        folder = os.path.dirname(output_summary_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(output_summary_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return report

# Example (commented):
# if __name__ == "__main__":
#     ctx = "Robot: What's something fun you did today?"
#     user = "I went to a coffee place."
#     resp = "That sounds nice! What did you have there?"
#     s = heuristic_score(user, resp, dialog_context=ctx)
#     row = to_csv_row("personalization_test", 3, user, ctx, resp, s, reference_response="Nice! What drink did you try?")
#     append_csv("llm_responses.csv", [row])
#     print(s, s.overall(), "-> saved to llm_responses.csv")
