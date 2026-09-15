import os
import sys
import json
from typing import List, Dict, Any, Optional

# Ensure local src is importable when run from repo root
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from rubric import heuristic_score, to_csv_row, append_csv

ALL_HISTORY_FILE = os.path.abspath(os.path.join(HERE, "..", "all_sessions_history.json"))
DEFAULT_OUT = os.path.abspath(os.path.join(HERE, "..", "llm_responses.csv"))


def load_all_sessions(path: str) -> List[List[Dict[str, Any]]]:
    if not os.path.exists(path):
        print(f"[ERROR] History file not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                print("[ERROR] History file root is not a list")
                return []
            return data
        except Exception as e:
            print(f"[ERROR] Failed to read history: {e}")
            return []


def context_from_window(window: List[Dict[str, Any]], max_turns: int = 6) -> str:
    # Build a compact context string from the last few robot/user turns
    parts: List[str] = []
    for ev in window[-max_turns:]:
        role = ev.get("role")
        t = ev.get("type")
        text = ev.get("text")
        if not text:
            continue
        if role == "robot" and t in {"say", "ask_open", "ask_yesno", "ask_options", "personalize"}:
            parts.append(f"Robot: {text}")
        elif role == "user" and t.startswith("answer_"):
            parts.append(f"Child: {text}")
    return "\n".join(parts)


def export_from_history(history: List[List[Dict[str, Any]]], out_csv: str) -> int:
    rows: List[Dict[str, Any]] = []
    total = 0
    for session_idx, session in enumerate(history):
        current_dialog_id: Optional[str] = None
        last_question: Optional[str] = None
        last_user_open: Optional[str] = None
        turn_index = 0
        # Slide window for context
        window: List[Dict[str, Any]] = []
        for ev in session:
            window.append(ev)
            if ev.get("type") == "dialog_start":
                current_dialog_id = ev.get("dialog_id")
                continue
            if ev.get("type") == "dialog_end":
                current_dialog_id = None
                continue
            role = ev.get("role")
            t = ev.get("type")
            if role == "robot" and t == "ask_open":
                last_question = ev.get("text")
            if role == "user" and t == "answer_open":
                last_user_open = ev.get("text")
            # Capture LLM-generated follow-ups recorded by runtime as 'personalize'
            if role == "robot" and t == "personalize":
                llm_resp = ev.get("text") or ""
                user_utt = last_user_open or ""
                robot_q = ev.get("source_question") or last_question or ""
                dialog_context = context_from_window(window)
                # Heuristic rubric score
                scores = heuristic_score(user_utt, llm_resp, dialog_context=dialog_context)
                dialog_id = current_dialog_id or ev.get("dialog_id") or f"session{session_idx}"
                row = to_csv_row(
                    dialog_id=dialog_id,
                    turn_index=turn_index,
                    user_utterance=user_utt,
                    dialog_context=f"{robot_q}\n{dialog_context}".strip(),
                    llm_response=llm_resp,
                    scores=scores,
                    reference_response="",
                    meta={"session_index": session_idx},
                )
                rows.append(row)
                turn_index += 1
                total += 1
    if rows:
        append_csv(out_csv, rows)
    print(f"[INFO] Exported {total} LLM rows to {out_csv}")
    return total


def main():
    # Args: [history_json] [out_csv]
    hist_path = sys.argv[1] if len(sys.argv) > 1 else ALL_HISTORY_FILE
    out_csv = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    sessions = load_all_sessions(hist_path)
    if not sessions:
        sys.exit(1)
    export_from_history(sessions, out_csv)


if __name__ == "__main__":
    main()
