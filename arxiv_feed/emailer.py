"""Composing and sending the daily email. The email is the whole interface.

Part 1 is every question from every paper in one list -- the part you actually
read. Part 2 is the papers, so a question that catches your eye can be traced
back to its source, its nearest anchor, and its scores.
"""

from __future__ import annotations

import html
import logging
import smtplib
from email.message import EmailMessage

from .guard import safe_header_value

log = logging.getLogger(__name__)


class EmailError(Exception):
    """The email could not be sent. The JSON file is still on disk."""


def _mark(entry: dict) -> str:
    return " [random]" if entry.get("from_random") else ""


def render_text(result: dict) -> str:
    papers = result["papers"]
    lines: list[str] = []
    day = result["date"]

    lines.append(f"arXiv open questions -- {day}")
    lines.append("=" * 60)
    counts = result["counts"]
    lines.append(
        f"{counts['fetched']} new papers, {counts['unseen']} unseen, "
        f"{counts['shortlisted']} shortlisted, {counts['kept']} kept."
    )
    lines.append("")

    # ---- Part 1: every question, in one list -------------------------------
    approachable = sum(
        1 for p in papers for q in p["open_questions"] if q["label"] == "approachable"
    )
    total_q = sum(len(p["open_questions"]) for p in papers)
    lines.append(f"PART 1 -- OPEN QUESTIONS ({total_q}, {approachable} approachable)")
    lines.append("-" * 60)
    n = 0
    for p in papers:
        for q in p["open_questions"]:
            n += 1
            flag = "[approachable]" if q["label"] == "approachable" else "[not approachable]"
            lines.append(f"{n:>3}. {flag} {q['question']}")
            lines.append(f"     why: {q['reason']}")
            lines.append(f"     from: {p['title']} ({p['arxiv_id']})")
            lines.append("")
    if n == 0:
        lines.append("No questions extracted today.")
        lines.append("")

    # ---- Part 2: the papers ------------------------------------------------
    lines.append(f"PART 2 -- PAPERS ({len(papers)})")
    lines.append("-" * 60)
    for i, p in enumerate(papers, 1):
        lines.append(f"{i}. {p['title']}{_mark(p)}")
        lines.append(f"   {', '.join(p['authors'])}")
        lines.append(f"   {p['arxiv_id']} -- {p['url']}")
        lines.append(
            f"   similarity {p['similarity']:.3f} to {p['nearest_anchor_id']} "
            f"({p['nearest_anchor_title']})"
        )
        lines.append(f"   significance {p.get('significance', '?')}/5, "
                     f"novelty {p.get('novelty', '?')}/5")
        if p.get("suspicious_markers"):
            lines.append(f"   note: text contains injection-like patterns "
                         f"({', '.join(p['suspicious_markers'])}) -- advisory, the "
                         f"paper may simply be about prompt injection")
        if p.get("one_sentence"):
            lines.append(f"   {p['one_sentence']}")
        if p["open_questions"]:
            lines.append("   open questions:")
            for q in p["open_questions"]:
                flag = "approachable" if q["label"] == "approachable" else "not approachable"
                lines.append(f"     - {q['question']}")
                lines.append(f"       {flag}: {q['reason']}")
        else:
            lines.append("   open questions: none extracted")
        lines.append("")

    # ---- Failures, never silent -------------------------------------------
    if result.get("problems"):
        lines.append("PROBLEMS")
        lines.append("-" * 60)
        for prob in result["problems"]:
            lines.append(f"- {prob}")
        lines.append("")

    lines.append(f"Filter: top {result['config']['shortlist_n']} by similarity to "
                 f"{result['counts']['anchors']} anchors, plus "
                 f"{result['config']['explore_n']} at random; "
                 f"top {result['config']['top_n']} after scoring.")
    return "\n".join(lines)


def render_html(result: dict) -> str:
    e = html.escape
    papers = result["papers"]
    out: list[str] = [
        "<html><body style=\"font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;"
        "line-height:1.45;color:#111;max-width:46em\">",
        f"<h1 style=\"font-size:1.3em\">arXiv open questions — {e(result['date'])}</h1>",
    ]
    c = result["counts"]
    out.append(
        f"<p style=\"color:#555\">{c['fetched']} new papers, {c['unseen']} unseen, "
        f"{c['shortlisted']} shortlisted, {c['kept']} kept.</p>"
    )

    total_q = sum(len(p["open_questions"]) for p in papers)
    approachable = sum(
        1 for p in papers for q in p["open_questions"] if q["label"] == "approachable"
    )
    out.append(f"<h2 style=\"font-size:1.1em\">Part 1 — open questions "
               f"({total_q}, {approachable} approachable)</h2><ol>")
    for p in papers:
        for q in p["open_questions"]:
            colour = "#0a7d33" if q["label"] == "approachable" else "#8a8a8a"
            out.append(
                f"<li style=\"margin-bottom:.7em\">{e(q['question'])}<br>"
                f"<span style=\"color:{colour};font-weight:600\">{e(q['label'])}</span>"
                f"<span style=\"color:#555\"> — {e(q['reason'])}</span><br>"
                f"<span style=\"color:#777;font-size:.9em\">from "
                f"<a href=\"{e(p['url'])}\">{e(p['title'])}</a></span></li>"
            )
    out.append("</ol>")
    if total_q == 0:
        out.append("<p>No questions extracted today.</p>")

    out.append(f"<h2 style=\"font-size:1.1em\">Part 2 — papers ({len(papers)})</h2>")
    for p in papers:
        badge = (
            "<span style=\"background:#eee;padding:1px 5px;border-radius:3px;"
            "font-size:.85em\">random</span> " if p.get("from_random") else ""
        )
        out.append(
            f"<div style=\"margin-bottom:1.4em\">"
            f"<div style=\"font-weight:600\">{badge}"
            f"<a href=\"{e(p['url'])}\">{e(p['title'])}</a></div>"
            f"<div style=\"color:#555;font-size:.92em\">{e(', '.join(p['authors']))}</div>"
            f"<div style=\"color:#555;font-size:.92em\">{e(p['arxiv_id'])} · "
            f"similarity {p['similarity']:.3f} to "
            f"<a href=\"https://arxiv.org/abs/{e(p['nearest_anchor_id'])}\">"
            f"{e(p['nearest_anchor_title'])}</a> · "
            f"significance {p.get('significance', '?')}/5 · "
            f"novelty {p.get('novelty', '?')}/5</div>"
        )
        if p.get("suspicious_markers"):
            out.append(
                f"<div style=\"color:#8a6d00;font-size:.9em\">note: injection-like "
                f"patterns in the text ({e(', '.join(p['suspicious_markers']))}) — "
                f"advisory; the paper may simply be about prompt injection</div>"
            )
        if p.get("one_sentence"):
            out.append(f"<div style=\"margin:.3em 0\">{e(p['one_sentence'])}</div>")
        if p["open_questions"]:
            out.append("<ul style=\"margin:.3em 0\">")
            for q in p["open_questions"]:
                colour = "#0a7d33" if q["label"] == "approachable" else "#8a8a8a"
                out.append(
                    f"<li>{e(q['question'])} "
                    f"<span style=\"color:{colour}\">({e(q['label'])}: {e(q['reason'])})</span></li>"
                )
            out.append("</ul>")
        else:
            out.append("<div style=\"color:#777\">no questions extracted</div>")
        out.append("</div>")

    if result.get("problems"):
        out.append("<h2 style=\"font-size:1.1em\">Problems</h2><ul>")
        for prob in result["problems"]:
            out.append(f"<li style=\"color:#a33\">{e(str(prob))}</li>")
        out.append("</ul>")

    out.append("</body></html>")
    return "".join(out)


def build_message(result: dict, cfg) -> EmailMessage:
    msg = EmailMessage()
    n = sum(len(p["open_questions"]) for p in result["papers"])
    msg["Subject"] = (
        f"arXiv open questions — {result['date']} — "
        f"{len(result['papers'])} papers, {n} questions"
    )
    msg["From"] = safe_header_value(cfg.email_from())
    msg["To"] = safe_header_value(cfg.email_to())
    msg.set_content(render_text(result))
    msg.add_alternative(render_html(result), subtype="html")
    return msg


def send(result: dict, cfg) -> None:
    """Send over SMTP with STARTTLS. Raises EmailError; the caller keeps the JSON."""
    if not cfg.email_to():
        raise EmailError("FEED_EMAIL_TO is not set")
    password = cfg.smtp_password()
    if not password:
        raise EmailError("SMTP_PASSWORD is not set")

    msg = build_message(result, cfg)
    try:
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=60) as smtp:
            smtp.starttls()
            smtp.login(cfg.smtp_user(), password)
            smtp.send_message(msg)
    except Exception as exc:
        # The address is deliberately absent from the message: this string ends
        # up in logs and in the committed archive.
        raise EmailError(f"send failed: {exc}") from exc
    log.info("email sent")
