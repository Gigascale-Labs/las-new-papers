"""Building each day's content as one HTML fragment.

Read by data/feed.xml, which wraps this in <content type="html"> for each
entry -- the only place this HTML is read. web/index.html renders the same
papers straight from the JSON, in the same shape, so nothing here should say
more about a paper than the page does.

Nothing here justifies a paper's presence. No score, no similarity value, no
screening reason, and no label or reason on a question -- just the question
itself, quoted. The title, the one-sentence summary and the questions are the
whole case for reading it.
"""

from __future__ import annotations

import html


def render_body_html(result: dict) -> str:
    """The content only, no <html>/<body> tags."""
    e = html.escape
    papers = result["papers"]
    out: list[str] = [
        f"<h1 style=\"font-size:1.3em\">arXiv open questions — {e(result['date'])}</h1>",
    ]
    c = result["counts"]
    out.append(
        f"<p style=\"color:#555\">{c['fetched']} new papers, {c['unseen']} unseen, "
        f"{c.get('screened', 0)} screened, {c.get('relevant', 0)} relevant, "
        f"{c['kept']} kept.</p>"
    )

    if not papers:
        out.append(
            "<p>Nothing passed the screen on this day. "
            "That is a normal outcome, not a failure.</p>"
        )

    for p in papers:
        out.append(
            f"<div style=\"margin-bottom:1.4em\">"
            f"<div style=\"font-weight:600\">"
            f"<a href=\"{e(p['url'])}\">{e(p['title'])}</a></div>"
            f"<div style=\"color:#555;font-size:.92em\">{e(', '.join(p['authors']))}</div>"
            f"<div style=\"color:#555;font-size:.92em\">{e(p['arxiv_id'])}"
            + (
                f" · nearest in the canon: "
                f"<a href=\"https://arxiv.org/abs/{e(p['nearest_anchor_id'])}\">"
                f"{e(p['nearest_anchor_title'])}</a>"
                if p.get("nearest_anchor_title") else ""
            )
            + "</div>"
        )
        if p.get("one_sentence"):
            out.append(f"<div style=\"margin:.3em 0\">{e(p['one_sentence'])}</div>")
        if p.get("open_questions"):
            out.append("<ul style=\"margin:.3em 0;padding-left:1.2em\">")
            for q in p["open_questions"]:
                out.append(f"<li>{e(q['question'])}</li>")
            out.append("</ul>")
        out.append("</div>")

    if result.get("problems"):
        out.append("<h2 style=\"font-size:1.1em\">Problems</h2><ul>")
        for prob in result["problems"]:
            out.append(f"<li style=\"color:#a33\">{e(str(prob))}</li>")
        out.append("</ul>")

    return "".join(out)


__all__ = ["render_body_html"]
