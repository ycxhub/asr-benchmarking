from __future__ import annotations

import html
import json


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value) -> str:
    if value is None:
        return "—"
    return f"{100 * value:.1f}%"


def _ci(block: dict | None) -> str:
    if not block or block.get("low") is None:
        return "—"
    return f"{_pct(block.get('estimate'))} [{_pct(block.get('low'))}, {_pct(block.get('high'))}]"


def render_report(summary: dict, utterances: list[dict]) -> str:
    rows = []
    points = []
    for p in summary.get("providers") or []:
        wer = (p.get("wer") or {}).get("wer")
        lat = ((p.get("latency") or {}).get("eos_to_final_ms") or {})
        block = p.get("wer") or {}
        n_clips = block.get("n_clips") or 0
        fail_share = (block.get("n_failed") or 0) / n_clips if n_clips else None
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(p['provider']))}</td>"
            f"<td>{_pct(wer)}</td>"
            f"<td>{_ci(p.get('wer_ci95'))}</td>"
            f"<td>{_pct(p.get('wer_duration_weighted'))}</td>"
            f"<td>{_fmt(lat.get('p50'), 0)}</td>"
            f"<td>{_fmt(lat.get('p90'), 0)}</td>"
            f"<td>{_fmt(lat.get('p95'), 0)}</td>"
            f"<td>{_pct((p.get('latency') or {}).get('error_rate'))}</td>"
            f"<td>{_pct(fail_share)}</td>"
            "</tr>"
        )
        if wer is not None and lat.get("p50") is not None:
            points.append({"provider": p["provider"], "wer": wer, "latency": lat.get("p50")})

    paired_rows = []
    for row in summary.get("paired_vs_murf") or []:
        paired_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['other']))}</td>"
            f"<td>{_pct(row.get('diff'))}</td>"
            f"<td>{_pct(row.get('low'))} to {_pct(row.get('high'))}</td>"
            f"<td>{_fmt(row.get('p_two_sided'), 3)}</td>"
            f"<td>{row.get('n')}</td>"
            "</tr>"
        )

    def slice_table(title: str, rows_in: list[dict]) -> str:
        if not rows_in:
            return ""
        body = "".join(
            "<tr>"
            f"<td>{html.escape(str(r['provider']))}</td>"
            f"<td>{html.escape(str(r['bucket']))}</td>"
            f"<td>{r['n']}</td>"
            f"<td>{_pct(r.get('wer'))}</td>"
            "</tr>"
            for r in rows_in
        )
        return (
            f"<h2>{html.escape(title)}</h2>"
            "<table><thead><tr><th>Provider</th><th>Bucket</th><th>Clips</th><th>WER</th></tr></thead>"
            f"<tbody>{body}</tbody></table>"
        )

    worst = []
    for utt in (summary.get("worst") or [])[:50]:
        worst.append(
            "<article class='utt'>"
            f"<header><strong>{html.escape(str(utt['provider']))}</strong> "
            f"{html.escape(str(utt['item_id']))} · WER {_pct(utt.get('wer'))}</header>"
            f"<p><span>ref</span> {html.escape(utt.get('ref_norm') or '')}</p>"
            f"<p><span>hyp</span> {html.escape(utt.get('hyp_norm') or '')}</p>"
            + (f"<p class='err'>{html.escape(str(utt.get('error')))}</p>" if utt.get("error") else "")
            + "</article>"
        )

    payload = json.dumps(points)
    title = f"{summary.get('dataset')} · {summary.get('mode')} · {summary.get('condition')}"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>asrbench {html.escape(str(summary.get('run_id')))}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ font: 15px/1.45 -apple-system, BlinkMacSystemFont, sans-serif; margin: 2rem; color: #1c1917; }}
  h1 {{ font-size: 1.4rem; }}
  table {{ border-collapse: collapse; margin: 1rem 0 2rem; }}
  th, td {{ border-bottom: 1px solid #e7e5e4; text-align: left; padding: 0.35rem 0.8rem; }}
  .utt {{ border-top: 1px solid #e7e5e4; padding: 0.6rem 0; }}
  .utt span {{ display: inline-block; width: 2.2rem; color: #78716c; }}
  .err {{ color: #b91c1c; }}
  #scatter {{ max-width: 720px; height: 420px; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<p>Run <code>{html.escape(str(summary.get('run_id')))}</code> · {summary.get('n_rows')} sessions · language {html.escape(str(summary.get('lang')))}</p>
<h2>Leaderboard</h2>
<table>
<thead><tr><th>Provider</th><th>WER</th><th>WER 95% CI</th><th>Duration-weighted</th><th>EOS→final p50 ms</th><th>p90</th><th>p95</th><th>Error rate</th><th>Failure share</th></tr></thead>
<tbody>{''.join(rows) or '<tr><td colspan="9">No scored sessions</td></tr>'}</tbody>
</table>
<div id="scatter"></div>
<h2>Paired difference vs Murf</h2>
<p>Difference is other WER minus Murf WER on clips scored for both. Positive means the other model has the higher error.</p>
<table>
<thead><tr><th>Provider</th><th>Δ WER</th><th>95% CI</th><th>Two-sided p</th><th>Paired clips</th></tr></thead>
<tbody>{''.join(paired_rows) or '<tr><td colspan="5">Murf was not in this run</td></tr>'}</tbody>
</table>
{slice_table('Duration slices', (summary.get('slices') or {}).get('duration') or [])}
{slice_table('Code-mix slices', (summary.get('slices') or {}).get('cmi') or [])}
<h2>Highest-error clips</h2>
{''.join(worst) or '<p>No utterances.</p>'}
<script>
const points = {payload};
if (points.length) {{
  Plotly.newPlot('scatter', [{{
    x: points.map(p => p.latency),
    y: points.map(p => p.wer),
    text: points.map(p => p.provider),
    mode: 'markers+text',
    textposition: 'top center',
    marker: {{ size: 12 }}
  }}], {{
    margin: {{ t: 24 }},
    xaxis: {{ title: 'End-of-speech to final, p50 (ms)' }},
    yaxis: {{ title: 'WER', tickformat: '.0%' }}
  }}, {{ displayModeBar: false }});
}}
</script>
</body>
</html>
"""
