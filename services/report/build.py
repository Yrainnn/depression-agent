from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List

from jinja2 import Environment, select_autoescape
from weasyprint import HTML


def build_report(
    session_id: str,
    summary: str,
    scores: List[Dict[str, object]],
    transcripts: List[Dict[str, object]],
) -> str:
    """Render a clinical summary PDF and return a file:// URI."""

    env = Environment(autoescape=select_autoescape(["html", "xml"]))
    template = env.from_string(
        """
        <html>
        <head>
            <meta charset="utf-8" />
            <style>
                body { font-family: 'Noto Sans', sans-serif; padding: 24px; }
                h1 { color: #2c3e50; }
                table { width: 100%; border-collapse: collapse; margin-top: 16px; }
                th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
                th { background: #f7f7f7; }
                .section { margin-top: 24px; }
            </style>
        </head>
        <body>
            <h1>访谈报告 - {{ session_id }}</h1>
            <div class="section">
                <h2>概要</h2>
                <p>{{ summary }}</p>
            </div>
            <div class="section">
                <h2>量表评分</h2>
                <table>
                    <thead>
                        <tr><th>项目</th><th>分数</th><th>证据</th></tr>
                    </thead>
                    <tbody>
                        {% for item in scores %}
                            <tr>
                                <td>{{ item.name }}</td>
                                <td>{{ item.score }}</td>
                                <td>{{ item.evidence_refs | join(', ') }}</td>
                            </tr>
                        {% else %}
                            <tr><td colspan="3">暂无评分数据</td></tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            <div class="section">
                <h2>对话记录</h2>
                <ul>
                    {% for seg in transcripts %}
                        <li><strong>{{ seg.get('utt_id', loop.index) }}</strong>：{{ seg.get('text', '') }}</li>
                    {% else %}
                        <li>暂无记录</li>
                    {% endfor %}
                </ul>
            </div>
        </body>
        </html>
        """
    )

    html = template.render(session_id=session_id, summary=summary, scores=scores, transcripts=transcripts)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        HTML(string=html).write_pdf(temp_file.name)
        return Path(temp_file.name).resolve().as_uri()
