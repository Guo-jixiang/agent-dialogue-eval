from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from core.models import AggregatedReport, SimulatedDialogue

TEMPLATES_DIR = Path(__file__).parent / "templates"


class ReportGenerator:
    def __init__(self, output_dir: Path | str = "reports"):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._jinja = Environment(
            loader=FileSystemLoader(str(TEMPLATES_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def generate(
        self,
        report: AggregatedReport,
        dialogues: list[SimulatedDialogue],
    ) -> dict[str, Path]:
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dialogues_map = {d.id: d for d in dialogues}

        ctx = {
            "report": report,
            "dialogues_map": dialogues_map,
            "generated_at": generated_at,
        }

        paths: dict[str, Path] = {}

        # Markdown
        md_path = self._output_dir / f"report_{report.id}.md"
        md_template = self._jinja.get_template("report.md.j2")
        md_path.write_text(md_template.render(**ctx), encoding="utf-8")
        paths["md"] = md_path

        # HTML
        html_path = self._output_dir / f"report_{report.id}.html"
        html_template = self._jinja.get_template("report.html.j2")
        html_path.write_text(html_template.render(**ctx), encoding="utf-8")
        paths["html"] = html_path

        # JSON — embed dialogue turns so the UI can show transcript
        report_dict = report.model_dump()
        # Attach turns to each dialogue_report entry
        for dr in report_dict.get("dialogue_reports", []):
            dia = dialogues_map.get(dr.get("dialogue_id", ""))
            if dia:
                dr["dialogue"] = [
                    {"role": t.role, "content": t.content, "turn_id": t.turn_id}
                    for t in dia.turns
                ]
        json_path = self._output_dir / f"report_{report.id}.json"
        json_path.write_text(
            json.dumps(report_dict, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        paths["json"] = json_path

        return paths
