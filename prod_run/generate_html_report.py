"""
Generate a standalone HTML report for match-result predictions.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd


def _format_value_table(df: pd.DataFrame) -> str:
	if df.empty:
		return "<p>No positive EV result bets found for this period.</p>"
	columns = ["Date", "Time", "League", "Home", "Away", "Result_Value_Side", "Result_EV"]
	return df[columns].to_html(index=False, classes="value-table")


def generate_html_report(predictions_df: pd.DataFrame, output_path: Path) -> None:
	"""Write a styled HTML summary of match-result predictions."""

	today_str = datetime.now().strftime("%Y-%m-%d")
	value_df = predictions_df[predictions_df["Result_EV"].notna()].copy()
	predictions_html = predictions_df.to_html(index=False, classes="predictions-table")
	value_html = _format_value_table(value_df)
	html = f"""<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8">
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<title>Football Predictions - {today_str}</title>
	<style>
		body {{ font-family: Arial, sans-serif; margin: 0; padding: 24px; background: #f5f7fb; color: #1f2937; }}
		.container {{ max-width: 1400px; margin: 0 auto; }}
		.card {{ background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08); }}
		h1, h2 {{ margin-top: 0; }}
		.subtitle {{ color: #64748b; }}
		table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
		th, td {{ border: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; }}
		th {{ background: #0f172a; color: white; position: sticky; top: 0; }}
		tr:nth-child(even) {{ background: #f8fafc; }}
		.table-wrap {{ overflow-x: auto; }}
		.summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
		.summary-item {{ background: #f8fafc; border-radius: 10px; padding: 14px; }}
		.summary-label {{ font-size: 12px; color: #64748b; margin-bottom: 6px; }}
		.summary-value {{ font-size: 22px; font-weight: 700; }}
	</style>
</head>
<body>
	<div class="container">
		<div class="card">
			<h1>Football Predictions</h1>
			<p class="subtitle">Match Result report for {today_str}</p>
			<div class="summary-grid">
				<div class="summary-item"><div class="summary-label">Matches</div><div class="summary-value">{len(predictions_df)}</div></div>
				<div class="summary-item"><div class="summary-label">Positive EV Picks</div><div class="summary-value">{len(value_df)}</div></div>
				<div class="summary-item"><div class="summary-label">Home Picks</div><div class="summary-value">{int((predictions_df['Result_Model_Pick'] == 'Home').sum())}</div></div>
				<div class="summary-item"><div class="summary-label">Draw Picks</div><div class="summary-value">{int((predictions_df['Result_Model_Pick'] == 'Draw').sum())}</div></div>
				<div class="summary-item"><div class="summary-label">Away Picks</div><div class="summary-value">{int((predictions_df['Result_Model_Pick'] == 'Away').sum())}</div></div>
			</div>
		</div>
		<div class="card">
			<h2>Value Picks</h2>
			<div class="table-wrap">{value_html}</div>
		</div>
		<div class="card">
			<h2>All Predictions</h2>
			<div class="table-wrap">{predictions_html}</div>
		</div>
	</div>
</body>
</html>"""
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(html, encoding="utf-8")
