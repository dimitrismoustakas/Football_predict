"""
Generate a standalone HTML report for match-result predictions.
"""

from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd


def _format_pct(value: float) -> str:
	return f"{value * 100:.2f}%"


def _format_decimal(value: float) -> str:
	return f"{value:.2f}"


def _format_interactive_table(df: pd.DataFrame) -> str:
	if df.empty:
		return "<p>No matches found for this report.</p>"

	df = df.sort_values(["Date", "League", "Time", "Home", "Away"]).reset_index(drop=True)

	rows = []
	for _, row in df.iterrows():
		best_bet = row["Result_Value_Side"] if pd.notna(row["Result_Value_Side"]) and row["Result_Value_Side"] else "No Bet"
		ev_now = _format_pct(row["Result_EV"]) if pd.notna(row["Result_EV"]) else ""
		split_now = _format_pct(row["Result_Budget_Share"]) if pd.notna(row["Result_Budget_Share"]) else "0.00%"
		row_class = "active-bet" if best_bet != "No Bet" else ""
		rows.append(
			"<tr class=\"{row_class}\" data-prob-home=\"{prob_home:.6f}\" data-prob-draw=\"{prob_draw:.6f}\" "
			"data-prob-away=\"{prob_away:.6f}\">"
			"<td>{date}</td><td>{league}</td><td>{time}</td><td>{match}</td>"
			"<td>{model_home}</td><td>{model_draw}</td><td>{model_away}</td>"
			"<td><input class=\"odds-input odds-home\" type=\"number\" min=\"1.01\" step=\"0.01\" value=\"{odds_home}\"></td>"
			"<td><input class=\"odds-input odds-draw\" type=\"number\" min=\"1.01\" step=\"0.01\" value=\"{odds_draw}\"></td>"
			"<td><input class=\"odds-input odds-away\" type=\"number\" min=\"1.01\" step=\"0.01\" value=\"{odds_away}\"></td>"
			"<td class=\"best-bet\">{best_bet}</td><td class=\"ev-now\">{ev_now}</td><td class=\"split-now\">{split_now}</td><td class=\"amount-now\">0.00</td>"
			"</tr>".format(
				row_class=row_class,
				prob_home=float(row["Prob_Home"]),
				prob_draw=float(row["Prob_Draw"]),
				prob_away=float(row["Prob_Away"]),
				date=escape(str(row["Date"])),
				league=escape(str(row["League"])),
				time=escape(str(row["Time"])),
				match=escape(f"{row['Home']} vs {row['Away']}"),
				model_home=_format_pct(float(row["Prob_Home"])),
				model_draw=_format_pct(float(row["Prob_Draw"])),
				model_away=_format_pct(float(row["Prob_Away"])),
				odds_home=_format_decimal(float(row["Odds_Home"])),
				odds_draw=_format_decimal(float(row["Odds_Draw"])),
				odds_away=_format_decimal(float(row["Odds_Away"])),
				best_bet=escape(best_bet),
				ev_now=ev_now,
				split_now=split_now,
			)
		)

	headers = "".join(
		f"<th>{label}</th>"
		for label in [
			"Date",
			"League",
			"Time",
			"Match",
			"Model Home %",
			"Model Draw %",
			"Model Away %",
			"Home Odds",
			"Draw Odds",
			"Away Odds",
			"Best Bet Now",
			"EV % Now",
			"Split % Now",
			"Amount Now",
		]
	)
	return f"<table class=\"predictions-table\"><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def generate_html_report(
	predictions_df: pd.DataFrame,
	output_path: Path,
	fixed_budget: float | None = None,
	budget_strategy: str | None = None,
	kelly_fraction: float | None = None,
) -> None:
	"""Write a styled HTML summary of match-result predictions."""

	today_str = datetime.now().strftime("%Y-%m-%d")
	strategy_label = budget_strategy or "n/a"
	kelly_label = f"{kelly_fraction:.2f}" if kelly_fraction is not None else "n/a"
	predictions_html = _format_interactive_table(predictions_df)
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
		tr.active-bet {{ background: #eefbf3; }}
		.table-wrap {{ overflow-x: auto; }}
		.odds-input {{ width: 72px; padding: 6px 8px; border: 1px solid #cbd5e1; border-radius: 8px; font: inherit; }}
		.budget-input {{ width: 96px; padding: 8px 10px; border: 1px solid #cbd5e1; border-radius: 8px; font: inherit; margin: 0 8px; }}
		.note {{ color: #475569; max-width: 900px; }}
	</style>
</head>
<body>
	<div class="container">
		<div class="card">
			<h1>Football Predictions</h1>
			<p class="subtitle">Match Result report for {today_str}</p>
			<p class="note">Edit the Home, Draw, and Away odds to recalculate best bet, EV, and split percentage. Model probabilities stay fixed from the production run. Split % is recomputed using the current {escape(strategy_label)} strategy{f' (kelly fraction {kelly_label})' if strategy_label == 'kelly' else ''}.</p>
			<p class="note">Total budget to split: <input id="total-budget" class="budget-input" type="number" min="0" step="0.01" value="10.00"> euros</p>
			<div class="table-wrap">{predictions_html}</div>
		</div>
	</div>
	<script>
		const STRATEGY = {strategy_label!r};
		const KELLY_FRACTION = {float(kelly_fraction) if kelly_fraction is not None else 0.5};
		const LABELS = ["Home", "Draw", "Away"];

		function parseOdds(input) {{
			const value = Number.parseFloat(input.value);
			return Number.isFinite(value) && value > 1.0 ? value : null;
		}}

		function computeRow(row) {{
			const probs = [
				Number.parseFloat(row.dataset.probHome),
				Number.parseFloat(row.dataset.probDraw),
				Number.parseFloat(row.dataset.probAway),
			];
			const odds = [
				parseOdds(row.querySelector('.odds-home')),
				parseOdds(row.querySelector('.odds-draw')),
				parseOdds(row.querySelector('.odds-away')),
			];
			if (odds.some((value) => value === null)) {{
				return {{ positive: false, bestIdx: -1, bestEv: 0, weight: 0 }};
			}}

			const inverseOdds = odds.map((value) => 1 / value);
			const norm = inverseOdds.reduce((sum, value) => sum + value, 0);
			const implied = inverseOdds.map((value) => value / norm);
			const evs = probs.map((prob, index) => prob * odds[index] - 1);
			let bestIdx = 0;
			for (let index = 1; index < evs.length; index += 1) {{
				if (evs[index] > evs[bestIdx]) {{
					bestIdx = index;
				}}
			}}
			const bestEv = evs[bestIdx];
			if (!(bestEv > 0)) {{
				return {{ positive: false, bestIdx, bestEv, weight: 0 }};
			}}

			const edge = probs[bestIdx] - implied[bestIdx];
			let fullKelly = 0;
			if (odds[bestIdx] > 1.0) {{
				fullKelly = Math.max((probs[bestIdx] * odds[bestIdx] - 1) / (odds[bestIdx] - 1), 0);
			}}

			let weight = 0;
			if (STRATEGY === 'flat') {{
				weight = 1;
			}} else if (STRATEGY === 'edge') {{
				weight = Math.max(edge, 0);
			}} else {{
				weight = fullKelly * Math.max(0, KELLY_FRACTION);
			}}
			if (!(weight > 0)) {{
				weight = 1;
			}}
			return {{ positive: true, bestIdx, bestEv, weight }};
		}}

		function updateTable() {{
			const rows = Array.from(document.querySelectorAll('.predictions-table tbody tr'));
			const results = rows.map((row) => computeRow(row));
			const totalWeight = results.reduce((sum, result) => sum + (result.positive ? result.weight : 0), 0);
			const totalBudgetInput = document.getElementById('total-budget');
			const totalBudget = Number.parseFloat(totalBudgetInput.value);
			const resolvedBudget = Number.isFinite(totalBudget) && totalBudget >= 0 ? totalBudget : 0;

			rows.forEach((row, index) => {{
				const result = results[index];
				const bestBetCell = row.querySelector('.best-bet');
				const evCell = row.querySelector('.ev-now');
				const splitCell = row.querySelector('.split-now');
				const amountCell = row.querySelector('.amount-now');
				if (!result.positive) {{
					bestBetCell.textContent = 'No Bet';
					evCell.textContent = '';
					splitCell.textContent = '0.00%';
					amountCell.textContent = '0.00';
					row.classList.remove('active-bet');
					return;
				}}
				const split = totalWeight > 0 ? result.weight / totalWeight : 0;
				const amount = split * resolvedBudget;
				bestBetCell.textContent = LABELS[result.bestIdx];
				evCell.textContent = `${{(result.bestEv * 100).toFixed(2)}}%`;
				splitCell.textContent = `${{(split * 100).toFixed(2)}}%`;
				amountCell.textContent = amount.toFixed(2);
				row.classList.add('active-bet');
			}});
		}}

		document.getElementById('total-budget').addEventListener('input', updateTable);
		document.querySelectorAll('.odds-input').forEach((input) => {{
			input.addEventListener('input', updateTable);
		}});
		updateTable();
	</script>
</body>
</html>"""
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(html, encoding="utf-8")
