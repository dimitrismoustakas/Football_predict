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


def _clamp(value: float, lower: float, upper: float) -> float:
	return max(lower, min(upper, value))


def _format_interactive_table(df: pd.DataFrame) -> str:
	if df.empty:
		return "<p>No matches found for this report.</p>"

	df = df.sort_values(["Date", "League", "Time", "Home", "Away"]).reset_index(drop=True)

	rows = []
	for _, row in df.iterrows():
		is_active = pd.notna(row["Result_Budget_Amount"]) and float(row["Result_Budget_Amount"]) > 0.0
		best_bet = row["Result_Value_Side"] if is_active and pd.notna(row["Result_Value_Side"]) and row["Result_Value_Side"] else "No Bet"
		ev_now = _format_pct(float(row["Result_EV"])) if is_active and pd.notna(row["Result_EV"]) else ""
		stake_now = _format_pct(float(row["Result_Budget_Share"])) if is_active and pd.notna(row["Result_Budget_Share"]) else "0.00%"
		amount_now = _format_decimal(float(row["Result_Budget_Amount"])) if is_active else "0.00"
		row_class = "active-bet" if is_active else ""
		rows.append(
			"<tr class=\"{row_class}\" data-prob-home=\"{prob_home:.6f}\" data-prob-draw=\"{prob_draw:.6f}\" "
			"data-prob-away=\"{prob_away:.6f}\">"
			"<td>{date}</td><td>{league}</td><td>{time}</td><td>{match}</td>"
			"<td>{model_home}</td><td>{model_draw}</td><td>{model_away}</td>"
			"<td><input class=\"odds-input odds-home\" type=\"number\" min=\"1.01\" step=\"0.01\" value=\"{odds_home}\"></td>"
			"<td><input class=\"odds-input odds-draw\" type=\"number\" min=\"1.01\" step=\"0.01\" value=\"{odds_draw}\"></td>"
			"<td><input class=\"odds-input odds-away\" type=\"number\" min=\"1.01\" step=\"0.01\" value=\"{odds_away}\"></td>"
			"<td class=\"best-bet\">{best_bet}</td><td class=\"ev-now\">{ev_now}</td><td class=\"stake-now\">{stake_now}</td><td class=\"amount-now\">{amount_now}</td>"
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
				stake_now=stake_now,
				amount_now=amount_now,
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
			"Stake % Now",
			"Amount Now",
		]
	)
	return f"<table class=\"predictions-table\"><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def generate_html_report(
	predictions_df: pd.DataFrame,
	output_path: Path,
	fixed_budget: float | None = None,
	kelly_fraction: float | None = None,
	min_bet_amount: float = 0.1,
) -> None:
	"""Write a styled HTML summary of match-result predictions."""

	html = render_html_report(
		predictions_df=predictions_df,
		fixed_budget=fixed_budget,
		kelly_fraction=kelly_fraction,
		min_bet_amount=min_bet_amount,
	)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(html, encoding="utf-8")


def render_html_report(
	predictions_df: pd.DataFrame,
	fixed_budget: float | None = None,
	kelly_fraction: float | None = None,
	min_bet_amount: float = 0.1,
) -> str:
	"""Render the standalone HTML report as a string."""

	today_str = datetime.now().strftime("%Y-%m-%d")
	predictions_html = _format_interactive_table(predictions_df)
	default_budget = float(fixed_budget) if fixed_budget is not None else 10.0
	default_kelly_fraction = _clamp(float(kelly_fraction) if kelly_fraction is not None else 0.5, 0.1, 1.0)
	note = (
		"Edit the Home, Draw, and Away odds to recalculate the best bet, expected value, and suggested stake. "
		"Enter your current bankroll and Kelly fraction below to adjust the risk level."
	)
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
		.summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 16px 0 20px; }}
		.summary-tile {{ background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px 14px; }}
		.summary-label {{ color: #64748b; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
		.summary-value {{ color: #0f172a; font-size: 22px; font-weight: 700; margin-top: 6px; }}
		table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
		th, td {{ border: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; }}
		th {{ background: #0f172a; color: white; position: sticky; top: 0; }}
		tr:nth-child(even) {{ background: #f8fafc; }}
		tr.active-bet {{ background: #eefbf3; }}
		.table-wrap {{ overflow-x: auto; }}
		.odds-input {{ width: 72px; padding: 6px 8px; border: 1px solid #cbd5e1; border-radius: 8px; font: inherit; }}
		.budget-input {{ width: 96px; padding: 8px 10px; border: 1px solid #cbd5e1; border-radius: 8px; font: inherit; margin: 0 8px; }}
		.controls {{ display: flex; flex-wrap: wrap; gap: 12px 20px; margin: 12px 0 20px; }}
		.control-label {{ color: #475569; display: flex; align-items: center; flex-wrap: wrap; gap: 8px; }}
		.note {{ color: #475569; max-width: 900px; }}
	</style>
</head>
<body>
	<div class="container">
		<div class="card">
			<h1>Football Predictions</h1>
			<p class="subtitle">Match Result report for {today_str}</p>
			<p class="note">{note}</p>
			<div class="controls">
				<label class="control-label">Current bankroll: <input id="total-budget" class="budget-input" type="number" min="0" step="0.01" value="{default_budget:.2f}"> euros</label>
				<label class="control-label">Kelly fraction: <input id="kelly-fraction" class="budget-input" type="number" min="0.1" max="1" step="0.01" value="{default_kelly_fraction:.2f}"></label>
			</div>
			<div class="summary-grid">
				<div class="summary-tile">
					<div class="summary-label">Suggested Bets</div>
					<div id="summary-bet-count" class="summary-value">0</div>
				</div>
				<div class="summary-tile">
					<div class="summary-label">Total Stake</div>
					<div id="summary-total-amount" class="summary-value">0.00 EUR</div>
				</div>
				<div class="summary-tile">
					<div class="summary-label">Percent Of Bankroll</div>
					<div id="summary-total-pct" class="summary-value">0.00%</div>
				</div>
			</div>
			<div class="table-wrap">{predictions_html}</div>
		</div>
	</div>
	<script>
		const MIN_BET_AMOUNT = {float(min_bet_amount)};
		const MIN_KELLY_FRACTION = 0.1;
		const MAX_KELLY_FRACTION = 1.0;
		const LABELS = ["Home", "Draw", "Away"];

		function parseOdds(input) {{
			const value = Number.parseFloat(input.value);
			return Number.isFinite(value) && value > 1.0 ? value : null;
		}}

		function parseNonNegativeNumber(input, fallback) {{
			const value = Number.parseFloat(input.value);
			return Number.isFinite(value) && value >= 0 ? value : fallback;
		}}

		function clampKellyFraction(input) {{
			const parsed = Number.parseFloat(input.value);
			const fallback = Number.parseFloat(input.defaultValue);
			const baseValue = Number.isFinite(parsed)
				? parsed
				: (Number.isFinite(fallback) ? fallback : MIN_KELLY_FRACTION);
			return Math.min(MAX_KELLY_FRACTION, Math.max(MIN_KELLY_FRACTION, baseValue));
		}}

		function normalizeKellyFractionInput(input) {{
			input.value = clampKellyFraction(input).toFixed(2);
		}}

		function computeRow(row, stakeFraction) {{
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

			let fullFraction = 0;
			if (odds[bestIdx] > 1.0) {{
				fullFraction = Math.max((probs[bestIdx] * odds[bestIdx] - 1) / (odds[bestIdx] - 1), 0);
			}}
			return {{
				positive: true,
				bestIdx,
				bestEv,
				weight: fullFraction * Math.max(0, stakeFraction),
			}};
		}}

		function computeStakePlan(results, totalBudget) {{
			let active = results.map((result) => result.positive);
			let shares = results.map(() => 0);
			let amounts = results.map(() => 0);
			while (true) {{
				const totalWeight = results.reduce((sum, result, index) => sum + (active[index] ? result.weight : 0), 0);
				if (!(totalWeight > 0) || !(totalBudget > 0)) {{
					return {{ active, shares, amounts }};
				}}
				shares = results.map((result, index) => {{
					if (!active[index]) {{
						return 0;
					}}
					return totalWeight > 1 ? result.weight / totalWeight : result.weight;
				}});
				amounts = shares.map((share) => share * totalBudget);
				const tooSmall = amounts.map((amount, index) => active[index] && amount > 0 && amount + 1e-9 < MIN_BET_AMOUNT);
				if (!tooSmall.some(Boolean)) {{
					return {{ active, shares, amounts }};
				}}
				active = active.map((flag, index) => flag && !tooSmall[index]);
			}}
		}}

		function updateTable() {{
			const rows = Array.from(document.querySelectorAll('.predictions-table tbody tr'));
			const totalBudgetInput = document.getElementById('total-budget');
			const kellyFractionInput = document.getElementById('kelly-fraction');
			const totalBudget = Number.parseFloat(totalBudgetInput.value);
			const stakeFraction = clampKellyFraction(kellyFractionInput);
			const results = rows.map((row) => computeRow(row, stakeFraction));
			const resolvedBudget = Number.isFinite(totalBudget) && totalBudget >= 0 ? totalBudget : 0;
			const plan = computeStakePlan(results, resolvedBudget);
			let activeCount = 0;
			let totalAmount = 0;

			rows.forEach((row, index) => {{
				const result = results[index];
				const isActive = plan.active[index] && plan.amounts[index] > 0;
				const bestBetCell = row.querySelector('.best-bet');
				const evCell = row.querySelector('.ev-now');
				const stakeCell = row.querySelector('.stake-now');
				const amountCell = row.querySelector('.amount-now');
				if (!isActive) {{
					bestBetCell.textContent = 'No Bet';
					evCell.textContent = '';
					stakeCell.textContent = '0.00%';
					amountCell.textContent = '0.00';
					row.classList.remove('active-bet');
					return;
				}}
				const stake = plan.shares[index];
				const amount = plan.amounts[index];
				bestBetCell.textContent = LABELS[result.bestIdx];
				evCell.textContent = `${{(result.bestEv * 100).toFixed(2)}}%`;
				stakeCell.textContent = `${{(stake * 100).toFixed(2)}}%`;
				amountCell.textContent = amount.toFixed(2);
				row.classList.add('active-bet');
				activeCount += 1;
				totalAmount += amount;
			}});
			const totalPct = resolvedBudget > 0 ? (totalAmount / resolvedBudget) * 100 : 0;
			document.getElementById('summary-bet-count').textContent = String(activeCount);
			document.getElementById('summary-total-amount').textContent = `${{totalAmount.toFixed(2)}} EUR`;
			document.getElementById('summary-total-pct').textContent = `${{totalPct.toFixed(2)}}%`;
		}}

		document.getElementById('total-budget').addEventListener('input', updateTable);
		document.getElementById('kelly-fraction').addEventListener('input', updateTable);
		document.getElementById('kelly-fraction').addEventListener('blur', () => {{
			normalizeKellyFractionInput(document.getElementById('kelly-fraction'));
			updateTable();
		}});
		document.querySelectorAll('.odds-input').forEach((input) => {{
			input.addEventListener('input', updateTable);
		}});
		normalizeKellyFractionInput(document.getElementById('kelly-fraction'));
		updateTable();
	</script>
</body>
</html>"""
	return html
