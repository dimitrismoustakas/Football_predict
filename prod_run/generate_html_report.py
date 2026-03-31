"""
Generate a standalone HTML report for match-result predictions.
"""

from __future__ import annotations

import json
from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd

from utils.portfolio import (
	DEFAULT_JOINT_OPTIMIZER_INITIAL_STEP,
	DEFAULT_JOINT_OPTIMIZER_MAX_ITERATIONS,
	DEFAULT_JOINT_OPTIMIZER_MIN_STEP,
	DEFAULT_JOINT_QUADRATURE_ORDER,
	get_joint_quadrature_rule,
)


def _format_pct(value: float) -> str:
	return f"{value * 100:.2f}%"


def _format_decimal(value: float) -> str:
	return f"{value:.2f}"


def _clamp(value: float, lower: float, upper: float) -> float:
	return max(lower, min(upper, value))


def _format_interactive_table(df: pd.DataFrame) -> str:
	if df.empty:
		return "<p>No matches found for this report.</p>"

	df = df.sort_values(["League", "Date", "Time", "Home", "Away"]).reset_index(drop=True)

	rows = []
	for _, row in df.iterrows():
		prob_home = float(row.get("Prob_Home", row.get("Pred_Home")))
		prob_draw = float(row.get("Prob_Draw", row.get("Pred_Draw")))
		prob_away = float(row.get("Prob_Away", row.get("Pred_Away")))
		home_team = str(row.get("Home", row.get("Home Team")))
		away_team = str(row.get("Away", row.get("Away Team")))
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
				prob_home=prob_home,
				prob_draw=prob_draw,
				prob_away=prob_away,
				date=escape(str(row["Date"])),
				league=escape(str(row["League"])),
				time=escape(str(row.get("Time", ""))),
				match=escape(f"{home_team} vs {away_team}"),
				model_home=_format_pct(prob_home),
				model_draw=_format_pct(prob_draw),
				model_away=_format_pct(prob_away),
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
	quadrature_nodes, quadrature_weights = get_joint_quadrature_rule(DEFAULT_JOINT_QUADRATURE_ORDER)
	note = (
		"Edit the Home, Draw, and Away odds to recalculate the single best side, expected value, "
		"and suggested stake. Enter your current bankroll and Kelly fraction below to adjust the risk level."
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
		const JOINT_OPTIMIZER_MAX_ITERATIONS = {int(DEFAULT_JOINT_OPTIMIZER_MAX_ITERATIONS)};
		const JOINT_OPTIMIZER_INITIAL_STEP = {float(DEFAULT_JOINT_OPTIMIZER_INITIAL_STEP)};
		const JOINT_OPTIMIZER_MIN_STEP = {float(DEFAULT_JOINT_OPTIMIZER_MIN_STEP)};
		const QUADRATURE_NODES = {json.dumps(quadrature_nodes.tolist())};
		const QUADRATURE_WEIGHTS = {json.dumps(quadrature_weights.tolist())};
		const QUADRATURE_LOG_NODES = QUADRATURE_NODES.map((value) => Math.log(value));
		const QUADRATURE_LOG_WEIGHTS = QUADRATURE_WEIGHTS.map((value) => Math.log(value));
		const QUADRATURE_CONSTANT = QUADRATURE_WEIGHTS.reduce((sum, weight, index) => sum + weight / QUADRATURE_NODES[index], 0);
		const LABELS = ["Home", "Draw", "Away"];

		function parseOdds(input) {{
			const value = Number.parseFloat(input.value);
			return Number.isFinite(value) && value > 1.0 ? value : null;
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

		function roundHalfEven(value, decimals = 2) {{
			if (!Number.isFinite(value)) {{
				return 0;
			}}
			const factor = 10 ** decimals;
			const scaled = value * factor;
			const sign = scaled < 0 ? -1 : 1;
			const absScaled = Math.abs(scaled);
			const floor = Math.floor(absScaled);
			const fraction = absScaled - floor;
			const epsilon = 1e-10;
			let roundedInt = floor;
			if (fraction > 0.5 + epsilon) {{
				roundedInt = floor + 1;
			}} else if (Math.abs(fraction - 0.5) <= epsilon) {{
				roundedInt = floor % 2 === 0 ? floor : floor + 1;
			}}
			return sign * roundedInt / factor;
		}}

		function roundBudgetAmounts(amounts, totalBudget) {{
			const rounded = amounts.map((amount) => roundHalfEven(amount, 2));
			const positiveIndices = amounts
				.map((amount, index) => amount > 0 ? index : -1)
				.filter((index) => index >= 0);
			if (!positiveIndices.length) {{
				return rounded;
			}}
			let deltaCents = Math.round(
				(roundHalfEven(totalBudget, 2) - roundHalfEven(rounded.reduce((sum, amount) => sum + amount, 0), 2)) * 100
			);
			if (deltaCents === 0) {{
				return rounded;
			}}
			const residuals = amounts.map((amount, index) => amount - rounded[index]);
			const order = deltaCents > 0
				? [...positiveIndices].sort((left, right) => (
					(residuals[right] - residuals[left]) || (amounts[right] - amounts[left]) || (right - left)
				))
				: positiveIndices
					.filter((index) => rounded[index] > 0)
					.sort((left, right) => (
						((rounded[right] - amounts[right]) - (rounded[left] - amounts[left]))
						|| (rounded[right] - rounded[left])
						|| (right - left)
					));
			if (!order.length) {{
				return rounded;
			}}
			while (deltaCents !== 0) {{
				let changed = false;
				for (const index of order) {{
					if (deltaCents === 0) {{
						break;
					}}
					if (deltaCents > 0) {{
						rounded[index] = roundHalfEven(rounded[index] + 0.01, 2);
						deltaCents -= 1;
						changed = true;
						continue;
					}}
					if (rounded[index] >= 0.01 - 1e-12) {{
						rounded[index] = roundHalfEven(rounded[index] - 0.01, 2);
						deltaCents += 1;
						changed = true;
					}}
				}}
				if (!changed) {{
					break;
				}}
			}}
			return rounded;
		}}

		function projectNonnegativeL1Ball(values, radius) {{
			const clipped = values.map((value) => Math.max(0, value));
			const total = clipped.reduce((sum, value) => sum + value, 0);
			if (total <= radius) {{
				return clipped;
			}}
			const sorted = [...clipped].sort((a, b) => b - a);
			let cumulative = 0;
			let rho = -1;
			for (let index = 0; index < sorted.length; index += 1) {{
				cumulative += sorted[index];
				const threshold = sorted[index] - (cumulative - radius) / (index + 1);
				if (threshold > 0) {{
					rho = index;
				}}
			}}
			if (rho < 0) {{
				return clipped.map(() => 0);
			}}
			const theta = (sorted.slice(0, rho + 1).reduce((sum, value) => sum + value, 0) - radius) / (rho + 1);
			return clipped.map((value) => Math.max(0, value - theta));
		}}

		function collectRows() {{
			return Array.from(document.querySelectorAll('.predictions-table tbody tr')).map((row) => {{
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
					return {{ row, positive: false, bestIdx: -1, bestEv: 0, bestProb: 0, selectedOdds: 0, fullKelly: 0 }};
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
					return {{
						row,
						positive: false,
						bestIdx,
						bestEv,
						bestProb: probs[bestIdx],
						selectedOdds: odds[bestIdx],
						fullKelly: 0
					}};
				}}
				const bestProb = probs[bestIdx];
				const selectedOdds = odds[bestIdx];
				const fullKelly = selectedOdds > 1.0
					? Math.max((bestProb * selectedOdds - 1) / (selectedOdds - 1), 0)
					: 0;
				return {{
					row,
					positive: true,
					bestIdx,
					bestEv,
					bestProb,
					selectedOdds,
					fullKelly
				}};
			}});
		}}

		function jointObjectiveAndGrad(weights, activeResults) {{
			const width = weights.length;
			if (!width) {{
				return {{ value: 0, gradient: new Float64Array(0) }};
			}}
			const ratios = QUADRATURE_NODES.map(() => new Float64Array(width));
			const objectiveLogs = new Float64Array(QUADRATURE_NODES.length);
			const gradientLogs = new Float64Array(QUADRATURE_NODES.length);
			let maxObjectiveLog = -Infinity;
			let maxGradientLog = -Infinity;

			for (let nodeIndex = 0; nodeIndex < QUADRATURE_NODES.length; nodeIndex += 1) {{
				const node = QUADRATURE_NODES[nodeIndex];
				let logProduct = 0;
				for (let betIndex = 0; betIndex < width; betIndex += 1) {{
					const result = activeResults[betIndex];
					const winReturn = result.selectedOdds - 1;
					const lossScale = node * weights[betIndex];
					const winScale = -node * winReturn * weights[betIndex];
					const shift = Math.max(lossScale, winScale);
					const lossTerm = (1 - result.bestProb) * Math.exp(lossScale - shift);
					const winTerm = result.bestProb * Math.exp(winScale - shift);
					const denominator = lossTerm + winTerm;
					logProduct += shift + Math.log(denominator);
					ratios[nodeIndex][betIndex] = (lossTerm - winReturn * winTerm) / denominator;
				}}
				objectiveLogs[nodeIndex] = QUADRATURE_LOG_WEIGHTS[nodeIndex] - QUADRATURE_LOG_NODES[nodeIndex] + logProduct;
				gradientLogs[nodeIndex] = QUADRATURE_LOG_WEIGHTS[nodeIndex] + logProduct;
				maxObjectiveLog = Math.max(maxObjectiveLog, objectiveLogs[nodeIndex]);
				maxGradientLog = Math.max(maxGradientLog, gradientLogs[nodeIndex]);
			}}

			let objectiveSecond = 0;
			for (let nodeIndex = 0; nodeIndex < QUADRATURE_NODES.length; nodeIndex += 1) {{
				objectiveSecond += Math.exp(objectiveLogs[nodeIndex] - maxObjectiveLog);
			}}
			objectiveSecond *= Math.exp(maxObjectiveLog);

			const gradient = new Float64Array(width);
			for (let nodeIndex = 0; nodeIndex < QUADRATURE_NODES.length; nodeIndex += 1) {{
				const factor = Math.exp(gradientLogs[nodeIndex] - maxGradientLog);
				for (let betIndex = 0; betIndex < width; betIndex += 1) {{
					gradient[betIndex] -= factor * ratios[nodeIndex][betIndex];
				}}
			}}
			const gradientScale = Math.exp(maxGradientLog);
			for (let betIndex = 0; betIndex < width; betIndex += 1) {{
				gradient[betIndex] *= gradientScale;
			}}
			return {{
				value: QUADRATURE_CONSTANT - objectiveSecond,
				gradient
			}};
		}}

		function optimizeJointStakePlan(results, totalBudget, stakeFraction) {{
			let active = results.map((result) => result.positive);
			let shares = results.map(() => 0);
			let amounts = results.map(() => 0);
			while (true) {{
				const activeIndices = active
					.map((flag, index) => flag ? index : -1)
					.filter((index) => index >= 0);
				if (!activeIndices.length || !(totalBudget > 0)) {{
					return {{ active, shares, amounts }};
				}}

				const activeResults = activeIndices.map((index) => results[index]);
				let weights;
				if (activeResults.length === 1) {{
					weights = [activeResults[0].fullKelly];
				}} else {{
					weights = projectNonnegativeL1Ball(activeResults.map((result) => result.fullKelly), 1 - 1e-12);
					let step = JOINT_OPTIMIZER_INITIAL_STEP;
					let current = jointObjectiveAndGrad(weights, activeResults);
					for (let iteration = 0; iteration < JOINT_OPTIMIZER_MAX_ITERATIONS; iteration += 1) {{
						const candidate = projectNonnegativeL1Ball(
							weights.map((weight, index) => weight + step * current.gradient[index]),
							1 - 1e-12
						);
						const next = jointObjectiveAndGrad(candidate, activeResults);
						if (Number.isFinite(next.value) && next.value >= current.value) {{
							weights = candidate;
							current = next;
							step *= 1.05;
						}} else {{
							step *= 0.5;
						}}
						if (step < JOINT_OPTIMIZER_MIN_STEP) {{
							break;
						}}
					}}
				}}

				const scaledWeights = projectNonnegativeL1Ball(
					weights.map((weight) => weight * Math.max(0, stakeFraction)),
					1 - 1e-12
				);
				const rawAmounts = scaledWeights.map((weight) => weight * totalBudget);
				const roundedActiveAmounts = roundBudgetAmounts(
					rawAmounts,
					rawAmounts.reduce((sum, amount) => sum + amount, 0)
				);
				amounts = results.map(() => 0);
				shares = results.map(() => 0);
				activeIndices.forEach((rowIndex, localIndex) => {{
					amounts[rowIndex] = roundedActiveAmounts[localIndex];
					shares[rowIndex] = totalBudget > 0 ? roundedActiveAmounts[localIndex] / totalBudget : 0;
				}});
				const tooSmall = amounts.map((amount, index) => active[index] && amount > 0 && amount + 1e-12 < MIN_BET_AMOUNT);
				if (!tooSmall.some(Boolean)) {{
					return {{ active, shares, amounts }};
				}}
				active = active.map((flag, index) => flag && !tooSmall[index]);
			}}
		}}

		function updateTable() {{
			const rows = collectRows();
			const totalBudgetInput = document.getElementById('total-budget');
			const kellyFractionInput = document.getElementById('kelly-fraction');
			const totalBudget = Number.parseFloat(totalBudgetInput.value);
			const stakeFraction = clampKellyFraction(kellyFractionInput);
			const resolvedBudget = Number.isFinite(totalBudget) && totalBudget >= 0 ? totalBudget : 0;
			const plan = optimizeJointStakePlan(rows, resolvedBudget, stakeFraction);
			let activeCount = 0;
			let totalAmount = 0;

			rows.forEach((result, index) => {{
				const row = result.row;
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
