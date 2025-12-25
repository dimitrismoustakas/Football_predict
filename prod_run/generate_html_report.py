"""
Generate interactive HTML report for betting predictions.
Users can adjust odds and recalculate allocations in their browser.
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime


def generate_html_report(predictions_df: pd.DataFrame, output_path: Path) -> None:
	"""
	Generate a standalone HTML file with interactive odds adjustment.
	
	Args:
		predictions_df: DataFrame with columns: Date, Time, Home, Away, Prob_Over, Prob_Under,
			Implied_Over, Implied_Under, Odds_Over, Odds_Under, Model_Odds_Over, Model_Odds_Under,
			Bet_Side, EV, Allocation_Pct
		output_path: Where to save the HTML file
	"""
	# Convert DataFrame to list of dicts for JSON embedding
	data = predictions_df.to_dict(orient="records")
	
	# Convert any non-JSON-serializable types
	for row in data:
		for key, val in row.items():
			if isinstance(val, (pd.Timestamp, datetime)):
				row[key] = str(val)
			elif pd.isna(val):
				row[key] = None
	
	data_json = json.dumps(data, indent=2)
	today_str = datetime.now().strftime("%Y-%m-%d")
	
	html = f"""<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8">
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<title>Football Predictions - {today_str}</title>
	<style>
		* {{
			box-sizing: border-box;
		}}
		body {{
			font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
			margin: 0;
			padding: 20px;
			background: #f5f5f5;
			color: #333;
		}}
		.container {{
			max-width: 1400px;
			margin: 0 auto;
		}}
		h1 {{
			color: #2c3e50;
			margin-bottom: 5px;
		}}
		.subtitle {{
			color: #7f8c8d;
			margin-bottom: 20px;
		}}
		.controls {{
			background: white;
			padding: 15px 20px;
			border-radius: 8px;
			margin-bottom: 20px;
			box-shadow: 0 2px 4px rgba(0,0,0,0.1);
			display: flex;
			flex-wrap: wrap;
			gap: 20px;
			align-items: center;
		}}
		.control-group {{
			display: flex;
			align-items: center;
			gap: 8px;
		}}
		.control-group label {{
			font-weight: 500;
			white-space: nowrap;
		}}
		.control-group input[type="number"] {{
			width: 100px;
			padding: 8px;
			border: 1px solid #ddd;
			border-radius: 4px;
			font-size: 14px;
		}}
		.control-group input[type="checkbox"] {{
			width: 18px;
			height: 18px;
			cursor: pointer;
		}}
		button {{
			padding: 8px 16px;
			border: none;
			border-radius: 4px;
			cursor: pointer;
			font-size: 14px;
			transition: background 0.2s;
		}}
		.btn-reset {{
			background: #e74c3c;
			color: white;
		}}
		.btn-reset:hover {{
			background: #c0392b;
		}}
		.table-container {{
			background: white;
			border-radius: 8px;
			box-shadow: 0 2px 4px rgba(0,0,0,0.1);
			overflow-x: auto;
		}}
		table {{
			width: 100%;
			border-collapse: collapse;
			font-size: 13px;
		}}
		th, td {{
			padding: 10px 8px;
			text-align: left;
			border-bottom: 1px solid #eee;
		}}
		th {{
			background: #34495e;
			color: white;
			font-weight: 500;
			position: sticky;
			top: 0;
			white-space: nowrap;
		}}
		tr:hover {{
			background: #f8f9fa;
		}}
		tr.excluded {{
			opacity: 0.5;
			background: #f5f5f5;
		}}
		tr.ev-positive {{
			background: #d4edda;
		}}
		tr.ev-negative {{
			background: #f8d7da;
		}}
		tr.ev-flipped-positive {{
			background: #c3e6cb;
		}}
		tr.ev-flipped-negative {{
			background: #f5c6cb;
		}}
		.odds-input {{
			width: 70px;
			padding: 5px;
			border: 1px solid #ddd;
			border-radius: 3px;
			text-align: center;
			font-size: 13px;
		}}
		.odds-input:focus {{
			outline: none;
			border-color: #3498db;
			box-shadow: 0 0 0 2px rgba(52,152,219,0.2);
		}}
		.odds-input.changed {{
			background: #fff3cd;
			border-color: #ffc107;
		}}
		.confirm-checkbox {{
			width: 16px;
			height: 16px;
			cursor: pointer;
		}}
		.positive {{
			color: #27ae60;
			font-weight: 600;
		}}
		.negative {{
			color: #e74c3c;
		}}
		.bet-side {{
			font-weight: 600;
		}}
		.bet-side.over {{
			color: #2980b9;
		}}
		.bet-side.under {{
			color: #8e44ad;
		}}
		.allocation {{
			font-weight: 600;
			color: #27ae60;
		}}
		.summary {{
			background: white;
			padding: 15px 20px;
			border-radius: 8px;
			margin-top: 20px;
			box-shadow: 0 2px 4px rgba(0,0,0,0.1);
		}}
		.summary h3 {{
			margin-top: 0;
			color: #2c3e50;
		}}
		.summary-grid {{
			display: grid;
			grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
			gap: 15px;
		}}
		.summary-item {{
			padding: 10px;
			background: #f8f9fa;
			border-radius: 4px;
		}}
		.summary-item .label {{
			font-size: 12px;
			color: #7f8c8d;
			margin-bottom: 4px;
		}}
		.summary-item .value {{
			font-size: 18px;
			font-weight: 600;
			color: #2c3e50;
		}}
		.legend {{
			margin-top: 15px;
			padding-top: 15px;
			border-top: 1px solid #eee;
			font-size: 12px;
			color: #7f8c8d;
		}}
		.legend span {{
			margin-right: 15px;
			padding: 2px 8px;
			border-radius: 3px;
		}}
		.legend .flipped-pos {{
			background: #c3e6cb;
		}}
		.legend .flipped-neg {{
			background: #f5c6cb;
		}}
		@media (max-width: 768px) {{
			body {{
				padding: 10px;
			}}
			.controls {{
				flex-direction: column;
				align-items: flex-start;
			}}
			th, td {{
				padding: 8px 5px;
				font-size: 11px;
			}}
			.odds-input {{
				width: 55px;
				font-size: 11px;
			}}
		}}
	</style>
</head>
<body>
	<div class="container">
		<h1>Football Predictions</h1>
		<p class="subtitle">Over/Under 2.5 Goals - {today_str}</p>
		
		<div class="controls">
			<div class="control-group">
				<label for="bankroll">Bankroll:</label>
				<input type="number" id="bankroll" value="100" min="1" step="10">
			</div>
			<div class="control-group">
				<input type="checkbox" id="onlyConfirmed">
				<label for="onlyConfirmed">Only confirmed games</label>
			</div>
			<button class="btn-reset" onclick="resetAllOdds()">Reset All Odds</button>
		</div>
		
		<div class="table-container">
			<table id="predictionsTable">
				<thead>
					<tr>
						<th>✓</th>
						<th>Date</th>
						<th>Time</th>
						<th>Home</th>
						<th>Away</th>
						<th>P(Over)</th>
						<th>P(Under)</th>
						<th>Odds Over</th>
						<th>Odds Under</th>

						<th>Bet Side</th>
						<th>EV</th>
						<th>Alloc %</th>
						<th>Bet Amt</th>
					</tr>
				</thead>
				<tbody id="tableBody">
				</tbody>
			</table>
		</div>
		
		<div class="summary">
			<h3>Summary</h3>
			<div class="summary-grid">
				<div class="summary-item">
					<div class="label">Games Included</div>
					<div class="value" id="summaryGames">0</div>
				</div>
				<div class="summary-item">
					<div class="label">Positive EV Bets</div>
					<div class="value" id="summaryBets">0</div>
				</div>
				<div class="summary-item">
					<div class="label">Total Allocation</div>
					<div class="value" id="summaryAlloc">0%</div>
				</div>
				<div class="summary-item">
					<div class="label">Total Bet Amount</div>
					<div class="value" id="summaryAmount">€0.00</div>
				</div>
			</div>
			<div class="legend">
				<strong>Legend:</strong>
				<span class="flipped-pos">Became +EV with your odds</span>
				<span class="flipped-neg">Lost +EV with your odds</span>
			</div>
		</div>
	</div>

	<script>
		// Original data from Python
		const ORIGINAL_DATA = {data_json};
		
		// Working copy of data (user can modify odds)
		let data = JSON.parse(JSON.stringify(ORIGINAL_DATA));
		
		// Track confirmed games
		let confirmedGames = new Set();
		
		// Initialize
		document.addEventListener('DOMContentLoaded', () => {{
			renderTable();
			recalculate();
			
			document.getElementById('bankroll').addEventListener('input', recalculate);
			document.getElementById('onlyConfirmed').addEventListener('change', recalculate);
		}});
		
		function renderTable() {{
			const tbody = document.getElementById('tableBody');
			tbody.innerHTML = '';
			
			data.forEach((row, idx) => {{
				const tr = document.createElement('tr');
				tr.id = `row-${{idx}}`;
				tr.innerHTML = `
					<td><input type="checkbox" class="confirm-checkbox" data-idx="${{idx}}" onchange="toggleConfirm(${{idx}})"></td>
					<td>${{row.Date}}</td>
					<td>${{row.Time}}</td>
					<td>${{row.Home}}</td>
					<td>${{row.Away}}</td>
					<td>${{(row.Prob_Over * 100).toFixed(1)}}%</td>
					<td>${{(row.Prob_Under * 100).toFixed(1)}}%</td>
					<td><input type="number" class="odds-input" id="odds-over-${{idx}}" 
						value="${{row.Odds_Over.toFixed(2)}}" step="0.01" min="1.01"
						data-original="${{row.Odds_Over.toFixed(2)}}"
						onchange="onOddsChange(${{idx}}, 'over')"></td>
					<td><input type="number" class="odds-input" id="odds-under-${{idx}}" 
						value="${{row.Odds_Under.toFixed(2)}}" step="0.01" min="1.01"
						data-original="${{row.Odds_Under.toFixed(2)}}"
						onchange="onOddsChange(${{idx}}, 'under')"></td>

					<td class="bet-side" id="bet-side-${{idx}}"></td>
					<td id="ev-${{idx}}"></td>
					<td id="alloc-${{idx}}" class="allocation"></td>
					<td id="amount-${{idx}}"></td>
				`;
				tbody.appendChild(tr);
			}});
		}}
		
		function toggleConfirm(idx) {{
			const checkbox = document.querySelector(`#row-${{idx}} .confirm-checkbox`);
			if (checkbox.checked) {{
				confirmedGames.add(idx);
			}} else {{
				confirmedGames.delete(idx);
			}}
			recalculate();
		}}
		
		function onOddsChange(idx, side) {{
			const input = document.getElementById(`odds-${{side}}-${{idx}}`);
			const newValue = parseFloat(input.value);
			
			if (isNaN(newValue) || newValue < 1.01) {{
				input.value = input.dataset.original;
				return;
			}}
			
			// Update data
			if (side === 'over') {{
				data[idx].Odds_Over = newValue;
			}} else {{
				data[idx].Odds_Under = newValue;
			}}
			
			// Mark as changed
			const original = parseFloat(input.dataset.original);
			if (Math.abs(newValue - original) > 0.001) {{
				input.classList.add('changed');
			}} else {{
				input.classList.remove('changed');
			}}
			
			// Auto-confirm when user changes odds
			if (!confirmedGames.has(idx)) {{
				confirmedGames.add(idx);
				const checkbox = document.querySelector(`#row-${{idx}} .confirm-checkbox`);
				checkbox.checked = true;
			}}
			
			recalculate();
		}}
		
		function resetAllOdds() {{
			data = JSON.parse(JSON.stringify(ORIGINAL_DATA));
			confirmedGames.clear();
			
			// Reset all inputs
			data.forEach((row, idx) => {{
				document.getElementById(`odds-over-${{idx}}`).value = row.Odds_Over.toFixed(2);
				document.getElementById(`odds-under-${{idx}}`).value = row.Odds_Under.toFixed(2);
				document.getElementById(`odds-over-${{idx}}`).classList.remove('changed');
				document.getElementById(`odds-under-${{idx}}`).classList.remove('changed');
				document.querySelector(`#row-${{idx}} .confirm-checkbox`).checked = false;
			}});
			
			recalculate();
		}}
		
		function recalculate() {{
			const bankroll = parseFloat(document.getElementById('bankroll').value) || 100;
			const onlyConfirmed = document.getElementById('onlyConfirmed').checked;
			
			// Calculate EV and variance for each game
			const gameStats = data.map((row, idx) => {{
				const p = row.Prob_Over;
				const oddsOver = row.Odds_Over;
				const oddsUnder = row.Odds_Under;
				
				// Expected value
				const muOver = p * oddsOver - 1;
				const muUnder = (1 - p) * oddsUnder - 1;
				
				// Variance
				const eX2Over = p * Math.pow(oddsOver - 1, 2) + (1 - p) * 1;
				const varOver = eX2Over - Math.pow(muOver, 2);
				
				const eX2Under = (1 - p) * Math.pow(oddsUnder - 1, 2) + p * 1;
				const varUnder = eX2Under - Math.pow(muUnder, 2);
				
				// Select better side
				const betterIsOver = muOver >= muUnder;
				const muBest = betterIsOver ? muOver : muUnder;
				const varBest = betterIsOver ? varOver : varUnder;
				const betSide = betterIsOver ? 'Over' : 'Under';
				
				// Check if included in calculation
				const isIncluded = !onlyConfirmed || confirmedGames.has(idx);
				const isEligible = muBest > 0 && isIncluded;
				
				// Original EV for comparison
				const origOddsOver = ORIGINAL_DATA[idx].Odds_Over;
				const origOddsUnder = ORIGINAL_DATA[idx].Odds_Under;
				const origMuOver = p * origOddsOver - 1;
				const origMuUnder = (1 - p) * origOddsUnder - 1;
				const origMuBest = Math.max(origMuOver, origMuUnder);
				
				return {{
					idx,
					muBest,
					varBest,
					betSide,
					isIncluded,
					isEligible,
					origMuBest,
				}};
			}});
			
			// Calculate Sharpe-weighted allocations for eligible games
			const eligible = gameStats.filter(g => g.isEligible);
			let sumWeights = 0;
			
			eligible.forEach(g => {{
				g.rawWeight = Math.max(0, g.muBest / (g.varBest + 1e-6));
				sumWeights += g.rawWeight;
			}});
			
			// Normalize weights
			eligible.forEach(g => {{
				g.allocPct = sumWeights > 0 ? (g.rawWeight / sumWeights) * 100 : 0;
			}});
			
			// Create lookup
			const allocByIdx = {{}};
			eligible.forEach(g => {{
				allocByIdx[g.idx] = g.allocPct;
			}});
			
			// Update UI
			let totalAlloc = 0;
			let totalAmount = 0;
			let positiveBets = 0;
			let includedGames = 0;
			
			gameStats.forEach(g => {{
				const row = document.getElementById(`row-${{g.idx}}`);
				const evCell = document.getElementById(`ev-${{g.idx}}`);
				const allocCell = document.getElementById(`alloc-${{g.idx}}`);
				const amountCell = document.getElementById(`amount-${{g.idx}}`);
				const betSideCell = document.getElementById(`bet-side-${{g.idx}}`);
				
				const alloc = allocByIdx[g.idx] || 0;
				const amount = (alloc / 100) * bankroll;
				
				// Update cells
				betSideCell.textContent = g.betSide;
				betSideCell.className = `bet-side ${{g.betSide.toLowerCase()}}`;
				
				evCell.textContent = (g.muBest * 100).toFixed(1) + '%';
				evCell.className = g.muBest > 0 ? 'positive' : 'negative';
				
				allocCell.textContent = alloc > 0 ? alloc.toFixed(1) + '%' : '-';
				amountCell.textContent = alloc > 0 ? '€' + amount.toFixed(2) : '-';
				
				// Row styling
				row.classList.remove('excluded', 'ev-positive', 'ev-negative', 'ev-flipped-positive', 'ev-flipped-negative');
				
				if (!g.isIncluded) {{
					row.classList.add('excluded');
				}} else {{
					includedGames++;
					
					// Check if EV flipped
					const wasPositive = g.origMuBest > 0;
					const isPositive = g.muBest > 0;
					
					if (!wasPositive && isPositive) {{
						row.classList.add('ev-flipped-positive');
					}} else if (wasPositive && !isPositive) {{
						row.classList.add('ev-flipped-negative');
					}} else if (isPositive) {{
						row.classList.add('ev-positive');
					}}
					
					if (g.muBest > 0) {{
						positiveBets++;
						totalAlloc += alloc;
						totalAmount += amount;
					}}
				}}
			}});
			
			// Update summary
			document.getElementById('summaryGames').textContent = includedGames;
			document.getElementById('summaryBets').textContent = positiveBets;
			document.getElementById('summaryAlloc').textContent = totalAlloc.toFixed(1) + '%';
			document.getElementById('summaryAmount').textContent = '€' + totalAmount.toFixed(2);
		}}
	</script>
</body>
</html>
"""
	
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, "w", encoding="utf-8") as f:
		f.write(html)
	
	print(f"Generated interactive HTML report: {output_path}")
