"""
Email utility functions for sending reports and predictions.
"""

import os
import smtplib
from pathlib import Path
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import pandas as pd


def send_email(
	csv_path: Path,
	html_path: Path,
	predictions_df: pd.DataFrame,
	bets_df: pd.DataFrame,
	recipients: list
):
	"""
	Send email with predictions CSV, interactive HTML, and betting recommendations.
	
	Args:
		csv_path: Path to CSV file with predictions
		html_path: Path to interactive HTML report
		predictions_df: DataFrame with all predictions
		bets_df: DataFrame with betting recommendations (or None)
		recipients: List of email addresses
	"""
	if not recipients:
		print("No email recipients defined. Skipping email.")
		return
	
	sender_email = os.environ.get("EMAIL_USER")
	sender_password = os.environ.get("EMAIL_PASS")
	
	if not sender_email or not sender_password:
		print("EMAIL_USER or EMAIL_PASS not set. Skipping email.")
		return
	
	print(f"Sending email to {recipients}...")
	
	# Build email body
	today_str = datetime.now().strftime("%Y-%m-%d")
	
	# Create display versions without clutter columns
	columns_to_hide = ["Implied_Over", "Implied_Under", "Model_Odds_Over", "Model_Odds_Under"]
	predictions_display = predictions_df.drop(columns=[col for col in columns_to_hide if col in predictions_df.columns])
	bets_display = bets_df.drop(columns=[col for col in columns_to_hide if col in bets_df.columns]) if bets_df is not None and not bets_df.empty else None
	
	html_body = f"""
	<html>
	<head>
		<style>
			table {{ border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 12px; }}
			th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
			th {{ background-color: #4CAF50; color: white; }}
			tr:nth-child(even) {{ background-color: #f2f2f2; }}
			.positive {{ color: green; font-weight: bold; }}
			.negative {{ color: red; }}
			h2 {{ color: #333; }}
		</style>
	</head>
	<body>
		<h2>Football Predictions - {today_str}</h2>
		<h3>Over/Under 2.5 Goals - Top 5 European Leagues</h3>
		
		<h4>All Predictions</h4>
		{predictions_display.to_html(index=False, classes='predictions')}
	"""
	
	if bets_display is not None and not bets_display.empty:
		html_body += f"""
		<h4>Betting Recommendations (Positive EV)</h4>
		<p>Budget allocation percentages based on Sharpe-weighted portfolio strategy:</p>
		{bets_display[["Date", "Time", "Home", "Away", "Bet_Side", "Odds_Over", "Odds_Under", "EV", "Allocation_Pct"]].to_html(index=False, classes='bets')}
		<p><strong>Total Allocation: {bets_df['Allocation_Pct'].sum():.2f}%</strong></p>
		<p><em>To use: If your total budget is €10, multiply each Allocation_Pct by €0.1 to get the bet amount.</em></p>
		"""
	else:
		html_body += """
		<h4>Betting Recommendations</h4>
		<p>No positive expected value bets found for this period.</p>
		"""
	
	html_body += """
		<hr>
		<h4>📊 Interactive Tool Attached</h4>
		<p>Open the attached <strong>upcoming_predictions.html</strong> file in your browser to:</p>
		<ul>
			<li>Adjust odds to match your bookmaker's prices</li>
			<li>See recalculated EV and allocations in real-time</li>
			<li>Filter to only games where you've confirmed the odds</li>
			<li>Set your bankroll to see exact bet amounts</li>
		</ul>
	</body>
	</html>
	"""
	
	# Create message
	msg = MIMEMultipart("alternative")
	msg["Subject"] = f"Football Predictions (NN) - {today_str}"
	msg["From"] = sender_email
	msg["To"] = ", ".join(recipients)
	
	# Attach HTML body
	msg.attach(MIMEText(html_body, "html"))
	
	# Attach CSV
	with open(csv_path, "rb") as f:
		part = MIMEBase("application", "octet-stream")
		part.set_payload(f.read())
		encoders.encode_base64(part)
		part.add_header("Content-Disposition", f"attachment; filename={csv_path.name}")
		msg.attach(part)
	
	# Attach interactive HTML
	with open(html_path, "rb") as f:
		part = MIMEBase("application", "octet-stream")
		part.set_payload(f.read())
		encoders.encode_base64(part)
		part.add_header("Content-Disposition", f"attachment; filename={html_path.name}")
		msg.attach(part)
	
	try:
		with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
			smtp.login(sender_email, sender_password)
			smtp.send_message(msg)
		print("Email sent successfully.")
	except Exception as e:
		print(f"Failed to send email: {e}")
