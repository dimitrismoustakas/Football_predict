"""
Email utility functions for sending match-result reports.
"""

import os
import smtplib
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd


def _build_email_bets_table(bets_df: pd.DataFrame) -> str:
	"""Render a compact positive-EV table for the email body."""

	display = bets_df.copy()
	display["Match"] = display["Home"] + " vs " + display["Away"]
	display["Bet"] = display["Result_Value_Side"]
	display["Odds"] = display.apply(
		lambda row: (
			row["Odds_Home"]
			if row["Result_Value_Side"] == "Home"
			else row["Odds_Draw"]
			if row["Result_Value_Side"] == "Draw"
			else row["Odds_Away"]
		),
		axis=1,
	).map(lambda value: f"{value:.2f}")
	display["Model %"] = (display["Result_Value_Prob"] * 100).map(lambda value: f"{value:.2f}%")
	display["Market %"] = (display["Result_Value_Implied"] * 100).map(lambda value: f"{value:.2f}%")
	display["Edge"] = (display["Result_Edge"] * 100).map(lambda value: f"{value:.2f} pts")
	display["EV %"] = (display["Result_EV"] * 100).map(lambda value: f"{value:.2f}%")
	columns = ["Date", "Time", "League", "Match", "Bet", "Odds", "Model %", "Market %", "Edge", "EV %"]
	return display[columns].to_html(index=False)


def build_email_html(
	predictions_df: pd.DataFrame,
	bets_df: pd.DataFrame | None,
	report_date: str,
) -> str:
	"""Render the HTML email body for prediction reports."""

	value_display = bets_df if bets_df is not None and not bets_df.empty else None
	match_count = len(predictions_df)
	value_count = len(value_display) if value_display is not None else 0
	if value_display is not None:
		value_section = f"""
		<h4>Positive EV Games</h4>
		<p>These are the games with positive expected value from this production run.</p>
		{_build_email_bets_table(value_display)}
		"""
	else:
		value_section = """
		<h4>Positive EV Games</h4>
		<p>No positive expected value result bets found for this period.</p>
		"""

	return f"""
	<html>
	<head>
		<style>
			table {{ border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 12px; }}
			th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
			th {{ background-color: #1f2937; color: white; }}
			tr:nth-child(even) {{ background-color: #f8fafc; }}
			h2 {{ color: #111827; }}
		</style>
	</head>
	<body>
		<h2>Football Predictions - {report_date}</h2>
		<h3>Match Result - Top 5 European Leagues</h3>
		<p>Scanned <strong>{match_count}</strong> matches and identified <strong>{value_count}</strong> games with positive expected value.</p>
		<p>To get bankroll allocation, download the attached HTML file and open it in a browser.</p>
		{value_section}
		<hr>
		<h4>HTML Attachment</h4>
		<p>The attached <strong>upcoming_predictions.html</strong> file lets you edit odds, enter your current bankroll, and recalculate the suggested stakes.</p>
	</body>
	</html>
	"""


def send_email(
	html_path: Path,
	predictions_df: pd.DataFrame,
	bets_df: pd.DataFrame | None,
	recipients: list[str],
):
	"""Send the result prediction report and HTML attachment."""

	if not recipients:
		print("No email recipients defined. Skipping email.")
		return

	sender_email = os.environ.get("EMAIL_USER")
	sender_password = os.environ.get("EMAIL_PASS")
	if not sender_email or not sender_password:
		print("EMAIL_USER or EMAIL_PASS not set. Skipping email.")
		return

	print(f"Sending email to {recipients}...")
	today_str = datetime.now().strftime("%Y-%m-%d")
	html_body = build_email_html(
		predictions_df=predictions_df,
		bets_df=bets_df,
		report_date=today_str,
	)

	msg = MIMEMultipart("alternative")
	msg["Subject"] = f"Football Predictions (NN) - {today_str}"
	msg["From"] = sender_email
	msg["To"] = ", ".join(recipients)
	msg.attach(MIMEText(html_body, "html"))

	with open(html_path, "rb") as file:
		part = MIMEBase("application", "octet-stream")
		part.set_payload(file.read())
		encoders.encode_base64(part)
		part.add_header("Content-Disposition", f"attachment; filename={html_path.name}")
		msg.attach(part)

	try:
		with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
			smtp.login(sender_email, sender_password)
			smtp.send_message(msg)
		print("Email sent successfully.")
	except Exception as exc:
		print(f"Failed to send email: {exc}")
