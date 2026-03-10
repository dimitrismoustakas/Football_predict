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


def build_email_html(
	predictions_df: pd.DataFrame,
	bets_df: pd.DataFrame | None,
	report_date: str,
	fixed_budget: float | None = None,
	budget_strategy: str | None = None,
	kelly_fraction: float | None = None,
) -> str:
	"""Render the HTML email body for prediction reports."""

	value_display = bets_df if bets_df is not None and not bets_df.empty else None
	budget_label = f"{fixed_budget:.2f}" if fixed_budget is not None else "n/a"
	strategy_label = budget_strategy or "n/a"
	kelly_label = f"{kelly_fraction:.2f}" if kelly_fraction is not None else "n/a"
	value_section = ""
	if value_display is not None:
		columns = [
			"Date",
			"Time",
			"League",
			"Home",
			"Away",
			"Result_Value_Side",
			"Result_Value_Prob",
			"Result_Value_Implied",
			"Result_Edge",
			"Result_EV",
			"Result_Budget_Share",
			"Result_Budget_Amount",
		]
		value_section = f"""
		<h4>Positive EV Result Picks</h4>
		{value_display[columns].to_html(index=False)}
		"""
	else:
		value_section = """
		<h4>Positive EV Result Picks</h4>
		<p>No positive EV result bets found for this period.</p>
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
		<p>Budget split strategy: <strong>{strategy_label}</strong> | Kelly fraction: <strong>{kelly_label}</strong> | Fixed budget: <strong>{budget_label}</strong></p>
		<h4>All Predictions</h4>
		{predictions_df.to_html(index=False)}
		{value_section}
		<hr>
		<h4>Interactive HTML Attached</h4>
		<p>Open the attached <strong>upcoming_predictions.html</strong> file in your browser for a formatted version of the report.</p>
	</body>
	</html>
	"""


def send_email(
	csv_path: Path,
	html_path: Path,
	predictions_df: pd.DataFrame,
	bets_df: pd.DataFrame | None,
	recipients: list,
	fixed_budget: float | None = None,
	budget_strategy: str | None = None,
	kelly_fraction: float | None = None,
):
	"""Send the result prediction report and attachments."""

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
		fixed_budget=fixed_budget,
		budget_strategy=budget_strategy,
		kelly_fraction=kelly_fraction,
	)

	msg = MIMEMultipart("alternative")
	msg["Subject"] = f"Football Predictions (NN) - {today_str}"
	msg["From"] = sender_email
	msg["To"] = ", ".join(recipients)
	msg.attach(MIMEText(html_body, "html"))

	for attachment_path in [csv_path, html_path]:
		with open(attachment_path, "rb") as file:
			part = MIMEBase("application", "octet-stream")
			part.set_payload(file.read())
			encoders.encode_base64(part)
			part.add_header("Content-Disposition", f"attachment; filename={attachment_path.name}")
			msg.attach(part)

	try:
		with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
			smtp.login(sender_email, sender_password)
			smtp.send_message(msg)
		print("Email sent successfully.")
	except Exception as exc:
		print(f"Failed to send email: {exc}")
