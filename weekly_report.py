"""
DIGIT Studio Bot — Weekly Report Runner

Run manually:
    python weekly_report.py

Or schedule via cron (every Monday 9am):
    0 9 * * 1 /path/to/venv/bin/python /path/to/weekly_report.py

Required env vars:
    # Slack (optional):
    SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

    # Email (optional):
    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USER=your@gmail.com
    SMTP_PASS=yourapppassword
    REPORT_EMAIL=recipient@example.com

At least one of SLACK_WEBHOOK_URL or REPORT_EMAIL must be set.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

from utils import generate_weekly_report, send_slack_report, send_email_report


def run_weekly_report(days: int = 7):
    print(f"\n{'='*50}")
    print("DIGIT Studio Bot — Weekly Report")
    print(f"{'='*50}\n")

    print(f"Generating report for the last {days} days...")
    report = generate_weekly_report(days=days)

    stats = report["stats"]
    print(f"\nStats:")
    print(f"  Total queries    : {stats.get('total', 0)}")
    print(f"  Satisfaction     : {stats.get('satisfaction', 0)}%")
    print(f"  Positive         : {stats.get('positive', 0)}")
    print(f"  Negative         : {stats.get('negative', 0)}")
    print(f"  From cache       : {stats.get('from_cache', 0)}")
    print(f"  From RAG         : {stats.get('from_rag', 0)}")
    print(f"  Auto-promoted    : {report['auto_promoted_count']}")
    print(f"  Flagged items    : {report['flagged_count']}")

    if report["flagged_items"]:
        print(f"\nFlagged questions (thumbs down):")
        for item in report["flagged_items"]:
            print(f"  [{item['created_at'][:10]}] {item['query'][:80]}")

    sent_somewhere = False
    errors = []

    # ── Slack ──
    slack_url = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    if slack_url:
        try:
            send_slack_report(slack_url, report)
            print("\n✅ Slack report sent")
            sent_somewhere = True
        except Exception as e:
            errors.append(f"Slack: {e}")
            print(f"\n❌ Slack failed: {e}")

    # ── Email ──
    report_email = os.environ.get("REPORT_EMAIL", "").strip()
    if report_email:
        try:
            send_email_report(report, to_email=report_email)
            print(f"✅ Email report sent to {report_email}")
            sent_somewhere = True
        except Exception as e:
            errors.append(f"Email: {e}")
            print(f"\n❌ Email failed: {e}")

    if not sent_somewhere and not errors:
        print(
            "\n⚠️  No delivery channel configured.\n"
            "   Set SLACK_WEBHOOK_URL and/or REPORT_EMAIL in your .env file."
        )

    print(f"\n{'='*50}\n")
    return report


if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    run_weekly_report(days=days)
