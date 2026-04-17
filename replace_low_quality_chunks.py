"""
Replace low-quality chunks in studio_manual with improved merged versions.

Run AFTER manually deleting these 6 rows:
  cq_12b_test_workflows
  cq_32_deploy_application
  cq_10_responsive_apps
  cq_11_mobile_apps
  cq_18_where_used
  cq_28_use_cases_supported_v2

This script upserts 3 improved replacements:
  cq_09_preview_changes        — extended (was good, minor addition)
  cq_10_responsive_and_mobile  — merges old cq_10 + cq_11
  cq_19_use_cases_supported    — merges old cq_19 + cq_28 + cq_18

Run: python replace_low_quality_chunks.py
"""
from utils import get_conn
from retrieval import get_embedding

TABLE = "studio_manual"

REPLACEMENTS = [
    {
        "id": "cq_09_preview_changes",
        "document": """DIGIT Studio provides a Preview feature that allows you to test workflows before publishing and making them live. It helps simulate the application behavior in a controlled environment without affecting real users or data.

Using Preview: While working in Draft mode, you can use Preview to quickly validate your workflow by:
- Simulating the end-to-end workflow flow
- Testing states and transitions
- Validating role-based actions and permissions
- Verifying form behavior and configurations
- Identifying issues before publishing

How It Works:
1. Configure your application in Draft mode
2. Launch it using the Preview option
3. Perform actions and observe state transitions
4. Test with different user roles (if supported for your configuration)

What This Ensures:
- All transitions are working correctly
- Roles are properly assigned
- No steps are missing or misconfigured

Summary: Use Preview to validate your workflow end-to-end before publishing, ensuring a smooth and error-free experience.""",
    },
    {
        "id": "cq_10_responsive_and_mobile",
        "document": """DIGIT Studio apps are responsive and work across mobile, tablet, and web platforms.

Key points:
- All applications built on DIGIT Studio are fully responsive — they adapt to mobile, tablet, and desktop/web screens without any extra configuration.
- Citizens and employees can use the generated apps on any device.

Can I build a native mobile app on DIGIT Studio?
DIGIT Studio does not currently have the functionality to build standalone native mobile apps. However, since all generated applications are mobile-responsive, they work well on mobile browsers and provide a mobile-like experience.

Summary: You do not need to build a separate mobile app — DIGIT Studio's generated apps are responsive and work on mobile, tablet, and web by default.""",
    },
    {
        "id": "cq_19_use_cases_supported",
        "document": """DIGIT Studio supports Public Service Delivery use cases that follow a structured process:

Apply — Form for citizen to apply, or for an employee to apply on behalf of the citizen
Process — Stages through which the application passes to reach completion
Inform/Track — Keep the applicant informed on the status of their application
Resolve — Close/Reopen/Reject an Application

Applications built on DIGIT Studio come with both an employee interface and a citizen interface.

Typical use cases include:
- License or permit systems
- Complaint or grievance management
- Registration and approval-based services
- Inspection or checklist-based workflows

What these use cases have in common:
- Form-based data collection (users submit requests or applications)
- Workflow steps (requests move through roles/stages for approval or action)
- Task management (officials act on requests via Inbox)
- Document handling (uploads and generated outputs like certificates/receipts)
- Notifications (SMS, email, or app updates)
- Payments (optional, if fees are involved)

When it may NOT be suitable:
- Highly custom UI interactions beyond configurable forms
- Systems without a clear workflow or request lifecycle
- Real-time heavy processing or complex computations

Where has DIGIT Studio been used?
DIGIT Studio is currently in MVP stage. A version of Studio has been used to build 15 use cases of Building Permits in Djibouti, which is currently in UAT stage.

Summary: If a use case can be structured as form → workflow → outcome (approval/rejection/record), it is a strong fit for DIGIT Studio.""",
    },
]


def replace():
    conn = get_conn()
    print(f"Upserting {len(REPLACEMENTS)} replacement chunks into {TABLE}...")

    try:
        with conn.cursor() as cur:
            for i, chunk in enumerate(REPLACEMENTS):
                chunk_id = chunk["id"]
                text = chunk["document"]
                try:
                    emb = get_embedding(text)
                    cur.execute(f"""
                        INSERT INTO {TABLE} (id, document, embedding)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            document = EXCLUDED.document,
                            embedding = EXCLUDED.embedding
                    """, (chunk_id, text, emb))
                    print(f"  [{i+1}/{len(REPLACEMENTS)}] ✅ {chunk_id}")
                except Exception as e:
                    print(f"  [{i+1}/{len(REPLACEMENTS)}] ❌ {chunk_id}: {e}")

        conn.commit()
        print("\n✅ Done.")

    finally:
        conn.close()


if __name__ == "__main__":
    replace()
