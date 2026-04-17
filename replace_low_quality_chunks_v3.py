"""
Third round of chunk quality cleanup for studio_manual.

Run AFTER manually deleting these 8 rows:
  arch/data-storage-flow
  cq_38b_data_storage
  arch/api-gateway
  arch/technology-stack
  pitch/delivery-metrics
  pitch/stakeholder-value
  cq_35_create_ui_screen
  cq_36_look_and_feel

SQL:
  DELETE FROM studio_manual WHERE id IN (
    'arch/data-storage-flow', 'cq_38b_data_storage',
    'arch/api-gateway', 'arch/technology-stack',
    'pitch/delivery-metrics', 'pitch/stakeholder-value',
    'cq_35_create_ui_screen', 'cq_36_look_and_feel'
  );

This script upserts 4 improved replacements:
  arch/data-storage         — merges arch/data-storage-flow + cq_38b_data_storage
  arch/tech-infrastructure  — merges arch/api-gateway + arch/technology-stack
  pitch/value-and-metrics   — merges pitch/delivery-metrics + pitch/stakeholder-value
  cq_35_ui_screens          — merges cq_35_create_ui_screen + cq_36_look_and_feel

Run: python replace_low_quality_chunks_v3.py
"""
from utils import get_conn
from retrieval import get_embedding

TABLE = "studio_manual"

REPLACEMENTS = [
    {
        "id": "arch/data-storage",
        "document": """DIGIT Studio — Data Storage, Data Flow, and Where Data Lives

Where Data Is Stored:
- Application and transactional data is stored in PostgreSQL databases used by different DIGIT backend services
- Search and analytics data is indexed in Elasticsearch
- Documents/files are stored in the File Store service (object storage like S3 or similar)
- Data is managed by platform services and is segregated by tenant (e.g., city or department)

Data Storage Types:
- Relational Database (PostgreSQL): Primary data store for transactional data
- Analytical Database (Elasticsearch): Analytics, full-text search, and log aggregation
- File Store: Object storage for uploaded documents and generated PDFs

Write/Update Data Flow:
Any Service → Queue Service → Persister → PostgreSQL
                            ↓
             Signed Audit → Indexer → Elasticsearch

Read Data Flow:
Any Registry → PostgreSQL (direct read)
Dashboard Backend Service → Elasticsearch (analytics queries)

Authentication Flow:
Client Request → API Gateway → User Service (Authenticate)
                             ↓
              Access Control (Authorise) → Target Service

Summary: Data is primarily stored in PostgreSQL for transactions, Elasticsearch for search/analytics, and File Store for documents. All data is tenant-segregated.""",
    },
    {
        "id": "arch/tech-infrastructure",
        "document": """DIGIT Studio — Technology Stack and API Gateway

Technology Stack:
Layer → Technology:
- Database: PostgreSQL
- Search/Analytics: Elasticsearch
- Message Queue: Kafka
- API Gateway: Zuul/Spring Gateway
- Container Orchestration: Kubernetes

Data is stored in PostgreSQL as the primary transactional store, and indexed in Elasticsearch for analytics and full-text search. Services communicate asynchronously via Kafka message queues for write/update operations. Kubernetes manages container deployment and scaling.

API Gateway:
The API Gateway acts as the single entry point for all client requests from the three front-end applications (Studio Designer, Studio Service Application, Studio Service Dashboard).

The API Gateway handles:
- Request routing to appropriate microservices
- Authentication token validation
- Rate limiting and throttling
- Request/response transformation
- Load balancing across service instances

All communication between front-end applications and backend services goes through the API Gateway.""",
    },
    {
        "id": "pitch/value-and-metrics",
        "document": """DIGIT Studio — Value Delivered and Stakeholder Benefits

Key Metrics and Benefits:
- Less than 1 Week to Build an App: Launch new services in days instead of months
- 50-70% Time Saving: Minimize back-and-forth communication delay between teams
- Reusable Templates with Built-in Best Practices: Pre-configured templates embed proven public service delivery practices, helping teams design services faster and more effectively
- Powered by DIGIT: Leverages the DIGIT platform's secure, scalable foundation to deliver citizen services with confidence and long-term reliability

Value for Every Stakeholder:

Government Department Leaders:
- Reduce risk and cost of digitization
- Improve citizen satisfaction
- Scale across regions
- Deliver visible impact within their tenure

Program Managers:
- Stay responsive to changing needs
- Align stakeholders with ease
- Deliver visible impact faster

Implementation Partners:
- Stay aligned with program managers
- Cut back-and-forth delays
- Deliver compliant services faster""",
    },
    {
        "id": "cq_35_ui_screens",
        "document": """UI Screens and Look & Feel in DIGIT Studio

How UI Screens Are Created:
Creating a new UI screen in DIGIT Studio does not require coding. A UI screen is what users interact with, such as a form or a page within a service.

DIGIT Studio automatically generates UI screens based on Service configurations. Forms are rendered as per the sections and fields configured by you. Predefined page formats include: Citizen landing page, view application page, employee inbox, and others.

Important: You cannot create standalone or custom screens outside this configuration flow. Screens are always generated from forms or checklists — you cannot manually add an independent "next screen" outside of this structure.

You can preview the screens in the same environment and publish the service once ready.

Look and Feel / UI Customization:
In DIGIT Studio, the UI is generated based on DIGIT platform standards. The look and feel can be customized by modifying the CSS configuration at the platform level.

The generated UI is:
- Accessible and compliant with WCAG 2.0 standards
- Designed based on experience from multiple public service delivery applications

Advanced layout options (such as more flexible or horizontal layouts) are planned for future releases.

It is recommended to keep UI changes minimal to maintain consistency, accessibility, and usability.

Summary: Screens are auto-generated from your service configuration — no coding needed. UI styling can be adjusted via CSS at the platform level, but minimal changes are recommended to maintain accessibility and consistency.""",
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
        print("\n✅ Done. 83 → 79 rows after deletions + 4 new replacements.")

    finally:
        conn.close()


if __name__ == "__main__":
    replace()
