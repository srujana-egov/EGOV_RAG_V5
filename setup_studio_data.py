import os
import json
import sys
from dotenv import load_dotenv

load_dotenv()

try:
    import psycopg2
    from pgvector.psycopg2 import register_vector
except ImportError:
    print("❌ Missing psycopg2 or pgvector. Run: pip install psycopg2-binary pgvector")
    sys.exit(1)


# ─────────────────────────────────────────────
# Studio Q&A data
# (kept short here — keep your full list)
# ─────────────────────────────────────────────
STUDIO_QA = [
    {
        "question": "What environments are available dev staging prod",
        "answer": """DIGIT Studio can be deployed in any DIGIT environment you choose. You can have dev, staging/UAT and production, or just dev and production.

On Studio, you can build and preview applications, test changes, and once satisfied, **Export** the service configuration and **Import** it to another Studio environment (UAT or production).

**Export/Import works for:**
- Service Configuration

**Export/Import does NOT work for:**
- Data you've added
- Users you've created"""
    },
    {
        "question": "What can I build using DIGIT Studio",
        "answer": """DIGIT Studio lets you build complete digital public services without heavy coding.

**You can build:**
- Public service applications (permits, grievance systems)
- User-friendly forms for citizens or employees
- Approval workflows through different officials
- Task dashboards (Inbox) for officials
- Document handling (uploads, certificates, receipts, PDFs)
- Notifications (SMS, email, app alerts)
- Multi-tenant systems for multiple departments or cities

**Not currently supported:**
- Payment-enabled services with fee calculation (API-based only for now)
- Dashboards and Reports

**Example:** A Trade License service where a user applies online, officials review and approve, fees are paid, and a license certificate is generated."""
    },
    {
        "question": "What is a Service in DIGIT Studio",
        "answer": """In DIGIT Studio, a **Service** is a logical grouping used to organize related modules under a common domain.

- A service represents a domain or category (e.g., Trade License, Complaint Management, Health)
- Each service can contain one or more modules
- It helps organize configurations, workflows, and data for related modules
- Used across configurations and APIs to identify where a module belongs

**Example:** A "Trade License" service can include modules for new license application, renewal, and modification."""
    },
    {
        "question": "What is a module in DIGIT Studio",
        "answer": """In DIGIT Studio, a **Module** is the actual digital use case or application that users interact with.

- Created within a service
- Represents a specific functionality (applying for a license, raising a complaint, requesting a permit)
- Currently there is typically a one-to-one relationship between service and module in the UI
- In future, a service can support multiple modules

**A module usually includes:**
- Forms to collect user input
- Workflow for processing and approvals
- Document handling (uploads and generated outputs)
- Inbox/tasks for officials
- Notifications
- Optional payment support"""
    },
    {
        "question": "What is the difference between a module and a Service",
        "answer": """In DIGIT Studio, service and module represent two different levels of organization.

**Service** = High-level grouping or domain
- Example: Trade License, Complaint Management

**Module** = The actual application or use case users interact with, created within a service
- Example: Applying for a new license, renewing a license, submitting a complaint

**Example:**
- Service: Trade License
  - Module 1: New Trade License Application
  - Module 2: Trade License Renewal
  - Module 3: Trade License Modification

Note: Currently the UI shows a one-to-one mapping, but the platform supports multiple modules under a single service."""
    },
    {
        "question": "What is the difference between a checklist and a form",
        "answer": """**Form:** Used to collect detailed information from users. Includes input fields where users enter data (name, address, business details, etc.).

**Checklist:** Used during verification, inspection, or feedback stages. Allows users or officials to mark responses against predefined items (Yes/No, ratings, remarks).

**Example (Trade License):**
- *Form (citizen)*: Applicant name, business type, address, required documents
- *Checklist (inspector)*: Are documents valid? Is location correct? Does applicant meet eligibility?
- *Checklist (feedback)*: Was the process smooth? Rate your experience.

**Key difference:**
- Form = Detailed data collection
- Checklist = Validation, inspection, or feedback"""
    },
    {
        "question": "How do I preview changes before publishing",
        "answer": """DIGIT Studio provides a **Preview** feature to test workflows before publishing.

**Using Preview:**
- Simulate the end-to-end workflow
- Test states and transitions
- Validate role-based actions and permissions
- Verify form behavior and configurations
- Identify issues before publishing

**How it works:**
1. Configure your application in Draft mode
2. Launch using the Preview option
3. Perform actions and observe state transitions
4. Test with different user roles

Use Preview to validate your workflow end-to-end before publishing, ensuring a smooth and error-free experience."""
    },
    {
        "question": "Are apps generated on Studio responsive can I build mobile apps",
        "answer": """**Responsive:** Yes, applications built on DIGIT Studio are responsive and work on mobile, tablet, and web.

**Mobile apps:** DIGIT Studio currently does not have the functionality to build native mobile apps. However, all applications are mobile-responsive and work well in mobile browsers."""
    },
    {
        "question": "Can I edit my app after publishing",
        "answer": """DIGIT Studio allows unlimited iterations on a **Draft** app — you can preview and edit as many times as needed.

**Once published:** Currently, a published application cannot be edited in DIGIT Studio. Publishing marks the configuration as final and immutable, ensuring consistency for all users and ongoing processes.

Support for editing published applications is planned as a future enhancement."""
    },
    {
        "question": "How will citizens see the services I created",
        "answer": """Once published in DIGIT Studio, services become available through the **Citizen Portal**.

**How it works:**
1. Application is published in DIGIT Studio
2. It gets mapped to a business service/module
3. The service is rendered on the Citizen Portal
4. Citizens select the service and initiate an application

Only published services are visible. Services are shown based on configured access. The Citizen URL is available on the Post Go-Live / Publish success screen in DIGIT Studio."""
    },
    {
        "question": "What is the default citizen role",
        "answer": """Applications built on DIGIT Studio come with a mobile-responsive interface for user type **'Citizen'**.

**Citizens can:**
- Self-register using mobile OTP or email OTP
- Apply for a public service
- Track current and past application status
- Perform tasks assigned to them in the workflow

There is currently no way to disable the citizen role."""
    },
    {
        "question": "How do I assign permissions to roles",
        "answer": """Permissions in DIGIT Studio are assigned to roles to control what actions users can perform.

**Steps:**
1. Go to **Manage Roles** section in DIGIT Studio
2. Create or select a role (e.g., Reviewer, Approver, Inspector)
3. Define permissions: Create application, View application, Edit application
4. Map role to workflow actions (e.g., APPROVE, REJECT)
5. Assign role to users via user management (HRMS)

**Example — Inspector role:**
- Permissions: View and Edit application (not Create)
- Workflow action: "Inspection" tagged to Inspector role
- Result: Inspector only sees the application when it's in the Inspection workflow state

Roles define what a user can see and do. Workflow mapping ensures users can act only at the right stage."""
    },
    {
        "question": "Can dropdown values be used across apps",
        "answer": """Yes. DIGIT apps are powered by the **MDMS service**. Master data configured in MDMS can be used across applications and added to form dropdowns using the option **'Pull data from Database'**.

For dropdown values specific to a use case, you can add the data directly within the field configuration itself."""
    },
    {
        "question": "How do I create a dependent dropdown",
        "answer": """A dependent dropdown shows or filters options in one field based on the value selected in another. No coding required.

**How it works:**
- Parent field (Field X) → controls the selection (e.g., Complaint Type)
- Child field (Field Y) → updates based on the parent (e.g., Complaint Subtype)

**Steps:**
1. Create the parent dropdown with all required options
2. Create the child dropdown with all possible options
3. Select the child field and enable **Dependent Field** under the Logic tab
4. Click **Add Display Logic** and define conditions:
   - If Complaint Type = Water → Show Leakage, No Supply
   - If Complaint Type = Garbage → Show Not Collected, Overflow

**Runtime behavior:**
- When parent field changes, child dropdown updates instantly
- Only relevant options are shown
- If no options available, field is automatically hidden"""
    },
    {
        "question": "What is SLA",
        "answer": """**SLA** (Service Level Agreement) is the time in which a user must complete a particular action.

An SLA can be defined at an action level (e.g., document verification must be completed within 72 hours). In the employee interface, officials can see tasks pending for them and the time remaining, helping them prioritize tasks nearing or past SLA.

**Future:** Rules with SLAs will be supported (e.g., escalate to department head if inspection is pending for more than 3 days)."""
    },
    {
        "question": "Where has DIGIT Studio been used in the past",
        "answer": """DIGIT Studio is currently in **MVP stage** and not used in Production.

However, a version of Studio has been used to build 15 use cases of **Building Permits in Djibouti**, which is currently in UAT stage."""
    },
    {
        "question": "What use cases can DIGIT Studio support",
        "answer": """DIGIT Studio supports **Public Service Delivery** use cases that follow a set process:

- **Apply** — Form for citizen or employee to apply
- **Process** — Stages through which the application passes
- **Inform/Track** — Keep applicant informed on status
- **Resolve** — Close, reopen, or reject an application

**Fits well if your use case has:**
- Form-based data collection
- Workflow or approval steps
- Task management (Inbox)
- Document handling
- Notifications
- Optional payments

**May not be suitable for:**
- Highly custom UI beyond configurable forms
- Real-time heavy processing or complex computations
- Systems without a defined request lifecycle

Applications come with both employee and citizen interfaces."""
    },
    {
        "question": "How do I configure approvals and rejections",
        "answer": """Approvals and rejections are configured through **workflow configuration**.

**Steps:**
1. Define workflow states: INITIATED, IN_REVIEW, APPROVED, REJECTED
2. Define actions (transitions):
   - APPROVE → moves to APPROVED
   - REJECT → moves to REJECTED
   - SEND_BACK → moves to previous state
3. Map actions to states (specify from/to states)
4. Assign roles to actions (only users with that role can perform that action)
5. Actions automatically appear as buttons in the UI based on current state and user role"""
    },
    {
        "question": "Can I have multiple languages in an app",
        "answer": """Yes. All text on apps generated from DIGIT Studio has localization keys. You can add as many languages as needed by adding values in that language against the localization key.

**Note:** Studio currently does not have a UI for adding localization. You add localization using a JSON format. Refer to the DIGIT documentation for details."""
    },
    {
        "question": "Are applications built on Studio accessible",
        "answer": """Yes. Applications generated from Studio follow **WCAG 2.0** accessibility guidelines.

The UI design is inspired by the **Gov.UK design system**, which is the gold standard for accessibility, simplicity, and user experience."""
    },
    {
        "question": "How do I add users",
        "answer": """**Three types of users in DIGIT:**

1. **System users** (Boundary Admin, Super Admin) — created during platform installation
2. **Employees** — created after a service is published via DIGIT User Management
3. **Citizens** — self-register using the Citizen URL

**To add employees:**
1. Go to DIGIT User Management module after publishing
2. Create users with name, mobile number, etc.
3. Assign the appropriate role
4. Users receive login credentials on their registered mobile number

**Citizens** don't need manual creation — they self-register via the Citizen URL shown on the Post Go-Live/Publish success screen."""
    },
    {
        "question": "What are the steps to go live",
        "answer": """**Steps to go live with DIGIT Studio:**

1. **Build in Development** — Create service, design screens, configure workflows
2. **Test using Preview** — Create test users, simulate real scenarios
3. **Finalize** — Ensure all configurations, fields, workflows, documents are complete
4. **Move configuration to Production** — Export config from Dev, import into Production
5. **Set up Production** — Recreate users, roles, and platform settings (not carried over)
6. **Verify in Production** — Test the service again
7. **Go live** — Service is ready for real users

⚠️ **Important:** Only service configuration is moved. Data is not transferred. Users must be created again in each environment."""
    },
    {
        "question": "How do I integrate with SMS email notification services",
        "answer": """DIGIT Studio uses the DIGIT platform's **Notification Service** for SMS and email.

**Steps:**
1. Configure notification templates in DIGIT Studio (SMS/email content with personalized variables)
2. Map notifications to workflow events (e.g., on APPROVE, REJECT)
3. Enable SMS and/or email channels in configuration
4. Configure external providers at platform level (SMS gateways like NIC, Twilio; email via SMTP)

⚠️ In many countries, SMS must be pre-approved by the government. Ensure SMS templates are approved or notifications may be blocked."""
    },
    {
        "question": "Can I roll back a deployment",
        "answer": """DIGIT Studio does not provide a direct one-click rollback option.

**Rollback is manual:**
- If you have a copy of an earlier configuration, re-import it to restore the previous version

**Important:**
- There is no built-in version history or automatic revert
- Recommended to keep backups of configurations before making changes"""
    },
    {
        "question": "Can I build grievance complaint applications",
        "answer": """Yes, you can build grievance or complaint management applications using DIGIT Studio.

**Typical flow:**
1. Citizen submits complaint (issue description, location, category, photos)
2. Complaint assigned to relevant department/official
3. Officials review and take action (resolve, escalate, reject)
4. Application moves through workflow states (Open → In Progress → Resolved)
5. Citizen receives notifications at each stage
6. Final outcome recorded and shared

**Configure in DIGIT Studio:**
- Form: complaint type, description, location, attachments
- Workflow: Submitted → Assigned → In Progress → Resolved/Rejected
- Roles: operator, supervisor with appropriate actions
- Notifications: SMS/email on status changes

A grievance application fits DIGIT Studio perfectly — it follows a clear form → workflow → resolution structure."""
    },
    {
        "question": "How is authentication handled in DIGIT Studio",
        "answer": """Authentication is handled via the **DIGIT platform's core authentication services**.

- Users authenticate by generating an auth token via `/user/oauth/token`
- The API Gateway validates the token and routes requests to backend services
- User Service manages credentials and identity
- **RBAC** (Role-Based Access Control) handles authorization

**For citizens:** Mobile number/email + OTP-based authentication
**For employees:** Username + password (or configured method)

Integration with external identity providers (SSO) can be enabled via API-based integration at the gateway level."""
    },
    {
        "question": "How do I configure notifications in DIGIT Studio",
        "answer": """**Option 1: From Notification Dashboard (Recommended)**
1. Open Notification Dashboard
2. Select Email or SMS
3. Click Create Notification
4. Select Workflow State, Channel (Email/SMS), write Message
5. Click Save — notification is now linked to that workflow state

**Option 2: From Workflow Designer**
1. Open Workflow Designer → Select a State
2. Go to Notifications section → Click + Create Notification
3. Fill Channel and Message → Save

**Important rules:**
- Must select one Workflow State and one Channel
- Notifications sent only to the Applicant
- Triggered automatically when state is reached
- Also requires: service providers for email/SMS configured at platform level"""
    },
    {
        "question": "Can I use my own domain when going live",
        "answer": """Yes, you can use your own custom domain when going live with DIGIT Studio.

The application can be hosted on a custom domain (e.g., `service.city.gov.in`). Domain configuration is managed during DIGIT platform deployment (DNS and hosting setup).

This enables a branded and official access point for end users."""
    },
    {
        "question": "How do I deploy my application in DIGIT Studio",
        "answer": """**Deployment process:**

1. **Build and test in Dev** — Create service, design screens, configure workflows, use Preview to test
2. **Finalize configuration** — Ensure all forms, workflows, settings are complete
3. **Export configuration** — Copy/export service configuration from Dev
4. **Import into Production** — Set up DIGIT Studio in production and import the configuration
5. **Set up environment-specific details** — Configure users, roles, and platform settings in production
6. **Go live** — Application becomes available for real users

⚠️ Only configuration is moved. Application data is not transferred. Users need to be created again."""
    },
    {
        "question": "Can I import export data service configuration",
        "answer": """**Service Configuration:**
- **Export:** Available as JSON for published services — copy/paste the config
- **Import:** Navigate to Studio landing page → click Import Service → upload/paste the JSON

**Use when:** Reusing an application, migrating between environments (Dev → UAT → Prod)

**Application Data:**
- Submitted application data can be downloaded/exported from the application/module section
- Useful for reporting, analysis, and audits

⚠️ Export/Import of service config does NOT include submitted data or users."""
    },
    {
        "question": "Can workflows integrate with external systems",
        "answer": """Currently, workflows in DIGIT Studio operate within the platform — **direct external system integration is not supported out of the box**.

**Current behavior:**
- Workflow execution, state transitions, and actions are handled internally
- No direct UI-based option to connect workflows with external systems

**Future scope:**
- Integrations can be enabled at backend or service layer
- External systems can interact through APIs or custom extensions
- Workflow events can potentially trigger external processes"""
    },
    {
        "question": "How do I create a new UI screen in DIGIT Studio",
        "answer": """Creating a new UI screen in DIGIT Studio requires **no coding**. UI screens are automatically generated based on service configurations.

- Forms are rendered as per the sections and fields you configure
- Citizen landing page, view application page, employee inbox are predefined page formats

**Important:** You cannot create standalone or custom screens outside this configuration flow. Screens are always generated from forms or checklists."""
    },
    {
        "question": "How do I change the look and feel UI of my app",
        "answer": """In DIGIT Studio, the UI is generated based on DIGIT platform standards. Look and feel can be customized by modifying **CSS configuration at the platform level**.

The default UI is:
- Accessible and compliant with **WCAG 2.0** standards
- Designed based on experience from multiple public service delivery applications
- Advanced layout options are planned for future releases

**Recommendation:** Keep UI changes minimal to maintain consistency, accessibility, and usability."""
    },
    {
        "question": "How does DIGIT Studio work with DIGIT platform",
        "answer": """**DIGIT Studio = "Design & Configure"**
**DIGIT Platform = "Execute & Run services at scale"**

DIGIT Studio is a configuration layer on top of the DIGIT platform:
1. You design and configure services (forms, workflows, roles) without writing code
2. Configurations are stored as schemas and JSON-based definitions
3. Once published, DIGIT Studio activates services on the DIGIT platform
4. Core DIGIT microservices execute the configured service:
   - Workflow engine (processing applications)
   - User Service (authentication)
   - Billing & Payment services
   - Notification services (SMS, Email)
5. End-user apps (Citizen & Employee) are auto-generated from configurations
6. All requests flow through the API Gateway"""
    },
    {
        "question": "What are states and actions in workflows",
        "answer": """**States** represent the current stage or status of an application in a workflow.
- Examples: DRAFT, SUBMITTED, IN_REVIEW, APPROVED, REJECTED

**Actions** are operations that move an entity from one state to another.
- Each action defines: From State → To State + Allowed Roles
- Examples: SUBMIT, APPROVE, REJECT, SEND_BACK

**Role-based visibility:**
- Actions are tagged to roles
- Users only see and can perform actions permitted by their role
- APPROVER role → sees Approve/Reject buttons
- CITIZEN role → sees Submit/Edit options

**Example table:**
| Current State | Action | Next State | Allowed Role |
|---|---|---|---|
| DRAFT | SUBMIT | SUBMITTED | CITIZEN |
| SUBMITTED | VERIFY | IN_REVIEW | EMPLOYEE |
| IN_REVIEW | APPROVE | APPROVED | APPROVER |
| IN_REVIEW | REJECT | REJECTED | APPROVER |"""
    },
    {
        "question": "How is DIGIT Studio different from traditional low code no code tools",
        "answer": """DIGIT Studio is purpose-built for **public service systems**, not general-purpose app development.

**Key differences:**
- **Service-first approach:** Builds complete public services with a defined lifecycle (application → workflow → approval → record)
- **Built-in workflow and governance:** Pre-integrated approval workflows, role-based access, and audit trails
- **End-to-end ecosystem:** Forms, workflow, inbox, document handling, notifications, payments — out of the box
- **Configuration-driven standardization:** Structured configurations ensure consistency and reusability
- **Multi-tenancy:** Supports multiple departments or cities with isolated configurations
- **Public sector aligned:** Built for compliance, structured data, and approval-based governance processes

Traditional low-code tools focus on generic apps. DIGIT Studio focuses on standardizing complete, workflow-driven public service systems."""
    },
    {
        "question": "How do I test workflows before going live",
        "answer": """DIGIT Studio provides a **Preview** feature to test workflows before publishing.

**Using Preview:**
- Simulate the end-to-end workflow
- Test states and transitions
- Validate role-based actions and permissions
- Verify form behaviour and configurations
- Identify issues before publishing

**How it works:**
1. Configure your application in Draft mode
2. Launch using the Preview option
3. Perform actions and observe state transitions
4. Test with different user roles

Use Preview to validate your workflow end-to-end before publishing, ensuring a smooth and error-free experience."""
    },
    {
        "question": "What is the downtime during deployment",
        "answer": """Downtime depends on what is being deployed:

**Service deployment (via DIGIT Studio):**
- Configuration-based, so downtime is typically **minimal or negligible**
- Updates are applied by importing the latest service configuration
- The system switches to the updated configuration without a full shutdown
- Users may experience only a brief moment during the update

**Initial DIGIT Studio / platform setup or upgrades:**
- May involve service restarts or infrastructure changes
- Typically planned during maintenance windows
- Downtime can be higher and should be communicated to users in advance

**Summary:** Service-level config updates have near-zero downtime. Platform-level deployments should be planned carefully."""
    },
    {
        "question": "How do I assign roles to workflow steps",
        "answer": """In a workflow, roles are assigned to **actions** (transitions), not directly to states.

**Steps:**
1. **Identify the workflow state** where access needs to be controlled (e.g., IN_REVIEW, SUBMITTED)
2. **Define the action** that moves the workflow (e.g., APPROVE, REJECT, SEND_BACK)
3. **Assign roles to the action** — select one or more roles (CITIZEN, EMPLOYEE, APPROVER) who can perform it
4. Only users with the assigned role will see and be able to perform that action

**Important:**
- A user without the required role will not see the action button in the UI
- Incorrect role configuration can cause workflow to get stuck (no one able to act)
- Roles should match the user-role mapping defined in HRMS/user management

**Example:**
- State: IN_REVIEW → Action: APPROVE → Role: APPROVER
- Only users with the APPROVER role can approve; others see no Approve button."""
    },
    {
        "question": "How does role based access work",
        "answer": """**Role-Based Access Control (RBAC)** ensures users can only perform actions and access features based on their assigned roles.

**How it works:**
- Each user is assigned one or more roles (e.g., CITIZEN, EMPLOYEE, APPROVER)
- Each workflow action is configured with allowed roles
- When a user accesses an application:
  - The system checks the current state
  - Filters available actions based on the user's roles
  - Only permitted action buttons are shown in the UI

**Examples:**
- APPROVER role → sees Approve/Reject buttons
- CITIZEN role → sees Submit/Track options
- EMPLOYEE role → sees Verify/Forward options

**Key points:**
- A user can have multiple roles
- Access is role-driven, not user-specific
- Incorrect role configuration can hide required actions or block workflow progression"""
    },
    {
        "question": "Where is the data stored",
        "answer": """In DIGIT Studio (part of the DIGIT platform), data is stored in the DIGIT platform's backend systems:

- **Application and transactional data** → stored in **PostgreSQL** databases used by DIGIT backend services
- **Search and analytics data** → indexed in **Elasticsearch**
- **Documents and files** → stored in the **File Store service** (object storage, e.g., S3 or equivalent)

Data is managed by the platform services and segregated by **tenant** (city or department).

**Summary:** Data is primarily in PostgreSQL, with Elasticsearch for search and File Store for documents."""
    },
    {
        "question": "Can I integrate with existing identity systems",
        "answer": """Yes, DIGIT supports integration with external identity systems.

- Authentication is handled via the platform's **API Gateway** and **User Service**, allowing extensibility
- You can integrate with **SSO, LDAP, or other identity providers** using API-based configuration
- External systems can be used for user authentication and token generation
- DIGIT's **RBAC** continues to manage authorization within the platform after authentication
- Integration typically requires environment-level configuration and setup

**For citizens:** Default is mobile/email OTP
**For employees:** Default is username + password
**Custom identity providers:** Configurable via the API Gateway"""
    },
    {
        "question": "Can I build permit and license applications",
        "answer": """Yes, permit and license applications are a core use case for DIGIT Studio.

**What you can configure:**
- **Service setup:** Create a service/module (e.g., Trade License, Building Permit) with types like New, Renewal, Modification
- **Form configuration:** Design application forms with fields, validations, and required document uploads
- **Workflow:** Define stages (Submission → Scrutiny → Approval → Issuance) with role-mapped actions
- **Roles:** Configure Reviewer, Approver roles with correct permissions
- **Notifications:** SMS/email updates at key milestones (submission, approval, rejection)
- **Citizen & Employee apps:** Auto-generated for end-to-end lifecycle management

**Current limitations:**
- Document generation (e.g., license certificate PDF) requires additional configuration
- Payment configuration is not yet available in Studio and requires external setup

**In summary:** The core permit/license workflow and application journey can be fully configured in DIGIT Studio."""
    },
    {
        "question": "Can I build any use case using DIGIT Studio",
        "answer": """You can build a use case on DIGIT Studio if it follows a **service-based workflow** with data collection, processing, and a defined outcome.

**A use case is a good fit if it has:**
- Form-based data collection (users submit applications or requests)
- Workflow or approval steps (requests move through roles/stages)
- Task management (Inbox) for officials to review and act
- Document handling (uploads and generated outputs)
- Notifications (SMS, email, app updates)
- Optional payments

**Examples of supported use cases:**
- License or permit systems
- Complaint or grievance management
- Registration and approval-based services
- Inspection or checklist-based workflows

**When it may not be suitable:**
- Highly custom UI beyond configurable forms
- Real-time heavy processing or complex computations
- Systems without a defined request lifecycle

**Summary:** If a use case can be structured as form → workflow → outcome (approval/rejection/record), it can be built using DIGIT Studio."""
    },
]

# ─────────────────────────────────────────────
# DB Connection
# ─────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(
        dbname=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT", "5432"),
        sslmode=os.getenv("PGSSLMODE", "require")
    )


# ─────────────────────────────────────────────
# Safe DB Helpers
# ─────────────────────────────────────────────
def clear_health_data(conn, table="studio_manual"):
    """Safely clear old data if table exists."""
    with conn.cursor() as cur:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            print(f"Found {count} rows in {table}")

            cur.execute(f"DELETE FROM {table}")
            print(f"✅ Cleared {count} rows from {table}")

            conn.commit()

        except Exception as e:
            conn.rollback()  # 🔥 critical fix
            print(f"   Note: {e} (table may not exist yet, that's ok)")


def create_tables(conn):
    """Create required tables safely."""
    with conn.cursor() as cur:
        try:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bot_feedback (
                    id SERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    query TEXT,
                    answer_snippet TEXT,
                    rating VARCHAR(10),
                    source VARCHAR(20),
                    comment TEXT,
                    is_flagged BOOLEAN DEFAULT FALSE
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS predetermined_qa (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    confidence FLOAT DEFAULT 1.0,
                    positive_votes INT DEFAULT 0,
                    negative_votes INT DEFAULT 0,
                    source VARCHAR(20) DEFAULT 'manual',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            conn.commit()
            print("✅ Tables created/verified")

        except Exception as e:
            conn.rollback()
            raise e


def load_qa_cache(conn):
    """Load Q&A safely."""
    with conn.cursor() as cur:
        try:
            # Faster + safer reset
            cur.execute("TRUNCATE predetermined_qa")
            print("Cleared existing Q&A cache")

            for qa in STUDIO_QA:
                cur.execute("""
                    INSERT INTO predetermined_qa (question, answer, confidence, source)
                    VALUES (%s, %s, %s, %s)
                """, (qa["question"], qa["answer"], 1.0, "manual"))

            conn.commit()
            print(f"✅ Loaded {len(STUDIO_QA)} Q&A pairs")

        except Exception as e:
            conn.rollback()
            raise e


def generate_studio_jsonl():
    """Generate JSONL for vector ingestion."""
    chunks = []

    for i, qa in enumerate(STUDIO_QA):
        combined = f"Q: {qa['question']}\n\nA: {qa['answer']}"

        chunks.append({
            "id": f"studio_qa/{i:03d}",
            "title": qa["question"][:80],
            "document": combined,
            "url": "https://docs.digit.org/studio",
            "tag": "studio",
            "version": "v1"
        })

    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", "studio_chunks.jsonl")

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"✅ Generated {output_path} with {len(chunks)} chunks")
    return output_path


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("DIGIT Studio — Data Setup")
    print("=" * 50)

    conn = get_conn()

    try:
        register_vector(conn)

        print("\n1. Creating tables...")
        create_tables(conn)

        print("\n2. Clearing old Health/HCM data...")
        clear_health_data(conn, table=os.getenv("DB_TABLE", "studio_manual"))

        print("\n3. Loading Studio Q&A...")
        load_qa_cache(conn)

        print("\n4. Generating JSONL...")
        generate_studio_jsonl()

        print("\n" + "=" * 50)
        print("✅ Setup complete!")
        print("=" * 50)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
