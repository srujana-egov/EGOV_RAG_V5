"""
Second round of chunk quality cleanup for studio_manual.

Run AFTER manually deleting these 5 rows:
  cq_22_accessibility
  cq_13_citizen_role
  cq_15_dropdown_values
  us_forms_edge_cases
  us_roles_validations

This script upserts 5 improved replacements:
  cq_21_multiple_languages   — merged with cq_22 (localization + accessibility)
  cq_23_add_users            — merged with cq_13 (citizen role + all user types)
  cq_15_dropdown_values      — expanded with more context
  us_forms_main_flow         — appended edge cases from deleted us_forms_edge_cases
  us_roles_overview          — appended validations from deleted us_roles_validations

Run: python replace_low_quality_chunks_v2.py
"""
from utils import get_conn
from retrieval import get_embedding

TABLE = "studio_manual"

REPLACEMENTS = [
    {
        "id": "cq_21_multiple_languages",
        "document": """Localization and Accessibility in DIGIT Studio

Localization (Multiple Languages):
Localization can be updated for any app built on DIGIT Studio. All text on apps generated have localization keys associated with them. You can add as many languages as you like by adding values in that language against their localization key.

Currently, Studio does not come with a UI to add localization. You can add localization using a JSON format by mapping values against the localization keys for each language you want to support.

Accessibility:
Applications generated from Studio follow WCAG 2.0 accessibility guidelines. The UI designed for end-user applications has been inspired by the Gov.UK design system, which is the gold standard for accessibility, simplicity, and user experience.

Summary:
- DIGIT Studio apps support multiple languages via localization keys — add translations in JSON format
- All generated apps are WCAG 2.0 compliant and designed for accessibility by default""",
    },
    {
        "id": "cq_23_add_users",
        "document": """Users in DIGIT Studio — Citizen Role and Adding Users

There are three types of users in DIGIT:
- System users (e.g., Boundary Admin, Super Admin) — created during platform installation
- Service-specific users (Employees) — created after a service is published
- Citizens — end users of the service

The Citizen Role:
Since applications built on DIGIT Studio are primarily built for Public Service Delivery, they come with a mobile-responsive interface for user type 'Citizen'.

Citizens are defined as users that can:
- Self-register using either a mobile OTP or email OTP
- Apply for a public service
- Track current and past application status
- Perform tasks assigned to them in the workflow

There is no way currently to disable the citizen role.

To add service-specific users (Employees):
- Go to the DIGIT User Management module after publishing the service
- Create users by entering basic details (name, mobile number, etc.)
- Assign the appropriate role during user creation
- Users receive their login credentials on their registered mobile number

Citizens do not need to be created manually:
- They can self-register using the Citizen URL
- The Citizen URL is available on the Post Go-Live / Publish success screen in DIGIT Studio
- Once registered, they can start using the service""",
    },
    {
        "id": "cq_15_dropdown_values",
        "document": """Dropdown Values in DIGIT Studio — MDMS and Field-Level Configuration

DIGIT Studio supports two ways to populate dropdown field values:

1. Pull from MDMS (Master Data Management Service):
DIGIT apps are powered by the MDMS service. Master data configured in the MDMS service can be used across applications and added to form dropdowns using the 'Pull data from Database' option. This is useful for shared reference data (e.g., complaint categories, district lists) that is reused across multiple services.

2. Configure within the field:
For dropdown values specific to a particular use case, you can add the options directly within the field configuration itself — no MDMS setup needed.

For dependent dropdowns (where one dropdown's options depend on another field's value), use Display Logic and Dependent Field settings. See also: how to create a dependent dropdown.

Summary: Use MDMS for shared/reusable dropdown data, and field-level configuration for use-case-specific options.""",
    },
    {
        "id": "us_forms_main_flow",
        "document": """Forms Flow - Main Flow

1. Landing on the Forms Screen
The user enters the "Forms" section of the Studio and is presented with:
- "Create New Form" button
- "Drafts" tab

2. Creating a New Form
User clicks on "Create New Form", which opens a unified Form Designer (all sections in one screen).

3. Designing the Form
Inside the Form Designer, the user:
- Enters a form name and description and sets up idGen. These fields appear in the properties panel when clicked on the form header.
- Adds, removes, or rearranges form sections and fields
- Configures field-level properties: label, type, validation, required/optional, default values
- Defines UI/UX elements and conditional logic
- Info icons on UI/UX elements open as toast messages.

4. Saving Progress
User clicks "Save" to store the form as a draft. Drafts are accessible from the "Drafts" section on the landing screen.

5. Previewing the Form
User clicks on a Form in the "My Forms" section, selects a form, and clicks "Preview" to see how the form appears to end-users.

6. Modifying an Existing Form
From Form preview, the user can click "Modify" to make changes. This reopens the form in the designer.

7. Publishing the Form
Once satisfied, the user clicks "Publish".

8. Validation Check
- If there are any errors (e.g., required metadata missing, invalid logic), the system displays an error message and highlights what needs to be fixed.
- Once errors are resolved, the user can retry publishing.

9. Success
On successful validation, the form is added to "My Forms" under the respective service group.

Negative Scenarios / Edge Cases:
- No form name entered: Prevent save/publish; show inline error: "Form name is required."
- Field added but no type selected: Prevent publish; show "Please select field type."
- Two fields with same name: Prevent publish; show "Field names must be unique."
- User closes window without saving: Prompt warning: "You have unsaved changes."
- Backend failure during publish: Show "Could not publish the form. Please try again." """,
    },
    {
        "id": "us_roles_overview",
        "document": """Roles - Overview, Default Behavior, and Validations

Overview:
This module enables users to create new roles or manage existing ones by defining permissions (actions within sections) and mapping them to workflows. The goal is to ensure that users only see and act upon what their role allows.

Default Behavior: Studio Admin Role
- The user who creates the service (project creator) is automatically assigned the Studio Admin role.
- Studio Admin is a system role with access to all studio features (forms, workflows, role config, deployment, data views, etc.).
- No option to edit/remove this role in V1.

Default Role Assignment:
- When a user creates a new service/project, assign them the "Studio Admin" role automatically in the HRMS.

Validations:
- Role Name: Required, must be unique (applicable only when custom roles are enabled)
- Actions: At least one section and action must be selected before saving

Edge Cases:
- Role name already exists: Show message "Role name already exists. Please use a different name."
- Attempt to delete Studio Admin role: Show message "This role is required and cannot be removed."
- No sections/actions selected: Disable Save Role button or show warning""",
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
