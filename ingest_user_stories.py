"""
One-time ingestion script for "Studio User Stories" PDF into studio_manual.
Every chunk gets url=https://app.supademo.com/demo/cmn6xbpcb01060l0jhatgyk9k?utm_source=link
Run: python ingest_user_stories.py
"""

from utils import get_conn
from retrieval import get_embedding

TABLE = "studio_manual"
URL = "https://app.supademo.com/demo/cmn6xbpcb01060l0jhatgyk9k?utm_source=link"
TAG = "user_stories"
VERSION = "v1"

CHUNKS = [
    {
        "id": "us_roles_overview",
        "document": """Roles - Overview

This module enables users to create new roles or manage existing ones by defining permissions (actions within sections) and mapping them to workflows. The goal is to ensure that users only see and act upon what their role allows.

Default Behavior: Studio Admin Role
- The user who creates the service (project creator) is automatically assigned the Studio Admin role.
- Studio Admin is a system role with access to all studio features (forms, workflows, role config, deployment, data views, etc.).
- No option to edit/remove this role in V1.

Default Role Assignment
- When a user creates a new service/project, assign them the "Studio Admin" role automatically in the HRMS.""",
    },
    {
        "id": "us_roles_flow",
        "document": """Roles - Flow Description

Access Role Management: Via Roles icon in the top nav OR from within workflow creation ("Add Role").

Choose an Option:
a. Create a New Role
b. Select an Existing Role (default: Studio Admin, could be more if they already saved any or picked an existing service group template)

Path A: Create a New Role
1. Create a New Role - Provide unique role name and optional description.
2. Add Section-Level Permissions - Select from sections (e.g., Form Builder, Workflow Config, Data View).
3. Choose Actions Within Each Section - Toggle specific actions (e.g., create, edit, delete).
4. Save Role - Role is saved and available in the role list on the Role Screen and adds to the workflow screen.

Path B: Select Existing Role (Not Studio Admin)
1. Review Permissions - Studio Admin has all permissions across sections. No edit/removal allowed in V1 for studio admin. Other Roles can be edited - Names/Access/Description.
2. Save Role - Role card is added/updated to Workflow screen and to the Roles screen.""",
    },
    {
        "id": "us_roles_validations",
        "document": """Roles - Validations and Edge Cases

Validations:
- Role Name: Required, must be unique (applicable only when custom roles enabled)
- Actions: At least one section and action must be selected

Negative Scenarios / Edge Cases:
- Role name already exists: Show message "Role name already exists. Please use a different name."
- Attempt to delete Studio Admin role: Show message "This role is required and cannot be removed."
- No sections/actions selected: Disable Save Role button or show warning""",
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
- We need info icons on the UI/UX elements which open as toast messages.

4. Saving Progress
User clicks "Save" to store the form as a draft. Drafts are accessible from the "Drafts" section on the landing screen.

5. Previewing the Form
User clicks on a Form in the "My Forms" section, selects a form, and clicks "Preview" to see how the form appears to end-users. (no need to show the section if no forms exist for any of the user's services)

6. Modifying an Existing Form
From Form preview, the user can click "Modify" to make changes. This reopens the form in the designer.

7. Publishing the Form
Once satisfied, the user clicks "Publish".

8. Validation Check
- If there are any errors (e.g., required metadata missing, invalid logic), the system displays an error message and highlights what needs to be fixed.
- Once errors are resolved, the user can retry publishing.

9. Success
On successful validation, the form is added to "My Forms" under the respective service group.""",
    },
    {
        "id": "us_forms_edge_cases",
        "document": """Forms Flow - Negative Scenarios / Edge Cases

Scenario | Expected Behavior
- No form name entered: Prevent save/publish; show inline error: "Form name is required."
- Field added but no type selected: Prevent publish; show "Please select field type."
- Two fields with same name: Prevent publish; show "Field names must be unique."
- User closes window without saving: Prompt warning: "You have unsaved changes."
- Backend failure during publish: Show "Could not publish the form. Please try again." """,
    },
    {
        "id": "us_checklists_main_flow",
        "document": """Checklists - Main Flow (Happy Path)

1. Landing on the Checklist Home Screen
The user sees:
- "Create a New Checklist" CTA card
- My Checklists tab with published checklists (name, created date, status)
- Drafts tab with saved but unpublished checklists

2. Creating a New Checklist
- On clicking "Create New Checklist", user is navigated to the Checklist Builder
- Header section asks for: Name of Checklist (display name), Help Text (optional instructions for field staff)

3. Adding Questions - Same as HCM Console
User adds one or more questions, each with:
- Question text
- Input type: Radio button, checkbox, text input
- Option to mark as mandatory
- Optional sub-questions (1.1, 2.1) for conditional logic
- "Link Nested Checklist" option to attach dependent checklist blocks

4. Adding Answer Choices - Same as HCM Console
For radio/checkbox types, user can add multiple options (e.g., "Shortages", "Quality complaints")

5. Reordering / Deleting Questions - Same as HCM Console
User can delete individual questions or sub-questions, and delete options within a question.

6. Previewing and Configuring (PUBLISH)
- Clicking "Preview Checklist" lets the user see how it looks to field staff
- On finalizing: Clicks "Publish" to validate and save

7. Saving as Draft / Publishing
User can choose to save the checklist as a draft or publish it after validations.

8. Managing Existing Checklists
From the home screen:
- My Checklists (Published): user can view, duplicate, or archive existing checklists
- Drafts: user can resume editing, delete, or publish""",
    },
    {
        "id": "us_checklists_criteria_edge",
        "document": """Checklists - Acceptance Criteria and Edge Cases

Acceptance Criteria:
- AC1: User can create a checklist with metadata and questions
- AC2: User can configure question type, mandatory status, and sub-questions
- AC3: User can preview the checklist before publishing
- AC4: Drafts are auto-saved and editable later
- AC5: Checklist shows up correctly under "Published" or "Drafts" tab

Edge Cases & Error Messaging:
- Missing checklist title or required metadata: "Please enter the checklist name."
- No questions added: "Checklist must have at least one question before publishing."
- Sub-question added without parent logic: "Sub-question must be linked to a valid answer."
- Attempt to publish with validation failure (e.g., Radio button type but no options added): "Some questions are incomplete. Please review before publishing."
- Duplicate checklist name within service group: "Checklist name already exists. Please use a unique name." """,
    },
    {
        "id": "us_landing_page",
        "document": """Studio Landing Page / Home Page - Main Features

1. Header Section
- "SERVICE DELIVERY STUDIO" Header
- Documentation & Support links for quick access to help materials - Will be in left panel/side bar according to DIGIT Design system.
- User avatar (A) indicating the logged-in user
- Logout button which logs out users and redirects them to the Login screen.

2. Create a New Service Group
- Primary CTA Box with a large "+" icon and a label: "Create New Service Group", Subtext: "Start building a service group from scratch"

3. My Service Groups
- A section listing user-created service groups
- Format of the Service Group Tile: Icon, Title, Description (if added), Creation date

4. Drafts Section
- A section that contains partially completed service groups (Name, Desc, Icon metadata submitted) but not yet published
- Format: Icon, Title, Description, Last modified date (instead of creation date)
- User interactions: Clicking on a tile resumes configuration; delete option available; duplicate option available

5. Explore Our Templates
- This section showcases a library of pre-configured service templates that users can adopt and customize.
- Format of template tile: Template name, Description, Representative icon
- User interactions: Clicking a template opens a preview screen (Preview User Story) showing the configuration structure and features included in the template. From the preview, users can click "Modify" to launch the Service Group Configuration flow, pre-filled with template data.

Negative Scenarios / Edge Cases:
- User closes metadata popup without saving: No group is created. Remain on landing screen.
- Required metadata field (e.g., name) left empty: Prevent submission, highlight required fields. Error: "Group name is required to create a service group."
- Duplicate service group name (draft or published): Block creation, inline validation. Error: "A service group with this name already exists. Please choose a different name." """,
    },
    {
        "id": "us_workflow_canvas_states",
        "document": """Workflow - Acceptance Criteria

Feature Use: As a Studio Admin (business user or service designer), we want to visually design, configure, and save workflows by adding states, connecting them with labeled actions, assigning roles, and setting properties.

1. Canvas Initialization
- On the first load, we should see an empty infinite canvas with helper instructions.
- We should see a quick start guide and component panel on the left with "Start", "Process", and "End" states.

2. Adding Workflow States
- We can click on "Start", "Transit", or "End" states from the left panel to add them to the canvas.
- The component should be placed smartly in the visible viewport, without overlapping existing states.
- Every node should have: A default icon and label (Label: Start, Transit, End - editable; Icon: Closest to start, In process and End in DIGIT design system), Delete icon on top right, Edit Icon, Properties section in the right panel (State Name - mandatory/editable, Description, Role assignment multiselect dropdown, SLA Timer, Mandatory Service Request Form Dropdown for start node, Optional stage actions like Notify, Assign, Comments, Checklist).

3. Dragging and Arranging States
- We can freely drag states around the canvas.
- We can "Zoom to Fit" all nodes.

4. Connecting Workflow States
- We can click and drag from the right side (output) of a node to connect to another existing node or drop on blank space to auto-create and connect a new node.
- Each connection can have a name (action), a description, and be edited or deleted via property panel.""",
    },
    {
        "id": "us_workflow_properties_roles",
        "document": """Workflow - Node Properties, Stage Actions, Role Management, and Publishing

5. Configuring Node Properties
- On clicking a node, the Properties panel on the right opens with: editable state name and description, role assignment (predefined or custom) multiselect dropdown, SLA timer input, stage actions with toggle/configure buttons.
- On clicking "Update Properties," changes reflect immediately.

6. Stage Actions
We can toggle actions like: Add Comments, Assign to User, Ask for Documents.
For advanced actions like "Notification", "Generate Documents", "Checklist": Clicking "Configure" opens a modal to add additional configuration.

7. Role Management
- We can assign multiple roles to each state or action.
- State Roles have view access to the artifacts attached in the state; Action Roles have the specified action access.
- If the desired role doesn't exist, we can choose "+ Add New Role": A modal appears to define the name, description, and access. Custom roles show up in future dropdowns and appear with badge on the states.

8. Connection Properties
- Clicking on a connection or its label opens connection properties: Edit action name and description [No role dropdown needed here].
- We can delete a connection from this panel or by right clicking on it.

9. Drafting and Publishing
- We can Save the workflow at any point, Auto Save ideally.
- We can also Clear the canvas to start over.
- Publish is not needed for the workflow.

10. Misc
- Info icons needed on all the fields in the property panel which open into toasts with configurable text.
- Badges for added configuration should appear on the node.
- Load Sample button - Modal opens on clicking (Image + text).
- Start node should not have input connection and End node should not have output connection.""",
    },
    {
        "id": "us_workflow_edge_cases",
        "document": """Workflow - Edge Cases & Negative Scenarios

- Trying to connect from an input handle shows a helpful message: "Start connections from output handles."
- Trying to publish a workflow with disconnected nodes or unnamed actions shows validation errors.
- Validation error if Start node is not the first node and error if End node is not the last node.
- Dragging from output to empty space smartly creates a new node and connects it.
- Form options are hidden unless the node type is Start.
- Start Node Usage Constraint: The Start node can only be added once to the canvas. If a Start node is already present, the "Start" option in the left panel is visually disabled (grayed out). The user cannot click or drag to add another. If the user deletes the Start node from the canvas, the Start node option in the panel becomes enabled again for reuse.
- SLA input rejects non-numeric values or values less than 1.
- Deleting a node removes its connected arrows and labels.""",
    },
    {
        "id": "us_notifications_overview",
        "document": """Notifications - Overview and Acceptance Criteria

We want to create, configure, and preview multi-channel notifications (Email, SMS, Push), so that they can inform citizens or officials at the right time during service delivery workflows.

Acceptance Criteria:
1. Users can create notifications for all three supported channels.
2. Previously created notifications appear in a list (drafts or linked).
3. Channel-specific validations are enforced.
4. Real-time preview is available.
5. Users can link notifications to workflow events after successful validation.
6. The system prevents errors through inline and summary validation messaging.""",
    },
    {
        "id": "us_notifications_manager",
        "document": """Notifications - View 1: Notification Manager (Landing Page)

Purpose: A dashboard where users see their created notifications and start creating a new notification per channel.

Layout:
- Header: "Notification Manager" with subtext "Create personalized notifications that reach users at the right moment through the right channel."
- Channel Cards: 3 Cards: Email, SMS, Push. Each with icon, short description, CTA: "Create Notification"
- Saved Notifications: List of previously created notifications by the user. Fields: Title, Channel, Status (Draft/Linked), Linked Workflow, Name, Last Updated
- Search & Filter: Optional filters by channel, keyword, or status (Not a P0)
- View Toggle: Grid/List toggle (Not a P0)
- No Templates Info: If no notifications created yet: "No notifications found. Start by choosing a channel to create your first notification."

Happy Flow:
- User lands on Notification Manager.
- Sees Email/SMS/Push cards.
- Clicks "Create Notification" > taken to the Configurator with selected channel prefilled.""",
    },
    {
        "id": "us_notifications_configurator",
        "document": """Notifications - View 2: Notification Configurator

Configuration Steps:
1. Choose Channel: Preselected from entry point (Email/SMS/Push). Cannot be changed mid-way.
2. Compose Message: Fields vary by channel. Content auto-saved.
3. Link to Workflow: Node ID (read-only) + Editable Display Name + Execution Conditions
4. Preview & Save: Real-time preview panel (channel-specific). Save Draft, Test, Link actions.

Message Fields by Channel:
- Title: Required for Email, SMS, Push
- Subject: Required for Email only
- Message Body: Required for all channels
- Char Limit: 160 characters for SMS, short message for Push
- Variables: Present for all channels (inserted using chips: user_name, application_id, etc.)

Workflow Linking:
- Node ID/Name: Multiselect Dropdown
- Display Name: Editable
- Validation Requirement: All message fields must be valid before enabling "Link to Workflow."

Preview Panel:
- Sticky on the right (desktop), collapsible on mobile
- Updates in real-time
- Channel-specific layouts: Email (Gmail-style desktop + mobile), SMS (iPhone-style chat bubble), Push (App toast mobile + desktop tray style)
- Expand Preview: Opens full modal
- Shows: Readiness badge (✅ if valid), Character count (for SMS), Validation summary (if any issues)""",
    },
    {
        "id": "us_notifications_flows_edge",
        "document": """Notifications - Happy Flows and Edge Cases

Flow A: Create New SMS Notification
1. Clicks "Create Notification" on SMS card
2. Enters Title: "Permit Application Received"
3. Types message: "Hi {user_name}, your application {application_id} has been received."
4. Adds variables using chip selector
5. Character count shows 105/160
6. Save > Saves the message on the landing page
7. Links to workflow > Selects "On Submission"
8. Preview shows green readiness badge
9. Clicks "Link to Workflow" > Selects one or more workflow node > Success message appears

Flow B: Create Email Notification
1. Selects "Create Notification" on Email
2. Fills Title: "Application Approved"
3. Subject: "Congrats! Your application is approved."
4. Message Body: "Hi {user_name}, your {application_type} has been approved."
5. Links to workflow "On Approval"
6. Live preview reflects email format
7. Clicks "Link to Workflow" > Select one or more nodes > Confirmation shown

Edge Cases / Negative Scenarios:
- SMS content > 160 characters: Live character count turns red, disables Save/Test/Link
- No Title provided: Inline error under Title, Save & Link disabled
- Missing Subject (Email): Inline error under Subject + Warning in Preview Footer
- Clicking "Link to Workflow" without completing fields: Button disabled + Tooltip: "Complete all required fields."
- Trying to test SMS with >160 characters: Test button disabled, preview footer shows red error
- Leaving message body blank: Inline error "Message content cannot be empty."
- Clicking "Save" with missing Title: Blocked with inline error and summary toast""",
    },
    {
        "id": "us_import_flow",
        "document": """Import - Acceptance Criteria & Flow

1. Trigger Import
- User clicks on the "Import Service" button on the My Applications section header.
- A modal opens with: A text area to paste the service configuration, Two input fields: Service Name (mandatory, user-editable, unique per tenant) and Module Name (mandatory, must map to an existing or new module), Helper text: "Imported services will be saved in Drafts. You can edit and publish them later.", Save and Cancel buttons.

2. Save Flow
- On clicking Save: Validate that all required fields are filled, validate the service config format (must be valid service config structure).
- If valid: The service is saved in Drafts. The Service Designer opens automatically with the imported service loaded.
- Success message: "Service imported successfully and saved to Drafts."

3. Cancel Flow
- On clicking Cancel: Modal closes, User is returned to the landing screen, No changes are saved.

Negative Scenarios & Edge Cases:
1. Invalid Config Format: User pastes invalid service config format. System shows inline error: "Invalid configuration format, please paste a valid service config."
2. Missing Mandatory Fields: If Service Name or Module Name is empty → Save button disabled or error message shown.
3. Network/Backend Error on Save: Show error toast "Import failed due to a network error. Please try again." Stay on modal with user inputs intact.
4. Empty Config: If user clicks Save without pasting any config, show inline error: "Service configuration cannot be empty." """,
    },
    {
        "id": "us_import_duplicate",
        "document": """Import - Duplicate Functionality

Each published service card has a "Duplicate" button (visible only on Published and draft cards).

On click: Open a Duplicate Service Modal with:
- Preloaded Service Config (not editable, hidden from user).
- Two input fields: New Service Name (mandatory, unique) and Module Name (mandatory).
- Helper text: "This service will be duplicated and saved in Drafts. You can edit and publish it separately."
- Save and Cancel buttons.

Edge Cases:
- If Service Name already exists in Drafts/Published for the tenant → show error: "A service with this name already exists. Please use a different name." """,
    },
    {
        "id": "us_configurable_applicant_details",
        "document": """Configurable Applicant Details - User Story

As a Service Designer, I want to configure the Applicant Details section in any form, so that I can define what applicant information needs to be collected for each service while ensuring essential identity fields are always captured.

Acceptance Criteria:

1. Default Section Creation
- When a new form is created, an Applicant Details section is automatically added.
- This section is non-mandatory and can be removed by the user.

2. Mandatory Fields
- If the Applicant Details section is present, it must always contain the following non-removable, mandatory fields: Name and Mobile Number.
- The mobile number prefix (e.g., country code) can be configured by the Admin.
- Everything apart from the field type and label of these fields can be edited.

3. Configurable Fields
Admins can:
- Add new fields (e.g., text, dropdown, date, file upload, etc.) within the Applicant Details section.
- Edit labels, placeholders, and validation rules for optional fields.
- Reorder optional fields within the section.

4. Validation
- The system must prevent deletion or editing of mandatory fields (Name, Mobile Number).
- Optional fields can be added, edited, or removed freely.

5. Address Details Section
- The Address Details section, while not configurable, should always be non-mandatory.
- If the Applicant Details section is not added, essential data points (Name, Mobile Number, Email, Address) should be auto-populated from the Citizen's registration profile.

Negative Scenarios / Edge Cases:
1. Admin tries to delete mandatory fields: System blocks deletion of Name or Mobile Number. Show error message: "This field is mandatory and cannot be removed."
2. Admin tries to edit mandatory fields: System prevents changing field type or removing validation for Name and Mobile Number. Only the Mobile Number prefix should be editable.
3. Duplicate field creation: Admin attempts to add a field that already exists (e.g., another 'Name' field). System should warn: "This field already exists. Please edit the existing one or add with another label." """,
    },
    {
        "id": "us_citizen_interface",
        "document": """Citizen Interface

As a Citizen, I want to self-register and access a standard dashboard, so that I can apply for services and track my applications without external assistance.

Studio - Role Configuration

Enable Self-Registration:
- Go to Roles in Studio.
- Toggle Self-Registration for a role (e.g., Citizen).
- The system should enforce that only one role can have self-registration enabled.
- Text prompt: "Is this the Citizen role? If yes, enable self-registration."

Additional Condition:
- If at least one role with self-registration (e.g., Citizen) is enabled, then on the login page, display the option: "Not a user? Register Now"
- This link should redirect to the Citizen registration interface.

End-User (Citizen) Journey

Registration / Login:
- When a Citizen logs in for the first time, they see the configured registration fields (V1 = only mobile/email + Name + OTP).
- The Citizen completes the form and performs OTP verification.
- The account is created and mapped to the Citizen role.

Citizen Dashboard (V1 = fixed, non-configurable):
- After login, the Citizen lands on a standard dashboard with all the services for which citizen role is enabled.
- Each service card will have two tiles:
  1. Apply for a Service: Opens a list of available services. Citizen selects a service and fills out the form configured by the Service Designer.
  2. Manage / Track Applications: Displays all applications submitted by the Citizen, with details: Date, Application Number, Open/Closed Status, Current Workflow Step. Clicking on an application opens its details and workflow progression.""",
    },
    {
        "id": "us_configurable_address_overview",
        "document": """Configurable Address Details - Overview

As a Service Designer, I want to configure the Address Details section (and Applicant Details section) in a form, so that I can manage default address fields, boundary hierarchy fields, and custom fields easily, while ensuring consistency with DIGIT's boundary logic.

DIGIT Studio provides a pre-defined Address Details section which can be added from Add Section CTA on the form designer interface. It includes boundary-level fields (derived from the tenant's hierarchy, e.g., State → District → Block → Ward), along with standard address fields like Street Name, Pincode, and Map Location.

The designer should be able to:
- Enable/disable default fields through toggles
- Decide which hierarchy type to be used to display the boundary fields
- Configure how many boundary levels (Highest → Lowest) should appear
- Have boundary configuration displayed in the Elements tab
- See the boundary fields appear vertically in the preview, based on the levels selected
- Add and configure custom fields without affecting the default logic

Boundary hierarchy selection dynamically constrains which levels can be chosen for Highest/Lowest and updates the preview accordingly.""",
    },
    {
        "id": "us_configurable_address_criteria",
        "document": """Configurable Address Details - Acceptance Criteria

1. Section & Default Fields

1.1 Adding Sections
- Applicant Details and Address Details sections load with their respective default fields. Applicant details appear by default while address have to be added from the Add Section CTA.
- All default fields in Address Section and Applicant section can be disabled using toggles except the Name and Mobile number in Applicant section.
- Only one instance of each default section is allowed for now.

1.2 Default Field Controls
- Default fields appear in Section Fields list with toggle switches.
- Toggling a default field: Shows/hides the field in preview. Does NOT delete or remove the fieldKey.
- Custom fields appear with a delete icon only (no toggle).

2. Boundary Cluster Behaviour
Boundary Cluster Field header should be: "Area Selection"

2.1 Elements Tab Configuration
When the user selects the Boundary Cluster field in preview:
- Field Properties panel opens on the right.
- Elements tab shows: Hierarchy Type dropdown (e.g., Admin/Revenue/Election), Highest Boundary Level dropdown, Lowest Boundary Level dropdown.

2.2 Dropdown Constraint Logic
- When Highest is selected first: Lowest Boundary Level dropdown only shows levels at same or below the selected Highest.
- When Lowest is selected first: Highest Boundary Level dropdown only shows levels at same or above the selected Lowest.

2.3 Preview Update
- Based on the Highest > Lowest path, the preview must show the exact boundary fields stacked vertically, in order. Example: Highest = State, Lowest = Ward → Preview shows: State → District → Block → Zone → Ward.
- Changing hierarchy settings updates preview in real time.

3. Custom Field Behavior
- Clicking "Add Field" opens modal with field type selection.
- Custom fields: Appear in Section Fields with a delete icon, are fully editable (label, placeholder, type), are removable with no restrictions, are stacked vertically like default fields.

4. Field Selection & Editing
- Clicking any field in the preview highlights it; Field Properties panel opens on the right.
- 4.4 Default Field Restrictions: Label and field type are locked for Pincode and Street Name. Label can be changed for Boundary and Map Field. Placeholder can be editable.

5. Modal Behaviors
5.2 Hierarchy Change Warning Modal: If tenant has already mapped data OR if changing hierarchy invalidates a previously selected boundary path, system shows: "Changing hierarchy will modify the boundary fields shown to applicants. Proceed?"

6. Data & FieldKey Rules
- Default fields use fixed fieldKeys (state, district, ulb, pincode, streetName, etc.).
- Toggling off does not remove the fieldKey.
- Deleting the Boundary Cluster is not allowed, it can be disabled though.
- Custom fields are assigned fieldKeys which get deleted on deletion.
- Re-adding default fields from toggle ON restores the same fieldKey.""",
    },
    {
        "id": "us_display_logic_overview",
        "document": """Display Logic - Overview and Entry Point

Display Logic allows Studio Designers to conditionally show or hide a field or its options, based on other field values, with operators driven by the condition field type and actions applied only to the configured field.

Entry Point:
1. User opens Form Designer
2. Selects a field (Field Y - target field)
3. Opens Configure Field Properties → Logic tab
4. Toggles Dependent Field = ON

Display Logic Setup:

Step 1: Add Display Logic
5. Under Display Logic, clicks + Add Display Logic
6. System opens Add Display Logic modal

Rule Builder (Core Interaction):

Step 2: Define Condition (IF)
7. User selects Field X (condition field)
8. System dynamically loads operators based on Field X type and renders correct value input control.

Examples:
- Number → ≤, ≥, = → numeric input
- Dropdown/Radio → Selected / Not Selected → multi option selector
- Text → Equals / Contains → text input

Operator Matrix by Field X Type:
- Number: =, ≠, <, ≤, >, ≥ → Numeric input
- Date: =, Before, After → Date picker
- Text: Equals, Not equals, Contains, Starts with, Is empty, Is not empty → Text input (case-insensitive by default)
- Dropdown: Selected, Not selected → Option selector (single value only)
- Radio: Selected, Not selected → Option selector (exactly one option)
- File Upload: Is uploaded, Is not uploaded → None
- Location: Is set, Is not set → None""",
    },
    {
        "id": "us_display_logic_actions_runtime",
        "document": """Display Logic - Actions, Multiple Conditions, Validation, and Runtime Behavior

Step 3: Define Action (THEN)
- Field Y is implicit (non-editable)
- System determines available actions based on Field Y type:
  - If Field Y = non-option field: Actions: Show field, Hide field
  - If Field Y = option-based field (dropdown/radio/checkbox): Actions: Show field, Hide field, Show selected options [multi choice], Hide selected options [multi choice]. Option selector shows Field Y's options. There has to be a select all option in this case on top of the options.

Action Matrix by Field Y Type:
- Text/Number/Date: Show field, Hide field (Value preserved unless field is hidden)
- Dropdown: Show field, Hide field (multi selection allowed, at least one option to be selected in case of Show/Hide options)
- Radio: Show field, Hide field (at least one option must remain)
- File Upload: Show field, Hide field (file cleared if field hidden)

Step 4: Multiple Conditions
- User clicks + Add Condition
- System adds logical connector selector (AND/OR)
- System evaluates all conditions top-to-bottom
- Note: The joins of logic can be either all "AND" or all "OR"

Step 5: Validation Rules
Before saving, system validates: Field X ≠ Field Y, Operator compatible with Field X type, Value not empty, At least one valid action defined.

Step 6: Save Logic
- User clicks Save Logic, modal closes.
- Logic appears as Logic 1 under Display Logic. User can edit or delete logic.

Runtime Behavior (End User App View):
- Form Load: All display logic evaluated, hidden fields not rendered, hidden options removed from dropdowns.
- On Field X Change: Logic re-evaluated instantly, Field Y visibility/options updated.

Edge Case Handling:
- If all options in Field Y are hidden: Field Y is auto-hidden.
- On back-navigation: Logic re-applied consistently.

Exit:
- User clicks Save Form
- Display Logic becomes part of service configuration""",
    },
]


def create_table_if_missing(conn):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                id TEXT PRIMARY KEY,
                document TEXT,
                embedding vector(1536),
                url TEXT,
                tag TEXT,
                version TEXT
            );
        """)
    conn.commit()


def ingest():
    conn = get_conn()
    create_table_if_missing(conn)

    print(f"Ingesting {len(CHUNKS)} chunks into {TABLE}...")

    inserted = 0
    skipped = 0

    try:
        with conn.cursor() as cur:
            for i, chunk in enumerate(CHUNKS):
                chunk_id = chunk["id"]
                text = chunk["document"]

                try:
                    emb = get_embedding(text)

                    cur.execute(f"""
                        INSERT INTO {TABLE} (id, document, embedding, url, tag, version)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            document = EXCLUDED.document,
                            embedding = EXCLUDED.embedding,
                            url = EXCLUDED.url,
                            tag = EXCLUDED.tag,
                            version = EXCLUDED.version
                    """, (chunk_id, text, emb, URL, TAG, VERSION))

                    inserted += 1
                    print(f"  [{i+1}/{len(CHUNKS)}] ✅ {chunk_id}")

                except Exception as e:
                    skipped += 1
                    print(f"  [{i+1}/{len(CHUNKS)}] ❌ {chunk_id}: {e}")

        conn.commit()

    finally:
        conn.close()

    print(f"\nDone. Inserted/updated: {inserted}, Failed: {skipped}")


if __name__ == "__main__":
    ingest()
