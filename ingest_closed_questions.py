"""
One-time ingestion of answers from "Closed Questions" document into studio_manual.
Only the answer text is stored — question text is excluded.
Run: python ingest_closed_questions.py
"""
from utils import get_conn
from retrieval import get_embedding

TABLE = "studio_manual"

CHUNKS = [
    {
        "id": "cq_01_environments",
        "document": """DIGIT Studio can be deployed in any DIGIT environment that you choose. So, as per your strategy, you can have a dev, staging/UAT and production environment or only a Dev and production environment.

On Studio, you can build and preview applications. This allows you to view changes and test the application as you make changes. Once you are satisfied with the application you have built, you can simply Export the service configuration and import it to another version of Studio (in your UAT or production environment).

The Export and Import Functionality work for the following:
- Export and Import of Service Configuration

What does not work for:
- Export and Import of any data you may have added
- Export and Import of any users you may have created""",
    },
    {
        "id": "cq_02_what_can_i_build",
        "document": """DIGIT Studio (part of the DIGIT platform by eGovernments Foundation) is a platform that allows you to build complete digital public services without heavy coding, using configuration.

You can use it to create services that handle the full journey of a request — from submission to approval and final output.

With DIGIT Studio, you can build:
- Public service applications such as permits, or grievance systems
- User-friendly forms where citizens or employees can submit information
- Approval workflows where requests move through different officials for verification and approval
- Task dashboards (Inbox) for officials to view and act on pending work
- Document handling including uploading documents and generating certificates, receipts, or predefined application PDFs
- Notifications through SMS, email, or app alerts to keep users informed
- Multi-tenant systems that can be used by multiple departments or cities with separate configurations

What you cannot build now:
- Payment-enabled services with fee calculation, billing, and online payments handled through API for now
- Dashboards and Reports

All of this is managed through simple configurations, which define how the service behaves, what data is collected, how approvals happen, and how users interact with the system.

Summary: DIGIT Studio helps you build complete digital services by combining forms, workflows, payments, documents, and notifications into a single system.

Example: You can build a Trade License service where a user applies online, officials review and approve the request, fees are paid, and a license certificate is generated and shared with the user.""",
    },
    {
        "id": "cq_03_can_i_build_xyz",
        "document": """Yes, you can build any use case using DIGIT Studio if it follows a service-based workflow involving data collection, processing, and outcome generation.

DIGIT Studio is designed to build end-to-end digital services, so any use case that includes the following can typically be implemented:
- Form-based data collection (users submit applications or requests)
- Workflow or approval steps (requests move through different roles or stages)
- Task management (Inbox) for officials to review and act on requests
- Document handling (uploading documents and generating outputs like certificates or receipts)
- Notifications (SMS, email, or app alerts to keep users informed)
- Payments (optional) for services that involve fees

If your use case includes most of the above components, it is a strong fit for DIGIT Studio.

Examples of supported use cases:
- License or permit systems
- Complaint or grievance management
- Registration and approval-based services
- Inspection or checklist-based workflows

Upcoming capabilities:
- Reporting and analytics dashboards for tracking service performance

When it may not be suitable — DIGIT Studio may not be ideal for use cases that require:
- Real-time heavy processing or complex computations
- Highly custom UI interactions beyond configurable forms
- Non-workflow-based systems without a defined request lifecycle

Summary: If the use case can be structured as a form + workflow + outcome (approval/rejection/record), it can be built using DIGIT Studio.""",
    },
    {
        "id": "cq_04_different_from_lowcode",
        "document": """DIGIT Studio (part of the DIGIT platform by eGovernments Foundation) is different from traditional low-code/no-code tools because it is purpose-built for designing and delivering public service systems, rather than general-purpose app development.

Key differences include:

Service-first approach: DIGIT Studio is designed to build complete public services with a defined lifecycle (application → workflow → approval → record), not just standalone apps or forms.

Built-in workflow and governance: It comes with pre-integrated approval workflows, role-based access, and audit trails, which are essential for government processes but usually need custom setup in other tools.

End-to-end ecosystem: It integrates multiple capabilities out of the box, such as forms, workflow, inbox (task management), document handling, notifications, and optional payments — whereas traditional tools often require separate integrations.

Configuration-driven standardization: Services are built using structured configurations (schemas), ensuring consistency and reusability across different services and departments.

Designed for scale and multi-tenancy: It supports multiple departments or cities (tenants) with isolated configurations and data, which is not always a default feature in typical low-code platforms.

Aligned with public sector needs: It is built to handle compliance, structured data, and approval-based processes commonly required in governance systems.

Summary: While traditional low-code/no-code tools focus on quickly building generic applications, DIGIT Studio focuses on standardizing and accelerating the creation of complete, workflow-driven public service systems.""",
    },
    {
        "id": "cq_05_what_is_service",
        "document": """In DIGIT Studio (part of the DIGIT platform by eGovernments Foundation), a service is a logical grouping used to organize related modules under a common domain or functional area.

- A service represents a domain or category, such as Trade License, Complaint Management, or Health
- Each service can contain one or more modules
- It helps in organizing configurations, workflows, and data for related modules
- It is used across configurations and APIs to consistently identify where a module belongs

Example: A service like Trade License can include modules such as new license application, renewal, and modification.

Summary: A service is a high-level grouping of related modules, used to structure and manage them within DIGIT Studio.""",
    },
    {
        "id": "cq_06_what_is_module",
        "document": """In DIGIT Studio, a module is the actual digital use case or application that users interact with.

It is created within a service and represents a specific functionality, such as applying for a license, raising a complaint, or requesting a permit.

- A module is one part of a service (service = grouping, module = use case)
- Currently, there is typically a one-to-one relationship between service and module in the UI
- In the future, a service can support multiple modules under the same domain

A module usually includes:
- Forms to collect user input
- Workflow for processing and approvals
- Document handling (uploads and generated outputs)
- Inbox/tasks for officials
- Notifications to keep users informed
- Optional payment support

Example: A License module allows users to submit an application, go through approval steps, and receive a license.

Summary: A module is the actual application or use case built within a service, handling the complete flow from request to outcome.""",
    },
    {
        "id": "cq_07_module_vs_service",
        "document": """In DIGIT Studio, service and module represent two different levels of organization.

Service: A service is a high-level grouping or domain that organizes related functionality.
Example: Trade License, Complaint Management

Module: A module is the actual application or use case that users interact with. It is created within a service.
Example: Applying for a new license, renewing a license, or submitting a complaint

Example to understand clearly:
- Service: Trade License
- Modules under it:
  - New Trade License Application
  - Trade License Renewal
  - Trade License Modification

Here, Trade License is the service (grouping), and each individual use case (new, renewal, modification) is a module.

Note: Currently, the UI may show a one-to-one mapping between service and module, but the platform is designed to support multiple modules under a single service.

Summary: A service is the category or domain, while a module is the specific application or functionality built within that category.""",
    },
    {
        "id": "cq_08_checklist_vs_form",
        "document": """In DIGIT Studio (part of the DIGIT platform by eGovernments Foundation), forms and checklists serve different purposes in a service.

Form: A form is used to collect detailed information from users. It includes input fields where users enter data such as name, address, business details, etc.

Checklist: A checklist is used during verification, inspection, or feedback stages. It allows users or officials to mark responses against predefined items (e.g., Yes/No, ratings, remarks).

Use Case Example (Trade License):
1. Form (used by citizen): A user fills out a Trade License application form with details like applicant name, business type, address, required documents.
2. Checklist (used by inspector/official): After submission, an inspector verifies the application using a checklist: Are documents valid? Is the business location correct? Does the applicant meet eligibility criteria?
3. Checklist (used for feedback): After service completion, a checklist can also be used to collect feedback: Was the process smooth? Are you satisfied with the service? Rate your experience.

Key Difference:
- Form = Detailed data collection
- Checklist = Validation, inspection, or feedback using predefined items

Summary: Forms capture structured information, while checklists are used to verify, evaluate, or gather feedback during different stages of the service lifecycle.""",
    },
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
- Configure your application in Draft mode
- Launch it using the Preview option
- Perform actions and observe state transitions
- Test with different user roles

What This Ensures:
- All transitions are working correctly
- Roles are properly assigned
- No steps are missing or misconfigured

Summary: Use Preview to validate your workflow end-to-end before publishing, ensuring a smooth and error-free experience.""",
    },
    {
        "id": "cq_10_responsive_apps",
        "document": """Yes, users can use the generated apps on platforms like mobile, tablet and web. Applications built on DIGIT Studio are responsive across all these platforms.""",
    },
    {
        "id": "cq_11_mobile_apps",
        "document": """DIGIT Studio currently does not have the functionality to build mobile apps. However, all applications built on DIGIT Studio are responsive over mobile, tablet, web.""",
    },
    {
        "id": "cq_12_edit_after_publishing",
        "document": """DIGIT Studio allows you to make as many iterations on a Draft App. A draft app can be previewed and edited. Once you are sure of the app, you can go ahead and publish it. Currently, once an application is published, it cannot be edited in DIGIT Studio.

Publishing marks the application configuration as final and immutable, ensuring consistency and stability for all users and ongoing processes.

We do understand that post-launch, there might be changes that come in and editing of apps may be required. Support for editing published applications is planned as a future enhancement in DIGIT Studio.""",
    },
    {
        "id": "cq_12b_test_workflows",
        "document": """DIGIT Studio provides a Preview feature that allows you to test workflows before publishing and making them live. It helps simulate the application behavior in a controlled environment without affecting real users or data.

Using Preview: While working in Draft mode, you can use Preview to quickly validate your workflow by:
- Simulating the end-to-end workflow flow
- Testing states and transitions
- Validating role-based actions and permissions
- Verifying form behavior and configurations
- Identifying issues before publishing

How It Works:
- Configure your application in Draft mode
- Launch it using the Preview option
- Perform actions and observe state transitions
- (If supported) test with different user roles

What This Ensures:
- All transitions are working correctly
- Roles are properly assigned
- No steps are missing or misconfigured

Summary: Use Preview to validate your workflow end-to-end before publishing, ensuring a smooth and error-free experience.""",
    },
    {
        "id": "cq_12c_citizen_sees_services",
        "document": """Once an application is configured and published in DIGIT Studio, it becomes available to citizens through the Citizen Portal.

Published services are exposed based on their configuration and are accessible to users with the appropriate role (typically CITIZEN).

How Services Are Visible: Citizens can access the created services:
- From the Citizen Portal interface
- Under the relevant module or service category

How It Works:
- The application is published in DIGIT Studio
- It gets mapped to a business service/module
- The service is then rendered on the Citizen Portal
- Citizens can select the service and initiate an application

What This Ensures:
- Only published services are visible to citizens
- Services are shown based on configured access and context
- Users get a structured and organized view of available services

Summary: Citizens can view and access your created services through the Citizen Portal once the application is published and properly configured.""",
    },
    {
        "id": "cq_13_citizen_role",
        "document": """Since applications built on DIGIT Studio are primarily built for Public Service Delivery, they come with a mobile responsive interface for user type 'Citizen'.

Citizens are defined as users that can:
- Self-register using either a mobile OTP/email OTP
- Have the ability to apply for a public service
- Can track current and past application status
- Can perform tasks assigned to them in the workflow

There is no way currently to disable the citizen role.""",
    },
    {
        "id": "cq_14_assign_permissions",
        "document": """Permissions in DIGIT Studio are assigned to roles to control what actions users can perform within an application.

Steps to assign permissions:
1. Go to the Roles Designer — Navigate to the "Manage Roles" section in DIGIT Studio
2. Create or select a role — Create a new role (e.g., Reviewer, Approver, Admin) or edit an existing role
3. Define permissions — Assign permissions such as: Create application, View application, Edit application. These permissions in combination to workflow actions control access to screens and operations.
4. Map role to workflow actions — Link the role to specific workflow actions (e.g., APPROVE, REJECT). Only users with that role can perform those actions.
5. Assign role to users — Map the role to the users via user management (e.g., HRMS). Citizen users have default self registration functionality.

Example: Assume you want to create a role called Inspector, who is a government employee responsible for Field Inspections.
- Role Created in the 'Manage Roles' Section = Inspector
- Permissions given would be to View an application and Edit an Application. This would allow them to see the application and update with necessary comments/fill a checklist. Create an application would not be required here as the inspector would not be logging applications in the system.
- Now, when we create the workflow, add an action called 'Inspection' and tag the Inspector role to it. This ensures that the Inspector will only see the application when the application is in the Workflow State where action 'Inspection' is tagged.

How it works:
- Roles define what a user can see and do
- Workflow mapping ensures users can act only at the right stage
- UI automatically adapts based on assigned permissions

What this ensures:
- Controlled access to features and data
- Secure and role-based operations
- Clear separation of responsibilities

Summary: Permissions are assigned by defining capabilities within a role, mapping those roles to workflow actions, and then assigning the roles to users — ensuring controlled and secure access across the application.""",
    },
    {
        "id": "cq_15_dropdown_values",
        "document": """DIGIT apps are powered by the MDMS service. Master data configured in the MDMS service can be used across applications — and can be added to form dropdowns by using the option of 'Pull data from Database'. For dropdown values specific to a use case, you can simply add the data within the field configuration itself.""",
    },
    {
        "id": "cq_16_dependent_dropdown",
        "document": """A dependent dropdown in DIGIT Studio allows you to show or filter options in one field based on the value selected in another field in a form. This is commonly used in scenarios like Complaint Type → Complaint Subtype, where the subtype options depend on the selected type.

You can configure this using Display Logic and Dependent Field settings, without writing any code.

How it works: A dependent dropdown consists of:
- A parent field (Field X) → controls the selection (e.g., Complaint Type)
- A child field (Field Y) → updates based on the parent (e.g., Complaint Subtype)

The system evaluates conditions and dynamically updates the child field's options or visibility.

Steps to configure:
- Create the parent dropdown (e.g., Complaint Type) with all required options
- Create the child dropdown (e.g., Complaint Subtype) with all possible options
- Select the child field and enable Dependent Field under the Logic tab
- Click Add Display Logic to define rules

Define conditions such as:
- If Complaint Type = Water → Show subtype options like Leakage, No Supply
- If Complaint Type = Garbage → Show subtype options like Not Collected, Overflow
- If Complaint Type = Road → Show subtype options like Pothole, Damage

You can add multiple conditions using AND / OR logic depending on your requirement.

Runtime behavior:
- When the form loads, all logic is evaluated
- When the parent field changes, the child dropdown updates instantly
- Only relevant options are shown to the user
- If no options are available, the field is automatically hidden

Important notes:
- The parent and child fields must be different
- All possible child options should be configured first
- At least one option should remain visible after applying logic
- Display Logic controls both field visibility and option filtering

Summary: A dependent dropdown is created by enabling Dependent Field on the child field and using Display Logic to control which options are shown based on the parent field's value.

Example — In a grievance system: A user selects Complaint Type = Water → The system shows only relevant Complaint Subtypes like Leakage or No Supply. This ensures users see only relevant options, improving accuracy and user experience.""",
    },
    {
        "id": "cq_17_sla",
        "document": """SLAs usually refer to service level agreements. Simply explained, it is the time in which a user has to complete a particular action.

An SLA can be defined at an action level (for example, document verification needs to be completed in 72 hours). In the employee interface, they can view tasks pending for them and the corresponding time left to finish the task. This helps them prioritize tasks that are nearing SLA or have crossed SLA timelines.

In the future, we will allow users to also build rules with SLAs (escalate to department head if inspection is pending for more than 3 days).""",
    },
    {
        "id": "cq_18_where_used",
        "document": """DIGIT Studio is currently in MVP stage and not used in Production. However, a version of Studio has been used to build out 15 use cases of Building Permits in Djibouti, which is currently in UAT stage.""",
    },
    {
        "id": "cq_19_use_cases_supported",
        "document": """DIGIT Studio currently supports Public Service Delivery use cases that following a set process. This includes:

Apply (Form for citizen to apply or employee to apply on behalf of the citizen)
Process (Stages through which the application need to pass through to be completed)
Inform/Track (Inform applicant on the status of their application)
Resolve (Close/Reopen/Reject an Application)

Applications built on DIGIT Studio come with both an employee and citizen interfaces.""",
    },
    {
        "id": "cq_20_approvals_rejections",
        "document": """Approvals and rejections in DIGIT Studio are configured through the workflow configuration.

Steps to configure:
1. Define workflow states — Create the stages of your process, such as: INITIATED, IN_REVIEW, APPROVED, REJECTED
2. Define actions (transitions) — Add actions that move the application between states, for example: APPROVE → moves to APPROVED, REJECT → moves to REJECTED, SEND_BACK → moves to a previous state
3. Map actions to states — Specify from which state an action can be performed and what the next state will be
4. Assign roles to actions — Assign roles (e.g., EMPLOYEE, APPROVER) to each action. Only users with the assigned role can perform that action.
5. Configure UI behavior (optional) — Actions automatically appear as buttons in the UI. Visibility is based on the current state and user role.

What this ensures:
- Controlled and role-based decision making
- Clear progression of applications through defined stages
- Standardized approval and rejection flows

Summary: Approvals and rejections are configured by defining workflow states, mapping actions like APPROVE/REJECT between those states, and assigning roles to control who can perform each action.""",
    },
    {
        "id": "cq_21_multiple_languages",
        "document": """Localization can be updated for any app built on DIGIT Studio. All text on apps generated have localization keys associated with them. You can add as many languages as you like by adding values in that language against their localization key.

Currently, Studio does not come with a UI to add localization. You can access details on how to add localization using a json format.""",
    },
    {
        "id": "cq_22_accessibility",
        "document": """Applications generated from Studio follow WCAG 2.0 accessibility guidelines. Additionally, the UI designed of end user applications generated has been inspired by the Gov.UK design system, which is the gold standard for accessibility, simplicity, and user experience.""",
    },
    {
        "id": "cq_23_add_users",
        "document": """There are three types of users in DIGIT:
- System users (e.g., Boundary Admin, Super Admin) – created during platform installation
- Service-specific users (Employees) – created after a service is published
- Citizens – end users of the service

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
        "id": "cq_24_steps_to_go_live",
        "document": """Going live in DIGIT Studio involves building your service, validating it, and then setting it up in the production environment.

The typical steps are:
1. Build the service in Development (Dev): Create your service, design screens, configure workflows, and set up all required details.
2. Test using preview: Use the preview option to test the complete flow. Create test users and simulate real scenarios to ensure everything works correctly.
3. Finalize the service: Make sure all configurations, fields, workflows, and documents are complete and working as expected.
4. Move configuration to Production: Export or copy the service configuration from Dev and import it into the production environment.
5. Set up production environment: Recreate required users (e.g., employees), roles, and other platform-level settings, as these are not carried over from Dev.
6. Verify in Production: Test the service again in production to ensure everything is correctly set up.
7. Go live: Once verified, the service is ready for actual users.

Important Note: Only the service configuration is moved between environments.
- Data is not transferred
- Users must be created again
- Platform setup needs to be done separately in each environment

Summary: Build and test in Dev, move the configuration to Production, set up required users and settings, verify, and then make the service live for users.""",
    },
    {
        "id": "cq_25_sms_email_integration",
        "document": """DIGIT Studio uses the DIGIT platform's Notification Service to handle SMS and email integrations.

How it works:
- Notifications are triggered based on workflow events (e.g., application submitted, approved, rejected)
- The system sends messages using configured SMS and email providers

Steps to integrate:
1. Configure notification templates in DIGIT Studio — Define message templates (SMS/email content). Use personalised variables to fill based on user given details in form (e.g., Name, Mobile Number, etc.)
2. Map notifications to workflow events — Link notifications to specific states (e.g., on APPROVE, REJECT)
3. Set up notification channels — Enable SMS and/or email in configuration
4. Configure external providers (platform level) — Integrate SMS gateways (e.g., NIC, Twilio). Configure email providers (e.g., SMTP services).

What this ensures:
- Automated communication at each stage of the workflow
- Consistent and configurable messaging
- Support for multiple channels (SMS, email, in-app)

Summary: You integrate SMS and email services by configuring templates and workflow triggers in Studio, while the actual delivery is handled by DIGIT's Notification Service connected to external providers.

Please note that you require the following to be done additionally:
- Configure service providers for email and SMS at the platform level
- In many countries, SMS have to be pre-approved by the government. Please ensure SMS templates configured in the system are approved or notifications may be blocked.""",
    },
    {
        "id": "cq_26_rollback",
        "document": """DIGIT Studio does not provide a direct one-click rollback option.

Rollback is handled by reverting to a previous service configuration:
- If you have a copy of an earlier configuration, you can re-import it
- This effectively restores the service to a previous version

Important points:
- Rollback is manual and configuration-based
- There is no built-in version history or automatic revert option
- It is recommended to keep backups of configurations before making changes

Summary: You can roll back a deployment by re-importing an older configuration, but it needs to be managed manually.""",
    },
    {
        "id": "cq_27_grievance_apps",
        "document": """Yes, you can build grievance or complaint management applications using DIGIT Studio.

How the use case typically works — A grievance system usually follows a structured flow:
1. A citizen submits a complaint through a form (e.g., issue description, location, category, photos)
2. The complaint is registered and assigned to a relevant department or official
3. Officials review the complaint and take action (resolve, escalate, or reject)
4. The complaint moves through different workflow stages (e.g., Open → In Progress → Resolved)
5. The citizen receives notifications at each stage
6. The final outcome (resolution or closure) is recorded and shared

How to build this on DIGIT Studio:
1. Create the form — Define fields like complaint type, description, location, and attachments. Allow upload of images if needed.
2. Configure the workflow — Define states such as Submitted, Assigned, In Progress, Resolved, Rejected. Add actions like Assign, Resolve, Reject, Reopen.
3. Set up roles and assignments — Configure which roles (e.g., operator, supervisor) can take actions at each step.
4. Configure notifications — Send SMS/email/app updates to citizens when status changes.

A grievance application fits well with DIGIT Studio because it follows a clear form → workflow → resolution structure, which is exactly what the platform is designed to support.""",
    },
    {
        "id": "cq_28_use_cases_supported_v2",
        "document": """DIGIT Studio can support use cases that follow a service-based workflow with data collection, processing, and a defined outcome.

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

When it may not be suitable:
- Highly custom UI beyond configurable forms
- Systems without a clear workflow or request lifecycle

Summary: If a use case can be structured as form → workflow → outcome, it can be supported by DIGIT Studio.""",
    },
    {
        "id": "cq_29_authentication",
        "document": """Authentication in DIGIT Studio is handled via the DIGIT platform's core authentication services:
- Users authenticate by generating an auth token (via /user/oauth/token), which is used in all API requests
- The API Gateway validates the token and routes the request to backend services
- User Service manages user credentials and identity
- Access Control (RBAC) handles authorization ensuring users can only access permitted actions/screens
- For citizens, authentication is typically mobile number/email + OTP-based (via OTP service)
- For employees, authentication is typically username + password (or configured method)
- Integration with external identity providers (e.g., SSO) can be enabled via API-based integration at the gateway level (platform level configurations)""",
    },
    {
        "id": "cq_30_configure_notifications",
        "document": """How to configure notifications in DIGIT Studio:

Option 1: From Notification Dashboard (Recommended)
1. Open Notification Dashboard
2. Select Email or SMS
3. Click Create Notification
4. Fill in: Select Workflow State, Select Channel (Email/SMS), Write Message
5. Click Save
Notification is now linked to that workflow state.

Option 2: From Workflow Designer
1. Open Workflow Designer
2. Select a State
3. Go to Notifications section
4. Click + Create Notification
5. Fill in: Channel (Email/SMS), Message
6. Click Save
Notification is attached to that state and appears in Dashboard.

Attach / Detach Notification:
- Select a State
- Attach an existing notification OR remove one
- If using a notification from another state → it will be duplicated

Important Rules:
- Must select: One Workflow State and One Channel
- Notifications are: Sent only to Applicant, Triggered automatically when state is reached

What Happens After Setup: When application reaches that state → Notification is automatically sent (Email/SMS).

Please note that you require the following to be done additionally:
- Configure service providers for email and SMS at the platform level
- In many countries, SMS have to be pre-approved by the government. Please ensure SMS templates configured in the system are approved or notifications may be blocked.""",
    },
    {
        "id": "cq_31_custom_domain",
        "document": """Yes, you can use your own domain when going live with DIGIT Studio, as it is part of the DIGIT platform.

When deploying to production, the application can be hosted on a custom domain (for example, a government or organization-specific URL). This is handled as part of the overall DIGIT platform deployment and infrastructure setup.

Key points:
- You can map the application to your own domain (e.g., service.city.gov.in)
- Domain configuration is managed during DIGIT platform deployment (DNS and hosting setup)
- It enables a branded and official access point for end users

Summary: Since DIGIT Studio runs on the DIGIT platform, it supports custom domains as part of the platform's deployment, allowing services to be accessed through organization-specific URLs.""",
    },
    {
        "id": "cq_32_deploy_application",
        "document": """In DIGIT Studio (part of the DIGIT platform by eGovernments Foundation), deployment is done by moving your service configuration from a development setup to a live (production) environment.

The typical process is:
1. Build and test in Development (Dev): Create your service, design screens, and configure workflows. Use the preview option to test the application and make changes as needed.
2. Finalize your configuration: Once the service is ready, ensure all forms, workflows, and settings are complete and working as expected.
3. Export and move configuration: Copy or export the service configuration from the Dev environment.
4. Import into Production: Set up DIGIT Studio in the production environment and import the configuration to recreate the service.
5. Set up environment-specific details: Configure users, roles, and any required platform settings again in production, as these are not carried over.
6. Go live: Once everything is verified, the application becomes available for real users.

Important Note: Only the configuration is moved between environments.
- Application data is not transferred
- Users and access need to be created again
- Platform-level setup must be done separately in each environment

Summary: You deploy an application by building and testing it in Dev, then exporting the configuration and importing it into Production, where final setup is completed before going live.""",
    },
    {
        "id": "cq_33_import_export_data",
        "document": """Yes, DIGIT Studio supports both exporting and importing service configurations and exporting application data.

Exporting and Importing Service Configuration: You can export/import a service by uploading its workflow configuration (JSON) into the system.

How to export:
- The service configuration is available in JSON format for a published service
- You can simply copy paste the service config

How to import:
- Prepare or obtain the workflow configuration JSON
- Navigate to the studio landing page and click on import service button in DIGIT Studio
- Upload or paste the configuration
- Save and validate the setup

When to use:
- Reusing an existing application
- Migrating configurations between environments (e.g., Dev → UAT → Prod)
- Setting up standardized services quickly

Exporting Application Data: Once users start submitting applications, the data can be downloaded for external use.

How to do it:
- Go to the application/module section where submissions are stored
- Search or filter the required applications
- Use the download/export option (if enabled)

What you get:
- Submitted application data
- Useful for reporting, analysis, and audits

Important Notes:
- Export and Import of service config is limited to configuration (not submitted data)
- Export is available only for submitted applications
- Access to import/export may depend on user roles and permissions

Summary:
- Import workflows using configuration JSON to set up or reuse services
- Export submitted application data for reporting and analysis""",
    },
    {
        "id": "cq_34_external_integrations",
        "document": """Currently, workflows in DIGIT Studio are designed to operate within the platform, and direct integration with external systems is not supported out of the box.

Current Behavior:
- Workflow execution, state transitions, and actions are handled internally within DIGIT
- All configurations such as states, actions, and role mappings are managed inside the platform
- There is no direct UI-based option to connect workflows with external systems

Flexibility and Future Scope: While direct integration is not currently available through DIGIT Studio, the platform is not restricted in terms of extensibility.
- Integrations can be enabled at the backend or service layer
- External systems can interact through APIs or custom extensions
- Workflow events can potentially be extended to trigger external processes

Summary:
- No direct external integration support within Studio currently
- Workflows operate internally within the platform
- The system is extensible and can support integrations through backend or API-level enhancements in the future""",
    },
    {
        "id": "cq_35_create_ui_screen",
        "document": """Creating a new UI screen in DIGIT Studio does not require coding. A UI screen is what users interact with, such as a form or a page within a service.

DIGIT Studio automatically generates UI screens based on Service configurations. Forms are rendered as per the sections and fields configured by you. Citizen landing page, view application page, employee inbox etc are predefined page formats.

It is important to note that you cannot create standalone or custom screens outside this configuration flow. Screens are always generated from forms or checklists, and you cannot manually add an independent "next screen" outside of this structure.

You can preview the screens in the same environment and publish the service once ready.""",
    },
    {
        "id": "cq_36_look_and_feel",
        "document": """In DIGIT Studio, the UI is generated based on DIGIT platform standards. The look and feel can be customized by modifying the CSS configuration at the platform level.

The generated UI is:
- Accessible and compliant with WCAG 2.0 standards
- Designed based on experience from multiple public service delivery applications

Advanced layout options (such as more flexible or horizontal layouts) are planned for future releases.

It is recommended to keep UI changes minimal to maintain consistency, accessibility, and usability.

Summary: UI styling can be adjusted via CSS, but the default UI is standardized, accessible, and optimized — so minimal customization is recommended.""",
    },
    {
        "id": "cq_36b_digit_studio_with_digit",
        "document": """DIGIT Studio is a configuration layer on top of the DIGIT platform.

It allows you to design and configure services (forms, workflows, roles, payments) without writing code.

These configurations are stored as schemas and JSON-based definitions.

Once published, DIGIT Studio activates services on the DIGIT platform.

The configured service is executed by core DIGIT microservices like:
- Workflow engine (for processing applications)
- User Service (for authentication and users)
- Billing & Payment services
- Notification services (SMS, Email, etc.)

The end-user applications (Citizen & Employee apps) are auto-generated based on these configurations.

All requests flow through the API Gateway, which handles authentication, routing, and communication between services.

In short:
DIGIT Studio = "Design & Configure"
DIGIT Platform = "Execute & Run services at scale" """,
    },
    {
        "id": "cq_37_states_and_actions",
        "document": """In a workflow system, states and actions define how a process progresses from start to completion. Together, they control the lifecycle, movement, and access within a workflow.

States: States represent the current stage or status of an entity (such as an application, request, or task) within a workflow.
- Each state indicates where the item currently is in the process lifecycle
- States are fixed checkpoints in the workflow
- Examples: DRAFT, SUBMITTED, IN_REVIEW, APPROVED, REJECTED

Actions: Actions are the operations or transitions that move an entity from one state to another.
- Each action defines: From State → To State, Allowed Roles (who can perform it)
- Actions are triggered by users or system logic
- Examples: SUBMIT, APPROVE, REJECT, SEND_BACK

Tagging Actions to Users (Role-Based Visibility): Actions are tagged to users through role assignment.
- Each action is configured with specific roles
- Users inherit roles from the system (e.g., CITIZEN, EMPLOYEE, APPROVER)
- Based on the logged-in user's role: Only relevant actions are visible in the UI, Users can only perform permitted transitions

Example:
- If a user has APPROVER role → they will see Approve/Reject buttons
- If a user has CITIZEN role → they may only see Submit/Edit options

Use Case Example:
- DRAFT + SUBMIT action → SUBMITTED (CITIZEN role)
- SUBMITTED + VERIFY action → IN_REVIEW (EMPLOYEE role)
- IN_REVIEW + APPROVE action → APPROVED (APPROVER role)
- IN_REVIEW + REJECT action → REJECTED (APPROVER role)
- REJECTED + RESUBMIT action → SUBMITTED (CITIZEN role)

Summary: States represent the current position in a workflow, while actions define the transitions between states. Actions are mapped to roles, ensuring that only authorized users can view and perform them, enabling a secure and structured workflow system.""",
    },
    {
        "id": "cq_37b_deployment_downtime",
        "document": """In DIGIT Studio (part of the DIGIT platform by eGovernments Foundation), downtime depends on what is being deployed — service configuration or the platform itself.

Service Deployment (via DIGIT Studio): Deployment is configuration-based, so downtime is typically minimal or negligible.
- Updates are applied by importing the latest service configuration
- The system switches to the updated configuration without requiring a full system shutdown
- Users may experience a brief moment during the update, but services generally remain available

DIGIT Studio and Platform Setup (initial deployment or upgrades): When setting up or deploying DIGIT Studio along with the underlying DIGIT platform services (backend services, infrastructure, etc.), downtime can be higher.
- May involve service restarts or infrastructure changes
- Typically planned and executed during maintenance windows

Important Note:
- Configuration updates are smooth and low-impact
- Platform-level deployments or upgrades should be planned carefully to minimize user impact

Summary: Service-level deployments in DIGIT Studio have minimal downtime, while initial setup or platform-level deployments may involve planned downtime depending on the changes.""",
    },
    {
        "id": "cq_38_assign_roles_workflow",
        "document": """In a workflow, roles are assigned to actions (transitions) rather than directly to states. This ensures that only authorized users can perform specific actions at each step and control how the workflow progresses.

Steps to Assign Roles:

A. Identify the Workflow Step (State): Determine the stage where access needs to be controlled (e.g., IN_REVIEW, SUBMITTED). This helps define where in the workflow the restriction applies.

B. Define the Action: Specify the transition that moves the workflow forward or backward (e.g., APPROVE, REJECT, SEND_BACK). Each action represents a possible operation from the current state.

C. Assign Roles to the Action: Select the relevant roles from the dropdown (e.g., CITIZEN, EMPLOYEE, APPROVER, etc.) who are allowed to perform that action.
- You can assign one or multiple roles to a single action
- Only users with the assigned roles will be able to view and perform that action
- If a user does not have the required role, the action will not be visible in the UI and cannot be executed

Additional Details:
- Role assignment ensures proper access control and workflow security
- It directly impacts which actions are available to which users at a given state
- Roles should be aligned with the user-role mapping defined in the system
- Incorrect role configuration may lead to: Actions not appearing in UI, Workflow getting stuck due to missing permissions

Summary: Roles are assigned to actions to control who can move the workflow between states, ensuring that only authorized users can perform specific operations at each step.""",
    },
    {
        "id": "cq_39_rbac",
        "document": """Role-Based Access Control (RBAC) ensures that users can only perform actions and access features based on their assigned roles.

Instead of assigning permissions directly to users, permissions are mapped to roles, and users are assigned those roles.

How It Works:
- Each user is assigned one or more roles (e.g., CITIZEN, EMPLOYEE, APPROVER)
- Each workflow action is configured with allowed roles
- When a user accesses an application: The system checks the current state, then filters available actions based on the user's roles

In Workflows:
- Roles are mapped to actions (transitions)
- Only users with the required role can: View the action in the UI, Perform the action

Example: If an action APPROVE is assigned to APPROVER, only users with that role can approve the application.

Key Points:
- A user can have multiple roles
- Access is role-driven, not user-specific
- Incorrect role configuration can: Hide required actions, Block workflow progression

Summary: Role-based access works by mapping roles to actions, ensuring that only authorized users can view and perform specific operations at each stage of the workflow.""",
    },
    {
        "id": "cq_38b_data_storage",
        "document": """In DIGIT Studio (part of the DIGIT platform by eGovernments Foundation), data is stored in the DIGIT platform's backend systems:
- Application and transactional data is stored in PostgreSQL databases used by different DIGIT backend services
- Search and analytics data is indexed in Elasticsearch
- Documents/files are stored in the File Store service (object storage like S3 or similar)

Data is managed by the platform services and is segregated by tenant (e.g., city or department).

Summary: Data is primarily stored in PostgreSQL, with Elasticsearch used for search and File Store for documents.""",
    },
    {
        "id": "cq_39b_identity_systems",
        "document": """Yes, DIGIT supports integration with existing external identity systems:
- Authentication is handled via the platform's API Gateway and User Service, allowing extensibility
- You can integrate with systems like SSO, LDAP, or other identity providers using API-based configuration
- External systems can be used for user authentication and token generation
- DIGIT's role-based access control (RBAC) continues to manage authorization within the platform
- Integration typically requires configuration and setup at the environment level""",
    },
    {
        "id": "cq_40_permit_license_apps",
        "document": """Yes, you can build permit and license applications using DIGIT Studio:

Service setup:
- Create a new service/module (e.g., Trade License, Building Permit)
- Configure multiple service types like New, Renewal, Edit/Renewal

Form configuration:
- Design application forms using the form builder
- Add fields, validations, and help text
- Capture required applicant and service details

Workflow configuration:
- Define the end-to-end approval workflow
- Set up stages like Submission → Scrutiny → Approval
- Map roles (Reviewer, Approver) and actions (approve, reject, forward)

Role & access control:
- Configure employee roles for processing applications
- Assign permissions based on workflow responsibilities

Notifications:
- Configure SMS/Email notifications for key events (e.g., submission, approval)

Application lifecycle:
- Citizens can apply and track applications via the Citizen app
- Employees can process applications via the Employee app
- The system manages the lifecycle from submission to approval

Current limitations:
- Document configuration (uploads/PDF generation) is not available in Studio yet
- Payment configuration is not available in Studio and requires external setup
- UI customization is limited and system-generated

In summary: You can configure the core permit/license workflow and application journey, while some advanced capabilities (documents, payments, UI customization) are not yet supported in Studio.""",
    },
]


def ingest():
    conn = get_conn()

    print(f"Ingesting {len(CHUNKS)} closed-question answer chunks into {TABLE}...")
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
                        INSERT INTO {TABLE} (id, document, embedding)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            document = EXCLUDED.document,
                            embedding = EXCLUDED.embedding
                    """, (chunk_id, text, emb))
                    inserted += 1
                    print(f"  [{i+1}/{len(CHUNKS)}] ✅ {chunk_id}")
                except Exception as e:
                    skipped += 1
                    print(f"  [{i+1}/{len(CHUNKS)}] ❌ {chunk_id}: {e}")

        conn.commit()
        print(f"\n✅ Done. Inserted/updated: {inserted}, Skipped: {skipped}")

    finally:
        conn.close()


if __name__ == "__main__":
    ingest()
