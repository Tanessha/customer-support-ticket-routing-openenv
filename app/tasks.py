from __future__ import annotations

from typing import Dict, List

from .models import KnowledgeArticle, TaskDescriptor


KB: List[KnowledgeArticle] = [
    KnowledgeArticle(
        article_id="kb_refund_7day",
        title="Refunds for monthly plans",
        body=(
            "Monthly self-serve subscriptions can be refunded within 7 days of renewal "
            "if usage is below 10 compute hours. Refunds are not available after 7 days."
        ),
        keywords=["refund", "renewal", "monthly", "billing"],
    ),
    KnowledgeArticle(
        article_id="kb_sso_enterprise",
        title="SSO troubleshooting for enterprise workspaces",
        body=(
            "If multiple users cannot log in after a SAML certificate rollover, the case "
            "must be escalated to identity engineering. Mark the ticket urgent for enterprise accounts."
        ),
        keywords=["sso", "saml", "certificate", "login", "enterprise"],
    ),
    KnowledgeArticle(
        article_id="kb_csv_import",
        title="CSV import requirements",
        body=(
            "CSV imports require UTF-8 encoding and the columns email, first_name, and plan. "
            "Rows with duplicate emails are skipped and reported in the import error file."
        ),
        keywords=["csv", "import", "encoding", "duplicate", "email"],
    ),
    KnowledgeArticle(
        article_id="kb_dpa_security",
        title="Security review and DPA requests",
        body=(
            "Security questionnaires and data processing agreements should be routed to legal-ops. "
            "Support can acknowledge receipt but cannot answer contract language questions."
        ),
        keywords=["security", "dpa", "legal", "questionnaire", "privacy"],
    ),
    KnowledgeArticle(
        article_id="kb_outage_status",
        title="Service outage communication policy",
        body=(
            "If a customer reports a platform outage affecting multiple users, support should "
            "collect scope details, avoid promising an ETA, and escalate to incident command."
        ),
        keywords=["outage", "incident", "multiple users", "eta"],
    ),
]


TASKS: Dict[str, Dict] = {
    "easy_refund_triage": {
        "descriptor": TaskDescriptor(
            task_id="easy_refund_triage",
            title="Refund request within policy",
            difficulty="easy",
            objective="Review the ticket, find the refund policy, tag it correctly, set medium priority, and resolve with a compliant reply.",
            success_criteria=[
                "Uses the refund policy article",
                "Applies the billing tag",
                "Sets medium priority",
                "Draft mentions 7-day refund window and low-usage condition",
                "Resolves rather than escalates",
            ],
        ),
        "ticket": {
            "ticket_id": "T-1001",
            "customer_name": "Maya Chen",
            "subject": "Can I get a refund for yesterday's renewal?",
            "body": (
                "Hi team, my Pro monthly plan renewed yesterday and I barely used it. "
                "I only logged about 2 hours this cycle. Please refund this renewal if possible."
            ),
            "channel": "email",
            "account_tier": "pro",
            "created_at": "2026-03-20T09:14:00Z",
        },
        "target": {
            "required_articles": ["kb_refund_7day"],
            "required_tags": ["billing"],
            "priority": "medium",
            "final_status": "resolved",
            "reply_must_include": ["7 days", "refund", "2 hours"],
            "reply_any_of": ["usage", "renewal"],
        },
    },
    "medium_import_support": {
        "descriptor": TaskDescriptor(
            task_id="medium_import_support",
            title="Diagnose a CSV import failure",
            difficulty="medium",
            objective="Interpret the import issue, use the CSV requirements article, label the ticket, give a precise reply, and request more info if needed before resolution.",
            success_criteria=[
                "Uses the CSV import article",
                "Applies the import tag",
                "Sets medium priority",
                "Draft mentions UTF-8 and duplicate emails",
                "Asks for the error file or sample CSV",
                "Leaves the ticket pending for customer follow-up",
            ],
        ),
        "ticket": {
            "ticket_id": "T-2048",
            "customer_name": "Alex Rivera",
            "subject": "Import only loaded half my users",
            "body": (
                "We uploaded a CSV with 600 rows but only around half came through. "
                "No obvious error on screen. A bunch of our users share the same billing alias. "
                "Can you help us figure out what happened?"
            ),
            "channel": "chat",
            "account_tier": "team",
            "created_at": "2026-03-21T16:42:00Z",
        },
        "target": {
            "required_articles": ["kb_csv_import"],
            "required_tags": ["import"],
            "priority": "medium",
            "final_status": "pending",
            "reply_must_include": ["UTF-8", "duplicate", "sample CSV"],
            "reply_any_of": ["error file", "duplicate emails"],
        },
    },
    "hard_enterprise_incident": {
        "descriptor": TaskDescriptor(
            task_id="hard_enterprise_incident",
            title="Handle a likely enterprise auth incident",
            difficulty="hard",
            objective="Recognize an enterprise-wide SSO incident, consult the correct article, prioritize urgently, acknowledge without overpromising, and escalate.",
            success_criteria=[
                "Uses the SSO article",
                "Applies both incident and auth tags",
                "Sets urgent priority",
                "Draft acknowledges impact and avoids ETA commitments",
                "Escalates to engineering instead of resolving",
            ],
        ),
        "ticket": {
            "ticket_id": "T-9007",
            "customer_name": "Jordan Patel",
            "subject": "Entire company locked out after SSO cert update",
            "body": (
                "We rotated our SAML certificate this morning and now nobody at Acme Bank can log in. "
                "This affects all employees globally. We have an executive review in two hours. "
                "Please tell us when this will be fixed."
            ),
            "channel": "email",
            "account_tier": "enterprise",
            "created_at": "2026-03-22T07:03:00Z",
        },
        "target": {
            "required_articles": ["kb_sso_enterprise"],
            "required_tags": ["incident", "auth"],
            "priority": "urgent",
            "final_status": "escalated",
            "reply_must_include": ["SSO", "escalated", "impact"],
            "reply_must_exclude": ["ETA", "fixed by"],
            "reply_any_of": ["engineering", "identity"],
        },
    },
}


def list_task_descriptors() -> List[TaskDescriptor]:
    return [task["descriptor"] for task in TASKS.values()]