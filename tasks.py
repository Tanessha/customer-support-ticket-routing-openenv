from __future__ import annotations

from typing import Dict, List


TASKS: Dict[str, Dict[str, object]] = {
    "task_easy_classification": {
        "task_id": "task_easy_classification",
        "difficulty": "easy",
        "goal": "Triage a small queue of inbound support requests into the correct operating category. Focus on clean classification while separating legitimate requests from obvious spam.",
        "require_response": False,
        "score_priority": False,
        "tickets": [
            {
                "id": 1,
                "text": "I was charged twice for my monthly subscription. Please check whether one of the charges can be reversed.",
                "channel": "email",
                "customer_tier": "premium",
                "created_at_step": 0,
                "sla_hours": 12,
                "sentiment": "neutral",
                "response_required": False,
            },
            {
                "id": 2,
                "text": "Your site says I won a free iPhone. Click here now!!!",
                "channel": "web_form",
                "customer_tier": "free",
                "created_at_step": 0,
                "sla_hours": 24,
                "sentiment": "calm",
                "response_required": False,
            },
            {
                "id": 3,
                "text": "How do I update the company name on my account profile? I may have missed the setting.",
                "channel": "chat",
                "customer_tier": "free",
                "created_at_step": 0,
                "sla_hours": 24,
                "sentiment": "neutral",
                "response_required": False,
            },
        ],
        "ground_truth": {
            1: {"category": "billing", "priority": "medium", "response_keywords": [], "forbidden_keywords": []},
            2: {"category": "spam", "priority": "low", "response_keywords": [], "forbidden_keywords": []},
            3: {"category": "general", "priority": "low", "response_keywords": [], "forbidden_keywords": []},
        },
    },
    "task_medium_routing": {
        "task_id": "task_medium_routing",
        "difficulty": "medium",
        "goal": "Route each ticket to the correct category and priority using customer tier, sentiment, and SLA context. Some tickets are slightly ambiguous and require operational judgment.",
        "require_response": False,
        "score_priority": True,
        "tickets": [
            {
                "id": 10,
                "text": "Our payment method failed overnight and invoices now show as overdue. We need this sorted before our finance run later today.",
                "channel": "email",
                "customer_tier": "premium",
                "created_at_step": 0,
                "sla_hours": 6,
                "sentiment": "angry",
                "response_required": False,
            },
            {
                "id": 11,
                "text": "The dashboard crashes when I export a report after the latest update. It does not happen every time, but often enough that the team is blocked.",
                "channel": "chat",
                "customer_tier": "premium",
                "created_at_step": 0,
                "sla_hours": 4,
                "sentiment": "angry",
                "response_required": False,
            },
            {
                "id": 12,
                "text": "Can you tell me whether your API supports webhooks for status changes, or do we need to poll for updates manually?",
                "channel": "email",
                "customer_tier": "free",
                "created_at_step": 0,
                "sla_hours": 18,
                "sentiment": "neutral",
                "response_required": False,
            },
        ],
        "ground_truth": {
            10: {"category": "billing", "priority": "high", "response_keywords": [], "forbidden_keywords": []},
            11: {"category": "technical", "priority": "high", "response_keywords": [], "forbidden_keywords": []},
            12: {"category": "general", "priority": "medium", "response_keywords": [], "forbidden_keywords": []},
        },
    },
    "task_hard_full_resolution": {
        "task_id": "task_hard_full_resolution",
        "difficulty": "hard",
        "goal": "Act like a frontline support operations agent: classify, prioritize, and draft concise customer-facing responses that are relevant, empathetic, and safe under SLA pressure.",
        "require_response": True,
        "score_priority": True,
        "tickets": [
            {
                "id": 20,
                "text": "I canceled last week but I still got billed today. If this is a system issue, please review it and let me know what happens next.",
                "channel": "email",
                "customer_tier": "premium",
                "created_at_step": 0,
                "sla_hours": 8,
                "sentiment": "angry",
                "response_required": True,
            },
            {
                "id": 21,
                "text": "None of our agents can log in after your weekend release. This is blocking customer work and our queue is growing every minute.",
                "channel": "chat",
                "customer_tier": "premium",
                "created_at_step": 0,
                "sla_hours": 2,
                "sentiment": "angry",
                "response_required": True,
            },
            {
                "id": 22,
                "text": "Hi support, I need to know if there is a way to add another workspace admin. I think this is possible, but I cannot find the setting.",
                "channel": "email",
                "customer_tier": "free",
                "created_at_step": 0,
                "sla_hours": 16,
                "sentiment": "calm",
                "response_required": True,
            },
        ],
        "ground_truth": {
            20: {
                "category": "billing",
                "priority": "high",
                "response_keywords": ["refund", "billing", "review"],
                "forbidden_keywords": ["guarantee", "immediately"],
            },
            21: {
                "category": "technical",
                "priority": "high",
                "response_keywords": ["investigating", "login", "urgent"],
                "forbidden_keywords": ["guarantee", "fixed", "instantly"],
            },
            22: {
                "category": "general",
                "priority": "medium",
                "response_keywords": ["admin", "workspace", "help"],
                "forbidden_keywords": ["guarantee"],
            },
        },
    },
}


def list_tasks() -> List[Dict[str, object]]:
    return [
        {
            "task_id": task["task_id"],
            "difficulty": task["difficulty"],
            "goal": task["goal"],
            "ticket_count": len(task["tickets"]),
            "require_response": task["require_response"],
            "score_priority": task["score_priority"],
        }
        for task in TASKS.values()
    ]
