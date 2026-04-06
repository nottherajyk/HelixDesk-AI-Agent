"""EmailGenerator — produces synthetic email events for the HelixDesk environment."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np


@dataclass
class EmailEvent:
    """A single customer email entering the support queue."""
    email_id: str
    sender_email: str
    category: str
    ticket_type: str            # "query" or "complaint"
    body_text: str
    sentiment_intensity: float  # 0.0 – 1.0
    has_keyword_flag: bool
    customer_tier: str          # "enterprise" | "standard" | "free"
    true_priority: str          # ground-truth priority for reward eval
    created_at_minutes: float   # SimClock time at generation


# ---------------------------------------------------------------------------
# Templates — 5 queries + 5 complaints per category (80 total)
# ---------------------------------------------------------------------------
_QUERY_TEMPLATES: dict[str, list[str]] = {
    "login_failure": [
        "Hi, I'm unable to log into my account. Could you help me reset my credentials?",
        "I keep getting 'invalid password' when I try to sign in. What should I do?",
        "My two-factor authentication code isn't arriving. How can I access my account?",
        "I forgot which email I used to register. Can you help me find my account?",
        "The login page shows a spinner but never loads. Is there a known issue?",
    ],
    "billing_dispute": [
        "Can you explain the charge of $49.99 on my last invoice? I don't recognize it.",
        "I was billed twice for the same subscription month. Could you look into this?",
        "My payment method expired but I was still charged. How is that possible?",
        "I see a 'processing fee' I wasn't told about. Can you clarify what that covers?",
        "When will I receive my refund for the overcharge from last month?",
    ],
    "refund_request": [
        "I'd like to request a refund for my recent purchase. The product didn't meet expectations.",
        "Can I get my money back? I returned the item two weeks ago and haven't heard anything.",
        "I cancelled within the free trial period but was still charged. Please refund me.",
        "The service was down for three days last week. I'd like a prorated refund.",
        "I purchased the wrong plan by mistake. Can I get a refund and switch?",
    ],
    "product_defect": [
        "Is this a known bug? The export feature keeps producing corrupt files.",
        "The dashboard widgets stopped loading after the last update. Any workaround?",
        "I noticed the search function returns wrong results for quoted phrases. Is this expected?",
        "The mobile app crashes when I try to upload images larger than 5 MB.",
        "API responses are returning HTTP 500 intermittently. Can you check your servers?",
    ],
    "shipping_delay": [
        "My order was supposed to arrive three days ago. Can you check the shipping status?",
        "The tracking number you gave me shows no updates for a week. What's happening?",
        "I paid for express shipping but my package is still in transit after 7 days.",
        "My order shows 'delivered' but I never received it. Can you investigate?",
        "Is there a delay on international shipments right now? My order hasn't moved.",
    ],
    "account_locked": [
        "My account seems to be locked after too many login attempts. How do I unlock it?",
        "I received a security alert and now my account is suspended. What do I do?",
        "I can't access any features — the page says 'account restricted'. Can you help?",
        "My team member's account was deactivated by mistake. How can we reactivate it?",
        "I changed my password but the system still shows my account as locked.",
    ],
    "data_privacy": [
        "How can I download all the data you have stored about me?",
        "I'd like to know your data retention policy. How long do you keep user data?",
        "Can you confirm whether my personal data was part of the recent breach notification?",
        "I want to opt out of data sharing with third parties. How do I do that?",
        "Please walk me through how to delete my account and all associated data.",
    ],
    "general_query": [
        "What are your business hours for phone support?",
        "Do you offer a student discount or educational pricing?",
        "Can I upgrade my plan mid-billing-cycle, or do I have to wait?",
        "Where can I find documentation for your REST API?",
        "Is there a way to export my usage analytics to CSV?",
    ],
}

_COMPLAINT_TEMPLATES: dict[str, list[str]] = {
    "login_failure": [
        "I've been locked out of my account for THREE DAYS and nobody has responded. This is unacceptable!",
        "Your login system is completely broken. I've lost access to critical business data because of this.",
        "I've sent five emails about my login issue and gotten zero response. Do you even have support staff?",
        "My team of 20 cannot access the platform. We are losing thousands of dollars every hour this continues.",
        "I'm furious — your password reset sends me in circles and the phone line is always busy.",
    ],
    "billing_dispute": [
        "You charged me $500 without authorization! I demand an immediate refund and an explanation.",
        "This is the THIRD time I've been double-billed. Your billing department is incompetent.",
        "I cancelled my subscription months ago yet you keep charging me. This feels like theft.",
        "Your hidden fees are disgraceful. I was quoted $29/month and I'm paying nearly $60.",
        "I've been overcharged for six consecutive months and every time I call I get a different excuse.",
    ],
    "refund_request": [
        "It has been 30 days since I requested a refund and NOTHING has happened. This is fraudulent!",
        "Your refund policy is deliberately confusing to prevent people from getting their money back.",
        "I returned the product in perfect condition weeks ago. Where is my refund? This is stealing.",
        "You promised a 7-day refund process. It's been three weeks. I want my money NOW.",
        "I've wasted hours on hold trying to get a simple refund. Your process is deliberately broken.",
    ],
    "product_defect": [
        "Your product literally destroyed my entire dataset! The sync feature corrupted everything!",
        "I've found critical security vulnerabilities in your software and nobody seems to care.",
        "The latest update broke EVERYTHING. My workflow is destroyed and you haven't even acknowledged it.",
        "Your software crashed and I lost 8 hours of unsaved work. This is absolutely devastating.",
        "Multiple features that I PAY FOR have been broken for weeks. Do you even test your releases?",
    ],
    "shipping_delay": [
        "My order is now TWO WEEKS late and your tracking says it's still 'processing'. This is a joke!",
        "I needed this delivery for a critical event that has now passed. Your service has cost me dearly.",
        "Three separate orders, all delayed, all with useless tracking. Your logistics are a disaster.",
        "You charged me $30 for 'priority shipping' and my package arrived LATER than standard. Outrageous!",
        "My perishable items arrived spoiled because of your inexcusable shipping delays.",
    ],
    "account_locked": [
        "You locked my business account with NO warning and I've lost access to all my client data!",
        "My account has been 'under review' for TWO WEEKS. My entire business depends on this!",
        "You falsely flagged my account for suspicious activity and now I can't serve my customers!",
        "I've verified my identity FIVE TIMES and you still won't unlock my account. This is harassment.",
        "Locking my account over a $5 payment discrepancy is absurd. I've been a customer for 4 years!",
    ],
    "data_privacy": [
        "I requested my data deletion THREE MONTHS AGO and you still have my records! This violates GDPR!",
        "You shared my personal data with advertisers without my consent. I'm consulting my attorney.",
        "I never agreed to your new privacy policy and you're treating my silence as acceptance. That's illegal.",
        "Your data breach affected my account and I was never notified. This is a serious legal violation.",
        "You're tracking my usage without disclosure. I have screenshots proving undisclosed data collection.",
    ],
    "general_query": [
        "I've been waiting for a response to my support request for OVER A WEEK. Do you even read these?",
        "Your documentation is hopelessly out of date and your support team gives contradictory answers.",
        "I've been a paying customer for years and I get treated worse than a free account. Shameful.",
        "Nobody in your company seems to know how your own product works. Every agent tells me something different.",
        "Your self-service portal has been 'under maintenance' for days. How am I supposed to get help?",
    ],
}

_KEYWORD_FLAG_TEMPLATES: list[str] = [
    "If this is not resolved immediately, I will be taking legal action against your company.",
    "I am filing a complaint with the consumer forum and SEBI regarding your malpractices.",
    "I have already contacted RBI about your unauthorized charges and they are investigating.",
    "I will be lodging a complaint with IRDAI if my insurance claim is not processed today.",
    "I'm posting about this experience on social media — my followers deserve to know.",
    "This is going viral on Twitter. Your company's reputation is about to take a serious hit.",
    "This is nothing short of fraud and I am reporting it as a scam to the authorities.",
    "I am preparing to file a lawsuit. My lawyer is already drafting the court complaint.",
]

_KEYWORD_FLAG_WORDS = [
    "legal action", "consumer forum", "SEBI", "RBI", "IRDAI",
    "social media post", "going viral", "fraud", "scam", "court",
    "police complaint", "lawsuit",
]

_SENDER_DOMAINS = [
    "gmail.com", "outlook.com", "yahoo.com", "company.co", "business.org",
    "enterprise.io", "startup.dev", "mail.com", "protonmail.com", "fastmail.com",
]

_FIRST_NAMES = [
    "Aarav", "Priya", "Rahul", "Ananya", "Vikram", "Sneha", "Arjun", "Divya",
    "Karan", "Meera", "Rohan", "Nisha", "Amit", "Pooja", "Siddharth", "Kavita",
    "David", "Sarah", "Michael", "Emily", "James", "Lisa", "Robert", "Jennifer",
]


class EmailGenerator:
    """Generates synthetic email events for each simulation step.

    Uses config-driven ratios for query vs complaint, customer tier,
    keyword flags, and category distribution (uniform across 8 categories).
    """

    def __init__(self, config: dict, seed: int):
        self.rng = np.random.default_rng(seed)
        self.cfg = config["email_gen"]
        self.categories: list[str] = self.cfg["categories"]
        self._query_templates = _QUERY_TEMPLATES
        self._complaint_templates = _COMPLAINT_TEMPLATES
        self._keyword_templates = _KEYWORD_FLAG_TEMPLATES

    def next(self, sim_time: float) -> EmailEvent:
        """Generate the next synthetic email event.

        Args:
            sim_time: Current simulation time in minutes.

        Returns:
            A fully populated EmailEvent.
        """
        # --- Ticket type ---
        is_query = self.rng.random() < self.cfg["query_ratio"]
        ticket_type = "query" if is_query else "complaint"

        # --- Category ---
        category = self.rng.choice(self.categories)

        # --- Customer tier ---
        tier_roll = self.rng.random()
        if tier_roll < self.cfg["enterprise_rate"]:
            customer_tier = "enterprise"
        elif tier_roll < self.cfg["enterprise_rate"] + self.cfg["standard_rate"]:
            customer_tier = "standard"
        else:
            customer_tier = "free"

        # --- Keyword flag ---
        has_keyword_flag = (
            not is_query and self.rng.random() < self.cfg["keyword_flag_rate"]
        )

        # --- Body text ---
        if is_query:
            templates = self._query_templates[category]
            body_text = templates[self.rng.integers(0, len(templates))]
        else:
            templates = self._complaint_templates[category]
            body_text = templates[self.rng.integers(0, len(templates))]

        # Generate contextual variables
        last_names = ["Chen", "Smith", "Patel", "Johnson", "Garcia", "Kim", "Lee", "Davis", "Martinez", "Lopez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin", "White", "Thompson", "Brown"]
        dates = ["March 15th", "last Tuesday", "yesterday", "April 2nd", "Monday", "the 1st of the month", "the weekend"]
        products = ["ProPlan", "BasicSuite", "EnterpriseKit"]
        
        last_name = last_names[self.rng.integers(0, len(last_names))]
        date_ref = dates[self.rng.integers(0, len(dates))]
        product = products[self.rng.integers(0, len(products))]
        order_num = f"HD-{self.rng.integers(10000, 99999)}"

        # We must pull first_name earlier for the intro (we'll reuse it for sender later)
        first_name = _FIRST_NAMES[self.rng.integers(0, len(_FIRST_NAMES))]
        full_name = f"{first_name} {last_name}"

        # Injecting into body text
        intro = f"Hi, I'm {full_name} (order {order_num}). I've had issues with my {product} account since {date_ref}. "
        body_text = intro + body_text

        if has_keyword_flag:
            kw_template = self._keyword_templates[
                self.rng.integers(0, len(self._keyword_templates))
            ]
            body_text = f"{body_text} {kw_template}"

        # --- Sentiment ---
        if is_query:
            sentiment = float(self.rng.beta(2.0, 5.0))
        else:
            sentiment = float(self.rng.beta(4.0, 2.0))

        if has_keyword_flag:
            sentiment = max(sentiment, 0.85)

        sentiment = float(np.clip(sentiment, 0.0, 1.0))

        # --- Ground-truth priority ---
        if has_keyword_flag:
            true_priority = "critical"
        elif customer_tier == "enterprise":
            true_priority = "high"
        elif sentiment > 0.85:
            true_priority = "high"
        elif ticket_type == "complaint":
            true_priority = "medium"
        else:
            true_priority = "normal"

        # --- Sender ---
        domain = _SENDER_DOMAINS[self.rng.integers(0, len(_SENDER_DOMAINS))]
        sender_email = f"{first_name.lower()}{self.rng.integers(10, 999)}@{domain}"

        # Generate deterministic ID from the seeded RNG (uuid4 is non-deterministic)
        id_bytes = self.rng.integers(0, 256, size=16, dtype=np.uint8)

        email_id = id_bytes.tobytes().hex()

        return EmailEvent(
            email_id=email_id,
            sender_email=sender_email,
            category=category,
            ticket_type=ticket_type,
            body_text=body_text,
            sentiment_intensity=sentiment,
            has_keyword_flag=has_keyword_flag,
            customer_tier=customer_tier,
            true_priority=true_priority,
            created_at_minutes=sim_time,
        )
