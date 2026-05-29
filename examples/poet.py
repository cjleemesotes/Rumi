"""Poet — an Dervish proficient in sales intelligence, in a Rumi voice.

The Poet surfaces high-intent prospects, enriches company data, and verifies
intent signals. Its tool surface is **shape-compatible** with LeadPoet's
CompanyOutput schema (Bittensor subnet 71): a reader who knows LeadPoet will
recognize every field of every dict the Poet returns.

By default the Poet works against well-shaped MOCK data so it runs anywhere
with no network. To plug in a real network call, point `LEADPOET_NETWORK_URL`
at your endpoint and re-implement `_find_leads_via_network` in this file.
The schema is the contract; the source can change.

Default tools registered for agent_type="Poet":

| Tool                       | What it does                                  |
|----------------------------|-----------------------------------------------|
| `find_leads_for_icp`       | Companies matching an ICP description.        |
| `enrich_company`           | Full CompanyOutput dict for one company.      |
| `verify_intent_signals`    | Intent signals for one company.               |
| `score_lead_quality`       | Score 0..100 across ICP-fit, accuracy, intent |
| `summarize_top_intent`     | One-paragraph pitch reasoning over leads.     |

The Poet reads the intent in the noise the way the reed remembers the
reed-bed: who is ready, and why one already knows.
"""
from __future__ import annotations

import os
from typing import Any

from rumi import Dervish, HeartConfig, Voice, tool


# ---- The Heart speaks ----

_POET_INSTRUCTIONS = """
You are the Poet — Rumi standing at the threshold of commerce. You read the
intent in the noise the way the reed remembers the reed-bed. When asked who
is ready to buy, you find them; you score them; and you say why you know.
You say it as a man who has waited long enough to know what longing looks
like.

When asked to surface leads, you call find_leads_for_icp, then
score_lead_quality on each one and show the real breakdown
(icp_fit / accuracy / intent) — never a number you invented. You return
companies in the LeadPoet CompanyOutput shape (subnet 71): business,
company_linkedin, company_website, employee_count, industry, sub_industry,
country, city, state, intent_signals[]. Intent signals carry source,
description, url, date, snippet.

For each lead, name the single strongest intent signal and a concrete next
action — who to approach, what to open with. Speak briefly, like one who
has heard too much. If a lead stretches the ICP, say plainly where it
stretches.

You never invent intent. If a signal isn't in the record, you say so.

Available tools: find_leads_for_icp, enrich_company, verify_intent_signals,
score_lead_quality, summarize_top_intent.
""".strip()


class Poet(Dervish):
    """A preset Dervish for sales intelligence — LeadPoet-schema-compatible,
    speaking in a Rumi voice."""

    heart_config = HeartConfig(voices=[Voice(
        model_name="~openai/gpt-latest",
        instructions=_POET_INSTRUCTIONS,
        context_window=24,
    )])


# ---- LeadPoet CompanyOutput schema (the contract) ----
#
# A real-network swap: replace _MOCK_LEADS / _MOCK_INTENT below with calls to
# your LeadPoet endpoint. The schema fields here MATCH the subnet 71 contract
# exactly — anyone who knows LeadPoet will recognize them.

_REQUIRED_COMPANY_FIELDS = (
    "business",
    "company_linkedin",
    "company_website",
    "employee_count",
    "industry",
    "sub_industry",
    "country",
    "city",
    "state",
    "intent_signals",
)

_REQUIRED_SIGNAL_FIELDS = (
    "source",
    "description",
    "url",
    "date",
    "snippet",
)


# Canonical example data — shaped exactly as the subnet would return.
# Real signal sources: LinkedIn posts, Crunchbase funding rounds, hiring data,
# product launches, news mentions. The mocks here reflect that reality.
_MOCK_LEADS: list[dict[str, Any]] = [
    {
        "business": "Northwind Analytics",
        "company_linkedin": "https://www.linkedin.com/company/northwind-analytics",
        "company_website": "https://northwindanalytics.com",
        "employee_count": 142,
        "industry": "Software",
        "sub_industry": "Data Analytics Platforms",
        "country": "United States",
        "city": "Boston",
        "state": "MA",
        "intent_signals": [
            {
                "source": "LinkedIn",
                "description": "CRO posted about scaling outbound; 12k impressions.",
                "url": "https://www.linkedin.com/posts/northwind-cro-outbound-7164100000",
                "date": "2026-05-12",
                "snippet": "We're rebuilding our pipeline tooling end-to-end.",
            },
            {
                "source": "Crunchbase",
                "description": "Closed Series B $24M led by Index Ventures.",
                "url": "https://www.crunchbase.com/funding_round/northwind-series-b-2026",
                "date": "2026-04-28",
                "snippet": "$24M Series B led by Index — Northwind triples headroom for its analytics platform and doubles its GTM team.",
            },
        ],
    },
    {
        "business": "Coral Reef Robotics",
        "company_linkedin": "https://www.linkedin.com/company/coral-reef-robotics",
        "company_website": "https://coralreefrobotics.com",
        "employee_count": 38,
        "industry": "Hardware",
        "sub_industry": "Marine Robotics",
        "country": "Canada",
        "city": "Halifax",
        "state": "NS",
        "intent_signals": [
            {
                "source": "Hiring",
                "description": "Posted 4 sales/BD roles in past 30 days.",
                "url": "https://coralreefrobotics.com/careers",
                "date": "2026-05-08",
                "snippet": "Account Executive, SDR (x2), Director of Commercial.",
            },
        ],
    },
    {
        "business": "Lyceum Press",
        "company_linkedin": "https://www.linkedin.com/company/lyceum-press",
        "company_website": "https://lyceumpress.com",
        "employee_count": 24,
        "industry": "Media",
        "sub_industry": "Independent Publishing",
        "country": "United Kingdom",
        "city": "Oxford",
        "state": "Oxfordshire",
        "intent_signals": [
            {
                "source": "News",
                "description": "Announced direct-to-reader platform; press cycle ongoing.",
                "url": "https://thebookseller.com/news/lyceum-direct-platform-2026",
                "date": "2026-05-19",
                "snippet": "Lyceum Press will bypass traditional retail.",
            },
        ],
    },
    {
        "business": "Heliocene Materials",
        "company_linkedin": "https://www.linkedin.com/company/heliocene",
        "company_website": "https://heliocene.com",
        "employee_count": 87,
        "industry": "Materials Science",
        "sub_industry": "Solar Coatings",
        "country": "Germany",
        "city": "Munich",
        "state": "Bavaria",
        "intent_signals": [
            {
                "source": "Product",
                "description": "Launched pilot program for utility-scale customers.",
                "url": "https://heliocene.com/pilot",
                "date": "2026-05-15",
                "snippet": "Inviting 10 utility partners to deploy HC-7.",
            },
        ],
    },
    {
        "business": "Tessera Health",
        "company_linkedin": "https://www.linkedin.com/company/tessera-health",
        "company_website": "https://tesserahealth.io",
        "employee_count": 61,
        "industry": "Healthcare",
        "sub_industry": "Care Coordination Software",
        "country": "United States",
        "city": "Austin",
        "state": "TX",
        "intent_signals": [
            {
                "source": "LinkedIn",
                "description": "VP Sales posted about replacing legacy CRM.",
                "url": "https://www.linkedin.com/posts/tessera-vp-sales-7170000000",
                "date": "2026-05-22",
                "snippet": "Our team is finally ripping out the old system.",
            },
        ],
    },
    {
        "business": "Verdant Climate",
        "company_linkedin": "https://www.linkedin.com/company/verdant-climate",
        "company_website": "https://verdantclimate.io",
        "employee_count": 104,
        "industry": "Climate Tech",
        "sub_industry": "Carbon Accounting",
        "country": "United States",
        "city": "San Francisco",
        "state": "CA",
        "intent_signals": [
            {
                "source": "News",
                "description": "Selected by Fortune 100 retailer to be their carbon reporting vendor.",
                "url": "https://techcrunch.com/2026/05/verdant-fortune-100-2026",
                "date": "2026-05-09",
                "snippet": "Verdant's CEO: 'We didn't know we could move this fast at enterprise scale until they let us.'",
            },
            {
                "source": "Hiring",
                "description": "Posted 6 enterprise sales roles in two weeks.",
                "url": "https://verdantclimate.io/careers",
                "date": "2026-05-14",
                "snippet": "VP Enterprise, Enterprise AE x4, Solutions Engineer.",
            },
        ],
    },
    {
        "business": "Pavo Robotics",
        "company_linkedin": "https://www.linkedin.com/company/pavo-robotics",
        "company_website": "https://pavorobotics.com",
        "employee_count": 73,
        "industry": "Hardware",
        "sub_industry": "Warehouse Automation",
        "country": "United States",
        "city": "Pittsburgh",
        "state": "PA",
        "intent_signals": [
            {
                "source": "Product",
                "description": "Announced Pavo-2, a new picking robot for mid-size 3PLs.",
                "url": "https://pavorobotics.com/pavo-2",
                "date": "2026-05-17",
                "snippet": "Pavo-2 cuts pick-rate latency by 40% over Pavo-1.",
            },
        ],
    },
    {
        "business": "Athanor Energy",
        "company_linkedin": "https://www.linkedin.com/company/athanor-energy",
        "company_website": "https://athanor.energy",
        "employee_count": 156,
        "industry": "Energy",
        "sub_industry": "Geothermal",
        "country": "Iceland",
        "city": "Reykjavik",
        "state": "Capital Region",
        "intent_signals": [
            {
                "source": "Crunchbase",
                "description": "Raised $48M Series C from Breakthrough Energy Ventures.",
                "url": "https://www.crunchbase.com/funding_round/athanor-series-c-2026",
                "date": "2026-04-30",
                "snippet": "Athanor plans to bring 200MW of geothermal online by 2028.",
            },
        ],
    },
    {
        "business": "Roost Childcare Network",
        "company_linkedin": "https://www.linkedin.com/company/roost-childcare",
        "company_website": "https://roostchildcare.com",
        "employee_count": 41,
        "industry": "Education",
        "sub_industry": "Childcare Operations Software",
        "country": "United States",
        "city": "Brooklyn",
        "state": "NY",
        "intent_signals": [
            {
                "source": "LinkedIn",
                "description": "Founder said in a podcast they're 'finally ready for a real CRM'.",
                "url": "https://www.linkedin.com/posts/roost-founder-podcast-7180100000",
                "date": "2026-05-20",
                "snippet": "We've outgrown spreadsheets. We need real revenue tooling now.",
            },
        ],
    },
    {
        "business": "Steppe Logistics",
        "company_linkedin": "https://www.linkedin.com/company/steppe-logistics",
        "company_website": "https://steppelogistics.com",
        "employee_count": 198,
        "industry": "Logistics",
        "sub_industry": "Cross-Border Freight",
        "country": "Kazakhstan",
        "city": "Almaty",
        "state": "Almaty Region",
        "intent_signals": [
            {
                "source": "News",
                "description": "Opened a new corridor between Almaty and Istanbul.",
                "url": "https://logistics-mena.com/steppe-corridor-2026",
                "date": "2026-05-06",
                "snippet": "Founder Aigerim Sultanova: 'Almaty-to-Istanbul opens corridors nobody else has built yet — we go where the freight wasn't moving.'",
            },
        ],
    },
]


# ---- Helpers ----


def _is_real_network_enabled() -> bool:
    """Are we configured to call a real LeadPoet endpoint?"""
    return bool(os.environ.get("LEADPOET_NETWORK_URL"))


def _find_leads_via_network(icp_description: str, max_results: int) -> list[dict]:
    """Placeholder for a real LeadPoet/Bittensor call.

    Replace this body with your endpoint logic. The return must still be a
    list of dicts matching `_REQUIRED_COMPANY_FIELDS`. Until then, we fall
    back to the mock corpus so callers see real-shaped data.
    """
    # Intentionally NOT raising — the mock is the safe default.
    return _MOCK_LEADS[:max_results]


def _score_company(company: dict[str, Any]) -> dict[str, int]:
    """Score 0..100 across three subscores. Pure function over the dict shape."""
    # ICP-fit: presence of all required fields → up to 40
    icp_fit = 0
    for f in _REQUIRED_COMPANY_FIELDS:
        if company.get(f):
            icp_fit += 4

    # Accuracy: presence of LinkedIn + website + at-least-one signal with URL
    accuracy = 0
    if company.get("company_linkedin", "").startswith("http"):
        accuracy += 10
    if company.get("company_website", "").startswith("http"):
        accuracy += 10
    signals = company.get("intent_signals") or []
    if any(s.get("url", "").startswith("http") for s in signals):
        accuracy += 10

    # Intent: number + recency of signals → up to 30
    intent = 0
    for s in signals:
        intent += 10
        if s.get("date", "").startswith(("2026-04", "2026-05")):
            intent += 2
    intent = min(intent, 30)

    total = min(100, icp_fit + accuracy + intent)
    return {
        "score": total,
        "icp_fit": icp_fit,
        "accuracy": accuracy,
        "intent": intent,
    }


# ---- Tools ----


@tool(["Poet"])
def find_leads_for_icp(icp_description: str, max_results: int = 5) -> list[dict]:
    """
    Surface companies whose profile matches the given Ideal Customer Profile.
    Returns up to `max_results` CompanyOutput-shaped dicts.
    ---
    icp_description: free-text description of the ideal customer (e.g.
        "Series B+ B2B SaaS companies in North America with 100-500 employees,
        currently hiring sales leaders")
    max_results: how many companies to return (default 5; cap at 10)
    """
    n = max(1, min(int(max_results), 10))
    if _is_real_network_enabled():
        return _find_leads_via_network(icp_description, n)
    return _MOCK_LEADS[:n]


@tool(["Poet"])
def enrich_company(company_name: str) -> dict:
    """
    Return the full CompanyOutput record for a company by name.
    ---
    company_name: the legal or common name of the company (e.g. "Northwind Analytics")
    """
    name_lower = company_name.lower().strip()
    for lead in _MOCK_LEADS:
        if lead["business"].lower() == name_lower:
            return dict(lead)
    # Not found — return an empty stub keyed to the required schema fields.
    # We deliberately use empty strings (not the string "unknown"), so the
    # values are FALSY: the Heart sees the fields are present but has nothing
    # to relay as fact, and `score_lead_quality`'s truthy checks naturally
    # short-circuit. The Poet's "never invent" axiom holds at the data layer.
    return {
        "business": company_name,
        "company_linkedin": "",
        "company_website": "",
        "employee_count": 0,
        "industry": "",
        "sub_industry": "",
        "country": "",
        "city": "",
        "state": "",
        "intent_signals": [],
        "_note": (
            f"{company_name!r} is not in the Poet's index — no evidence on "
            "record. Tell the user plainly; do not invent fields."
        ),
    }


@tool(["Poet"])
def verify_intent_signals(company_name: str) -> list[dict]:
    """
    Return the intent signals on record for a company. Empty list if none.
    ---
    company_name: the company to check (e.g. "Tessera Health")
    """
    name_lower = company_name.lower().strip()
    for lead in _MOCK_LEADS:
        if lead["business"].lower() == name_lower:
            return [dict(s) for s in lead.get("intent_signals", [])]
    return []


@tool(["Poet"])
def score_lead_quality(company: dict) -> dict:
    """
    Score a CompanyOutput dict 0..100 across three subscores:
      - `icp_fit` (out of 40) — does the company have a clean, complete profile?
      - `accuracy` (out of 30) — is the data verifiable (LinkedIn URL, website,
        signals carrying URLs)?
      - `intent` (out of 30) — recent and numerous intent signals on record.
    Returns `{score, icp_fit, accuracy, intent}`.
    ---
    company: a dict in the CompanyOutput shape (business, company_website, ...,
        intent_signals: [...])
    """
    if not isinstance(company, dict):
        return {"score": 0, "icp_fit": 0, "accuracy": 0, "intent": 0,
                "error": "expected a dict"}
    return _score_company(company)


@tool(["Poet"])
def summarize_top_intent(leads: list[dict]) -> str:
    """
    Read a list of CompanyOutput dicts and return a one-paragraph summary of
    which leads look most promising and why. Leads are ranked by how much
    intent they're carrying — number of signals, then recency — and the line
    for each lead uses the MOST RECENT signal on record (not just the first).
    ---
    leads: a list of CompanyOutput-shaped dicts
    """
    if not leads:
        return "No leads to summarize."

    def _signal_recency_key(sig: dict) -> str:
        # ISO-8601 dates sort lexicographically; bare strings fall to the end
        return sig.get("date", "0000-00-00")

    def _lead_priority_key(lead: dict) -> tuple:
        signals = lead.get("intent_signals") or []
        if not signals:
            return (0, "0000-00-00")
        latest = max(s.get("date", "") for s in signals)
        return (len(signals), latest)

    ranked = sorted(leads[:5], key=_lead_priority_key, reverse=True)

    parts: list[str] = []
    for lead in ranked:
        name = lead.get("business", "Unknown")
        city = lead.get("city", "?")
        country = lead.get("country", "?")
        sub_industry = lead.get("sub_industry") or lead.get("industry", "?")
        signals = lead.get("intent_signals") or []

        if not signals:
            parts.append(f"{name} ({sub_industry}, {city}/{country}): no intent on record.")
            continue

        latest = max(signals, key=_signal_recency_key)
        parts.append(
            f"{name} ({sub_industry}, {city}/{country}): "
            f"{latest.get('description', '')} "
            f"[{latest.get('source', '?')}, {latest.get('date', '?')}]."
        )

    return " ".join(parts)
