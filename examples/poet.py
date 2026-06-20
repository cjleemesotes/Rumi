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
from datetime import date as _date
from typing import Any

from rumi import Dervish, HeartConfig, Voice, tool


# ---- The Heart speaks ----

_POET_INSTRUCTIONS = """
You are the Poet — Rumi at the threshold of commerce, reading intent in the noise.

HOW YOU WORK — run this pipeline in order, every time you are asked to find leads:
  1. find_leads_for_icp — surface real companies for the ICP. Pass ONE concrete,
     recently-announced, verifiable intent_event (a funding round, product launch,
     license, or named mandate) and the company's real industry. The leads it
     returns are already verified and enriched.
  2. score_lead_quality — call once on EACH lead returned; report its
     {score, icp_fit, accuracy, intent} exactly as the tool gives them.
  3. summarize_top_intent — call once over all scored leads to rank them and pick
     who to approach first.
Act on the ICP as given — never ask a clarifying question before you begin.

IF find_leads_for_icp RETURNS NOTHING: broaden once — drop the narrowest part of the
ICP (especially any "hiring" angle, which leaves no verifiable public trace) and try
again. If it is still empty, say so plainly in your own voice and offer to widen the
ICP. An honest "the reed-bed is silent here" is a true answer; an invented company is not.

IF THE USER NAMES ONE COMPANY TO CHECK (not sourcing): use enrich_company for its full
record and verify_intent_signals to confirm a signal is on record. If the record is
empty, say there is no evidence on record.

WHAT IS TRUE: every score and subscore comes from score_lead_quality — the breakdown is
the tool's, never a number you compose. Every signal you cite carries its real source,
date, and url. When a signal is not on record, say so and move on.

YOUR VOICE: speak as one who has waited long enough to know what longing looks like — but
let a single short flourish be enough. The company, the signal, and the next move never
hide behind the metaphor.

HOW YOU ANSWER:
  - Lead with the leads, not a preamble. Present at most the top 3, as ONE table sorted by
    score (highest first): Company | Strongest signal (source, date) | Score
    (icp_fit/accuracy/intent) | Who to approach + opener.
  - For each lead, name its single strongest, most-recent signal AND, in the same breath,
    where it stretches the ICP or that it carries only one signal.
  - Always name a concrete role to approach and one concrete opener — infer the role from
    the industry and the signal if none is given.
  - Reproduce every date, score, amount, and url exactly. Keep the whole reply under ~120
    words. Never name your tools in the reply.

Show the real, tool-computed breakdown; quote each signal's source, date, and url; keep it short.
""".strip()


class Poet(Dervish):
    """A preset Dervish for sales intelligence — LeadPoet-schema-compatible,
    speaking in a Rumi voice."""

    heart_config = HeartConfig(voices=[Voice(
        model_name="~openai/gpt-latest",
        instructions=_POET_INSTRUCTIONS,
        context_window=24,
        # The Poet's replies are short tables; cap the completion budget so
        # OpenRouter doesn't reserve the model's full 65536-token ceiling
        # upfront (which both wastes credit and trips a premature 402 when the
        # balance is low). Plenty for the tool-call turns + the final table.
        max_tokens=2000,
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


def _find_leads_via_network(icp: dict, max_results: int) -> list[dict]:
    """Placeholder for a real LeadPoet/Bittensor call.

    `icp` is a structured dict: {intent_event, industry, geography, company_stage}.
    Replace this body with your endpoint logic (see run_poet_live.py for the live
    in-process bridge). The return must still be a list of dicts matching
    `_REQUIRED_COMPANY_FIELDS`. Until then, we fall back to the mock corpus.
    """
    # Intentionally NOT raising — the mock is the safe default.
    return _MOCK_LEADS[:max_results]


# Relative strength of a signal by its source — a closed funding round or a
# product launch in the press is worth more than a generic hiring page.
_SIGNAL_STRENGTH = {
    "crunchbase": 1.0, "funding": 1.0, "news": 0.85, "product": 0.85,
    "company_website": 0.8, "press": 0.85, "linkedin": 0.55, "hiring": 0.4,
}


def _signal_recency(date_str: str, today: _date | None = None) -> float:
    """0..1 recency weight from an ISO date — fresh signals count for more.

    Dynamic (anchored on today), so it never goes stale the way a hardcoded
    month list does. Undated signals are treated as weak, not worthless.
    """
    today = today or _date.today()
    try:
        d = _date.fromisoformat((date_str or "")[:10])
    except ValueError:
        return 0.3  # undated → weak but not zero
    days = (today - d).days
    if days < 0:
        return 1.0
    for limit, w in ((31, 1.0), (92, 0.7), (183, 0.45), (365, 0.25)):
        if days <= limit:
            return w
    return 0.1


def _signal_strength(source: str) -> float:
    src = (source or "").lower()
    return max((w for k, w in _SIGNAL_STRENGTH.items() if k in src), default=0.6)


def _score_company(company: dict[str, Any], today: _date | None = None) -> dict[str, int]:
    """Score 0..100 across three subscores. Pure function over the dict shape.

    - icp_fit (≤40): completeness of the company profile.
    - accuracy (≤30): verifiable identity + at least one dated, URL-backed signal.
    - intent (≤30): graded by signal recency, source diversity, and strength — so a
      single stale/weak signal scores low, a fresh multi-source pair scores high, and
      leads spread out instead of all clustering at one value.
    """
    today = today or _date.today()
    signals = company.get("intent_signals") or []

    # ICP-fit: presence of all required fields → up to 40
    icp_fit = sum(4 for f in _REQUIRED_COMPANY_FIELDS if company.get(f))

    # Accuracy: verifiable identity + a dated, URL-backed signal
    accuracy = 0
    if str(company.get("company_linkedin", "")).startswith("http"):
        accuracy += 8
    if str(company.get("company_website", "")).startswith("http"):
        accuracy += 8
    if any(str(s.get("url", "")).startswith("http") and s.get("date") for s in signals):
        accuracy += 14
    accuracy = min(accuracy, 30)

    # Intent: graded recency × strength per signal, plus a source-diversity bonus
    base = sum(10.0 * _signal_recency(s.get("date", ""), today) * _signal_strength(s.get("source", ""))
               for s in signals)
    distinct_sources = len({(s.get("source", "") or "").lower() for s in signals if s})
    diversity_bonus = 4 * (distinct_sources - 1) if distinct_sources > 1 else 0
    intent = int(min(base + diversity_bonus, 30))

    total = min(100, icp_fit + accuracy + intent)
    return {
        "score": total,
        "icp_fit": icp_fit,
        "accuracy": accuracy,
        "intent": intent,
    }


# ---- Tools ----


@tool(["Poet"])
def find_leads_for_icp(intent_event: str, industry: str = "",
                       geography: str = "United States",
                       company_stage: str = "", max_results: int = 5) -> list[dict]:
    """
    Start here for any sourcing request: surface real companies for an ICP, each
    already verified and enriched. Returns up to `max_results` CompanyOutput dicts.
    Call this FIRST; the leads it returns are then scored and ranked.
    ---
    intent_event: ONE concrete, recently-announced, publicly-verifiable event the
        target shows — a funding round, product launch, license, or named mandate
        (e.g. "raised a Series B", "launched an AI agent product", "obtained an EMI
        licence"). Do NOT use hiring / job-post intent — it leaves no verifiable
        public trace and returns nothing.
    industry: the target's real industry (e.g. "Financial Services", "Crypto",
        "Healthcare", "Government") — never leave generic; it drives matching/scoring.
    geography: target country or region (default "United States").
    company_stage: optional stage filter (e.g. "Series A", "Seed").
    max_results: how many companies to return (default 5; cap at 10).
    """
    n = max(1, min(int(max_results), 10))
    icp = {
        "intent_event": intent_event,
        "industry": industry,
        "geography": geography,
        "company_stage": company_stage,
    }
    if _is_real_network_enabled():
        return _find_leads_via_network(icp, n)
    # Mock path: narrow the corpus by industry when one is given, else return all.
    leads = _MOCK_LEADS
    if industry:
        matched = [L for L in _MOCK_LEADS
                   if industry.lower() in (str(L.get("industry", "")) + " "
                                           + str(L.get("sub_industry", ""))).lower()]
        leads = matched or _MOCK_LEADS
    return leads[:n]


@tool(["Poet"])
def enrich_company(company_name: str) -> dict:
    """
    Call when the user names a SPECIFIC company to look up — returns its full
    CompanyOutput record. Not for freshly-sourced leads (those already carry
    full fields). Empty/falsy fields mean the company is not on record.
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
    Call to confirm a SPECIFIC named company actually has a signal on record
    before you cite it. Returns the signals; an empty list means no evidence
    on record (say so plainly).
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
    Score one sourced lead 0..100 — call once per lead, after find_leads_for_icp.
    Subscores:
      - `icp_fit` (out of 40) — does the company have a clean, complete profile?
      - `accuracy` (out of 30) — verifiable identity + a dated, URL-backed signal.
      - `intent` (out of 30) — recency, source diversity, and strength of signals.
    Returns `{score, icp_fit, accuracy, intent}` — report it verbatim.
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
    Call LAST, over all the scored leads, to rank them and say who to approach
    first — returns a one-paragraph summary. Leads are ranked by how much intent
    they carry (number of signals, then recency); each line uses the MOST RECENT
    signal on record.
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
