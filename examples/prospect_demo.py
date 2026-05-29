"""prospect_demo — the reference Poet + Sage flow.

Two reference Dervishes work together on a small prospecting task. The Poet
enriches and scores a company from its LeadPoet-shape index; the Sage scouts
the company's website to confirm what the page actually says.

Live run requires `OPENROUTER_API_KEY` in your environment, and the Sage tools
require `botasaurus` (`pip install botasaurus`). Mocked tests skip both.

Run from the repo root:

    python -c "from examples.prospect_demo import prospect_demo; print(prospect_demo())"
"""
from __future__ import annotations

from rumi import World

from .poet import Poet
from .sage import Sage  # noqa: F401  — imported for registry side effects


def prospect_demo(target_company: str = "Northwind Analytics",
                  target_url: str = "https://northwindanalytics.com") -> dict:
    """Poet enriches; Sage scouts the web. Returns a small dict with both halves.

    On a live system the Poet hits the LLM (via OpenRouter) and the Sage hits
    the web (via botasaurus). In tests this is fully mocked.
    """
    world = World()  # OPENROUTER_API_KEY + RUMI_DATA_DIR from env

    # The Poet reads the intent — enriches the company from its index and
    # scores it across the three subscores.
    poet = Poet(world=world, agent_id="poet-1")
    company = poet.use_tool("enrich_company", {"company_name": target_company})
    score = poet.use_tool("score_lead_quality", {"company": company})

    # The Sage travels — visits the company's website to see what the page
    # actually says, as a guest, leaving no trace.
    sage = poet.get_agent("Sage", "scout-1")
    observation = sage.use_tool("visit_page", {"url": target_url})

    return {
        "poet_company": company,
        "poet_score": score,
        "sage_observation": observation,
    }


if __name__ == "__main__":
    print(prospect_demo())
