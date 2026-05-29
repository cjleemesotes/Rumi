"""Reference Dervishes — example implementations that USE the `rumi` library.

These Dervishes live alongside the library (not inside it) because they're
*consumers* of rumi, not part of its core. Copy them into your project as
templates, or import them directly when running from the repo root.

- `Poet` — proficient in sales intelligence, in a Rumi voice. Surfaces
  high-intent prospects, enriches companies, verifies intent signals.
  Shape-compatible with the LeadPoet (Bittensor subnet 71) CompanyOutput
  schema.
- `Sage` — proficient in the web. Visits pages, extracts text, clicks,
  types, searches. Wraps `botasaurus` (install separately:
  `pip install botasaurus`) for anti-bot-detection scraping.
"""
from .poet import Poet
from .sage import Sage

__all__ = ["Poet", "Sage"]
