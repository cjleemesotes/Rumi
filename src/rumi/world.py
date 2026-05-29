"""World — the shared place all Dervishes live in.

A World holds the things every Dervish needs: one OpenRouter client, one logger,
one Tablet factory. You'll usually want one World per process; pass it into
every Dervish.

Zero-config: `World()` reads `OPENROUTER_API_KEY`, `RUMI_DATA_DIR`,
`RUMI_LOG_LEVEL`, `OPENROUTER_HTTP_REFERER`, `OPENROUTER_X_TITLE` from the
environment. Explicit kwargs override anything found there.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

import openai

from rumi.log import agent_logger
from rumi.tablet.backend import ITablet
from rumi.tablet.filesystem import FilesystemTablet

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class World:
    """The container an Dervish breathes in.

    One openai client (pointed at OpenRouter), one Tablet factory (per-Dervish
    persistence), one logger. Pass the same World to every Dervish that should
    share resources.
    """

    def __init__(
        self,
        *,
        openrouter_api_key: str | None = None,
        data_dir: str | Path | None = None,
        log_level: str | None = None,
        tablet_factory: Callable[[str, str], ITablet] | None = None,
        http_referer: str | None = None,
        x_title: str | None = None,
        base_url: str | None = None,
    ) -> None:
        # Resolve from env with kwargs overriding
        self._api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        self._data_dir = Path(data_dir or os.environ.get("RUMI_DATA_DIR") or "./data")
        self._log_level = (log_level or os.environ.get("RUMI_LOG_LEVEL") or "INFO").upper()
        self._http_referer = http_referer or os.environ.get("OPENROUTER_HTTP_REFERER")
        self._x_title = x_title or os.environ.get("OPENROUTER_X_TITLE")
        self._base_url = base_url or os.environ.get("OPENROUTER_BASE_URL") or OPENROUTER_BASE_URL

        # Custom tablet factory or default to filesystem
        self._tablet_factory = tablet_factory or self._default_tablet_factory

        # Lazy openai client; created on first access
        self._openrouter_client: openai.OpenAI | None = None

        # Configure logging level once
        logging.getLogger("rumi").setLevel(self._log_level)

    # --- Properties ---

    @property
    def api_key(self) -> str | None:
        return self._api_key

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def log_level(self) -> str:
        return self._log_level

    @property
    def http_referer(self) -> str | None:
        return self._http_referer

    @property
    def x_title(self) -> str | None:
        return self._x_title

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def openrouter_client(self) -> openai.OpenAI:
        """The shared openai client pointed at OpenRouter. Built once, on first access."""
        if self._openrouter_client is None:
            default_headers: dict[str, str] = {}
            if self._http_referer:
                default_headers["HTTP-Referer"] = self._http_referer
            if self._x_title:
                # OpenRouter accepts both X-Title and X-OpenRouter-Title; use the documented one.
                default_headers["X-Title"] = self._x_title

            kwargs: dict = {
                "base_url": self._base_url,
                "api_key": self._api_key or "missing",
            }
            if default_headers:
                kwargs["default_headers"] = default_headers
            self._openrouter_client = openai.OpenAI(**kwargs)
        return self._openrouter_client

    @property
    def log(self) -> logging.Logger:
        return logging.getLogger("rumi")

    def tablet_for(self, agent_type: str, agent_id: str) -> ITablet:
        """A Tablet scoped to one (agent_type, agent_id) — its place to inscribe."""
        return self._tablet_factory(agent_type, agent_id)

    def logger_for(self, agent_type: str, agent_id: str) -> logging.Logger:
        """A logger named `rumi.{agent_type}.{agent_id}`."""
        return agent_logger(agent_type, agent_id, level=self._log_level)

    # --- Defaults ---

    def _default_tablet_factory(self, agent_type: str, agent_id: str) -> ITablet:
        return FilesystemTablet(self._data_dir, agent_type, agent_id)
