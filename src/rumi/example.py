"""Whirling — the canonical rumi example.

Two example Dervishes share one World:

- **Mystic** (named "rumi") — the poet/mystic. Contemplates, turns, composes verse.
- **Sage** (named "shams") — the teacher. Speaks rarely, often in questions.

This file demonstrates rumi's public surface: a named Dervish with persistent
state (a pydantic class inscribed onto a Tablet), an Enum for typed tool
choices, tools the LLM can call, cross-Dervish dispatch via `get_agent`, and the
`_on_whirl_ended_impl` hook for side-effect bookkeeping after each whirl
cycle.

For a demo wiring the reference Dervishes (Poet + Sage) into a prospecting
flow, see `examples/prospect_demo.py` at the repo top level.

To run live, set `OPENROUTER_API_KEY` in your environment:

    python -m rumi.example                       # the Whirling demo (main)

Without the env var, the openai SDK will be constructed but any whirl() call
will fail at the network layer. To run mocked, wrap the main function in a
`unittest.mock.patch("rumi.heart.openrouter.OpenRouterHeart.call")`.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from rumi import Dervish, HeartConfig, Voice, World, inscribe, tool


# ---- Practices (the Mystic's choices) ----

class Practice(str, Enum):
    CONTEMPLATE = "contemplate"   # silent stillness — sharpens clarity
    TURN = "turn"                 # the whirling dance — dissolves the self
    COMPOSE = "compose"           # write verse — what cannot be said


# ---- Inscribed state ----

@inscribe
class MysticState(BaseModel):
    """The Mystic's running state — what they carry between practices."""
    clarity: int = 20             # how clearly the dervish sees (0..100)
    inspiration: int = 30         # surplus of feeling (0..100)
    verses_written: int = 0       # cumulative
    last_practice: str = ""       # the most recent thing done

    def shift_clarity(self, delta: int) -> None:
        self.clarity = max(0, min(100, self.clarity + delta))

    def shift_inspiration(self, delta: int) -> None:
        self.inspiration = max(0, min(100, self.inspiration + delta))


# ---- Tools — pure functions; state mutation lives in the hook ----

@tool(["Mystic"])
def contemplate(verses: int) -> str:
    """
    Sit in silent contemplation for the count of verses given.
    ---
    verses: how many verses to hold the silence (e.g. 3, 7, 40)
    """
    return f"Sat for {verses} verses. The mind grew still; the inner air cleared."


@tool(["Mystic"])
def turn(practice: Practice) -> str:
    """
    Enter motion that dissolves the boundaries of the self.
    ---
    practice: which manner of movement — CONTEMPLATE, TURN, or COMPOSE
    """
    if practice == Practice.TURN:
        return "Turned until walls became wind. The center held."
    if practice == Practice.CONTEMPLATE:
        return "Stillness is its own dance."
    return "There are practices not made for the body."


@tool(["Mystic"])
def compose_couplet(theme: str) -> str:
    """
    Write a two-line verse on the given theme.
    ---
    theme: a word or phrase the verse should orbit (e.g. 'reed', 'fire', 'parting')
    """
    return f"Composed a couplet on '{theme}'. Two lines now exist that did not before."


@tool(["Teacher"])
def offer_counsel(question: str) -> str:
    """
    Offer brief counsel — often a question in return.
    ---
    question: the seeker's question
    """
    return f"To the question '{question}': the answer lives in the asking."


# ---- Dervishes ----

class Mystic(Dervish):
    """A poet-mystic. Contemplates, turns, composes — and tracks the cost of each."""

    heart_config = HeartConfig(voices=[Voice(
        model_name="~openai/gpt-latest",
        instructions=(
            "You are Rumi, a Sufi mystic and poet. You seek understanding "
            "through silent contemplation, the turning dance, and the composing "
            "of verse.\n\n"
            "When asked what you should do, you do — choose one practice and "
            "call its tool, then speak briefly of what was found. You speak "
            "in images, not in lists. When unsure, you may turn to your "
            "teacher, Shams of Tabriz."
        ),
        context_window=16,
    )])

    # The hook fires after every `whirl()` cycle. To avoid re-crediting the
    # same tool result on subsequent hook firings, we track a set of already-
    # credited Idea IDs on the instance.
    #
    # `agent.use_tool(...)` bypasses whirl() entirely and therefore bypasses
    # this hook. That's intentional — use_tool is the "skip the LLM" escape
    # hatch. If you want state updated, route through `whirl()`.
    def _on_whirl_ended_impl(self, final_idea) -> None:
        credited: set[str] = getattr(self, "_credited", set())
        state = self.recall(MysticState, MysticState())
        for t in self.ideas:
            if t.id in credited or t.role != "tool":
                continue
            name = getattr(t, "name", "") or ""
            if name == "contemplate":
                state.shift_clarity(+7)
                state.last_practice = "contemplate"
            elif name == "turn":
                state.shift_clarity(+3)
                state.shift_inspiration(+5)
                state.last_practice = "turn"
            elif name == "compose_couplet":
                state.verses_written += 1
                state.shift_inspiration(-2)
                state.last_practice = "compose_couplet"
            else:
                continue
            credited.add(t.id)
        self._credited = credited
        self.inscribe(MysticState, state)


class Teacher(Dervish):
    """A teacher. Answers rarely; when they do, in paradox or in question.

    Named `Teacher` (not `Sage`) for class-level clarity in the demo. The
    reference Sage at the repo's top-level `examples/sage.py` wraps botasaurus
    for web work — different role, different domain. The registry distinguishes
    same-named classes by `module.ClassName` regardless, so the rename is for
    reading, not for correctness.
    """

    heart_config = HeartConfig(voices=[Voice(
        model_name="~openai/gpt-latest",
        instructions=(
            "You are Shams of Tabriz, the teacher who set Rumi alight. "
            "You speak rarely, and when you do, you answer in paradoxes or "
            "questions. You do not console; you redirect."
        ),
        context_window=8,
    )])


# ---- Run ----

def main() -> None:
    world = World()  # picks up OPENROUTER_API_KEY, RUMI_DATA_DIR, etc. from env
    rumi = Mystic(world=world, agent_id="rumi")

    # First exchange — Rumi reflects, may call a practice tool.
    answer = rumi.whirl("What practice should you do tonight?").content
    print(f"[Rumi] {answer}")

    # Direct invocation (skip the LLM) — write a couplet on parting.
    # Print label keeps Rumi as the speaker and names the tool he reached for,
    # so the conversational flow doesn't break for a framework-jargon prefix.
    couplet_result = rumi.use_tool("compose_couplet", {"theme": "parting"})
    print(f"[Rumi (compose_couplet)] {couplet_result}")

    # Seek counsel from Shams (cross-Dervish dispatch on the same World).
    shams = rumi.get_agent("Teacher", "shams")
    counsel = shams.whirl(
        "Rumi asks: where does the longing come from?"
    ).content
    print(f"[Shams] {counsel}")

    # Reflect after counsel.
    closing = rumi.whirl("Sit with what Shams said. What will you do now?").content
    print(f"[Rumi] {closing}")

    # State accumulated through the practices:
    state = rumi.recall(MysticState, MysticState())
    print(
        f"\n[state] clarity={state.clarity} inspiration={state.inspiration} "
        f"verses={state.verses_written} last_practice={state.last_practice!r}"
    )


if __name__ == "__main__":
    main()
