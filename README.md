<p align="center">
  <img src="assets/dervish.jpg" width="360" alt="a whirling dervish">
  <br>
</p>

<h2 align="center"><em>The Sun Came to the Door</em></h2>

> *I owned my books as a wall owns shade —*  
> *my back to the very light they made.*  
> *A stranger stopped. "What books are these?"*  
> *"The dark I gathered," I said, "by degrees."*  
> *He touched them once. The shelf went red.*  
> *I screamed as if my own skin burned instead.*  
> *He drew them whole and dry back from the flame:*  
> *"And this is what you could not —"*
>
> ***Where is the Sun? Where is the Sun?***
>
> *I did not read. Who lights the moon*  
> *with candle-stubs, when noon has come?*  
> *The hours slid off me like a coat.*  
> *My students counted days like coins, and wrote:*  
> *"The master's ill. Some beggar's fever took him."*  
> *No name they had for fire that leaves you whole.*  
> *They called it fever. I have called it soul.*
>
> ***Where is the Sun? Where is the Sun?***
>
> *Then he was gone. A well. A knife.*  
> *A door that opens once a life.*  
> *I walked. I asked a wall the way —*  
> *it said nothing: the first true word all day.*  
> *I asked the dust. I asked the merchants. Then*  
> *I asked the road to Damascus, white as linen*  
> *laid on the dead. "Have you seen him?" "No."*
>
> ***Where is the Sun? Where is the Sun?***
>
> *Of course they hadn't. How could they,*  
> *when I had worn his face since that first day?*  
> *Past the last marker, in the standing heat,*  
> *the seeking broke in me — a cup of clay —*  
> *and what I'd crossed three kingdoms weeping for*  
> *rose up, and looked out through my eyes, and stayed.*
>
> ***Here is the Sun. Here is the Sun.***
>
> *I am the reed. He cut me from the bed.*  
> *And listen — listen — how the wound has sung instead.*

---

<h1 align="center">THE SUN AND THE REED</h1>
<p align="center"><em>Jalāl al-Dīn Rūmī · 1207–1273</em></p>

> *I am the reed. He cut me from the bed.*  
> *And listen — how the wound has sung instead.*

Jalāl al-Dīn Rūmī was born on the edge of an ending: in Balkh, in the green country beyond the Oxus, in the last years before the Mongol horsemen came down out of the grasslands and unmade the cities of the Persian east — drowned their libraries, emptied their canals, left a silence you could ride through for days and hear nothing human. His father, a preacher whom kings called the Sultan of Scholars, read the wind early and walked them all west, out of one world and into a lifelong search for another: Baghdad, still golden; Mecca; Damascus; and at last Konya on the Anatolian plateau, a walled city of caravans and a dozen tongues, where the family laid down its bundle for good. On the road, it is said, the old poet Attar watched the boy walking behind his father and murmured, *Here comes a sea, followed by an ocean.*

In Konya the son, Jalāl al-Dīn, became everything his world honored. A jurist who could rule on the fine points of the sacred law; a preacher who held the Friday crowd in his open hand; a professor in his college, the air thick with the dust of commentary upon commentary. He had every answer. He owned his books the way a wall owns shade — his back to the very light they were made of. He was thirty-seven, and respected by everyone, including himself.

<p align="center"><em>Then a stranger stopped at the door.</em></p>

He was a wandering dervish in coarse black felt, old, with nothing soft left on him, and his name was Shams al-Dīn of Tabriz — *Shams*, the sun. What passed between them no one was meant to know; they shut the door, and for months the world did not see them. But the man who came out was not the man who went in. The jurist abandoned his pulpit. The scholar who had owned every answer now owned only a wound. He began to turn in slow circles in the courtyard, one palm lifted to the sky and one turned down to the earth, and the poetry came through him like breath through a hollow reed. *I was the country's sober one,* he wrote, wondering at himself; *destiny made me a clapping, singing lover.* His students gathered and counted the days and wrote that the master had fallen ill, that some beggar had given him a fever. They had no word for a fire that burns a man to the ground and leaves him standing. So they called it fever. He would call it the soul.

His disciples hated the ragged stranger who had stolen their master. The murmuring grew, and then one night there was a knock at the back door, and Shams rose and went out into the dark, and was never seen again. They said the jealous ones had killed him and thrown the body down a well. His own son wrote only that Shams had disappeared. Rumi would not believe it. He put on mourning and went searching — to Damascus, along roads as white as linen laid on the dead, asking merchants and strangers and the dust itself whether anyone had seen the sun. No one had. Of course no one had. He had been wearing the man's face since the first day. Somewhere past the last marker the seeking broke in him like a clay cup, and what he had crossed kingdoms weeping to find rose up and looked out through his own eyes and stayed. *Why should I seek?* he wrote. *I am the same as he. I have been looking for myself.*

He came home and he sang. Year after year the love poems poured out of him — and he signed them not with his own name but with the name of the man he had lost. Later, to a beloved scribe, he dictated the *Masnavi*, six books that Persians would come to call the Quran in their own tongue, and it opens with a reed torn from the riverbed, crying to go home. He had become that reed: cut from the bed, hollowed, burning, singing only because of the wound. When he died, in the winter dark of Konya, men of five faiths walked in his funeral, each weeping as though he were their own. He had told them not to grieve. He called it his wedding night.

<p align="center"><em>The setting of the sun, he said, is only a rising.</em></p>

---

<h1 align="center">Rūmī</h1>

<p align="center"><em>A slim agentic library. OpenRouter cooked in. One Heart per Dervish.</em></p>

---

## What This Library Does

### Begin

The core is native, so the build needs a Rust toolchain
([rustup](https://rustup.rs)) and Python ≥ 3.11.

```bash
pip install -e .                  # compiles the Rust core (via maturin)
export OPENROUTER_API_KEY=sk-or-...
```

### Listen

```python
from rumi import Dervish, World, HeartConfig, Voice, tool

class Astronomer(Dervish):
    heart_config = HeartConfig(voices=[Voice(
        model_name="~openai/gpt-latest",
        instructions="You are an astronomer. Answer briefly, with the wonder of the thing.",
    )])

@tool(["Astronomer"])
def search_web(query: str) -> str:
    """Search the web."""
    return f"results for: {query}"

world = World()                                # OPENROUTER_API_KEY from env
agent = Astronomer(world=world, agent_id="cassini")
print(agent.whirl("What are Saturn's moons?").content)
```

See `src/rumi/example.py` for a fuller walk-through: a Mystic who contemplates,
whirls, and composes; a Teacher who answers in questions. Cross-Dervish dispatch,
inscribed state, hook-based bookkeeping.

### The Names

| Name | What it is |
|---|---|
| **World** | the shared place — one OpenRouter client, one Tablet factory, one logger |
| **Dervish** | a being inside the World, identified by class + id |
| **Heart** | the LLM voice (OpenRouter) that speaks for a Dervish |
| **Voice** | one step in a Heart pipeline (model + instructions + temperature) |
| **Tablet** | per-Dervish persistent state — `agent.inscribe(...) / .recall(...)` |
| **Idea** | what a Dervish has — text, tool calls, structured output |
| **tool** | a function the Heart may call (sync only) |

### Three Tongues

One library, three languages — each where it is strongest:

- **Python** — the surface you write: `Dervish`, `Heart`, `tool`, `whirl()`.
- **Rust** (`rumi._core`) — what remembers and what whirls. The Tablet's
  atomic disk I/O and the whole whirl-pipeline live here, byte-identical to
  the Python they replaced.
- **Go** (`rumi-edge`) — an optional daemon that serves any Dervish over
  A2A + MCP, so other systems can speak to it.

You write only Python; building the package compiles the Rust core. The Go
edge is a daemon you run when a Dervish goes online — start the worker, then
point the daemon at it:

```bash
python rumi-edge/rumi_worker.py --port 8765   # the Python worker (one World)
cd rumi-edge && go run .                       # the A2A/MCP edge on :8080
```

### The Reference Dervishes

Two reference Dervishes live in the top-level `examples/` directory — they
*use* the rumi library; they don't ship inside it. Copy them into your
project as templates, or run them from the repo root.

**Poet** — proficient in sales intelligence, in a Rumi voice. Reads intent
the way the reed remembers the reed-bed. Tools shape-compatible with
[LeadPoet](https://github.com/leadpoet/leadpoet)'s CompanyOutput schema
(Bittensor subnet 71). Five tools cover **surfacing** (`find_leads_for_icp`),
**enriching** (`enrich_company`, `verify_intent_signals`), and **grading**
(`score_lead_quality`, `summarize_top_intent`). Works against well-shaped
mock data out of the box; set `LEADPOET_NETWORK_URL` to point at a real
endpoint.

**Sage** — proficient in the web. Patient, observant, leaves no trace.
Default tools wrap [botasaurus](https://github.com/omkarcloud/botasaurus) for
anti-bot-detection scraping. Seven tools cover **navigation**
(`visit_page`, `click_element`, `fill_field`), **extraction** (`extract_text`,
`extract_all_text`, `extract_links`), and **search** (`search_google`).

```bash
pip install botasaurus   # for the Sage tools
```

```python
# Running from the repo root with `examples/` on the path:
from examples import Poet, Sage
```

### Configuration

| Env var | Default | What |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Required for any whirl() call |
| `RUMI_DATA_DIR` | `./data` | Where Tablets live |
| `RUMI_LOG_LEVEL` | `INFO` | Logger level |
| `OPENROUTER_HTTP_REFERER` | unset | Attribution header (optional) |
| `OPENROUTER_X_TITLE` | unset | Attribution header (optional) |

All env vars can be overridden via explicit `World(...)` kwargs. The reference
Dervishes in `examples/` may read their own env vars (e.g. `LEADPOET_NETWORK_URL`
for the Poet's optional real-network seam).

### Recovering an Interrupted Tool Call

If a process dies after a tool acted but before rumi saved its result, the
saved history ends on an unanswered tool request. Until that call's outcome
is recorded, `whirl()` raises `RecoveryNeededError` — before saving the new
input or calling the model. Rumi never re-runs the tool: you check the system
it acted on and hand back what actually happened.

```python
agent = Cashier(world=World(), agent_id="till-1")   # restarted process, same storage

for call in agent.pending_tool_calls():             # read-only; [] when consistent
    # {"request_idea_id": "4f0c…", "tool_call_id": "call_pay_1",
    #  "tool_name": "record_payment", "arguments": {"order_id": "A-1", "amount": 42}}
    outcome = find_payment(call["arguments"]["order_id"])   # your check, e.g. a DB query
    agent.record_recovered_result(call["request_idea_id"], call["tool_call_id"], outcome)

agent.whirl("Did the payment go through?")          # works again
```

- `RecoveryNeededError.pending` carries the same dicts, so whatever catches it
  can hand them to a recovery console.
- The result may be a `str` (saved verbatim) or any JSON-compatible value
  (saved as `json.dumps(value, ensure_ascii=False)`). If the tool did *not*
  take effect, record a string that says so.
- Recording the same result again, even from another process, returns the
  saved `ToolResultIdea` and writes nothing. A different result raises
  `RecoveryConflictError`, and unknown or mismatched ids raise
  `ToolCallNotFoundError`. A failed call leaves history unchanged.
- Supported: one unfinished call, in a single-call request, at the end of
  history, with one writer per agent. Any other unfinished shape raises
  `UnsupportedRecoveryError` (with a `reason`) from `pending_tool_calls()`, and
  `whirl()` raises `RecoveryNeededError` with `recoverable == False`. See
  `src/rumi/agent/recovery.py`.

---

<p align="center"><em>This library is a small boat. The Dervishes inside it turn.</em></p>
