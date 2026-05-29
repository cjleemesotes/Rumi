//! The whirl-pipeline — ported from `src/rumi/agent/pipeline.py`.
//!
//! Rust owns the orchestration control flow: input append, the per-voice tool
//! loop, the `>= MAX_TOOL_ITERATIONS` cap, the atomic commit + reverse-order
//! rollback of a tool cycle, cooperative cancel, and message building
//! (truncation + leading-orphan skip). It calls BACK into Python for the
//! genuine "crossings" on objects the agent already owns — the heart, the
//! tools, the ideas, the lifecycle hooks, the tablet — so behavior is
//! byte-for-byte identical to the Python pipeline it replaces.
//!
//! GIL discipline (scoping doc §3): the GIL is held through orchestration —
//! NO `allow_threads` in the hot loop. CPython auto-drops the GIL at the
//! socket syscall inside `heart.call`, so a separate thread's cancel
//! (`threading.Event.set`) is still observed at the next loop-top check. This
//! is what makes the rumi-edge daemon's `tasks/cancel` cooperative.
//!
//! Single-owner by design, like the rest of the crate (not `Sync`): scale by
//! process, not by sharing one agent across threads.

use pyo3::exceptions::{PyException, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::errors;

/// Run one full whirl cycle and return the final IIdea.
///
/// Mirrors the whirl pipeline `(agent, message, *, response_model=None,
/// cancel_check=None)`. `response_model` applies only to the LAST voice step;
/// `cancel_check`, if given, is polled at each tool-loop boundary and raises
/// the Python `WhirlCancelled` when truthy.
#[pyfunction]
#[pyo3(signature = (agent, message, response_model=None, cancel_check=None))]
pub fn run_whirl_pipeline<'py>(
    py: Python<'py>,
    agent: &Bound<'py, PyAny>,
    message: &Bound<'py, PyAny>,
    response_model: Option<&Bound<'py, PyAny>>,
    cancel_check: Option<&Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    // Live Python modules + classes, imported once and reused (cheap — hits
    // sys.modules). Calling back into these keeps the verbatim error strings,
    // the pydantic Idea construction, and the hook logging in Python.
    let pipeline_mod = py.import("rumi.agent.pipeline")?;
    let types_mod = py.import("rumi.ideas.types")?;
    let registry_mod = py.import("rumi.tools.registry")?;
    let tool_request_cls = types_mod.getattr("ToolRequestIdea")?;
    let tool_result_cls = types_mod.getattr("ToolResultIdea")?;
    let text_idea_cls = types_mod.getattr("TextIdea")?;
    let safe_hook = pipeline_mod.getattr("_safe_hook")?;
    // MAX_TOOL_ITERATIONS stays authoritative in Python (user-tunable); read it.
    let max_iter: u64 = pipeline_mod.getattr("MAX_TOOL_ITERATIONS")?.extract()?;

    // --- normalize incoming + persist (delegated: rich verbatim messages +
    // pydantic TextIdea construction stay in Python) ---
    let incoming = pipeline_mod
        .getattr("_normalize_incoming")?
        .call1((message,))?;
    for t in incoming.try_iter()? {
        let t = t?;
        agent.call_method1("_append_idea", (&t,))?;
    }
    // _safe_hook(agent, "_on_whirl_start_impl", incoming)
    safe_hook.call1((agent, "_on_whirl_start_impl", &incoming))?;

    // cfg = type(agent).heart_config  (None should be impossible post-__init__)
    let cfg = agent.get_type().getattr("heart_config")?;
    if cfg.is_none() {
        return Err(PyRuntimeError::new_err(errors::HEART_CONFIG_MISSING));
    }
    let voices_attr = cfg.getattr("voices")?;
    let voices: Vec<Bound<'py, PyAny>> = voices_attr.try_iter()?.collect::<PyResult<Vec<_>>>()?;
    let last_idx = voices.len().saturating_sub(1);

    let mut final_idea: Option<Bound<'py, PyAny>> = None;
    for (idx, voice) in voices.iter().enumerate() {
        let step_rm = if idx == last_idx { response_model } else { None };
        final_idea = Some(run_voice_step(
            py,
            agent,
            voice,
            step_rm,
            cancel_check,
            &pipeline_mod,
            &safe_hook,
            &registry_mod,
            &tool_request_cls,
            &tool_result_cls,
            max_iter,
        )?);
    }

    let final_idea = match final_idea {
        Some(fi) => fi,
        None => {
            // Defensive: HeartConfig.voices has min_length=1, so the loop above
            // always ran. Keep the contract safe (mirrors the Python fallback).
            let kw = PyDict::new(py);
            kw.set_item("content", "")?;
            let fi = text_idea_cls.call((), Some(&kw))?;
            agent.call_method1("_append_idea", (&fi,))?;
            fi
        }
    };

    // _safe_hook(agent, "_on_whirl_ended_impl", final_idea)
    safe_hook.call1((agent, "_on_whirl_ended_impl", &final_idea))?;
    Ok(final_idea.unbind())
}

/// Execute one voice step including its tool loop; return the final idea.
///
/// `cancel_check` is polled at the TOP of each iteration — before the LLM call,
/// never between a ToolRequestIdea and its results — so a tool cycle is never
/// interrupted halfway (no orphaned ToolRequestIdea on disk).
#[allow(clippy::too_many_arguments)]
fn run_voice_step<'py>(
    py: Python<'py>,
    agent: &Bound<'py, PyAny>,
    voice: &Bound<'py, PyAny>,
    response_model: Option<&Bound<'py, PyAny>>,
    cancel_check: Option<&Bound<'py, PyAny>>,
    pipeline_mod: &Bound<'py, PyAny>,
    safe_hook: &Bound<'py, PyAny>,
    registry_mod: &Bound<'py, PyAny>,
    tool_request_cls: &Bound<'py, PyAny>,
    tool_result_cls: &Bound<'py, PyAny>,
    max_iter: u64,
) -> PyResult<Bound<'py, PyAny>> {
    let heart = agent.getattr("heart")?;
    let tools = agent.getattr("tools")?;
    let execute_tool_loop = registry_mod.getattr("execute_tool_loop")?;
    let mut iterations: u64 = 0;

    loop {
        check_cancel(cancel_check, pipeline_mod)?;

        let messages = build_messages(py, agent, voice, tool_result_cls)?;
        // tools_schemas = agent.tools.schemas() or None  (empty list → None)
        let raw_schemas = tools.call_method0("schemas")?;
        let tools_schemas: Option<Bound<'py, PyAny>> = if raw_schemas.is_truthy()? {
            Some(raw_schemas)
        } else {
            None
        };

        // payload_in dict for the start hook
        let payload_in = PyDict::new(py);
        payload_in.set_item("model", voice.getattr("model_name")?)?;
        payload_in.set_item("messages", &messages)?;
        set_opt(&payload_in, "tools", tools_schemas.as_ref(), py)?;
        match response_model {
            // response_model.__name__ if response_model else None
            Some(rm) => payload_in.set_item("response_model", rm.getattr("__name__")?)?,
            None => payload_in.set_item("response_model", py.None())?,
        }
        // _safe_hook(agent, "_on_heart_call_start_impl", payload_in)
        safe_hook.call1((agent, "_on_heart_call_start_impl", &payload_in))?;

        // heart.call(voice, messages, tools=..., response_model=...)
        let call_kwargs = PyDict::new(py);
        set_opt(&call_kwargs, "tools", tools_schemas.as_ref(), py)?;
        set_opt(&call_kwargs, "response_model", response_model, py)?;
        let response = heart.call_method("call", (voice, &messages), Some(&call_kwargs))?;

        // _safe_hook(agent, "_on_heart_call_ended_impl", payload_in, response)
        // FLAT positionals — `*args` must receive (payload_in, response) as TWO
        // args so the hook's (payload_in, payload_out) arity is satisfied.
        safe_hook.call1((agent, "_on_heart_call_ended_impl", &payload_in, &response))?;

        // idea = heart.parse_response(response, response_model=response_model)
        let parse_kwargs = PyDict::new(py);
        set_opt(&parse_kwargs, "response_model", response_model, py)?;
        let idea = heart.call_method("parse_response", (&response,), Some(&parse_kwargs))?;
        agent.call_method1("_append_idea", (&idea,))?;

        if idea.is_instance(tool_request_cls)? {
            let results = execute_tool_loop.call1((&idea, &tools))?;

            // Atomicity: the request and ALL its results persist as a unit. If
            // a result append fails mid-batch, unwind the already-committed
            // results in reverse, then the request, then re-raise the ORIGINAL
            // error — so the next whirl() never builds a trailing assistant
            // tool_calls with no matching tool responses (OpenAI 400).
            let mut committed: Vec<Bound<'py, PyAny>> = Vec::new();
            if let Err(orig) = commit_results(agent, &results, &mut committed) {
                for r in committed.iter().rev() {
                    rollback_one(py, agent, r)?;
                }
                rollback_one(py, agent, &idea)?;
                set_current_idea(py, agent)?;
                return Err(orig);
            }

            iterations += 1;
            // `>=` so the cap fires AT the limit, not one past it.
            if iterations >= max_iter {
                return Err(PyRuntimeError::new_err(errors::max_tool_iterations_msg(
                    max_iter,
                )));
            }
            continue;
        }

        return Ok(idea);
    }
}

/// Append each tool result, tracking committed ones for rollback. Propagates
/// the first append failure (its original PyErr, unchanged).
fn commit_results<'py>(
    agent: &Bound<'py, PyAny>,
    results: &Bound<'py, PyAny>,
    committed: &mut Vec<Bound<'py, PyAny>>,
) -> PyResult<()> {
    for r in results.try_iter()? {
        let r = r?;
        agent.call_method1("_append_idea", (&r,))?;
        committed.push(r);
    }
    Ok(())
}

/// Best-effort unwind of one persisted idea (mirrors the Python rollback's
/// nested try/except): `_tablet.delete` swallows ALL errors; `_ideas.remove`
/// swallows ONLY `ValueError` (any other error propagates, exactly as Python's
/// bare `except ValueError` would let it escape the handler).
fn rollback_one<'py>(
    py: Python<'py>,
    agent: &Bound<'py, PyAny>,
    idea: &Bound<'py, PyAny>,
) -> PyResult<()> {
    if let Ok(id) = idea.getattr("id").and_then(|v| v.extract::<String>()) {
        if let Ok(tablet) = agent.getattr("_tablet") {
            let _ = tablet.call_method1("delete", (format!("history/{id}"),));
        }
    }
    let ideas = agent.getattr("_ideas")?;
    match ideas.call_method1("remove", (idea,)) {
        Ok(_) => Ok(()),
        Err(e) if e.is_instance_of::<PyValueError>(py) => Ok(()),
        Err(e) => Err(e),
    }
}

/// `agent._current_idea = agent._ideas[-1] if agent._ideas else None`
fn set_current_idea<'py>(py: Python<'py>, agent: &Bound<'py, PyAny>) -> PyResult<()> {
    let ideas = agent.getattr("_ideas")?;
    let len = ideas.len()?;
    if len == 0 {
        agent.setattr("_current_idea", py.None())?;
    } else {
        let last = ideas.get_item(len - 1)?;
        agent.setattr("_current_idea", last)?;
    }
    Ok(())
}

/// Build the OpenAI messages list from the agent's idea history, truncated to
/// `context_window` and snapped past leading orphan ToolResultIdeas so the
/// slice always begins on a self-contained message.
fn build_messages<'py>(
    py: Python<'py>,
    agent: &Bound<'py, PyAny>,
    voice: &Bound<'py, PyAny>,
    tool_result_cls: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let msgs = PyList::empty(py);

    let instructions = voice.getattr("instructions")?;
    if instructions.is_truthy()? {
        let sys = PyDict::new(py);
        sys.set_item("role", "system")?;
        sys.set_item("content", &instructions)?;
        msgs.append(&sys)?;
    }

    // history = list(agent._ideas)  (a copy — never mutate the agent's list)
    let history: Vec<Bound<'py, PyAny>> = agent
        .getattr("_ideas")?
        .try_iter()?
        .collect::<PyResult<Vec<_>>>()?;
    let len = history.len();
    // context_window: int, ge=0 (0 = unlimited). `if N and N > 0 and len > N`.
    let n: i64 = voice.getattr("context_window")?.extract()?;
    let mut start: usize = 0;
    if n > 0 && len > n as usize {
        start = len - n as usize;
        while start < len && history[start].is_instance(tool_result_cls)? {
            start += 1;
        }
    }

    for t in &history[start..] {
        match t.call_method0("to_openai_message") {
            Ok(m) => {
                msgs.append(&m)?;
            }
            Err(e) => {
                // Best-effort: skip malformed ideas. Match Python's
                // `except Exception` — let BaseException (KeyboardInterrupt,
                // SystemExit) propagate rather than swallowing it.
                if e.is_instance_of::<PyException>(py) {
                    continue;
                }
                return Err(e);
            }
        }
    }
    Ok(msgs)
}

/// Raise the Python `WhirlCancelled` if `cancel_check` is present and truthy.
/// Imports the *live* class so `except WhirlCancelled` (test + edge worker)
/// catches it — must NOT be a PyO3-native exception type.
fn check_cancel<'py>(
    cancel_check: Option<&Bound<'py, PyAny>>,
    pipeline_mod: &Bound<'py, PyAny>,
) -> PyResult<()> {
    if let Some(cb) = cancel_check {
        if cb.call0()?.is_truthy()? {
            let cls = pipeline_mod.getattr("WhirlCancelled")?;
            return Err(PyErr::from_value(cls.call0()?));
        }
    }
    Ok(())
}

/// Set `dict[key] = value` where a missing value becomes Python `None`
/// (rumi always passes `tools=`/`response_model=` explicitly, possibly None).
fn set_opt<'py>(
    dict: &Bound<'py, PyDict>,
    key: &str,
    value: Option<&Bound<'py, PyAny>>,
    py: Python<'py>,
) -> PyResult<()> {
    match value {
        Some(v) => dict.set_item(key, v),
        None => dict.set_item(key, py.None()),
    }
}
