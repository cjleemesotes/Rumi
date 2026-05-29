//! Verbatim error strings for the whirl-pipeline, ported from
//! `src/rumi/agent/pipeline.py`.
//!
//! These MUST match the Python source byte-for-byte — they are the
//! differential-oracle equality bar (the Python suite asserts on substrings
//! like `match="whirled past the floor"`, and a careless reword would silently
//! change a user-facing message). Watch the em-dash (U+2014, `—`).
//!
//! `MAX_TOOL_ITERATIONS` is read from the live Python module at runtime so the
//! constant stays single-source-of-truth in Python; the cap message therefore
//! interpolates the resolved value, exactly like the Python f-string.

pub const HEART_CONFIG_MISSING: &str =
    "heart_config missing — should have been caught at __init__";

/// The `RuntimeError` raised by `run_voice_step` when an Agent calls tools
/// past the cap. Kept byte-exact (em-dash U+2014; default max 10).
pub fn max_tool_iterations_msg(max: u64) -> String {
    format!("The Agent kept calling tools past {max} turns — it whirled past the floor. Tighten the prompt, or raise MAX_TOOL_ITERATIONS if this is expected.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heart_config_string_matches_python() {
        assert_eq!(
            HEART_CONFIG_MISSING,
            "heart_config missing — should have been caught at __init__"
        );
    }

    #[test]
    fn max_iterations_string_matches_python() {
        // The exact concatenation of the three-line Python f-string with the
        // default MAX_TOOL_ITERATIONS = 10.
        assert_eq!(
            max_tool_iterations_msg(10),
            "The Agent kept calling tools past 10 turns — it whirled past the floor. Tighten the prompt, or raise MAX_TOOL_ITERATIONS if this is expected."
        );
    }
}
