//! Filename codec — a byte-exact port of rumi's Python `_safe_filename` /
//! `_decode_safe_filename` (src/rumi/tablet/filesystem.py).
//!
//! The contract is byte-identity with the Python encoder so a tablet written
//! by one language is read by the other. Quirks are reproduced faithfully
//! (e.g. the decoder reads exactly two hex digits and uses a strict
//! `i + 2 < len` bound), NOT "fixed" — fixing them would break compatibility.

/// Encode a key into a filesystem-safe filename stem.
///
/// Mirrors Python:
///   - empty key            -> "__root__"
///   - '_'                  -> "%5f"   (so "__" stays reserved for '/')
///   - alphanumeric / - / . -> verbatim
///   - '/'                  -> "__"    (segment separator)
///   - anything else        -> "%{codepoint:02x}" (lowercase hex, min 2 digits)
///
/// `char::is_alphanumeric` matches Python's Unicode-aware `str.isalnum()` for
/// ASCII and common letters/digits (so e.g. "café" passes through unescaped,
/// exactly as Python does).
pub fn safe_filename(key: &str) -> String {
    if key.is_empty() {
        return "__root__".to_string();
    }
    let mut out = String::new();
    for c in key.chars() {
        if c == '_' {
            out.push_str("%5f");
        } else if c.is_alphanumeric() || c == '-' || c == '.' {
            out.push(c);
        } else if c == '/' {
            out.push_str("__");
        } else {
            out.push_str(&format!("%{:02x}", c as u32));
        }
    }
    out
}

/// Inverse of `safe_filename`. Char-indexed to mirror Python's str indexing.
pub fn decode_safe_filename(name: &str) -> String {
    if name == "__root__" {
        return String::new();
    }
    let chars: Vec<char> = name.chars().collect();
    let n = chars.len();
    let mut out = String::new();
    let mut i = 0usize;
    while i < n {
        let c = chars[i];
        if c == '_' && i + 1 < n && chars[i + 1] == '_' {
            out.push('/');
            i += 2;
        } else if c == '%' && i + 2 < n {
            // Python reads name[i+1:i+3] — exactly two chars — as hex.
            let hex: String = chars[i + 1..i + 3].iter().collect();
            match u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                Some(ch) => {
                    out.push(ch);
                    i += 3;
                }
                None => {
                    out.push(c);
                    i += 1;
                }
            }
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}
