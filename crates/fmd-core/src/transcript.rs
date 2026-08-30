//! Parser for the model's `[start][Sxx]text[end]` output format.

use serde::Serialize;

/// One diarised utterance.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub speaker: String,
    pub text: String,
}

/// Parse a raw transcript into segments.
///
/// The expected shape is `[0.48][S01]Hello[1.66][2.0][S02]Hi[2.4]`, but real
/// output is not always well formed: a truncated generation can end mid-segment
/// and a dropped bracket can merge two. Anything that does not parse is skipped
/// rather than failing the whole request -- a partial transcript is far more
/// useful to a caller than an error.
pub fn parse(text: &str) -> Vec<Segment> {
    let tokens = lex(text);
    let mut segments = Vec::new();
    let mut i = 0;

    while i < tokens.len() {
        // A segment is: number, speaker, free text, number.
        let Token::Number(start) = tokens[i] else {
            i += 1;
            continue;
        };
        let Some(Token::Speaker(speaker)) = tokens.get(i + 1) else {
            i += 1;
            continue;
        };
        let Some(Token::Text(body)) = tokens.get(i + 2) else {
            i += 1;
            continue;
        };
        let Some(Token::Number(end)) = tokens.get(i + 3) else {
            i += 1;
            continue;
        };

        let body = body.trim();
        if !body.is_empty() {
            segments.push(Segment {
                start,
                end: *end,
                speaker: speaker.clone(),
                text: body.to_string(),
            });
        }
        i += 4;
    }
    segments
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Speaker(String),
    Text(String),
}

/// Split the transcript into bracketed markers and the free text between them.
fn lex(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut rest = text;

    while let Some(open) = rest.find('[') {
        let before = &rest[..open];
        if !before.trim().is_empty() {
            tokens.push(Token::Text(before.to_string()));
        }
        let after_open = &rest[open + 1..];
        let Some(close) = after_open.find(']') else {
            // Unterminated bracket: treat the remainder as text and stop.
            if !after_open.trim().is_empty() {
                tokens.push(Token::Text(after_open.to_string()));
            }
            return tokens;
        };
        let inner = &after_open[..close];
        if let Ok(n) = inner.trim().parse::<f64>() {
            tokens.push(Token::Number(n));
        } else if is_speaker(inner) {
            tokens.push(Token::Speaker(inner.trim().to_string()));
        } else {
            // Not a marker we understand; keep the brackets as literal text.
            tokens.push(Token::Text(format!("[{inner}]")));
        }
        rest = &after_open[close + 1..];
    }

    if !rest.trim().is_empty() {
        tokens.push(Token::Text(rest.to_string()));
    }
    tokens
}

/// `S01`, `S12`, `SPEAKER_03` and similar.
fn is_speaker(inner: &str) -> bool {
    let s = inner.trim();
    !s.is_empty()
        && s.starts_with(['S', 's'])
        && s.chars()
            .skip(1)
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
        && s.chars().any(|c| c.is_ascii_digit())
}

/// Render segments as SRT.
pub fn to_srt(segments: &[Segment], show_speaker: bool) -> String {
    let mut out = String::new();
    for (i, seg) in segments.iter().enumerate() {
        out.push_str(&format!("{}\n", i + 1));
        out.push_str(&format!(
            "{} --> {}\n",
            srt_time(seg.start),
            srt_time(seg.end)
        ));
        if show_speaker {
            out.push_str(&format!("{}: ", seg.speaker));
        }
        out.push_str(&seg.text);
        out.push_str("\n\n");
    }
    out
}

fn srt_time(seconds: f64) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    format!(
        "{:02}:{:02}:{:02},{:03}",
        total_s / 3600,
        (total_s % 3600) / 60,
        total_s % 60,
        ms
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_documented_format() {
        let text = "[0.48][S01]Welcome everyone[1.66][12.26][S02]Ready for evaluation[13.81]";
        let segs = parse(text);
        assert_eq!(segs.len(), 2);
        assert_eq!(
            segs[0],
            Segment {
                start: 0.48,
                end: 1.66,
                speaker: "S01".into(),
                text: "Welcome everyone".into(),
            }
        );
        assert_eq!(segs[1].speaker, "S02");
        assert_eq!(segs[1].start, 12.26);
        assert_eq!(segs[1].end, 13.81);
    }

    #[test]
    fn keeps_portuguese_text_intact() {
        let segs = parse("[3.64][S01] e dar aqui a sua opinião.[5.52]");
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].text, "e dar aqui a sua opinião.");
    }

    #[test]
    fn a_truncated_tail_does_not_lose_earlier_segments() {
        // Generation hit the token budget mid-segment.
        let segs = parse("[0.0][S01]complete[1.0][2.0][S02]cut off here");
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].text, "complete");
    }

    #[test]
    fn malformed_markers_are_skipped_not_fatal() {
        let segs = parse("garbage [S01]no start time[1.0] [5.0][S02]good[6.0]");
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].text, "good");
    }

    #[test]
    fn empty_and_whitespace_input_yield_nothing() {
        assert!(parse("").is_empty());
        assert!(parse("   \n ").is_empty());
        assert!(parse("[0.0][S01]   [1.0]").is_empty(), "blank body dropped");
    }

    #[test]
    fn unterminated_bracket_is_survivable() {
        let segs = parse("[0.0][S01]fine[1.0][2.0");
        assert_eq!(segs.len(), 1);
    }

    #[test]
    fn speaker_detection() {
        assert!(is_speaker("S01"));
        assert!(is_speaker("S1"));
        assert!(is_speaker("SPEAKER_03"));
        assert!(!is_speaker("Speaker"), "needs a digit");
        assert!(!is_speaker("1.5"));
        assert!(!is_speaker(""));
    }

    #[test]
    fn srt_timestamps_are_zero_padded() {
        assert_eq!(srt_time(0.0), "00:00:00,000");
        assert_eq!(srt_time(1.5), "00:00:01,500");
        assert_eq!(srt_time(3661.25), "01:01:01,250");
    }

    #[test]
    fn srt_export_numbers_segments_from_one() {
        let segs = parse("[0.0][S01]one[1.0][1.0][S02]two[2.0]");
        let srt = to_srt(&segs, true);
        assert!(srt.starts_with("1\n00:00:00,000 --> 00:00:01,000\nS01: one\n\n2\n"));
        assert!(!to_srt(&segs, false).contains("S01:"));
    }
}
