//! Prompt construction: chat template rendering and audio-span expansion.
//!
//! The audio span is not a flat run of placeholders. Every
//! `time_marker_every_seconds` the processor injects the elapsed second count as
//! literal digit tokens, which is how the model learns to emit calibrated
//! timestamps. Reproducing that interleaving exactly matters: get the spacing
//! wrong and the transcript's timestamps drift.

use tokenizers::Tokenizer;

use crate::config::ProcessorConfig;
use crate::error::{Error, Result};

/// The default transcribe-and-diarize instruction shipped with the model.
pub const DEFAULT_PROMPT: &str = concat!(
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号",
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，",
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
);

const AUDIO_PAD: &str = "<|audio_pad|>";
const AUDIO_START: &str = "<|audio_start|>";
const AUDIO_END: &str = "<|audio_end|>";
const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";
const SYSTEM_MESSAGE: &str = "You are a helpful assistant.";

/// Builds token sequences for the model.
pub struct PromptBuilder {
    tokenizer: Tokenizer,
    audio_token_id: u32,
    digit_ids: [u32; 10],
    cfg: ProcessorConfig,
}

impl PromptBuilder {
    pub fn new(tokenizer: Tokenizer, cfg: ProcessorConfig) -> Result<Self> {
        let audio_token_id = tokenizer
            .token_to_id(AUDIO_PAD)
            .ok_or_else(|| Error::Tokenizer(format!("tokenizer has no {AUDIO_PAD} token")))?;

        // The time markers are written digit by digit, so each digit must be a
        // single token for the span arithmetic to line up.
        let mut digit_ids = [0u32; 10];
        for (d, slot) in digit_ids.iter_mut().enumerate() {
            let text = d.to_string();
            let ids = tokenizer
                .encode(text.as_str(), false)
                .map_err(Error::tokenizer)?;
            let ids = ids.get_ids();
            if ids.len() != 1 {
                return Err(Error::Tokenizer(format!(
                    "digit {d:?} is not a single token: {ids:?}"
                )));
            }
            *slot = ids[0];
        }

        Ok(Self {
            tokenizer,
            audio_token_id,
            digit_ids,
            cfg,
        })
    }

    pub fn audio_token_id(&self) -> u32 {
        self.audio_token_id
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(Error::tokenizer)?
            .get_ids()
            .to_vec())
    }

    /// The token ids that fill the audio span, placeholders and time markers.
    ///
    /// Mirrors the reference `_audio_span_ids`, including its quirk that the
    /// marker for second `s` lands at token index `(s / every) * tokens_per_marker`
    /// rather than at the proportional position.
    pub fn audio_span_ids(&self, audio_tokens: usize) -> Vec<u32> {
        let every = self.cfg.time_marker_every_seconds;
        if !self.cfg.enable_time_marker || audio_tokens == 0 || every == 0 {
            return vec![self.audio_token_id; audio_tokens];
        }
        let tokens_per_marker = (self.cfg.audio_tokens_per_second * every as f64) as usize;
        if tokens_per_marker == 0 {
            return vec![self.audio_token_id; audio_tokens];
        }

        let duration = audio_tokens as f64 / self.cfg.audio_tokens_per_second;
        let mut out = Vec::with_capacity(audio_tokens + audio_tokens / 16);
        let mut consumed = 0usize;

        let mut sec = every;
        while sec <= duration as usize {
            let pos = (sec / every) * tokens_per_marker;
            if pos > consumed {
                out.extend(std::iter::repeat_n(self.audio_token_id, pos - consumed));
                consumed = pos;
            }
            for ch in sec.to_string().chars() {
                let d = ch.to_digit(10).expect("decimal digits only") as usize;
                out.push(self.digit_ids[d]);
            }
            sec += every;
        }

        if audio_tokens > consumed {
            out.extend(std::iter::repeat_n(
                self.audio_token_id,
                audio_tokens - consumed,
            ));
        }
        out
    }

    /// Render the chat template around an audio placeholder.
    ///
    /// Returns the text before and after the placeholder, so each half can be
    /// tokenised independently -- exactly as the reference processor does, which
    /// keeps the token boundaries identical.
    fn render(&self, instruction: &str) -> (String, String) {
        let before =
            format!("{IM_START}system\n{SYSTEM_MESSAGE}{IM_END}\n{IM_START}user\n{AUDIO_START}");
        let after = format!("{AUDIO_END}\n{instruction}{IM_END}\n{IM_START}assistant\n");
        (before, after)
    }

    /// Build the full prompt token sequence for one audio input.
    pub fn build(&self, audio_tokens: usize, instruction: Option<&str>) -> Result<Vec<u32>> {
        let instruction = instruction
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(DEFAULT_PROMPT);
        let (before, after) = self.render(instruction);

        let mut ids = self.encode(&before)?;
        ids.extend(self.audio_span_ids(audio_tokens));
        ids.extend(self.encode(&after)?);
        Ok(ids)
    }

    /// Decode generated ids back to text, dropping special tokens.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer.decode(ids, true).map_err(Error::tokenizer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A builder with synthetic ids, to test the span arithmetic without a
    /// tokenizer file. Token ids here are arbitrary but distinct.
    fn span_only(cfg: ProcessorConfig) -> PromptBuilder {
        PromptBuilder {
            // Never used by `audio_span_ids`.
            tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            audio_token_id: 999,
            digit_ids: [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            cfg,
        }
    }

    #[test]
    fn markers_appear_every_five_seconds() {
        let b = span_only(ProcessorConfig::default());
        // 12.5 tokens/s, markers every 5 s -> one marker per 62 audio tokens.
        let ids = b.audio_span_ids(375); // exactly 30 s
        let audio = ids.iter().filter(|&&i| i == 999).count();
        assert_eq!(audio, 375, "every audio token must survive");

        // Markers for 5, 10, 15, 20, 25, 30 seconds.
        let markers: Vec<u32> = ids.iter().copied().filter(|&i| i != 999).collect();
        assert_eq!(
            markers,
            vec![105, 101, 100, 101, 105, 102, 100, 102, 105, 103, 100]
        );
    }

    #[test]
    fn first_marker_sits_after_the_first_62_tokens() {
        let b = span_only(ProcessorConfig::default());
        let ids = b.audio_span_ids(375);
        assert!(ids[..62].iter().all(|&i| i == 999));
        assert_eq!(ids[62], 105, "the '5' of second 5");
    }

    #[test]
    fn short_audio_gets_no_markers() {
        let b = span_only(ProcessorConfig::default());
        // 4 s of audio never reaches the 5 s mark.
        let ids = b.audio_span_ids(50);
        assert_eq!(ids, vec![999; 50]);
    }

    #[test]
    fn disabling_markers_yields_a_flat_span() {
        let cfg = ProcessorConfig {
            enable_time_marker: false,
            ..ProcessorConfig::default()
        };
        assert_eq!(span_only(cfg).audio_span_ids(400), vec![999; 400]);
    }

    #[test]
    fn empty_audio_yields_an_empty_span() {
        assert!(span_only(ProcessorConfig::default())
            .audio_span_ids(0)
            .is_empty());
    }

    /// Element-for-element parity with the reference processor.
    ///
    /// Regenerate with `python scripts/dump_reference_prompt.py testdata/`.
    /// Ignored by default: it needs both the fixture and the cached tokenizer.
    #[test]
    #[ignore = "requires fixtures from scripts/dump_reference_prompt.py"]
    fn matches_the_reference_processor() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let fixture = std::fs::read_to_string(root.join("testdata/prompt_reference.json"))
            .expect("run scripts/dump_reference_prompt.py first");
        let doc: serde_json::Value = serde_json::from_str(&fixture).unwrap();

        let files = crate::hub::ModelSource::default().resolve().unwrap();
        let tokenizer = Tokenizer::from_file(&files.tokenizer).unwrap();
        let builder = PromptBuilder::new(tokenizer, ProcessorConfig::default()).unwrap();

        assert_eq!(
            builder.audio_token_id() as u64,
            doc["audio_token_id"].as_u64().unwrap(),
            "audio placeholder id disagrees"
        );

        let cases = doc["cases"].as_array().unwrap();
        assert!(!cases.is_empty());
        for case in cases {
            let audio_tokens = case["audio_tokens"].as_u64().unwrap() as usize;
            let instruction = case["instruction"].as_str().unwrap();
            let expected: Vec<u32> = case["input_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();

            let got = builder.build(audio_tokens, Some(instruction)).unwrap();
            assert_eq!(
                got.len(),
                expected.len(),
                "{} / {audio_tokens} audio tokens: length {} vs {}",
                case["name"],
                got.len(),
                expected.len()
            );
            if got != expected {
                let at = got
                    .iter()
                    .zip(expected.iter())
                    .position(|(a, b)| a != b)
                    .unwrap();
                panic!(
                    "{} / {audio_tokens} audio tokens: first difference at index {at}: \
                     got {} want {}",
                    case["name"], got[at], expected[at]
                );
            }
        }
        eprintln!("{} prompt cases matched exactly", cases.len());
    }

    #[test]
    fn ten_minutes_keeps_every_audio_token() {
        let b = span_only(ProcessorConfig::default());
        let tokens = 7500; // 600 s at 12.5 tokens/s
        let ids = b.audio_span_ids(tokens);
        assert_eq!(ids.iter().filter(|&&i| i == 999).count(), tokens);
        // 120 markers, each 1-3 digits, so the span grows but stays close.
        assert!(ids.len() > tokens && ids.len() < tokens + 400);
    }
}
