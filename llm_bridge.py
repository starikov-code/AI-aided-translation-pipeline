#!/usr/bin/env python3
"""
llm_bridge.py — Bridge between sdlxliff_tool TSV files and an LLM API.

Reads the TSV produced by `sdlxliff_tool extract`, translates empty
`translated_masked` cells via an LLM (in batches), validates placeholder
integrity per segment, and writes the TSV back — ready for
`sdlxliff_tool check` and `inject`.

Supported backends:
  --backend openai      any OpenAI-compatible endpoint (OpenAI, vLLM,
                        LM Studio, Ollama's OpenAI compat, OpenRouter...)
  --backend anthropic   Claude API

Usage examples:
  # Groq — only set the key, everything else is preset
$env:GROQ_API_KEY = "gsk_..."
python llm_bridge.py test_segments.tsv --provider groq

# Gemini
$env:GEMINI_API_KEY = "AQ..."
python llm_bridge.py test_segments.tsv --provider gemini

# OpenRouter free tier
$env:OPENROUTER_API_KEY = "sk-or-..."
python llm_bridge.py test_segments.tsv --provider openrouter

# Override the preset's model when needed
python llm_bridge.py test_segments.tsv --provider groq --model llama-3.1-8b-instant

# Local, no key ever
python llm_bridge.py test_segments.tsv --provider ollama

POWERSHELL:
# Groq
$env:OPENAI_API_KEY = "gsk_..."
python llm_bridge.py test_segments.tsv --backend openai `
  --base-url https://api.groq.com/openai/v1 `
  --model llama-3.3-70b-versatile

# Google Gemini (if accessible in your country)
$env:OPENAI_API_KEY = "AQ..."
python llm_bridge.py test_segments.tsv --backend openai `
  --base-url https://generativelanguage.googleapis.com/v1beta/openai/ `
  --model gemini-3.6-flash

# Local Ollama — no key at all
python llm_bridge.py test_segments.tsv --backend openai `
  --base-url http://localhost:11434/v1 `
  --model qwen2.5:7b-instruct

API keys are read from environment variables:
  OPENAI_API_KEY   /   ANTHROPIC_API_KEY
OPENAI_API_KEY: sk-proj-
The script is resumable: rows that already have translated_masked filled
are skipped, so you can stop and re-run at any time.
"""

import argparse
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import urllib.request
import urllib.error
import json

PLACEHOLDER_RE = re.compile(
    r"\[\[(\d+):(BPT|EPT|PH|IT|X|G|/G)(?::([^\]]*))?\]\]")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback: parse .env manually if python-dotenv isn't installed
    from pathlib import Path as _Path
    _env = _Path(__file__).with_name(".env")
    if _env.exists():
        for _line in _env.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

# ============================================================================
# Provider presets (OpenAI-compatible endpoints)
# ============================================================================

PRESETS = {
    "groq": {
    "base_url": "https://api.groq.com/openai/v1",
    "model": "openai/gpt-oss-120b",
    "env_var": "GROQ_API_KEY",
    "needs_key": True,
    },

    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-3.6-flash",
        "env_var": "GEMINI_API_KEY",
        "needs_key": True,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "env_var": "OPENROUTER_API_KEY",
        "needs_key": True,
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "env_var": "DEEPSEEK_API_KEY",
        "needs_key": True,
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:7b-instruct",
        "env_var": None,                 # no key needed
        "needs_key": False,
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "model": "local-model",          # LM Studio ignores/accepts any name
        "env_var": None,
        "needs_key": False,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "env_var": "OPENAI_API_KEY",
        "needs_key": True,
    },
}


log = logging.getLogger("llm_bridge")

# ============================================================================
# Data model
# ============================================================================

@dataclass
class Row:
    seg_id: str
    source: str
    target: str
    translated: str

    @property
    def needs_translation(self) -> bool:
        return bool(self.source.strip()) and not self.translated.strip()


def read_tsv(path: str) -> List[Row]:
    import csv
    rows: List[Row] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(Row(
                seg_id=r["seg_id"],
                source=r.get("source_masked", ""),
                target=r.get("target_masked", ""),
                translated=r.get("translated_masked", ""),
            ))
    return rows


def write_tsv(path: str, rows: List[Row]) -> None:
    import csv
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["seg_id", "source_masked", "target_masked",
                    "translated_masked"])
        for r in rows:
            w.writerow([r.seg_id, r.source, r.target, r.translated])


# ============================================================================
# Placeholder validation
# ============================================================================

def placeholder_signature(text: str) -> Optional[frozenset]:
    """Multiset of placeholder strings, or None if malformed ones found."""
    if re.search(r"\[\[(?!\d+:)", text):
        return None  # stray/malformed bracket
    return frozenset(PLACEHOLDER_RE.findall(text))


def validate_translation(source: str, translated: str) -> List[str]:
    problems: List[str] = []
    src_sig = placeholder_signature(source)
    if src_sig is None:
        return ["source itself has malformed placeholders"]
    if translated is None:
        return ["empty translation"]
    tgt_sig = placeholder_signature(translated)
    if tgt_sig is None:
        problems.append("malformed placeholder(s) in translation "
                        "(broken [[ ]] brackets)")
        return problems
    if src_sig != tgt_sig:
        missing = set(src_sig) - set(tgt_sig)
        extra = set(tgt_sig) - set(src_sig)
        if missing:
            problems.append(f"missing placeholders: {sorted(missing)}")
        if extra:
            problems.append(f"unknown placeholders: {sorted(extra)}")
    # G nesting balance — by order, ignoring close numbers
    stack: list = []
    for m in PLACEHOLDER_RE.finditer(translated):
        kind = m.group(2)
        if kind == "G":
            stack.append(m.group(1))
        elif kind == "/G":
            if not stack:
                problems.append("stray /G in translation")
            else:
                stack.pop()
    if stack:
        problems.append(f"unclosed G tags: {stack}")
    return problems



# ============================================================================
# LLM backends (stdlib only — no SDK dependencies)
# ============================================================================

def _http_json(url: str, headers: dict, payload: dict, timeout: int = 180) -> dict:
    headers = {"Content-Type": "application/json",
               "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                             "llm_bridge/1.0",
               **headers}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    last_err: Optional[Exception] = None
    for attempt in range(4):  # retry with backoff on 429/5xx
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:500]
            if e.code == 429 and "quota" in body.lower():
                raise RuntimeError(f"Quota exhausted: {body}") from e
            if e.code in (429, 500, 502, 503, 529) and attempt < 3:
                wait = 2 ** (attempt + 1)
                log.warning("HTTP %s, retrying in %ss: %s", e.code, wait, body)
                time.sleep(wait)
                last_err = e
                continue
            raise RuntimeError(f"HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            last_err = e
            if attempt < 3:
                time.sleep(2 ** (attempt + 1))
                continue
            raise
    raise RuntimeError(f"Request failed: {last_err}")


SYSTEM_PROMPT = """You are a professional translator. Translate the user's \
text from {src_lang} to {tgt_lang}.

Rules:
1. The text contains special placeholders enclosed in double square brackets, \
such as [[1:BPT]], [[2:EPT]], [[3:PH:tag]], [[4:X:x-bd]], and paired group \
placeholders [[5:G]] ... [[5:/G]]. You MUST copy every placeholder into the \
translation exactly as-is: same numbering, same type, and same extra part \
after the second colon (if present). Never translate, rename, renumber, \
merge, split, drop, or invent placeholders.
   - Opening group placeholders look like [[n:G]] and their matching closing \
placeholders look like [[n:/G]]. Always reproduce both members of each pair.
2. Placeholders represent inline formatting tags. Their POSITION in the \
sentence may change to match natural {tgt_lang} word order, but paired \
[[n:BPT]] ... [[n:EPT]] and [[n:G]] ... [[n:/G]] must both appear, in the \
correct order (open before close), and must not overlap incorrectly. Text \
placed between [[n:G]] and its [[n:/G]] is the text carrying that \
formatting — put inside it only the words that are formatted in the source.
3. Translate only the human-readable text between placeholders. Never \
output a placeholder adjacent to another placeholder if there was text \
between them in the source, and never insert text directly inside a \
placeholder's brackets.
4. Output ONLY the translation — no explanations, no quotes around it, \
no original text."""


def build_batch_prompt(pairs: List[tuple], src_lang: str, tgt_lang: str) -> str:
    """pairs: list of (index, source_text). Returns one user message."""
    lines = [f"Translate the following {len(pairs)} numbered segments "
             f"from {src_lang} to {tgt_lang}.",
             ""]
    for i, text in pairs:
        lines.append(f"### {i}")
        lines.append(text)
        lines.append("")
    lines.append("Reply with each segment in the same format:")
    lines.append("### <index>")
    lines.append("<translation>")
    lines.append("")
    lines.append("Do not skip any index. Do not add commentary.")
    return "\n".join(lines)


def parse_batch_response(response: str, expected_ids: List[int]) -> dict:
    """Parse '### N\n<translation>' blocks. Returns {id: translation}."""
    out = {}
    # Split on ### N headers
    blocks = re.split(r"^###\s*(\d+)\s*$", response, flags=re.MULTILINE)
    # blocks = [preamble, id1, text1, id2, text2, ...]
    for i in range(1, len(blocks) - 1, 2):
        idx = int(blocks[i].strip())
        text = blocks[i + 1].strip()
        out[idx] = text
    return out


def call_openai(texts: List[tuple], args, src_lang: str, tgt_lang: str) -> dict:
    url = (args.base_url or "https://api.openai.com/v1").rstrip("/") \
          + "/chat/completions"
    payload = {
        "model": args.model,
        "temperature": args.temperature,
        "messages": [
            {"role": "system",
             "content": SYSTEM_PROMPT.format(src_lang=src_lang,
                                             tgt_lang=tgt_lang)},
            {"role": "user", "content": build_batch_prompt(texts, src_lang,
                                                           tgt_lang)},
        ],
    }
    if args.backend == "anthropic":
        raise RuntimeError("use call_anthropic")
    data = _http_json(url, {"Authorization": f"Bearer {args.api_key}"}, payload)
    return parse_batch_response(data["choices"][0]["message"]["content"],
                                [i for i, _ in texts])


def call_anthropic(texts: List[tuple], args, src_lang: str, tgt_lang: str) -> dict:
    url = (args.base_url or "https://api.anthropic.com").rstrip("/") \
          + "/v1/messages"
    payload = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "system": SYSTEM_PROMPT.format(src_lang=src_lang, tgt_lang=tgt_lang),
        "messages": [
            {"role": "user",
             "content": build_batch_prompt(texts, src_lang, tgt_lang)},
        ],
    }
    data = _http_json(url, {
        "x-api-key": args.api_key,
        "anthropic-version": "2023-06-01",
    }, payload)
    content = "".join(b.get("text", "") for b in data.get("content", []))
    return parse_batch_response(content, [i for i, _ in texts])
# ============================================================================
# G-pair repair
# ============================================================================
def repair_g_pairs(text: str) -> str:
    """Rewrite /G placeholders to match the most recent unclosed G,
    pairing by nesting order (close numbers are functionally irrelevant)."""
    parts = re.split(r"(\[\[\d+:(?:G|/G)\]\])", text)
    stack: list = []          # numbers of currently-open Gs
    out = []
    for p in parts:
        m = re.fullmatch(r"\[\[(\d+):(G|/G)\]\]", p)
        if not m:
            out.append(p)
            continue
        num, kind = m.group(1), m.group(2)
        if kind == "G":
            stack.append(num)
            out.append(p)
        else:  # /G
            if stack:
                out.append(f"[[{stack.pop()}:/G]]")   # renumbered to correct open
            else:
                out.append(p)                          # stray close: leave, validator will flag
    return "".join(out)

# ============================================================================
# Main loop
# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="LLM translation bridge for "
                                             "sdlxliff_tool TSV files")
    ap.add_argument("tsv", help="TSV file from sdlxliff_tool extract")
    ap.add_argument("--backend", choices=["openai", "anthropic"],
                    default="openai")
    ap.add_argument("--provider", choices=sorted(PRESETS), default=None,
                    help="provider preset: sets base URL and default model. "
                         "Explicit --base-url/--model override it.")
    ap.add_argument("--base-url", default=None,
                    help="custom OpenAI-compatible API base URL "
                         "(overrides preset)")
    ap.add_argument("--model", default=None,
                    help="model name (default: preset's model, "
                         "required if no preset)")
    ap.add_argument("--api-key", default=None,
                    help="API key (default: provider-specific or "
                         "OPENAI_API_KEY env var)")

    ap.add_argument("--src-lang", default="Russian")
    ap.add_argument("--tgt-lang", default="English")
    ap.add_argument("--batch-size", type=int, default=20,
                    help="segments per API call")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--retries-per-seg", type=int, default=2,
                    help="retries for a batch whose validation failed; "
                         "failed segments are retried individually")
    ap.add_argument("-o", "--out", default=None,
                    help="output TSV (default: overwrite input; original "
                         "is backed up to <name>.bak.tsv)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")

    # --- resolve preset ------------------------------------------------
    preset = PRESETS.get(args.provider) if args.provider else None

    if args.base_url is None and preset:
        args.base_url = preset["base_url"]
    if args.model is None:
        if preset:
            args.model = preset["model"]
        else:
            ap.error("--model is required (or use --provider)")

    # --- resolve API key -------------------------------------------------
    if args.api_key is None:
        env_candidates = []
        if preset and preset["env_var"]:
            env_candidates.append(preset["env_var"])
        env_candidates.append("OPENAI_API_KEY")
        if args.backend == "anthropic":
            env_candidates.append("ANTHROPIC_API_KEY")
        for env_var in env_candidates:
            val = os.environ.get(env_var)
            if val:
                args.api_key = val
                log.debug("Using API key from %s", env_var)
                break

    needs_key = preset["needs_key"] if preset else args.base_url is None
    if args.api_key is None and needs_key:
        hint = env_candidates[0] if preset else "OPENAI_API_KEY"
        log.error("No API key. Set %s (or the provider's own variable), "
                  "pass --api-key, or use a keyless preset "
                  "(--provider ollama / lmstudio).", hint)
        return 1

    rows = read_tsv(args.tsv)
    todo = [r for r in rows if r.needs_translation]
    log.info("%d rows, %d need translation", len(rows), len(todo))
    if not todo:
        print("Nothing to translate.")
        return 0

    call = call_anthropic if args.backend == "anthropic" else call_openai

    translated_count = 0
    failed: List[Row] = []

    def translate_one(row: Row) -> bool:
        """Translate a single segment with validation + retries."""
        for attempt in range(args.retries_per_seg + 1):
            try:
                res = call([(0, row.source)], args,
                           args.src_lang, args.tgt_lang)
                text = repair_g_pairs(res.get(0, "").strip())
                probs = validate_translation(row.source, text)
                if not probs:
                    row.translated = text
                    return True
                log.warning("Seg %s: validation failed (attempt %d): %s | RAW: %r",
                            row.seg_id, attempt + 1, "; ".join(probs), text)
            except Exception as e:  # noqa: BLE001
                log.warning("Seg %s: API error (attempt %d): %s",
                            row.seg_id, attempt + 1, e)
        return False

    def translate_batch(batch: List[Row]) -> None:
        nonlocal translated_count
        pairs = [(i, r.source) for i, r in enumerate(batch)]
        try:
            res = call(pairs, args, args.src_lang, args.tgt_lang)
        except Exception as e:  # noqa: BLE001
            log.error("Batch API call failed (%s) — falling back to "
                      "one-by-one", e)
            for r in batch:
                if translate_one(r):
                    translated_count += 1
                else:
                    failed.append(r)
            return
        for i, r in enumerate(batch):
            text = repair_g_pairs(res.get(i, "").strip())
            probs = validate_translation(r.source, text)
            if text and not probs:
                r.translated = text
                translated_count += 1
            elif text:
                # retry individually with fresh context
                log.warning("Seg %s failed batch validation (%s) | RAW: %r",
                            r.seg_id, "; ".join(probs), text)

                if translate_one(r):
                    translated_count += 1
                else:
                    failed.append(r)
            else:
                log.warning("Seg %s missing from batch response — "
                            "retrying solo", r.seg_id)
                if translate_one(r):
                    translated_count += 1
                else:
                    failed.append(r)

    batch: List[Row] = []
    for n, row in enumerate(todo, 1):
        batch.append(row)
        if len(batch) >= args.batch_size or n == len(todo):
            log.info("Translating %d/%d ...", n, len(todo))
            translate_batch(batch)
            # checkpoint: save progress after every batch
            out_path = args.out or args.tsv
            if args.out is None:
                backup = Path(args.tsv).with_suffix(".bak.tsv")
                if Path(args.tsv).exists() and not backup.exists():
                    backup.write_bytes(Path(args.tsv).read_bytes())
            write_tsv(out_path, rows)
            batch = []

    write_tsv(args.out or args.tsv, rows)
    print(f"\nDone: {translated_count} translated, "
          f"{len(failed)} failed.")
    if failed:
        print("Failed segments (translate manually):")
        for r in failed:
            print(f"  {r.seg_id}\t{r.source[:70]}")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
