#!/usr/bin/env python3
"""
llm_bridge2.py — Bridge between sdlxliff_tool TSV files and LLM APIs,
WITH provider failover/cycling.

Reads the TSV produced by `sdlxliff_tool extract`, translates empty
`translated_masked` cells via an LLM (in batches), validates placeholder
integrity per segment, and writes the TSV back — ready for
`sdlxliff_tool check` and `inject`.

NEW in v2 — provider cycling:
  If a provider hits a quota limit, a dead key, or an outage, the bridge
  automatically rotates to the next provider in the chain and continues.
  Quota'd providers are put on cooldown and skipped for a while.

Provider chain resolution (in order of priority):
  1. --provider flags, repeatable, in CLI order:
        python llm_bridge2.py tsv.tsv --provider groq --provider gemini
  2. LLM_API_ORDER env var (comma-separated):
        LLM_API_ORDER=groq,gemini,openrouter,ollama
  3. single --provider as in v1 (one provider, no fallback)

API keys are read from environment variables / .env:
  GROQ_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY, DEEPSEEK_API_KEY,
  OPENAI_API_KEY, ANTHROPIC_API_KEY   (keyless: ollama, lmstudio)

Cooldown control (.env, optional):
  COOLDOWN_SECONDS=600        how long a quota'd provider is skipped

The script is resumable: rows that already have translated_masked filled
are skipped, so you can stop and re-run at any time.
"""

import argparse
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import urllib.request
import urllib.error
import json

PLACEHOLDER_RE = re.compile(
    r"\[\[(\d+):(BPT|EPT|PH|IT|X|G|/G)(?::([^\]]*))?\]\]")

# ---------------------------------------------------------------------------
# .env loading (python-dotenv if available, else manual fallback)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    _env = Path(__file__).with_name(".env")
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
        "backend": "openai",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-3.6-flash",
        "env_var": "GEMINI_API_KEY",
        "needs_key": True,
        "backend": "openai",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "env_var": "OPENROUTER_API_KEY",
        "needs_key": True,
        "backend": "openai",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "env_var": "DEEPSEEK_API_KEY",
        "needs_key": True,
        "backend": "openai",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:7b-instruct",
        "env_var": None,
        "needs_key": False,
        "backend": "openai",
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "model": "local-model",
        "env_var": None,
        "needs_key": False,
        "backend": "openai",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "env_var": "OPENAI_API_KEY",
        "needs_key": True,
        "backend": "openai",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "model": "claude-sonnet-4-5",   # adjust as needed
        "env_var": "ANTHROPIC_API_KEY",
        "needs_key": True,
        "backend": "anthropic",
    },
}

log = logging.getLogger("llm_bridge")

# ============================================================================
# Exceptions for failover classification
# ============================================================================

class QuotaExceeded(Exception):
    """Provider is unusable for a long time: quota, billing, dead key.
    -> rotate immediately, put on cooldown."""


class TransientError(Exception):
    """Temporary failure: timeout, connection, 5xx, rate limit.
    -> backoff-and-retry same provider, then rotate."""


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


@dataclass
class Provider:
    """A fully resolved provider configuration."""
    name: str
    backend: str            # "openai" | "anthropic"
    base_url: str
    model: str
    api_key: Optional[str]
    blocked_until: float = 0.0   # cooldown timestamp (monotonic)

    def is_blocked(self) -> bool:
        return time.monotonic() < self.blocked_until

    def block(self, seconds: float) -> None:
        self.blocked_until = time.monotonic() + seconds
        log.warning("Provider '%s' on cooldown for %.0fs", self.name, seconds)


@dataclass
class ProviderChain:
    providers: List[Provider] = field(default_factory=list)
    cooldown_seconds: float = 600.0

    def usable(self) -> List[Provider]:
        return [p for p in self.providers if not p.is_blocked()]

    def rotate(self, failed: Provider, reason: str) -> None:
        """Mark a provider as failed and (for quota) start its cooldown."""
        if isinstance_reason_is_quota(reason):
            failed.block(self.cooldown_seconds)


def isinstance_reason_is_quota(reason: str) -> bool:
    return "quota" in reason.lower() or "key" in reason.lower() \
        or "billing" in reason.lower() or "401" in reason or "402" in reason


# ============================================================================
# TSV I/O
# ============================================================================

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
        return None
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
# HTTP layer (stdlib only) — raises QuotaExceeded / TransientError
# ============================================================================

def _http_json(url: str, headers: dict, payload: dict,
               timeout: int = 180,
               retries: int = 2) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                               "llm_bridge/2.0",
                 **headers},
        method="POST",
    )
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:500]
            low = body.lower()
            # --- long-term unusable: rotate away immediately -------------
            if e.code in (401, 402, 403):
                raise QuotaExceeded(f"HTTP {e.code}: {body}") from e
            if e.code == 429 and ("quota" in low or "billing" in low
                                  or "exceeded" in low):
                raise QuotaExceeded(f"Quota exhausted: {body}") from e
            # --- transient: backoff, retry same provider, then rotate ----
            if e.code in (429, 500, 502, 503, 504, 529) and attempt < retries:
                wait = 2 ** (attempt + 1)
                log.warning("HTTP %s, retrying in %ss: %s", e.code, wait, body)
                time.sleep(wait)
                last_err = e
                continue
            raise TransientError(f"HTTP {e.code}: {body}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 ** (attempt + 1))
                continue
            raise TransientError(f"network: {e}") from e

    raise TransientError(f"Request failed: {last_err}")


def _extract_content(data: dict) -> str:
    """Extract assistant text from an OpenAI-compatible or Anthropic response."""
    # OpenAI-compatible
    if "choices" in data:
        return data["choices"][0]["message"]["content"] or ""
    # Anthropic
    if "content" in data and isinstance(data["content"], list):
        return "".join(b.get("text", "") for b in data["content"])
    raise TransientError(f"Unrecognized response shape: {list(data.keys())}")


# ============================================================================
# LLM backends
# ============================================================================

def _call_openai(p: Provider, system: str, user: str,
                 temperature: float) -> str:
    headers = {}
    if p.api_key:
        headers["Authorization"] = f"Bearer {p.api_key}"
    payload = {
        "model": p.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }
    data = _http_json(f"{p.base_url.rstrip('/')}/chat/completions",
                      headers, payload)
    return _extract_content(data)


def _call_anthropic(p: Provider, system: str, user: str,
                    temperature: float) -> str:
    headers = {
        "x-api-key": p.api_key or "",
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": p.model,
        "max_tokens": 8192,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    data = _http_json(f"{p.base_url.rstrip('/')}/v1/messages",
                      headers, payload)
    return _extract_content(data)


BACKENDS = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
}


def call_llm(chain: ProviderChain, system: str, user: str,
             temperature: float = 0.3) -> str:
    """
    Call the first usable provider in the chain.
    On QuotaExceeded -> block provider, rotate to the next.
    On TransientError -> rotate to the next (retries were already done
    inside _http_json).
    Raises the last error if every provider in the chain fails.
    """
    if not chain.providers:
        raise RuntimeError("Empty provider chain")
    last_exc: Optional[Exception] = None
    tried = 0
    for p in list(chain.providers):
        if p.is_blocked():
            continue
        tried += 1
        try:
            log.info("Calling provider '%s' (model=%s)", p.name, p.model)
            result = BACKENDS[p.backend](p, system, user, temperature)
            return result
        except QuotaExceeded as e:
            log.error("Provider '%s' quota/dead: %s — rotating", p.name, e)
            p.block(chain.cooldown_seconds)
            last_exc = e
        except TransientError as e:
            log.error("Provider '%s' transient failure: %s — rotating",
                      p.name, e)
            last_exc = e
    if tried == 0:
        raise RuntimeError(
            "All providers are on cooldown. Wait for cooldown to expire "
            "or restart the script.")
    raise RuntimeError(f"All providers failed. Last error: {last_exc}")


# ============================================================================
# Prompting + batch translation
# ============================================================================

SYSTEM_PROMPT = """\
You are a professional translator working inside a Trados Studio SDLXLIFF.
You receive numbered source segments whose inline tags are replaced by
placeholders like [[3:PH:...]] or [[5:G:...]].

Rules:
1. Translate every segment into the requested target language.
2. Copy EVERY placeholder EXACTLY as-is, byte for byte, into the
   translation at its natural position.
3. Do NOT translate, reorder, rename, drop, or modify any placeholder.
4. If a G tag pair opens and closes in the source, keep the paired
   [[n:G:...]] and [[n:/G:...]] correctly nested around the translated
   inner text.
5. Output ONLY the translations, in the exact same numbering format:

### 1
<translation of segment 1>
### 2
<translation of segment 2>

No explanations, no markdown fences, no extra text.
"""


def build_batch_prompt(segments: List[Row], target_lang: str) -> str:
    lines = [f"Target language: {target_lang}", ""]
    for i, r in enumerate(segments, 1):
        lines.append(f"### {i}")
        lines.append(r.source)
    lines.append("")
    lines.append("Translate all segments now, following the numbering "
                 "format exactly.")
    return "\n".join(lines)


def parse_batch_reply(reply: str, expected: int) -> dict:
    """Parse '### N' delimited reply -> {index: text}. 1-based indices."""
    out: dict = {}
    cur_idx: Optional[int] = None
    buf: List[str] = []
    for line in reply.splitlines():
        m = re.match(r"^###\s*(\d+)\s*$", line.strip())
        if m:
            if cur_idx is not None:
                out[cur_idx] = "\n".join(buf).strip()
            cur_idx = int(m.group(1))
            buf = []
        elif cur_idx is not None:
            buf.append(line)
    if cur_idx is not None:
        out[cur_idx] = "\n".join(buf).strip()
    if len(out) != expected:
        raise TransientError(
            f"Batch reply had {len(out)} segments, expected {expected}")
    return out


def repair_g_pairs(source: str, translated: str) -> str:
    """
    Best-effort repair: if the source has an unmatched G-tag pair
    (same id with :G and :/G), ensure both appear in the translation.
    If translated lost one half, re-append it at the end. (v1 behaviour)
    """
    src_pairs = {}
    for m in PLACEHOLDER_RE.finditer(source):
        if m.group(2) in ("G", "/G"):
            src_pairs.setdefault(m.group(1), {})[m.group(2)] = m.group(0)
    for gid, halves in src_pairs.items():
        if "G" in halves and "/G" in halves:
            if halves["G"] not in translated and halves["/G"] in translated:
                translated = translated.replace(halves["/G"],
                                                halves["/G"], 1)
            if halves["/G"] not in translated and halves["G"] in translated:
                translated += halves["/G"]
            if halves["G"] not in translated and halves["/G"] not in translated:
                translated = f"{halves['G']}{translated}{halves['/G']}"
    return translated


def translate_batch(chain: ProviderChain, segments: List[Row],
                    target_lang: str, max_repair_rounds: int = 1) -> dict:
    """
    Translate a batch. Returns {seg_id: translated_text}.
    Raises on total failure (caller decides what to do).
    """
    prompt = build_batch_prompt(segments, target_lang)
    reply = call_llm(chain, SYSTEM_PROMPT, prompt)
    parsed = parse_batch_reply(reply, len(segments))

    # First-pass validation + G repair
    result: dict = {}
    for i, r in enumerate(segments, 1):
        text = parsed.get(i, "")
        problems = validate_translation(r.source, text)
        if problems:
            text = repair_g_pairs(r.source, text)
            problems = validate_translation(r.source, text)
        result[r.seg_id] = text
        if problems:
            log.warning("seg %s (%s): %s", r.seg_id, "batch",
                        "; ".join(problems))
    return result


def translate_one_with_fallback(chain: ProviderChain, row: Row,
                                target_lang: str) -> str:
    """
    Per-segment fallback when a batch fails mid-way: try each usable
    provider for this single segment. Raises if all fail.
    """
    prompt = f"Target language: {target_lang}\n\n### 1\n{row.source}\n"
    for p in chain.usable():
        try:
            tmp_chain = ProviderChain(providers=[p],
                                      cooldown_seconds=chain.cooldown_seconds)
            reply = call_llm(tmp_chain, SYSTEM_PROMPT, prompt)
            parsed = parse_batch_reply(reply, 1)
            text = parsed.get(1, "")
            problems = validate_translation(row.source, text)
            if problems:
                text = repair_g_pairs(row.source, text)
            return text
        except Exception as e:      # noqa: BLE001 — any failure -> next provider
            log.error("Single-segment fallback on '%s' failed: %s",
                      p.name, e)
    raise RuntimeError(f"All providers failed for segment {row.seg_id}")


# ============================================================================
# Orchestration
# ============================================================================

def backup_tsv(path: str) -> str:
    bak = path + ".bak.tsv"
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as src, \
             open(bak, "w", encoding="utf-8-sig") as dst:
            dst.write(src.read())
    return bak


def run(args) -> int:
    rows = read_tsv(args.tsv)
    todo = [r for r in rows if r.needs_translation]
    log.info("Loaded %d rows, %d need translation", len(rows), len(todo))

    # Build provider chain
    order = args.provider or [
        s.strip() for s in
        os.environ.get("LLM_API_ORDER", "groq,gemini,ollama").split(",")
        if s.strip()
    ]
    cooldown = float(os.environ.get("COOLDOWN_SECONDS", "600"))
    chain = ProviderChain(cooldown_seconds=cooldown)
    for name in order:
        if name not in PRESETS:
            log.error("Unknown provider '%s' (known: %s)",
                      name, ", ".join(PRESETS))
            return 2
        preset = PRESETS[name]
        key = os.environ.get(preset["env_var"], "") if preset["env_var"] else ""
        if preset["needs_key"] and not key.strip():
            log.warning("Provider '%s' skipped: %s not set",
                        name, preset["env_var"])
            continue
        chain.providers.append(Provider(
            name=name,
            backend=preset["backend"],
            base_url=preset["base_url"],
            model=preset["model"],
            api_key=key.strip() or None,
        ))
    if not chain.providers:
        log.error("No usable providers (missing keys or none selected).")
        return 2
    log.info("Provider chain: %s",
             " -> ".join(p.name for p in chain.providers))

    # Batch loop
    batch_size = args.batch_size
    done = 0
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        try:
            results = translate_batch(chain, batch, args.lang)
        except Exception as e:      # noqa: BLE001
            log.error("Batch failed (%s) — falling back per segment", e)
            results = {}
            for r in batch:
                try:
                    results[r.seg_id] = translate_one_with_fallback(
                        chain, r, args.lang)
                except Exception as e2:     # noqa: BLE001
                    log.error("Segment %s failed everywhere: %s — "
                              "leaving empty for re-run", r.seg_id, e2)
                    results[r.seg_id] = ""
        # Apply results
        by_id = {r.seg_id: r for r in rows}
        for seg_id, text in results.items():
            if text.strip():
                by_id[seg_id].translated = text
                done += 1
        # Checkpoint after every batch
        write_tsv(args.tsv, rows)
        log.info("Progress: %d/%d translated (batch at %d done)",
                 done, len(todo), start)

    backup_tsv(args.tsv)
    write_tsv(args.tsv, rows)
    still_empty = sum(1 for r in rows if r.needs_translation)
    log.info("Finished: %d translated this run, %d remain empty "
             "(re-run to retry those)", done, still_empty)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Translate sdlxliff_tool TSV via LLMs with "
                    "provider failover.")
    ap.add_argument("tsv", help="TSV file from sdlxliff_tool extract")
    ap.add_argument("--lang", required=True, help="Target language, "
                    "e.g. 'German (Germany)'")
    ap.add_argument("--batch-size", type=int,
                    default=int(os.environ.get("BATCH_SIZE", "20")))
    ap.add_argument("--provider", action="append", default=None,
                    metavar="NAME",
                    help="Provider to use; repeatable for a failover chain "
                         "(e.g. --provider groq --provider gemini). "
                         "Default: LLM_API_ORDER env or groq,gemini,ollama. "
                         f"Known: {', '.join(PRESETS)}")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())

