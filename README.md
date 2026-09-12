sdlxliff_tool
A Python utility for AI-assisted translation of SDLXLIFF files (Trados): extracts masked segments, validates AI output, and injects translations back with full tag preservation.

Pipeline
test.sdlxliff ──extract──▶ TSV (masked source) ──AI bridge──▶ TSV (translations)
                                                                    │
        test_translated.sdlxliff ◀──inject── check ◀────────────────┘
extract — pulls every <mrk mtype="seg"> segment, replacing inline tags (bpt, ept, ph, it, x, g) with [[n:TYPE]] placeholders. Output is safe to send to an LLM.
check — validates translations: unknown segment IDs, malformed placeholders, unpaired BPT/EPT (and G//G after the pending patch).
inject — rebuilds <target> content: text runs from the translation, tag elements deep-copied from the source tree via placeholder maps, preserving formatting and tag-defs integrity.
Usage
python main.py file.sdlxliff extract -o segments.tsv
python main.py file.sdlxliff check segments.tsv
python main.py file.sdlxliff inject segments.tsv -o translated.sdlxliff
Auxiliary scripts:

relink.py — re-attaches translations from an older TSV to a fresh extract by matching source_masked text (survives file re-exports; warns on conflicting translations for duplicate sources).
compare_ids*.py, placeholder_loss.py — diagnostics: TSV↔file ID alignment, placeholder multiset comparison between source and translation columns.
Status
Working end-to-end on a real 847-segment file. File opens in Trados without errors; exotic scripts (IPA, Hebrew) survive translation intact.

Fixed
#	Bug	Fix
1	Random uuid4() seg_ids — mrk elements had neither id nor mid, so extract/check/inject never agreed across runs (every check returned 847 UNKNOWN)	Stable IDs: {tu_id}#{mid}, position-index fallback
2	Nested <mrk><mrk> in targets on injection — Trados crash ("ключ отсутствует в словаре" / "непредвиденный контент")	Wrap in <mrk> only when target container is a bare <target>; reuse existing matching mrk otherwise
3	<target> appended after SDL extension elements (sdl:seg-defs)	Insert directly after seg-source/source
In progress
<g> inline groups (bold/italic) dropped at extract time. _serialize only converts bpt/ept/ph/it/x; the 439 <g> elements (paired XLIFF groups carrying <cf> character formatting) were recursed into with tags discarded — the AI never saw them. Designed fix, not yet applied:
paired placeholders [[n:G:ctype]] / [[n:/G]]
regex + validate_masked updated (G pairs validated like BPT/EPT)
container-stack in _build_target_content for nested rebuild; hard-fail on unclosed G
⚠️ masks change ⇒ re-extract → relink → check → inject after patching
12 segments drop footnote anchor [[2:X:x]] in AI output (currently WARN-only). Plan: scripted repair or quarantine-to-review for production.
Known issues (out of scope)
Trados DOCX generation crashes ("ссылка на объект не указывает на экземпляр объекта") after any external-review round-trip — pre-existing Trados quirk, unrelated to this tool. Workaround: populate TM from the sdlxliff → reopen original → populate from TM → save as.
Lessons learned
Never generate segment IDs — always derive stable keys from the document itself; a text-match relink is the robust fallback.
Placeholder-loss checks must compare against source, not just validate syntax — a vacuous 0/847 match hid the <g> problem.
Trados is stricter than well-formed XLIFF: element order (target before sdl:seg-defs) and mrk structure matter.
For production: segments failing placeholder checks should be quarantined, not injected with a warning.
Requirements
Python 3.10+, lxml
