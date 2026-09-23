#!/usr/bin/env python3
"""
sdlxliff_tool.py — SDLXLIFF extraction, masking, and injection utility.

  1. Extract source/target segment text (per <mrk mtype="seg">) with inline
     tags replaced by stable placeholders.
  2. Masked text is safe to send to an AI / MT engine.
  3. Validate AI output (bpt/ept pairing, malformed placeholders).
  4. Inject translated targets back, deep-copying tag elements from source.

Requires: lxml  (pip install lxml)
"""

import argparse
import copy
import csv
import json
import logging
import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lxml import etree

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

XLIFF_NS = "urn:oasis:names:tc:xliff:document:1.2"
SDL_NS = "http://sdl.com/FileTypes/SdlXliff/1.0"
XML_NS = "http://www.w3.org/XML/1998/namespace"


def q(local: str, ns: str = XLIFF_NS) -> str:
    return f"{{{ns}}}{local}"


INLINE_TAGS = {"bpt", "ept", "ph", "it", "x"}
PAIRED_TAGS = {"g"}                      # container-style inline tags

# [[n:G:...]] opens a <g>; [[n:/G]] closes it. extra = serialized attributes.
PLACEHOLDER_RE = re.compile(
    r"\[\[(\d+):(BPT|EPT|PH|IT|X|G|/G)(?::([^\]]*))?\]\]")


log = logging.getLogger("sdlxliff_tool")


# ----------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------

@dataclass
class Segment:
    """One translatable segment (a <mrk mtype="seg"> or a bare <source>)."""
    tu_id: str
    seg_id: str
    source_text: str = ""
    target_text: str = ""
    translated_text: str = ""
    # placeholder string -> live lxml element in the SOURCE tree
    placeholder_map: Dict[str, etree._Element] = field(default_factory=dict)
    counter: int = 0

    def register(self, elem: etree._Element, kind: str, extra: str = "") -> str:
        self.counter += 1
        ph = f"[[{self.counter}:{kind}" + (f":{extra}]]" if extra else "]]")
        self.placeholder_map[ph] = elem
        return ph


# ----------------------------------------------------------------------------
# Core tool
# ----------------------------------------------------------------------------

class SdlxliffTool:

    def __init__(self, path: str):
        self.path = Path(path)
        parser = etree.XMLParser(strip_cdata=False,
                                 remove_blank_text=False,
                                 resolve_entities=False)
        self.tree = etree.parse(str(self.path), parser)
        self.root = self.tree.getroot()
        if self.root.tag != q("xliff"):
            raise ValueError(
                f"Not an XLIFF file: root element is {self.root.tag}")
        self.version = self.root.get("version", "1.2")
        if not self.version.startswith("1."):
            log.warning("XLIFF version %s — tool targets 1.x (SDLXLIFF).",
                        self.version)
        self.segments: List[Segment] = []

    # ------------------------------------------------------------------ #
    #  EXTRACTION                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _inline_to_placeholder(elem: etree._Element, seg: Segment) -> str:
        tag = etree.QName(elem).localname.upper()
        extra = ""
        if tag == "X":
            extra = elem.get("ctype", "x")
        elif tag == "PH":
            extra = elem.get("ctype", "ph")
        elif tag == "IT":
            extra = elem.get("pos", "it")

        return seg.register(elem, tag, extra)

    def _serialize(self, container: etree._Element, seg: Segment) -> str:
        parts: List[str] = []
        if container.text:
            parts.append(container.text)
        for child in container:
            tag = etree.QName(child).localname
            log.debug("Handling <%s> in seg %s", tag, seg.seg_id)
            if tag == "mrk":
                parts.append(self._serialize(child, seg))
            elif tag in INLINE_TAGS:
                parts.append(self._inline_to_placeholder(child, seg))
            elif tag in PAIRED_TAGS:
                ph_open = self._inline_to_placeholder(child, seg)
                ph_close = f"[[{seg.counter}:/G]]"  # capture id BEFORE recursion
                parts.append(ph_open)
                parts.append(self._serialize(child, seg))
                seg.placeholder_map[ph_close] = child
                parts.append(ph_close)
            else:
                # Unknown element: recurse into it to be safe
                parts.append(self._serialize(child, seg))
            if child.tail:
                parts.append(child.tail)
        return "".join(parts)

    def _iter_segment_containers(self, tu: etree._Element):
        """
        Yield (seg_id, source_container, target_container).
        Pattern A: <seg-source> with <mrk mtype="seg"> children.
        Pattern B: plain <source>/<target>.
        Pattern C: no <seg-source>, no <source> — target-only files, segmented
        on the target's own <mrk mtype="seg"> children (or the bare <target>
        as a single whole segment if it has none).
        """
        seg_source = tu.find(q("seg-source"))
        if seg_source is not None:
            target = tu.find(q("target"))
            tgt_mrks: Dict[str, etree._Element] = {}
            if target is not None:
                for m in target.findall(".//" + q("mrk")):
                    mid = m.get("mid")
                    if mid is not None:
                        tgt_mrks[mid] = m
            for m in seg_source.findall(".//" + q("mrk")):
                if m.get("mtype") != "seg":
                    continue
                # Stable ID: trans-unit id + mrk mid, falling back to tu id + position
                tu_id = tu.get("id", "")
                mid = m.get("mid")
                if mid is None:
                    mids = [x for x in seg_source.findall(".//" + q("mrk"))
                            if x.get("mtype") == "seg"]
                    mid = str(mids.index(m))
                seg_id = f"{tu_id}#{mid}"
                yield seg_id, m, tgt_mrks.get(m.get("mid"))
            return

        source = tu.find(q("source"))
        target = tu.find(q("target"))
        tu_id = tu.get("id") or str(uuid.uuid4())

        if source is not None:
            yield f"{tu_id}#whole", source, target
        elif target is not None:
            # Fallback: no <seg-source>, no <source> — segment on the
            # target's own <mrk mtype="seg"> children (target-side only).
            mrks = [m for m in target.findall(".//" + q("mrk"))
                    if m.get("mtype") == "seg"]
            if mrks:
                for i, m in enumerate(mrks):
                    yield f"{tu_id}#t{i}", m, m
            else:
                yield f"{tu_id}#whole", target, target

    def extract(self) -> List[Segment]:
        self.segments.clear()
        for tu in self.root.iter(q("trans-unit")):
            if (tu.get("translate") or "yes").lower() == "no":
                continue
            for seg_id, src_c, tgt_c in self._iter_segment_containers(tu):
                seg = Segment(tu_id=tu.get("id", ""), seg_id=seg_id)
                seg.source_text = self._serialize(src_c, seg)
                if tgt_c is not None:
                    seg.target_text = self._serialize(tgt_c, seg)
                self.segments.append(seg)
        log.info("Extracted %d segments from %s",
                 len(self.segments), self.path.name)
        return self.segments

    # ------------------------------------------------------------------ #
    #  VALIDATION                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def validate_masked(text: str) -> List[str]:
        problems: List[str] = []
        for m in re.finditer(r"\[\[(?!\d+:)", text):
            problems.append(
                f"Malformed placeholder at offset {m.start()}: "
                f"{text[m.start():m.start() + 20]!r}")
        opens, closes = set(), set()
        g_stack: List[str] = []
        for m in PLACEHOLDER_RE.finditer(text):
            pid, kind = m.group(1), m.group(2)
            if kind == "BPT":
                opens.add(pid)
            elif kind == "EPT":
                closes.add(pid)
            elif kind == "G":
                g_stack.append(pid)
            elif kind == "/G":
                if not g_stack:
                    problems.append(f"stray /G [[{pid}]] has no open G")
                else:
                    g_stack.pop()
        if g_stack:
            problems.append(f"Unclosed G tag(s): {g_stack}")
        for pid in closes - opens:
            problems.append(f"EPT [[{pid}]] has no matching BPT")
        for pid in opens - closes:
            problems.append(
                f"BPT [[{pid}]] has no matching EPT (may be intentional)")
        return problems

    # ------------------------------------------------------------------ #
    #  INJECTION                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _clean_target(tgt_container, parent):
        if tgt_container is None:
            tgt = etree.Element(q("target"))
            # insert right after seg-source (or after source), before SDL extras
            anchor = parent.find(q("seg-source")) or parent.find(q("source"))
            parent.insert(list(parent).index(anchor) + 1, tgt)
        else:
            tgt = tgt_container
            for child in list(tgt):
                tgt.remove(child)
            tgt.text = None
        return tgt

    def _build_target_content(self, tgt_container, seg, translated) -> bool:
        ph_map = seg.placeholder_map

        # Pre-validate G balance (fail BEFORE mutating the tree)
        stack: List[str] = []
        for m in PLACEHOLDER_RE.finditer(translated):
            pid, kind = m.group(1), m.group(2)
            if kind == "G":
                stack.append(pid)
            elif kind == "/G":
                if not stack:
                    log.error("Seg %s: stray /G [[%s]]", seg.seg_id, pid)
                    return False
                stack.pop()
        if stack:
            log.error("Seg %s: unclosed G %s", seg.seg_id, stack)
            return False

        parts: List[Tuple[str, str]] = []
        pos = 0
        for m in PLACEHOLDER_RE.finditer(translated):
            if m.start() > pos:
                parts.append(("text", translated[pos:m.start()]))
            parts.append(("tag", m.group(0)))
            pos = m.end()
        if pos < len(translated):
            parts.append(("text", translated[pos:]))

        for kind, val in parts:
            if kind == "tag" and val not in ph_map:
                log.error("Seg %s: unknown placeholder %s", seg.seg_id, val)
                return False

        stack: List[etree._Element] = [tgt_container]

        def cur() -> etree._Element:
            return stack[-1]

        def append_text(text: str) -> None:
            if not text:
                return
            c = cur()
            if len(c) == 0 and not c.text:
                c.text = text
            else:
                last = c[-1]
                last.tail = (last.tail or "") + text

        for kind, val in parts:
            if kind == "text":
                append_text(val)
                continue
            m = PLACEHOLDER_RE.match(val)
            pid, gkind = m.group(1), m.group(2)
            if gkind == "/G":
                if len(stack) > 1:
                    stack.pop()
                continue
            new_elem = copy.deepcopy(ph_map[val])
            if gkind == "G":
                for ch in list(new_elem):
                    new_elem.remove(ch)
                new_elem.text = None
                cur().append(new_elem)
                new_elem.tail = None
                stack.append(new_elem)
            else:
                cur().append(new_elem)
                new_elem.tail = None

        used = {v for k, v in parts if k == "tag"}
        missing = set(ph_map) - used
        if missing:
            log.warning("Seg %s: placeholders missing in AI output: %s",
                        seg.seg_id, sorted(missing))
        return True

    @staticmethod
    def _inject_plain_fallback(tgt_container: etree._Element,
                               translated: str) -> None:
        plain = PLACEHOLDER_RE.sub("", translated)
        plain = re.sub(r"\s+", " ", plain).strip()
        tgt_container.text = plain
        log.warning("Injected plain text (tags dropped) — review manually.")

    def inject(self, seg_id: str, translated_masked: str,
               allow_fallback: bool = True) -> bool:
        seg = next((s for s in self.segments if s.seg_id == seg_id), None)
        if seg is None:
            log.error("Segment %s not found (run extract() first)", seg_id)
            return False

        for tu in self.root.iter(q("trans-unit")):
            if tu.get("id") != seg.tu_id:
                continue
            for sid, src_c, tgt_c in self._iter_segment_containers(tu):
                if sid != seg_id:
                    continue

                parent = src_c.getparent()
                tgt = self._clean_target(tgt_c, parent)

                # Wrap in an mrk ONLY if we're filling a bare <target> element.
                # If tgt is already the matching <mrk mtype="seg">, fill it directly.
                if (etree.QName(src_c).localname == "mrk"
                        and etree.QName(tgt).localname == "target"):
                    tgt_mrk = etree.SubElement(tgt, q("mrk"))
                    tgt_mrk.set("mtype", "seg")
                    seg_source = parent.find(q("seg-source"))
                    segs = [x for x in seg_source.findall(q("mrk"))
                            if x.get("mtype") == "seg"]
                    tgt_mrk.set("mid", src_c.get("mid")
                                or src_c.get("id") or str(segs.index(src_c)))
                    tgt = tgt_mrk

                ok = self._build_target_content(tgt, seg, translated_masked)
                if not ok:
                    if allow_fallback:
                        self._inject_plain_fallback(tgt, translated_masked)
                    else:
                        return False
                seg.translated_text = translated_masked
                return True

        log.error("Segment %s not found in tree", seg_id)
        return False

    # ------------------------------------------------------------------ #
    #  IO                                                                #
    # ------------------------------------------------------------------ #

    def save(self, path: Optional[str] = None) -> None:
        out = Path(path) if path else self.path
        self.tree.write(str(out), xml_declaration=True,
                        encoding="utf-8", standalone=True)
        log.info("Saved %s", out)

    def export_segments_tsv(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["seg_id", "source_masked", "target_masked",
                        "translated_masked"])
            for s in self.segments:
                w.writerow([s.seg_id, s.source_text, s.target_text,
                            s.translated_text])

    def import_segments_tsv(self, path: str) -> int:
        n = 0
        with open(path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                tr = (row.get("translated_masked") or "").strip()
                if not tr:
                    continue
                if self.inject(row["seg_id"], tr):
                    n += 1
        log.info("Injected %d segments", n)
        return n


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="SDLXLIFF extract / mask / inject utility")
    ap.add_argument("file", help="input .sdlxliff file")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ex = sub.add_parser("extract", help="extract masked segments to TSV")
    p_ex.add_argument("-o", "--out", default=None)
    p_ex.add_argument("--json", action="store_true")

    p_in = sub.add_parser("inject", help="inject translations from TSV")
    p_in.add_argument("tsv")
    p_in.add_argument("-o", "--out", required=True)

    p_ch = sub.add_parser("check", help="validate masked text in a TSV")
    p_ch.add_argument("tsv")

    p_dg = sub.add_parser("diagnose",
                          help="inspect file structure (why 0 segments?)")

    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")

    tool = SdlxliffTool(args.file)

    if args.cmd == "extract":
        segs = tool.extract()
        out = args.out or (str(Path(args.file).with_suffix(""))
                           + "_segments.tsv")
        tool.export_segments_tsv(out)
        print(f"Wrote {len(segs)} segments -> {out}")
        if args.json:
            jout = str(Path(out).with_suffix(".json"))
            dump = [{k: (list(v) if k == "placeholder_map" else v)
                     for k, v in vars(s).items()} for s in segs]
            with open(jout, "w", encoding="utf-8") as f:
                json.dump(dump, f, ensure_ascii=False, indent=2)
            print(f"Wrote {jout}")

    elif args.cmd == "inject":
        tool.extract()  # build placeholder maps from the ORIGINAL file
        n = tool.import_segments_tsv(args.tsv)
        tool.save(args.out)
        print(f"Injected {n} segments -> {args.out}")

    elif args.cmd == "check":
        tool.extract()
        by_id = {s.seg_id: s for s in tool.segments}
        bad = 0
        with open(args.tsv, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                tr = (row.get("translated_masked") or "").strip()
                if not tr:
                    continue
                if row["seg_id"] not in by_id:
                    print(f"UNKNOWN seg {row['seg_id']}")
                    bad += 1
                    continue
                probs = tool.validate_masked(tr)
                if probs:
                    bad += 1
                    print(f"Seg {row['seg_id']}:")
                    for p in probs:
                        print(f"   - {p}")
        print("OK" if bad == 0 else f"{bad} problematic segment(s)")

    elif args.cmd == "diagnose":
        tus = list(tool.root.iter(q("trans-unit")))
        print(f"Root tag:        {tool.root.tag} (version {tool.version})")
        print(f"Trans-units:     {len(tus)}")
        n_segsource = sum(1 for t in tus if t.find(q("seg-source")) is not None)
        n_source    = sum(1 for t in tus if t.find(q("source")) is not None)
        n_target    = sum(1 for t in tus if t.find(q("target")) is not None)
        n_skip      = sum(1 for t in tus
                          if (t.get("translate") or "yes").lower() == "no")
        n_mrk_tgt   = sum(1 for t in tus for m in t.iter(q("mrk"))
                          if m.get("mtype") == "seg")
        print(f"  with <seg-source>: {n_segsource}")
        print(f"  with <source>:     {n_source}")
        print(f"  with <target>:     {n_target}")
        print(f"  translate=no:      {n_skip}")
        print(f"  mrk mtype=seg:     {n_mrk_tgt}")
        # show all namespaces actually used
        ns = set()
        for el in tool.root.iter():
            if isinstance(el.tag, str) and el.tag.startswith("{"):
                ns.add(el.tag.split("}")[0][1:])
        print("Namespaces seen:")
        for n in sorted(ns):
            print(f"  {n}")
        if tus:
            print("\nFirst trans-unit (truncated):")
            print(etree.tostring(tus[0], pretty_print=True).decode()[:2000])

    return 0


if __name__ == "__main__":
    sys.exit(main())
