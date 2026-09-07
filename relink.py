import csv, sys

def load(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter="\t"))

tsv = load("test_segments.tsv")
new = load("fresh.tsv")   # TARGET file's fresh extract (see note below)

# translated rows from old TSV, keyed by source text
translated = {}
dupes = 0
for r in tsv:
    src = r["source_masked"].strip()
    tr  = r["translated_masked"].strip()
    if not tr:
        continue
    if src in translated and translated[src] != tr:
        dupes += 1
        print(f"WARNING: conflicting translations for source: {src[:60]}...")
    translated[src] = tr

print(f"Translated rows in old TSV: {len(translated)}, conflicts: {dupes}")

matched, unmatched = 0, 0
for r in new:
    src = r["source_masked"].strip()
    tr = translated.get(src, "")
    if tr:
        matched += 1
    else:
        unmatched += 1
    r["translated_masked"] = tr

with open("test_segments_fresh.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=new[0].keys(), delimiter="\t")
    w.writeheader()
    w.writerows(new)

print(f"Re-linked: {matched}, without translation: {unmatched}")
print("Wrote test_segments_fresh.tsv")
