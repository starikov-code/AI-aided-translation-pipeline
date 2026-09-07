import csv, re
PH = re.compile(r"\[\[(\d+):([A-Z]+)(?::[^\]]*)?\]\]")
def phs(t): return sorted(PH.findall(t))
rows = list(csv.DictReader(open("test_segments_fresh.tsv", encoding="utf-8-sig"), delimiter="\t"))
mismatch = 0
for r in rows:
    if not r["translated_masked"].strip():
        continue
    if phs(r["source_masked"]) != phs(r["translated_masked"]):
        mismatch += 1
print("rows where translated placeholders != source placeholders:", mismatch, "/", len(rows))
