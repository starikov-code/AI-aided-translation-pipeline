import csv

def ids(path):
    with open(path, encoding="utf-8-sig") as f:
        return [r["seg_id"] for r in csv.DictReader(f, delimiter="\t")]

a = ids("test_segments.tsv")
for f in ["verify.tsv", "verify2.tsv"]:
    b = ids(f)
    print(f"{f}: rows={len(b)}, TSV-not-in-file={len(set(a)-set(b))}, "
          f"file-not-in-TSV={len(set(b)-set(a))}")


    def ids(path):
        with open(path, encoding="utf-8-sig") as f:
            return [r["seg_id"] for r in csv.DictReader(f, delimiter="\t")]


    x, y = ids("verify.tsv"), ids("verify2.tsv")
    print("verify vs verify2:", len(set(x) - set(y)), "/", len(set(y) - set(x)))

