import csv

def ids(path):
    with open(path, encoding="utf-8-sig") as f:
        return [r["seg_id"] for r in csv.DictReader(f, delimiter="\t")]

a = ids("test_segments_new.tsv")
b = ids("verify.tsv")
print("new TSV rows:          ", len(a))
print("verify.tsv rows:       ", len(b))
print("In new TSV, not in file:", len(set(a) - set(b)))
print("In file, not in new TSV:", len(set(b) - set(a)))
print("First new TSV id:", a[0])
print("First verify id: ", b[0])
