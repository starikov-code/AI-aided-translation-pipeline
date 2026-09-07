from lxml import etree
t = etree.parse("test_translated.sdlxliff")
ns = {"x": "urn:oasis:names:tc:xliff:document:1.2"}
for tgt in t.findall(".//x:target", ns):
    if tgt.findall(".//x:bpt", ns):
        print(etree.tostring(tgt, encoding="unicode")[:500])
        break
