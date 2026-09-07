from lxml import etree
from collections import Counter
t = etree.parse("test.sdlxliff")
ns = {"x": "urn:oasis:names:tc:xliff:document:1.2"}
mrks = t.findall(".//x:seg-source/x:mrk", ns)
c = Counter(etree.QName(ch).localname for m in mrks for ch in m)
print("direct children of mrk:", c)
c2 = Counter(etree.QName(ch).localname for m in mrks for ch in m.iter())
print("all descendants:", c2)
