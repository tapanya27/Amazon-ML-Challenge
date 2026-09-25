import sys
import unicodedata

sys.stdout.reconfigure(encoding='utf-8')
text = 'राम मार्केटिंग प्रा. लि. (B+ Retail) & Co!'

def strip_punct(s):
    res = []
    for c in s:
        if c == '&':
            res.append(c)
        elif unicodedata.category(c).startswith(('P', 'S')):
            res.append(' ')
        else:
            res.append(c)
    return ''.join(res)

print(repr(strip_punct(text)))
