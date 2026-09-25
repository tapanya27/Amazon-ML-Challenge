import sys
import unicodedata

sys.stdout.reconfigure(encoding='utf-8')

text = 'राम'
print('Original:', repr(text))
nfkd = unicodedata.normalize('NFKD', text)
print('NFKD:', repr(nfkd))
for c in text:
    print(f'  {repr(c)} cat={unicodedata.category(c)} name={unicodedata.name(c, "")}')
print()
print('NFKD chars:')
for c in nfkd:
    print(f'  {repr(c)} cat={unicodedata.category(c)} name={unicodedata.name(c, "")}')
