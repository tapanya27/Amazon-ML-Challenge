import sys
import re

sys.stdout.reconfigure(encoding='utf-8')
punct_pattern = re.compile(r'[^\w\s&]', re.UNICODE)
text = 'राम'
res = punct_pattern.sub(' ', text)
print('Regex sub result:', repr(res))
