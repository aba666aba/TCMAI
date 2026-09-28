"""验证PPT小数位分布"""
import re
from pathlib import Path

text = Path(r'E:\TOM\TCMAI\ppt_text_dump.txt').read_text(encoding='utf-8')

pcts = re.findall(r'\d+\.\d+%', text)
print('=== All X.X% values ===')
for v in sorted(set(pcts), key=lambda x: float(x.replace('%', ''))):
    count = pcts.count(v)
    print(f'  {v} ({count}x)')

from collections import Counter
digits = Counter()
for v in pcts:
    m = re.match(r'\d+\.(\d)%', v)
    if m:
        digits[int(m.group(1))] += 1

print()
print('=== Last digit distribution ===')
for d in range(10):
    bar = '#' * digits.get(d, 0)
    print(f'  .{d}: {digits.get(d, 0):2d} {bar}')
print(f'  Total: {sum(digits.values())}')
