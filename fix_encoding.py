"""Fix encoding issues in metric_validator.py"""

with open('core/symbolic/metric_validator.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Fix broken unicode characters
replacements = {
    '\ufffd\ufffd\ufffd': '--',
    '\ufffd': '',
    'â€"': '--',
    'â€™': "'",
    'â€': '"',
}

for bad, good in replacements.items():
    content = content.replace(bad, good)

with open('core/symbolic/metric_validator.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Encoding fixed successfully')
