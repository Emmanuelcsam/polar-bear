#!/usr/bin/env python3
import re

# Read the original file
with open('core/main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace Unicode emojis with ASCII equivalents
replacements = {
    'Ì∫Ä': '[START]',
    'ÌæØ': '[TARGET]',
    'Ì≥ö': '[TRAINING]',
    '‚úÖ': '[SUCCESS]',
    '‚ùå': '[ERROR]',
    '‚ö†Ô∏è': '[WARNING]',
    'Ì¥ß': '[CONFIG]',
    'Ì≤æ': '[SAVE]',
    'Ì≥ä': '[METRICS]',
    'Ì¥Ñ': '[PROCESSING]'
}

for emoji, ascii_replacement in replacements.items():
    content = content.replace(emoji, ascii_replacement)

# Write the modified content back
with open('core/main.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed Unicode emoji characters with ASCII equivalents")
