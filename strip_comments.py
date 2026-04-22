import re

with open('self_pruning_network.py', 'r') as f:
    code = f.read()

# Remove triple quote docstrings
code = re.sub(r'"""[\s\S]*?"""', '', code)
code = re.sub(r"'''[\s\S]*?'''", '', code)

# Remove full line comments
lines = code.split('\n')
clean_lines = []
for line in lines:
    stripped = line.strip()
    if stripped.startswith('#'):
        continue
    # Remove inline comments, but be careful with strings
    if '#' in line and not ('"' in line or "'" in line):
        line = line.split('#')[0].rstrip()
    elif '#' in line:
        pass # keep it for safety if it might be in a string (though unlikely)
        line = line.split('#')[0].rstrip() # Actually just strip all '#' since no strings have #
    clean_lines.append(line)

# Remove multiple blank lines
final_code = re.sub(r'\n\s*\n\s*\n', '\n\n', '\n'.join(clean_lines))

with open('self_pruning_network.py', 'w') as f:
    f.write(final_code)
