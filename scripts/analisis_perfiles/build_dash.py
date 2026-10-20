import json, re, shutil
d = open('dashboard_data.json', encoding='utf-8').read().replace('</', r'<\/')
t = open('dash_template.html', encoding='utf-8').read()
open('dashboard.html', 'w', encoding='utf-8').write(t.replace('%%DATA%%', d))
js = re.search(r'<script>(.*)</script>', t.replace('%%DATA%%', 'null'), re.S).group(1)
open('dash_check.js', 'w', encoding='utf-8').write(js)
print('ok', len(t) // 1024, 'KB plantilla')
