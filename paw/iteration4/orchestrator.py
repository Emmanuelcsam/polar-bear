# orchestrator.py
import pkgutil, importlib
mods=[m.name for m in pkgutil.iter_modules(['.'])]
for m in mods: importlib.import_module(m.replace('.py',''))
print('Loaded modules:',mods)

