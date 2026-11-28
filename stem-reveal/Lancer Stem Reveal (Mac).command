#!/bin/bash
# Double-click to start Stem Reveal (macOS). Also works on Linux: bash "Lancer Stem Reveal (Mac).command"
cd "$(dirname "$0")" || exit 1
PY=""
for c in python3.12 python3.13 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python n'est pas installé. Installe Python 3.12 depuis https://www.python.org/downloads/ puis relance."
  open "https://www.python.org/downloads/" 2>/dev/null
  read -r -p "Appuie sur Entrée pour fermer…"; exit 1
fi
if [ ! -f .venv/installed.ok ]; then
  echo ""; echo "  Première installation : 5 à 10 minutes, une seule fois."; echo ""
  rm -rf .venv
  "$PY" -m venv .venv && .venv/bin/python -m pip install --upgrade pip && \
  .venv/bin/python -m pip install -r requirements.txt && touch .venv/installed.ok || {
    echo "L'installation a échoué (vérifie ta connexion internet) puis relance."
    read -r -p "Appuie sur Entrée pour fermer…"; exit 1; }
fi
if ! cmp -s requirements.txt .venv/requirements.installed; then
  echo "Mise à jour des composants…"
  .venv/bin/python -m pip install -r requirements.txt && cp requirements.txt .venv/requirements.installed
fi
.venv/bin/python app.py
