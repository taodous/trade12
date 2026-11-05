"""Protocole d'audit de la stratégie."""

import os

# Les graphiques de quantstats passent par matplotlib. Sans affichage non interactif, Windows ouvre
# une fenêtre qui bloque le rapport. Fixé avant tout import de quantstats.
os.environ.setdefault("MPLBACKEND", "Agg")
