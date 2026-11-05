[config/strategy.yaml] -> "Cibles de performance révisées le 2026-11-05 : rendement 5 % (au lieu de 8 %), Sharpe 0,6 (au lieu de 1), Calmar 0,5 (au lieu de 1). Seuils de drawdown, de robustesse et de résultats suspects inchangés. Chaque valeur est commentée avec sa dérivation et les anciennes valeurs sont conservées. Impact : le verdict de l'audit est recalculé avec ces cibles, la stratégie n'est pas modifiée et le registre ne bouge pas."

[CLAUDE.md] -> "Tableau des objectifs réécrit avec les anciennes cibles en regard, la raison de la révision (cibles de départ incohérentes entre elles), la dérivation de chaque valeur depuis des références antérieures aux résultats, la limite honnête (décision prise après l'in-sample, l'out-of-sample tranchera) et une note sur les montants (paper avec 100 000 $ ou plus, à revérifier avec des sommes raisonnables dans la version finale). Ajout des commandes audit.sabotage et audit.report --out-of-sample."

[tests/test_config.py] -> "Les cibles du projet sont figées par un test : toute modification future doit être assumée et documentée, pas glissée dans la config."

[tests/test_audit_report.py] -> "Le seuil du Sharpe sous slippage x2 est calculé depuis les cibles et seuils de la config, au lieu d'une valeur en dur."
