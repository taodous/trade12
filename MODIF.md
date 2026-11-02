[config/strategy.yaml] -> "Ajout des sections backtest et targets. La fin de l'in-sample est fixée au 2022-12-31 avant tout résultat : les séances suivantes sont réservées à la validation finale. Coûts retenus : 0,01 % de frais et 0,1 % de slippage par ordre."

[src/bot/config.py] -> "Ajout de BacktestConfig et PerformanceTargets, validés au chargement. Le bot refuse de démarrer si ces sections manquent, si le benchmark est hors univers ou si la coupure précède le début des données."

[src/bot/data/store.py] -> "Ajout de load_column pour charger n'importe quelle colonne, dont les ouvertures nécessaires au backtest. load_closes garde le même comportement."

[audit/backtest.py] -> "Nouveau. Backtest net de coûts avec exécution à l'ouverture de t+1, à 10 % du capital par ETF, comparé à SPY sur la même période. Ne peut pas lire l'out-of-sample : les données sont coupées avant tout calcul. N'applique ni le kill switch ni les contrôles de bot.risk. Sans rééquilibrage, un achat peut être réduit ou ignoré quand le cash manque."

[audit/metrics.py] -> "Nouveau. Calcule les métriques sur 252 séances par an et un taux sans risque de 0, compare aux objectifs critère par critère, signale les résultats suspects et repère les séances où le kill switch se serait déclenché."

[audit/registry.py] -> "Nouveau. Inscrit chaque nouvelle variante testée et son premier résultat, pour tenir compte du biais de tests multiples. Relancer une variante identique ne l'ajoute pas."

[audit/variants.jsonl] -> "Nouveau, à commiter. Registre des variantes, qui contient pour l'instant la stratégie de référence 50/200."

[pyproject.toml] -> "Ajout du module audit dans la mesure de couverture. Le seuil de 80 % s'y applique maintenant aussi."

[tests/test_audit_backtest.py] -> "Nouveau. Vérifie l'exécution à t+1 et les coûts sur des exemples calculés à la main, et que modifier les prix de l'out-of-sample ne change rien au rapport."

[tests/test_audit_metrics.py] -> "Nouveau. Vérifie les métriques à la main, les seuils exacts de chaque critère et la relecture du kill switch."

[tests/test_config.py] -> "Ajout des tests sur le benchmark et la date de coupure."

[MODIF.md] -> "Réécrit pour l'étape 5. Le récapitulatif de l'étape 4 est effacé, comme le prévoit la règle."
