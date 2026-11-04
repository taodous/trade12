[src/bot/main.py] -> "Nouveau. Boucle principale : kill switch, suivi des ordres, signaux, risque, journal puis envoi, une fois par séance. Options --once, --dry-run et --reset-kill-switch. Un lancement sans option envoie de vrais ordres sur le compte paper et doit rester actif pour suivre le kill switch."

[src/bot/execution/broker.py] -> "Nouveau. Seul point de contact avec l'API d'ordres. Un identifiant stable par ordre et par séance empêche les doublons, y compris après un redémarrage ou une réponse perdue. Un refus du broker est journalisé au lieu d'arrêter le cycle."

[src/bot/execution/planner.py] -> "Nouveau. Traduit les signaux en ordres, les ventes en premier, à 10 % du capital en parts entières. Une donnée manquante ne produit jamais d'ordre : une position n'est donc pas vendue à cause d'un trou de données."

[src/bot/execution/client.py] -> "Ajout de last_equity dans l'état du compte, nécessaire au contrôle de la perte journalière."

[src/bot/journal/store.py] -> "Nouveau. Journal SQLite de chaque décision : signal, données d'entrée, décision du risque, ordre, prix d'exécution et slippage observé. La décision est écrite avant l'envoi : si l'écriture échoue, l'ordre ne part pas."

[src/bot/data/market.py] -> "Ajout des derniers prix IEX et de AlpacaMarketData, qui recharge les clôtures SIP à chaque cycle. Une clôture manquante reste vide et aucune valeur n'est reportée."

[src/bot/risk/checks.py] -> "Un prix peut maintenant être en avance sur l'horloge dans la limite de max_clock_skew_seconds. Sans cette tolérance, l'écart d'horloge pouvait faire refuser des prix valides."

[src/bot/risk/state.py] -> "Ajout d'un stockage en mémoire, pour qu'un --dry-run ne modifie jamais l'état réel du risque."

[src/bot/config.py] -> "Ajout de LiveConfig et max_clock_skew_seconds. Ces réglages deviennent obligatoires dans strategy.yaml : le bot refuse de démarrer s'ils manquent."

[config/strategy.yaml] -> "Ajout de la section live (journal, fréquence, pannes tolérées, historique du signal, flux de prix) et de max_clock_skew_seconds à 5 secondes."

[tests/test_main.py] -> "Nouveau. Vérifie la boucle complète avec un faux broker : doublons, redémarrage, ordre refusé, panne réseau, kill switch, trous de données et simulation."

[tests/test_execution_units.py] -> "Nouveau. Vérifie le planificateur, le broker et le journal séparément."

[tests/test_data_market.py] -> "Ajout des tests sur les derniers prix et les clôtures du bot."

[tests/test_execution_client.py] -> "Ajout de last_equity au compte de test."

[tests/test_risk_checks.py] -> "Ajout du test de tolérance d'horloge et de ses limites."

[tests/test_risk_manager.py] -> "Ajout de max_clock_skew_seconds aux limites de test."

[tests/test_config.py] -> "Ajout de max_clock_skew_seconds aux limites de test."

[tests/test_audit_metrics.py] -> "Ajout de max_clock_skew_seconds aux limites de test."

[tests/__init__.py] -> "Nouveau, vide. Fait de tests un paquet pour partager les faux objets entre fichiers de test."

[MODIF.md] -> "Réécrit pour l'étape 6. Le récapitulatif de l'étape 5 est effacé, comme le prévoit la règle."
