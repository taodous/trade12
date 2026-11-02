[config/strategy.yaml] -> "Ajout de la section strategy avec les fenêtres 50 et 200. Valeurs conventionnelles non optimisées : toute modification devient une nouvelle variante à inscrire au registre d'audit."

[src/bot/config.py] -> "Ajout de MovingAverageParams, qui refuse une fenêtre rapide supérieure ou égale à la lente. La section strategy devient obligatoire dans strategy.yaml : le bot refuse de démarrer si elle manque ou est invalide."

[src/bot/strategy/base.py] -> "Nouveau. Définit l'interface Strategy et les signaux LONG et FLAT. Toute stratégie future s'y conforme, pour que le backtest et le paper trading utilisent le même code."

[src/bot/strategy/moving_average.py] -> "Nouveau. Stratégie de référence, à l'achat tant que la moyenne 50 est au-dessus de la moyenne 200, sinon hors marché. Reste hors marché pendant les 199 premières séances ou après un prix manquant. Un signal de la séance t ne peut être exécuté qu'à partir de t+1 : l'étape 5 doit respecter ce décalage."

[src/bot/data/store.py] -> "Ajout de load_closes, qui regroupe les clôtures par symbole sans combler les trous et vérifie l'empreinte de chaque fichier. C'est le lien entre les données stockées et la stratégie."

[tests/test_strategy_moving_average.py] -> "Nouveau. Vérifie les signaux sur un exemple calculé à la main, l'absence de look-ahead (troncature et prix futurs modifiés), l'indépendance à l'échelle des prix, les prix manquants et le refus de séances mal triées."

[tests/test_config.py] -> "Ajout des tests de validation des fenêtres."

[tests/test_data_store.py] -> "Ajout des tests de load_closes : alignement sans comblement, liste vide et fichier modifié."

[MODIF.md] -> "Réécrit pour l'étape 3. Le récapitulatif de l'étape 2 est effacé, comme le prévoit la règle."
