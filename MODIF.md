[src/bot/main.py] -> "Nouvelles options --until-close (une séance puis arrêt, ou arrêt immédiat sans séance ce jour-là) et --log-dir. Le capital est relevé dans le journal à chaque tick, marché ouvert. La mise en veille du PC est bloquée pendant une séance planifiée. Sans option, le comportement ne change pas. Une panne de l'API ne fait jamais croire la séance terminée."

[src/bot/execution/broker.py] -> "MarketClock gagne next_open, next_close et is_session_over, calculé sur le jour de New York. Les jours fériés et les heures d'été viennent de l'horloge d'Alpaca, sans calendrier en dur."

[src/bot/journal/store.py] -> "Nouvelle table equity (capital et cash par séance, le dernier relevé l'emporte), avec record_equity, equity_history et last_completed_cycle. Mode lecture seule pour les outils de suivi : le journal ne peut alors être ni créé ni modifié."

[src/bot/logs.py] -> "Les logs JSON vont aussi dans un fichier quand il est demandé. Chaque ligne est vidée aussitôt. Une tâche planifiée n'a pas de console : sans fichier, ses logs seraient perdus."

[src/bot/power.py] -> "Nouveau. Empêche la mise en veille de Windows pendant une séance, avec SetThreadExecutionState (bibliothèque standard, aucune dépendance). Ailleurs que sous Windows, il ne fait rien."

[src/bot/status.py] -> "Nouveau. Commande python -m bot.status, en lecture seule. Elle affiche : le capital, l'exposition, la variation du jour et le drawdown face aux seuils du kill switch ; l'état du kill switch ; les positions, avec leur poids visé et les positions hors univers signalées ; le dernier cycle et les séances manquées ; les ordres du dernier cycle ; les erreurs du log. Code de sortie 2 si le kill switch est déclenché."

[src/bot/config.py, config/strategy.yaml] -> "Nouveau réglage live.log_dir (logs). Le commentaire de signal_lookback_days est corrigé : il parlait encore de slow_window."

[audit/report.py] -> "Le texte des limites décrit maintenant le rééquilibrage réel de la stratégie, au lieu de « pas de rééquilibrage », faux pour V1."

[scripts/install_task.ps1, scripts/uninstall_task.ps1] -> "Nouveaux. Installent ou retirent la tâche planifiée TradeBotPaper. Elle se lance du lundi au vendredi à 14h15 (heure de Paris), avec réveil du PC, rattrapage d'un lancement manqué, une seule instance et une limite de 9 h, dans la session de l'utilisateur, sans mot de passe stocké. Les erreurs Python imprévues vont dans logs/stderr.log."

[docs/paper_trading.md] -> "Nouveau. Mode d'emploi de l'étape 9 : préparation du compte, tests, lancement en veille, suivi, procédure en cas de kill switch, arrêt et limites."

[CLAUDE.md] -> "Ajout des commandes --dry-run, --until-close, bot.status et de l'installation de la tâche."

[.gitignore] -> "Le dossier logs/ n'est pas commité."

[tests/test_main.py] -> "Nouveaux tests : attente avant l'ouverture puis arrêt après la clôture, arrêt immédiat un jour férié, panne de l'API pendant la séance, kill switch toujours signalé par le code 2, relevé du capital, fichier de log de main."

[tests/test_operations.py, tests/test_status.py] -> "Nouveaux. Fin de séance selon le jour de New York, journal en lecture seule, logs console et fichier, mise en veille, et commande status (poids visés, ordres, séances manquées, positions hors univers, mise en forme, lecture sans écriture, codes de sortie)."
