[src/bot/alerts.py] -> "Nouveau. Alertes sur deux canaux : notification Windows (PowerShell, sans dépendance, texte passé par l'environnement pour ne jamais être exécuté) et téléphone via ntfy (JSON UTF-8, le canal secret est dans le corps et jamais dans l'URL ni les logs). Une alerte qui échoue est journalisée sans bloquer le bot. Commande de test : python -m bot.alerts --test."

[src/bot/main.py] -> "Le bot envoie une alerte critique au kill switch (puis à chaque lancement tant qu'il n'est pas réarmé), après des pannes répétées de l'API et en cas d'échec au démarrage. Il envoie aussi un avertissement groupé pour les ordres refusés par le risque, et un résumé à chaque rééquilibrage. Aucune alerte en dry-run. Les contrôles de risque ne changent pas."

[src/bot/watchdog.py] -> "Nouveau. Chien de garde : si le marché est ouvert et que le bot ne donne plus signe de vie, alerte critique. Couvre le plantage, le PC redémarré, la tâche arrêtée. Lecture seule."

[src/bot/heartbeat.py] -> "Nouveau. Signe de vie du bot (dernier log et dernier relevé du capital), sorti du dashboard pour être partagé avec le chien de garde."

[src/bot/dashboard/data.py] -> "Utilise bot.heartbeat au lieu de son propre code de signe de vie. Affichage inchangé."

[src/bot/config.py, config/strategy.yaml] -> "Nouvelle section alerts (toast Windows, serveur ntfy, délai, alertes de refus et de rééquilibrage). NTFY_TOPIC, facultatif, est lu dans .env. Une variable vide de .env vaut absente, pour qu'un NTFY_TOPIC= copié de .env.example ne bloque pas le bot."

[.env.example] -> "Ajout de NTFY_TOPIC, facultatif, avec la consigne de choisir un nom imprévisible."

[scripts/install_task.ps1, scripts/uninstall_task.ps1] -> "Installent et retirent aussi la tâche TradeBotWatchdog : toutes les 30 minutes de 14h45 à 22h45 en semaine, sans fenêtre, avec réveil du PC. À relancer pour l'installer."

[docs/paper_trading.md, CLAUDE.md] -> "Section Alertes : événements, chien de garde, mise en place de ntfy sur le téléphone, test. Nouvelles commandes bot.alerts --test et bot.watchdog."

[pyproject.toml] -> "Les modules d'alerte, de signe de vie et de chien de garde entrent dans le calcul de couverture."

[tests/test_alerts.py] -> "Nouveau. ntfy (JSON, canal hors URL, statut en erreur), toast (texte par l'environnement), alerteur qui survit à un canal en panne sans afficher le canal, choix des canaux, variable vide ignorée, commande de test, chien de garde (marché fermé, bot actif ou silencieux, signe de vie lu dans le journal)."

[tests/test_main.py] -> "Nouveaux tests : résumé du rééquilibrage, refus groupés, alertes désactivables, kill switch, pannes répétées (et pas d'alerte en --once), échec au démarrage, aucune alerte en dry-run."

[tests/test_dashboard.py] -> "Tests du signe de vie déplacés vers bot.heartbeat."
