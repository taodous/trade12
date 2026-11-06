[src/bot/dashboard/app.py] -> "Nouveau. Page Streamlit du dashboard, affichage seulement. Partie live, rafraîchie toutes les 15 s : capital, exposition, drawdown, état du bot et du marché, kill switch avec deux jauges, et des onglets (séance intraday, positions avec P&L latent et poids visé, 50 dernières décisions avec slippage, logs). Partie historique, rafraîchie toutes les 5 min : capital face à SPY, drawdown avec le seuil d'arrêt, métriques face à SPY et aux cibles, avec un avertissement de taille d'échantillon avant 63 séances. Une panne d'API affiche un message dans sa section, sans faire tomber la page."

[src/bot/dashboard/data.py] -> "Nouveau. Toutes les données du dashboard, sans Streamlit, en lecture seule : état du bot (réutilise bot.status), historique du capital Alpaca, SPY normalisé, drawdown, métriques (réutilise audit.metrics), tableau des décisions, slippage moyen. Le signe de vie du bot combine le dernier log et le dernier relevé du capital, car le bot ne logue plus une fois son cycle du jour fait. Les barres de SPY s'arrêtent à la veille, car l'abonnement gratuit refuse le SIP récent : le point du jour vient du dernier prix."

[src/bot/dashboard/__main__.py] -> "Nouveau. Commande python -m bot.dashboard : lance Streamlit sur localhost uniquement, sans télémétrie, et ouvre le navigateur."

[src/bot/dashboard/__init__.py] -> "Nouveau paquet du dashboard."

[.streamlit/config.toml] -> "Nouveau. Serveur local uniquement, sans télémétrie, thème clair."

[src/bot/execution/broker.py] -> "Ajout de position_details (prix moyen, prix courant, P&L latent) et de portfolio_history (historique du capital, sans les fenêtres inconnues ou à 0), avec backoff. Lecture seule : aucun changement sur les ordres."

[src/bot/status.py] -> "collect_status est découpé : build_status reçoit la configuration et le client, ce qui permet au dashboard de réutiliser ses clients. Le comportement de bot.status ne change pas."

[src/bot/config.py, config/strategy.yaml] -> "Nouvelle section dashboard : rafraîchissement, cache de l'historique, nombre de décisions, timeframe intraday, seuil de signe de vie, port, seuil d'échantillon des métriques."

[scripts/install_task.ps1] -> "La tâche tourne sans fenêtre console (conhost --headless). Le 2026-11-06, la fenêtre visible a été fermée, ce qui a arrêté le bot en pleine séance (code 0xC000013A). La tâche a été réinstallée et relancée."

[pyproject.toml, uv.lock] -> "Dépendances ajoutées : streamlit, et plotly, déjà installé via vectorbt mais désormais importé directement. Override mypy pour plotly, qui n'a pas de types. Le dashboard entre dans le calcul de couverture."

[docs/paper_trading.md, CLAUDE.md] -> "Section Dashboard et commande python -m bot.dashboard. La tâche sans fenêtre est documentée."

[tests/test_dashboard.py] -> "Nouveau. 24 tests : broker (détails des positions, historique), calculs (signe de vie, capital par séance, benchmark aligné, drawdown, métriques, décisions, slippage), chargement avec de faux clients, journal jamais créé, page rendue avec AppTest (cas normal, kill switch déclenché, panne d'API et historique vide), lanceur sur localhost."

[tests/test_main.py] -> "Le faux client Alpaca sait renvoyer un historique du capital."
