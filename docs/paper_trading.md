# Paper trading en veille (étape 9)

Le bot fait tourner la stratégie V1 (variante `aa32f0c7ab3e`) sur le compte **paper** Alpaca, chaque jour de bourse, sans surveillance. Il doit tourner au moins 3 mois pour mesurer l'écart entre le backtest et l'exécution réelle. La stratégie ne change pas pendant cette période.

Toutes les commandes se lancent depuis la racine du projet.

## 1. Préparer le compte

1. Sur https://app.alpaca.markets, section Paper Trading, remettre le compte à zéro avec un capital d'au moins 100 000 $. Il ne doit rester aucune position ni aucun ordre ouvert. Les positions hors univers sont signalées par `bot.status`, mais elles faussent l'exposition et la comparaison au backtest.
2. Copier `.env.example` vers `.env` et renseigner les clés du compte **paper**. Ne jamais commiter ce fichier.
3. Vérifier la connexion, sans envoyer d'ordre :

   ```bash
   uv run python -m bot.check_connection
   ```

## 2. Tester avant de lancer

```bash
uv run pytest                        # tests unitaires et scénarios, sans réseau
uv run python -m bot.main --dry-run  # un cycle sur le vrai compte, aucun ordre envoyé
uv run python -m bot.status          # état du compte, du kill switch et du journal
```

Ce qu'il faut voir :
- **Dry-run.** Les ordres simulés apparaissent dans les logs (`order_simulated`). Le journal de simulation est `data/journal.dry-run.db` : il ne touche ni au vrai journal ni à l'état du risque. Hors séance, le module risque refuse les prix trop anciens. C'est normal.
- **Status.** Le kill switch est armé, il n'y a aucune position, et le journal est absent tant que le bot n'a pas tourné.

## 3. Lancer en veille

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_task.ps1
```

L'installation crée deux tâches : `TradeBotPaper`, le bot, et `TradeBotWatchdog`, le chien de garde (voir « Alertes »). La tâche `TradeBotPaper` démarre du lundi au vendredi à 14h15, heure de Paris. Voici ce qu'elle fait :
- elle lance `bot.main --until-close` ;
- le bot attend l'ouverture de New York ;
- il rééquilibre si c'est sa première séance du mois, et vérifie le kill switch chaque minute ;
- il s'arrête après la clôture ;
- un jour férié, il s'arrête aussitôt.

Conditions côté PC :
- **Session ouverte.** Le PC doit rester allumé, ou en veille, mais pas éteint, et ta session Windows doit être ouverte. Elle peut être verrouillée.
- **Réveil.** La tâche réveille le PC s'il dort. Il faut que les « minuteurs de réveil » soient autorisés : Options d'alimentation, puis Paramètres avancés, puis Veille, puis Autoriser les minuteurs de réveil.
- **Pas de veille pendant la séance.** Le bot l'empêche lui-même, l'écran peut s'éteindre.
- **Aucune fenêtre.** La tâche tourne sans fenêtre console, pour qu'on ne puisse pas la fermer par erreur : fermer cette fenêtre arrêtait le bot. Pour vérifier qu'il tourne, utilise le dashboard ou `Get-ScheduledTask TradeBotPaper` (état `Running`).
- **Lancement manqué ou redémarrage.** Si le PC était éteint à 14h15, ou s'il a été éteint ou redémarré pendant la séance, la tâche repart à l'ouverture de ta session. Le bot reprend alors la séance en cours. Un week-end ou après la clôture, il s'arrête aussitôt.
- **Batterie.** Sur batterie, Windows n'autorise pas les minuteurs de réveil par défaut. Laisse le PC branché.

Pour essayer la tâche tout de suite :

```powershell
Start-ScheduledTask TradeBotPaper
Get-ScheduledTaskInfo TradeBotPaper   # LastTaskResult : 0 = normal, 2 = kill switch, 1 = erreur
```

Hors séance, le bot s'arrête en quelques secondes et écrit son fichier de log.

## 4. Suivre

### Dashboard

```bash
uv run python -m bot.dashboard
```

Le dashboard s'ouvre dans le navigateur sur http://localhost:8501. On peut le lancer et le fermer (Ctrl+C) à tout moment : il est indépendant du bot, en lecture seule, et n'envoie aucun ordre. Il n'écoute que sur ce PC.

**Partie live, rafraîchie toutes les 15 s :**
- le capital, la variation du jour, l'exposition, le cash et le drawdown ;
- l'état du bot, actif ou arrêté, d'après son dernier signe de vie, et l'état du marché ;
- le kill switch, avec une jauge pour la perte du jour face à 2 % et une pour le drawdown face à 10 ;
- quatre onglets :
  - **Aujourd'hui** : le capital de la séance en intraday ;
  - **Positions** : P&L latent, poids réel et poids visé ;
  - **Décisions** : les 50 dernières, avec la décision du risque et le slippage ;
  - **Logs** : erreurs et avertissements.

**Performance depuis le début, rafraîchie toutes les 5 min :**
- la courbe du capital face à SPY, avec le même capital de départ ;
- le drawdown, avec le seuil d'arrêt ;
- les métriques face à SPY et aux cibles de CLAUDE.md.

Tant qu'il y a moins de 63 séances (3 mois), un avertissement rappelle que les chiffres annualisés ne sont pas significatifs.

### Alertes

Le bot prévient tout seul, sur ce PC (notification Windows) et sur le téléphone (application ntfy) :

| Événement | Gravité |
|---|---|
| Kill switch déclenché, puis à chaque lancement tant qu'il n'est pas réarmé | critique |
| Bot arrêté après des pannes répétées de l'API, ou en échec au démarrage | critique |
| Bot sans signe de vie pendant une séance ouverte (chien de garde) | critique |
| Ordres refusés par le module risque, regroupés par cycle | avertissement |
| Rééquilibrage effectué, avec la liste des ordres | info |

Le chien de garde est la tâche `TradeBotWatchdog`. Elle vérifie toutes les 30 minutes, de 14h45 à 22h45 en semaine, que le bot donne signe de vie. Hors séance, elle ne fait rien. Elle couvre ce que le bot ne peut pas signaler lui-même : un plantage, un PC redémarré, une tâche arrêtée.

**Mise en place sur le téléphone :**
1. Installer l'application ntfy (Android ou iOS).
2. Choisir un nom de canal long et imprévisible : quiconque le connaît lit les alertes. Pour en générer un :

   ```bash
   uv run python -c "import secrets; print('tradebot-' + secrets.token_urlsafe(16))"
   ```

3. L'ajouter dans `.env` : `NTFY_TOPIC=tradebot-...`.
4. S'abonner à ce canal dans l'application.
5. Tester les deux canaux :

   ```bash
   uv run python -m bot.alerts --test
   ```

Les messages passent par le serveur public ntfy.sh. Ils contiennent des montants du compte paper, mais jamais de clé. Une alerte qui échoue est notée dans les logs, et elle ne bloque jamais le bot.

### En ligne de commande

```bash
uv run python -m bot.status
```

La commande affiche :
- le capital, l'exposition, la variation du jour et le drawdown, avec les seuils du kill switch ;
- les positions, avec leur poids réel et le poids visé au dernier cycle ;
- les séances où le bot n'a pas tourné ;
- les ordres du dernier cycle et leur sort ;
- les erreurs du jour.

Les logs sont dans `logs/` :
- `bot-AAAA-MM-JJ.jsonl` : un fichier JSON par jour ;
- `stderr.log` : les erreurs Python imprévues.

Le journal `data/journal.db` contient chaque décision et le capital de chaque séance.

Rythme conseillé :
- **Chaque semaine :** un `bot.status`.
- **À chaque début de mois :** vérifier les ordres du rééquilibrage.

## 5. Si le kill switch se déclenche

Le kill switch se déclenche sur une perte de 2 % dans la journée, ou une baisse de 10 % depuis le plus haut. Le bot annule alors les ordres, ferme les positions et s'arrête. La tâche finit avec le code 2, et chaque lancement suivant s'arrête aussitôt tant qu'il n'est pas réarmé.

1. Comprendre ce qui s'est passé :
   - `bot.status` ;
   - le fichier de log du jour ;
   - les décisions `kill_switch` dans le journal.
2. Décider : une reprise est une **décision humaine**, jamais automatique. V1 a fait 10,9 % de drawdown en in-sample : un arrêt sur le seuil de 10 % est donc possible sans que le bot soit en panne.
3. Réarmer, si la reprise est décidée :

   ```bash
   uv run python -m bot.main --reset-kill-switch
   ```

## 6. Arrêter

```powershell
powershell -ExecutionPolicy Bypass -File scripts\uninstall_task.ps1
```

Les positions du compte paper restent en place. Pour un arrêt d'urgence pendant une séance, on peut aussi fermer les positions depuis le tableau de bord Alpaca, ou lancer `Stop-ScheduledTask TradeBotPaper`.

## Limites connues

- **Premier mois.** Lancé en milieu de mois, le bot rééquilibre dès sa première séance. Le backtest ne le fait qu'en début de mois : le premier mois diffère un peu.
- **PC indisponible.** Un jour sans PC, le kill switch n'est pas surveillé. Si c'était le jour du rééquilibrage, il a lieu à la séance suivante du mois.
- **Exécutions simulées.** Alpaca simule les exécutions paper, ce qui sous-estime le slippage réel. Les prix temps réel viennent du flux IEX.
- **Alertes dépendantes du PC.** Les alertes partent de ce PC : s'il est éteint, ni le bot ni le chien de garde ne peuvent prévenir. Une séance sans aucune notification de rééquilibrage, en début de mois, est donc un signal à vérifier.
