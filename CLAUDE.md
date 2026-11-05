# CLAUDE.md

Guide de travail pour Claude Code sur ce projet. À lire avant toute modification.

## Objectif du projet

Construire un **bot de trading rentable** en Python, testé en **paper trading uniquement** sur Alpaca.

La philosophie est celle d'une **croissance régulière et maîtrisée**, pas de la performance spectaculaire. On cherche une courbe de capital qui monte de façon stable, avec des pertes contenues, plutôt que des gains de +100 % obtenus au prix de risques excessifs. Entre une stratégie à +40 % avec 35 % de drawdown et une à +10 % avec 6 % de drawdown, on choisit la seconde.

L'**audit** fait partie intégrante du projet : c'est le garde-fou qui vérifie que la rentabilité est réelle, durable et obtenue sans prise de risque déraisonnable. Une stratégie ne passe à l'étape suivante que si elle valide les critères ci-dessous.

Objectifs secondaires :
- Code Python de niveau production (typé, testé, lisible, modulaire)
- Projet présentable sur GitHub comme démonstration de compétences

## Objectifs de performance (révisés le 2026-11-05)

Mesurés **out-of-sample**, frais et slippage inclus :

| Critère | Cible | Cible de départ |
|---|---|---|
| Rendement annualisé | au moins 5 % | 8 % à 15 % |
| Max drawdown | inférieur à 10 % (seuil d'alerte), 15 % (seuil de rejet) | inchangé |
| Volatilité annualisée | inférieure à celle de SPY | inchangé |
| Sharpe ratio | au moins 0,6 | supérieur à 1 |
| Calmar ratio (rendement / max drawdown) | au moins 0,5 | supérieur à 1 |
| Mois positifs | au moins 60 % | inchangé |
| Comparaison à SPY | meilleur rendement ajusté du risque (Sharpe), même si le rendement brut est inférieur | inchangé |

**Pourquoi la révision.** Les cibles de départ étaient incohérentes entre elles. Une volatilité inférieure à celle de SPY et un drawdown sous 10 % limitent la volatilité à environ 8 %. Viser 8 à 15 % de rendement avec cette volatilité exige un Sharpe de 1 à 2, ce que les stratégies ETF longues, sans levier et diversifiées n'atteignent pas durablement (fourchette habituelle de 0,6 à 0,9).

**Comment elles sont dérivées.** Elles viennent de références fixées avant les résultats de V1, et non des chiffres de V1 :
- **Sharpe :** le bas de la fourchette 0,6 à 0,9, annoncée dans le plan de l'étape 8 avant tout résultat.
- **Rendement :** ce Sharpe multiplié par la volatilité visée de 8 % (0,6 x 8 % = 4,8 %), arrondi à 5 %.
- **Calmar :** ce rendement divisé par le seuil d'alerte du drawdown (5 % / 10 %).
- **Inchangés :** les seuils de drawdown, les critères de robustesse et les seuils de résultats suspects. Ils protègent contre le risque, on ne les assouplit pas.

**Limite honnête.** Cette décision a été prise après avoir vu les résultats in-sample de V1. Elle n'est donc pas totalement indépendante. Le garde-fou est l'out-of-sample, encore jamais ouvert : c'est lui qui tranchera.

**Montants.** Le paper trading se fait avec un capital important (100 000 $ ou plus) pour que l'arrondi en parts entières et la bande de rééquilibrage pèsent peu. Les cibles sont en pourcentage et ne dépendent pas du capital. Dans la version finale, avec des montants raisonnables, il faudra revérifier l'écart dû aux parts entières et aux frais fixes éventuels.

Des résultats nettement au-dessus de ces cibles (rendement supérieur à 30 % par an, Sharpe supérieur à 2) ne sont pas une victoire : ils sont **suspects par défaut** (surajustement, look-ahead, erreur de calcul) et doivent être investigués avant toute autre chose.

Préférer les approches reconnues pour leur régularité : diversification sur plusieurs actifs, dimensionnement des positions selon la volatilité, suivi de tendance ou retour à la moyenne sur ETF liquides, filtres de régime de marché. Éviter le levier, les penny stocks, les actifs illiquides et le trading haute fréquence.

## Règles absolues (non négociables)

1. **Paper trading uniquement.** Le code ne doit jamais pointer vers `api.alpaca.markets` (live). Toujours `paper=True` dans `TradingClient`. Toute fonction qui crée un client doit vérifier explicitement le mode paper et lever une exception sinon.
2. **Aucun secret dans le code.** Les clés API sont lues depuis `.env` (via `pydantic-settings`). `.env` est dans `.gitignore`. Ne jamais afficher, logger ou commiter une clé.
3. **La couche risque a le dernier mot.** Aucun ordre ne part sans être validé par le module `risk`. La stratégie propose, le risque dispose.
4. **Pas de look-ahead bias.** Une décision au temps t n'utilise que des données disponibles avant t. Toute feature doit être vérifiable sur ce point.
5. **Ne jamais modifier la stratégie pour améliorer un résultat de test out-of-sample.** Les données de test sont sacrées. Si un paramètre change, on le documente et on relance toute la validation.
6. Ne pas installer de dépendance sans l'ajouter à `pyproject.toml` et sans le signaler.

## Plateforme : Alpaca

- SDK : `alpaca-py` (pas l'ancien `alpaca-trade-api`, déprécié)
- Endpoint paper : `paper-api.alpaca.markets`
- Données : flux IEX gratuit par défaut. Garder en tête qu'il ne couvre qu'une partie du volume du marché.
- Heures de marché US : 15h30 à 22h00 heure de Paris (hors décalages d'heure d'été). Le bot doit vérifier l'état du marché via l'API (`get_clock`) avant d'agir.
- Respecter les rate limits de l'API (backoff exponentiel sur erreur 429).

## Stack technique

- Python 3.12+, gestion des dépendances avec `uv`
- `alpaca-py`, `pandas`, `numpy`
- `pydantic` et `pydantic-settings` pour la configuration et la validation des données
- `vectorbt` pour le backtesting
- `quantstats` pour les métriques et rapports de performance
- `pytest` (avec `pytest-cov`) pour les tests
- `ruff` pour le lint et le formatage, `mypy` en mode strict pour le typage
- `structlog` pour les logs structurés (JSON)

## Architecture

```
trading-bot/
├── CLAUDE.md
├── pyproject.toml
├── .env.example          # modèle sans vraies clés
├── config/
│   └── strategy.yaml     # paramètres de stratégie et limites de risque
├── src/bot/
│   ├── config.py         # chargement settings + validation
│   ├── data/             # récupération et nettoyage des données de marché
│   ├── strategy/         # génération de signaux (pure, sans effet de bord)
│   ├── risk/             # validation des ordres, limites, kill switch
│   ├── execution/        # envoi des ordres à Alpaca, gestion des erreurs
│   ├── journal/          # journalisation de chaque décision (SQLite)
│   └── main.py           # boucle principale
├── audit/
│   ├── backtest.py       # backtest historique
│   ├── walk_forward.py   # validation walk-forward
│   ├── monte_carlo.py    # robustesse sur l'ordre des trades
│   ├── sensitivity.py    # sensibilité aux paramètres
│   ├── stress_tests.py   # scénarios de crise et pannes
│   └── report.py         # génération du rapport d'audit
├── reports/              # rapports générés (HTML, PNG), non commités
└── tests/
```

Principes :
- Le module `strategy` est **pur** : il reçoit des données, retourne des signaux. Pas d'appel API, pas d'état caché. C'est ce qui permet d'utiliser exactement le même code en backtest et en paper trading.
- `execution` est le seul module qui parle à l'API d'ordres d'Alpaca.
- Chaque module dépend d'interfaces (protocoles), pas d'implémentations, pour pouvoir simuler l'API dans les tests.

## Gestion du risque (exigences minimales)

Toutes les valeurs sont configurables dans `config/strategy.yaml` :
- Taille maximale par position (en % du capital)
- Exposition totale maximale
- Perte journalière maximale : au-delà, **kill switch** qui ferme les positions et arrête le bot
- Drawdown maximal depuis le plus haut : au-delà, arrêt
- Nombre maximal d'ordres par jour (protection contre une boucle folle)
- Refus de tout ordre si les données sont périmées, manquantes ou aberrantes (variation de prix anormale)
- Pas de vente à découvert ni de levier sauf activation explicite dans la config

## Journalisation

Chaque décision est enregistrée en base SQLite avec : timestamp, symbole, signal produit, données d'entrée utilisées, décision du module risque (acceptée ou refusée, et pourquoi), ordre envoyé, prix d'exécution, slippage observé. L'audit doit pouvoir reconstituer pourquoi chaque trade a eu lieu.

## Protocole d'audit

L'audit sert de **porte de validation** : il confirme que la stratégie atteint les objectifs de performance de façon fiable. Chaque rapport se termine par un verdict explicite (validée, à retravailler, rejetée) critère par critère. Il couvre trois axes, chacun avec ses scripts dans `audit/`.

**1. Performance ajustée du risque**
- Sharpe, Sortino, Calmar, max drawdown et durée, profit factor, taux de réussite, gain moyen sur perte moyenne
- Comparaison obligatoire à un benchmark buy and hold (SPY)
- Intégration des frais et d'un slippage réaliste (le paper trading le sous-estime)

**2. Robustesse statistique**
- Séparation stricte in-sample / out-of-sample
- Validation walk-forward
- Monte Carlo sur l'ordre des trades (distribution des drawdowns possibles)
- Sensibilité des paramètres : la performance doit rester stable autour des valeurs choisies
- Tenir un registre de toutes les variantes testées (pour tenir compte du biais de tests multiples)

**3. Sûreté opérationnelle**
- Tests de stress : krach soudain, gap à l'ouverture, données manquantes, API indisponible, ordre rejeté, ordre en double
- Vérification que chaque limite de risque se déclenche bien
- Comparaison entre résultats de backtest et résultats paper sur la même période (écart d'exécution)

Le rapport final doit énoncer clairement ses limites (période testée, biais du paper trading, taille d'échantillon).

## Conventions de code

- Typage complet, `mypy --strict` doit passer
- Docstrings sur toutes les fonctions publiques
- Pas de nombre magique : tout paramètre va dans la config
- Gestion explicite des erreurs : pas de `except Exception: pass`
- Commits petits et explicites, un sujet par commit
- Rédaction (commentaires, docs, messages) : pas de tirets cadratins dans les phrases

## Tests

- Toute fonction de `strategy` et `risk` est testée unitairement
- Les appels à Alpaca sont simulés (mocks) dans les tests, jamais d'appel réseau
- Cible : 80 % de couverture minimum sur `strategy`, `risk` et `execution`
- Avant chaque commit : `ruff check`, `ruff format --check`, `mypy`, `pytest`

## Commandes

```bash
uv sync                          # installer les dépendances
uv run pytest                    # lancer les tests
uv run ruff check . && uv run ruff format .
uv run mypy src
uv run python -m bot.main        # lancer le bot (paper)
uv run python -m bot.main --dry-run      # un cycle sans envoi d'ordre
uv run python -m bot.main --until-close  # une séance puis arrêt (mode de la tâche planifiée)
uv run python -m bot.status      # état du bot en lecture seule (capital, kill switch, positions, logs)
powershell -ExecutionPolicy Bypass -File scripts\install_task.ps1  # paper en veille, voir docs/paper_trading.md
uv run python -m audit.report    # générer le rapport d'audit
uv run python -m audit.candidates  # compare les variantes pré-enregistrées (in-sample)
uv run python -m audit.sabotage  # test de sabotage, à relancer après tout changement de code
uv run python -m audit.report --out-of-sample   # ouvre l'out-of-sample (journalisé, à réserver à la validation finale)
```

## Feuille de route

1. Ossature du projet, config, connexion paper vérifiée (lecture du compte)
2. Récupération et stockage des données historiques
3. Une stratégie simple de référence (ex. croisement de moyennes mobiles) pour valider toute la chaîne
4. Module risque et ses tests
5. Backtest et premières métriques
6. Exécution paper et journalisation
7. Protocole d'audit complet avec verdict par critère
8. Itération vers une stratégie qui valide les objectifs de performance, en respectant les règles sur l'out-of-sample
9. Période de paper trading en conditions réelles (au moins 3 mois) pour confirmer les résultats du backtest

## Comment travailler avec moi

- Proposer un plan avant tout changement qui touche plusieurs modules
- Signaler tout doute sur un biais possible (look-ahead, surajustement, survivorship) plutôt que de l'ignorer
- Être critique sur les résultats : un Sharpe supérieur à 2 en backtest est suspect par défaut et doit être questionné
- Entre deux améliorations, privilégier celle qui réduit le risque plutôt que celle qui augmente le rendement
- apres tout travail sur des fichier, ecrire dans le fichier MODIF.md une courte explication pour les modifs apportés fichier par fichier dans le format suivant : [nom du fichier] -> "explication de la modif et ses impacts". effacer l'ancien contenu de MODIF.md a chaque nouvlele ecriture.
