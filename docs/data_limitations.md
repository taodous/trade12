# Limites connues des données historiques

À reprendre dans la section « limites » de chaque rapport d'audit.

## Source

- Barres journalières Alpaca, flux SIP, ajustées des splits et des dividendes.
- Historique disponible à partir du 4 janvier 2016. Il couvre donc 2018, 2020 et 2022, mais pas 2008.
- L'abonnement gratuit refuse les données SIP récentes : le téléchargement s'arrête à la dernière séance terminée.
- Le flux IEX n'est pas utilisé pour le backtest : il commence en novembre 2018 et son volume médian sur SPY représente environ 1,6 % de celui du SIP. Le bot en paper trading verra donc d'autres volumes que le backtest s'il utilise IEX.

## Clôtures erronées en mars 2020

Comparaison des clôtures non ajustées d'Alpaca et de Yahoo Finance sur les 10 ETF, du 4 janvier 2016 au 30 octobre 2026 (27 230 barres, contrôle ponctuel du 2 novembre 2026). Seules 15 barres s'écartent de plus de 0,5 %, toutes entre le 28 février et le 18 mars 2020, en pleine période de krach :

| Date | Symbole | Alpaca | Yahoo | Écart |
|---|---|---|---|---|
| 2020-02-28 | IWM | 147,42 | 146,33 | +0,74 % |
| 2020-03-09 | SPY | 276,32 | 274,23 | +0,76 % |
| 2020-03-09 | VNQ | 86,00 | 83,68 | +2,77 % |
| 2020-03-12 | GLD | 150,44 | 147,79 | +1,79 % |
| 2020-03-12 | SPY | 255,24 | 248,11 | +2,87 % |
| 2020-03-16 | EEM | 32,01 | 31,63 | +1,20 % |
| 2020-03-16 | EFA | 46,49 | 47,45 | -2,02 % |
| 2020-03-16 | GLD | 137,56 | 141,64 | -2,88 % |
| 2020-03-16 | IWM | 105,51 | 103,62 | +1,82 % |
| 2020-03-16 | SPY | 241,06 | 239,85 | +0,51 % |
| 2020-03-16 | VNQ | 71,90 | 65,39 | +9,95 % |
| 2020-03-17 | SPY | 254,19 | 252,80 | +0,55 % |
| 2020-03-18 | IWM | 100,86 | 99,97 | +0,89 % |
| 2020-03-18 | SPY | 235,69 | 240,00 | -1,80 % |
| 2020-03-18 | VNQ | 63,52 | 62,59 | +1,49 % |

Conséquences :

- Le contrôle qualité ne détecte pas ces écarts, car chaque barre est cohérente prise isolément.
- Les rendements journaliers de ces séances sont faux. Celui de VNQ le 17 mars 2020 change même de signe (-3,79 % contre +5,78 %).
- Sur toute la période, l'effet reste faible : le drawdown maximal de SPY est de -33,8 % contre -33,7 %, son rendement total de +343 % contre +347 %.
- Une règle sensible aux mouvements journaliers (stop, kill switch, filtre de volatilité) peut en revanche réagir différemment pendant ce krach, justement la période la plus utile pour les tests de stress.

Les données stockées ne sont pas corrigées. La cause n'a pas été établie.

**À faire à l'étape 7 :** relancer le backtest retenu sur une seconde source et traiter tout écart notable comme une limite du résultat.

## Convention d'horodatage

Une barre porte la date de sa séance, mais contient la clôture de cette séance. Elle n'est connue qu'après la clôture. Une décision exécutée pendant la séance t ne doit utiliser que les barres antérieures à t : un signal calculé sur la clôture de t s'exécute au plus tôt à l'ouverture de t+1.

## Univers

L'univers est fixé a priori et ne contient que des ETF qui existent encore aujourd'hui. Il reste donc exposé au biais du survivant, et son choix est postérieur à la période testée. Il ne doit pas être modifié au vu de résultats de backtest.
