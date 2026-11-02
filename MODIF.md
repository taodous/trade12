[src/bot/risk/models.py] -> "Nouveau. Définit les types du risque : ordre proposé, état du compte, prix de référence, décision et motifs de refus. Le pouvoir d'achat n'existe pas dans ces types : le risque se mesure toujours sur le capital, donc la marge du compte ne peut pas servir."

[src/bot/risk/checks.py] -> "Nouveau. Contrôles purs : prix manquant, périmé ou aberrant, signal qui ne vient pas de la dernière séance terminée, vente à découvert, taille de position, exposition totale, perte journalière et drawdown. Tout ce qui est invérifiable, comme un NaN, est refusé. Un ordre qui réduit une position passe même au-dessus des limites. Fournit aussi les ordres de fermeture du kill switch."

[src/bot/risk/manager.py] -> "Nouveau. Point de passage obligé de tout ordre : applique les contrôles, compte les ordres de la séance et déclenche le kill switch. Celui-ci reste actif jusqu'à un réarmement manuel. L'étape 6 devra faire passer chaque ordre par review et appeler update à chaque cycle."

[src/bot/risk/state.py] -> "Nouveau. Conserve le kill switch, le plus haut du capital et le compteur d'ordres dans un fichier JSON, pour qu'un redémarrage ne les efface pas. Un fichier illisible arrête le bot au lieu de repartir d'un état vierge. Supprimer ce fichier à la main efface donc ces protections."

[src/bot/config.py] -> "Ajout du champ risk_state_file. Il devient obligatoire dans strategy.yaml : le bot refuse de démarrer s'il manque."

[config/strategy.yaml] -> "Ajout de risk_state_file, dans data/ et donc hors de git. Le commentaire de max_data_age_seconds précise qu'il s'agit de l'âge du prix de référence de l'ordre, et non de celui du signal."

[tests/test_risk_checks.py] -> "Nouveau. Vérifie chaque contrôle à sa limite exacte, les valeurs NaN et infinies, le refus de la marge, et le passage des ordres qui réduisent le risque."

[tests/test_risk_manager.py] -> "Nouveau. Vérifie le déclenchement du kill switch, sa persistance après redémarrage, le compteur d'ordres, le réarmement, et le refus d'approuver un ordre si l'état ne peut pas être enregistré."

[MODIF.md] -> "Réécrit pour l'étape 4. Le récapitulatif de l'étape 3 est effacé, comme le prévoit la règle."
