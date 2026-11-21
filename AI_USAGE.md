## Utilisation de l'IA

Nous avons utilisé une IA (Claude, d'Anthropic) à trois endroits du projet.

**Rédaction de fonctions de `ml_workflow.py`.** L'IA nous a aidés à écrire certaines fonctions du workflow, par exemple la sélection de variables, la comparaison de modèles ou le réglage des hyperparamètres. Nous l'avons fait pour gagner du temps sur le code répétitif et pour partir d'une structure commune à trois. Chaque membre a ensuite relu, adapté et intégré le code dans sa section.

**Rédaction de tests unitaires.** Elle a proposé des tests pour vérifier nos fonctions (nombre de variables gardées par les sélecteurs, absence de fuite de données, réentraînement du meilleur modèle sur toutes les lignes). Nous les avons exécutés avec `pytest` et corrigés quand ils ne correspondaient pas à notre code.

**Débogage.** Nous lui avons soumis nos erreurs et nos résultats inattendus pour en trouver la cause. Chaque correction a été relancée et, quand c'était possible, protégée par un test.

Nous avons relu et testé tout ce que l'IA nous a proposé avant de le garder, et chaque membre du groupe peut expliquer le code rendu. Les choix de méthode (métriques, modèle final, seuil de décision) ont été discutés et validés par l'équipe.