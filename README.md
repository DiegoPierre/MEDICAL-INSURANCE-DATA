## Prédiction des dépenses médicales et analyse des profils de risque à partir de données démographiques et socioéconomiques

<img src="Notebooks/Kaggle.png" width="4000" style="display: block; margin: 0 auto;">
<p style='text-align: center; font-style: italic; color: #7f8c8d;'>
</p>


### Contexte du projet
Le secteur de la santé produit aujourd’hui une quantité massive de données sur les individus : âge, sexe,revenu, region, état de santé,
couverture d’assurance, etc.
 Ces informations sont précieuses pour les compagnies d’assurance et les organismes de santé, car elles permettent d’identifier les
facteurs qui influencent le coût des soins médicaux et d’anticiper les dépenses futures.
Dans ce contexte, l’utilisation de méthodes d’analyse de données et de modélisation statistique devient un atout essentiel pour mieux
comprendre le comportement médical des assurés, évaluer les risques et optimiser la tarification des assurances santé.

### Problématique
Les coûts médicaux varient fortement d’un individu à un autre. Cette variation dépend de nombreux éléments : âge, sexe,revenu,
region, état de santé, couverture d’assurance, etc.
 La question centrale du projet est donc :
Comment prédire les dépenses médicales annuelles d’un individu à partir de ses caractéristiques personnelles,
économiques et médicales ?

### Objectifs du projet
- Développer un modèle capable d’estimer les dépenses annuelles par patient.
Identifier les variables les plus influentes sur les coûts, telles que l’âge, l’IMC, le statut de fumeur ou le nombre de visites médicales.
-Regrouper les patients selon des caractéristiques médicales et démographiques similaires.
Identifier les groupes à haut risque pour cibler la prévention et l’allocation des ressources.
-Comparer plusieurs modèles de machine learning (régression linéaire, Random Forest, XGBoost, Logistic Regression) pour sélectionner celui
offrant la meilleure performance.
Utiliser des métriques complètes : précision (accuracy), rappel (recall), score F1, matrice de confusion, AUC, R² et RMSE
-Représenter la structure des clusters via PCA 2D/3D, UMAP.
Créer des boxplots, heatmaps et graphiques interactifs pour comprendre la distribution des coûts et des caractéristiques par cluster.
-Prévoir les coûts pour les assureurs et hôpitaux.
Segmentation pour la prévention ciblée et la planification des traitements.
Allocation optimale des ressources médicales selon le risque.
Surveillance de la santé des populations à haut risque et contribution indirecte à la lutte contre les maladies chroniques ou infectieuses, comme le
VIH.

### Exploration et compréhension du jeu de données
Cette première phase d’exploration permet donc de :
● Comprendre la structure et la composition du jeu de données.
● Identifier les principales caractéristiques statistiques des variables.
● Détecter les problèmes de qualité des données, comme les valeurs manquantes ou les anomalies.
Ces vérifications sont essentielles pour garantir la fiabilité des résultats et assurer la réussite des étapes suivantes d’analyse et
de modélisation.

### Analyse exploratoire des coûts médicaux
```python
# Visualisations simples
# ------------------------------
# Histogramme du coût médical annuel
plt.figure(figsize=(8,5))
sns.histplot(df['annual_medical_cost'], bins=50, kde=True)
plt.title("Distribution des coûts médicaux annuels")
plt.xlabel("Coût annuel")
plt.ylabel("Nombre de personnes")
plt.show()
```

<img src="Images/1.png" width="600" style="display: block; margin: 0 auto;">
<p style='text-align: center; font-style: italic; color: #7f8c8d;'>
</p>

### Distribution des coûts médicaux annuels
Ce graphique montre la répartition des dépenses médicales annuelles dans l’ensemble du jeu de données.
Chaque barre représente le nombre de personnes dont les dépenses se situent dans une certaine plage de coûts.
Interprétation:
La plupart des individus dépensent relativement peu en soins médicaux chaque année, mais une minorité présente des coûts très élevés,
souvent liés à des maladies chroniques ou à des hospitalisations importantes.


### Coût moyen selon le statut de fumeur
```python

# Coût moyen par statut fumeur
plt.figure(figsize=(6,4))
sns.barplot(x='smoker', y='annual_medical_cost', data=df)
plt.title("Coût moyen selon le statut de fumeur")
plt.show()

```
<img src="Images/2.png" width="600" style="display: block; margin: 0 auto;">
<p style='text-align: center; font-style: italic; color: #7f8c8d;'>
</p>

### Coût moyen selon le statut de fumeur
Ce graphique compare le coût médical moyen entre les fumeurs et les non-fumeurs.
Chaque barre correspond à la dépense moyenne d’un groupe.
Interprétation:
On observe généralement que les fumeurs ont des coûts médicaux moyens plus élevés que les non-fumeurs.
Cela peut s’expliquer par un risque accru de maladies respiratoires, cardiovasculaires ou cancéreuses, entraînant des soins plus coûteux
