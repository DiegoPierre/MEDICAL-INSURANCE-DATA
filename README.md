## Prédiction des dépenses médicales et analyse des profils de risque à partir de données démographiques et socioéconomiques

<img src="Notebooks/Kaggle.png" width="4000" style="display: block; margin: 0 auto;">
<p style='text-align: center; font-style: italic; color: #7f8c8d;'>
</p>

## Contexte du projet
Le système de santé en Haïti fait face à de nombreux défis structurels et économiques qui affectent directement l’accès, la
qualité et la gestion des soins médicaux.
La majorité de la population haïtienne dispose de ressources financières limitées, tandis que le coût des soins, des
médicaments et des services hospitaliers reste relativement élevé.
Dans ce contexte, les dépenses de santé représentent un fardeau important pour les ménages, en particulier pour les
personnes atteintes de maladies chroniques telles que l’hypertension, le diabète ou les maladies cardiaques.
Cependant, le secteur de l’assurance santé en Haïti reste encore peu développé et manque d’outils modernes de gestion des
risques et de prévision des coûts.
Les compagnies d’assurance, les hôpitaux et les autorités sanitaires manquent souvent de données consolidées pour analyser
les dépenses médicales, identifier les facteurs de risque et anticiper les coûts futurs.
## Contexte
C’est dans ce cadre que s’inscrit le présent projet :
 il vise à exploiter les données de santé et socioéconomiques des individus pour prédire les dépenses médicales annuelles et
évaluer les profils de risque.
 Grâce aux outils d’analyse de données et aux techniques de machine learning, ce projet fournit une base scientifique pour :
● améliorer la planification des budgets médicaux,
● renforcer la prévention et la couverture santé,
● et appuyer les décisions stratégiques des acteurs du secteur.
## Problématique
Les coûts médicaux varient fortement d’un individu à un autre. Cette variation dépend de nombreux éléments : âge, sexe,revenu,
region, état de santé, couverture d’assurance, etc.
 La question centrale du projet est donc :
Comment prédire les dépenses médicales annuelles d’un individu à partir de ses caractéristiques personnelles,
économiques et médicales ?
## Objectifs du projet
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
## À propos de l'ensemble de données
Cet ensemble de données fournit des informations sur 100 000 personnes, notamment sur leurs données démographiques,
leur statut socio-économique, leur état de santé, leurs facteurs de style de vie, leurs régimes d’assurance et leurs dépenses
médicales.
À propos de ce fichier
Ce fichier contient :
**Lignes : **100 000
**Colonnes : **54+

## Exploration et compréhension du jeu de données
Cette phase d’exploration permet donc de :
● Comprendre la structure et la composition du jeu de données.
● Identifier les principales caractéristiques statistiques des variables.
● Détecter les problèmes de qualité des données, comme les valeurs manquantes ou les anomalies.
Ces vérifications sont essentielles pour garantir la fiabilité des résultats et assurer la réussite des étapes suivantes d’analyse et
de modélisation.
## Compréhension analytique
Le projet permet de réaliser plusieurs approches analytiques :
● Régression : prédire les dépenses médicales selon les caractéristiques des individus.
● Classification : attribuer un niveau de risque (faible, moyen, élevé).
● Clustering : regrouper les individus par similarité de profil santé.
● Analyse exploratoire : étudier les corrélations entre les variables et les coûts.
● Analyse de corrélation approfondie: Il serait intéressant d’étudier plus en détail les relations entre les différentes variables du jeu de données
● Analyse de la variance (ANOVA) : comparer les moyennes de dépenses médicales entre différents groupes d’individus : par exemple selon le type
d’assurance, le sexe, la région, ou la présence de maladies chroniques.
● Importance des variables et analyse de sensibilité : évaluer le poids de chaque facteur dans la prédiction du coût médical.

## Intérêt pour le domaine de la santé
Ce type d’analyse peut aider à :
● Identifier les comportements à risque pour la santé.
● Comprendre comment certains facteurs influencent les coûts médicaux.
● Développer des politiques d’assurance plus justes et adaptées.
● Promouvoir la prévention et la sensibilisation à la santé.
● Une carte de corrélation (ou heatmap) permettrait de visualiser ces liens et d’identifier les facteurs les plus liés aux
dépenses de santé.
● Cela permettrait de déterminer si les différences observées entre les groupes sont statistiquement significatives.
● Une analyse de sensibilité permettra aussi d’évaluer comment de petites variations dans les données (comme une légère
hausse de l’IMC) impactent le coût estimé.
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

### Compréhension analytique
Le projet permet de réaliser plusieurs approches analytiques :
● Régression : prédire les dépenses médicales selon les caractéristiques des individus.
● Classification : attribuer un niveau de risque (faible, moyen, élevé).
● Clustering : regrouper les individus par similarité de profil santé.
● Analyse exploratoire : étudier les corrélations entre les variables et les coûts.
● Analyse de corrélation approfondie: Il serait intéressant d’étudier plus en détail les relations entre les différentes
variables du jeu de données
● Analyse de la variance (ANOVA) : comparer les moyennes de dépenses médicales entre différents groupes d’individus : par
exemple selon le type d’assurance, le sexe, la région, ou la présence de maladies chroniques.
● Importance des variables et analyse de sensibilité : évaluer le poids de chaque facteur dans la prédiction du coût médical

### Intérêt pour le domaine de la santé
Ce type d’analyse peut aider à :
● Identifier les comportements à risque pour la santé.
● Comprendre comment certains facteurs influencent les coûts médicaux.
● Développer des politiques d’assurance plus justes et adaptées.
● Promouvoir la prévention et la sensibilisation à la santé.
● Une carte de corrélation (ou heatmap) permettrait de visualiser ces liens et d’identifier les facteurs les plus liés aux dépenses de
santé.
● Cela permettrait de déterminer si les différences observées entre les groupes sont statistiquement significatives.
● Une analyse de sensibilité permettra aussi d’évaluer comment de petites variations dans les données (comme une légère hausse de
l’IMC) impactent le coût estimé.



