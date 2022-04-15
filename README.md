# Détectez les Bad Buzz grâce au Deep Learning

Vous êtes ingénieur IA chez MIC (Marketing Intelligence Consulting), une entreprise de conseil spécialisée sur les problématiques de marketing digital.

Dans deux semaines, vous avez rendez-vous avec Mme Aline, directrice marketing de la compagnie aérienne “Air Paradis”. 

Air Paradis a missionné votre cabinet pour créer un produit IA permettant d’anticiper les bad buzz sur les réseaux sociaux. Il est vrai que “Air Paradis” n’a pas toujours bonne presse sur les réseaux…

En sortant d’un rendez-vous de cadrage avec les équipes de Air Paradis, vous avez noté les éléments suivants :

* Air Paradis veut un prototype d’un produit IA permettant de prédire le sentiment associé à un tweet.
* Données : pas de données clients chez Air Paradis. Solution : utiliser des données Open Source (https://www.kaggle.com/datasets/kazanova/sentiment140)
  * Description des données : des informations sur les tweets (utilisateur ayant posté, contenu, moment du post) et un label binaire (tweet exprimant un sentiment négatif ou non). 

* TO-DO :
  * Préparer un prototype fonctionnel du modèle. Le modèle envoie un tweet et récupère la prédiction de sentiment. 
  * Préparer un support de présentation explicitant la méthodologie utilisée pour l’approche “modèle sur mesure avancé” (attention : audience non technique).

Après avoir reçu votre compte-rendu, Marc, votre manager, vous a contacté pour, selon ses mots, “faire d’une pierre deux coups”.

##Livrables 

* Le “Modèle sur mesure avancé” exposé grâce au service Azure Machine Learning qui recevra en entrée un tweet et retournera le sentiment associé au tweet prédit par le modèle.
  * Ce livrable permettra d’illustrer votre travail auprès du client.
* L’ensemble des scripts développés sur Azure pour réaliser les trois approches. 
  * Ce livrable vous servira à présenter les détails de votre travail à une audience technique (par exemple des lecteurs de votre post de blog qui voudraient en savoir plus).
* Un article de blog de 800-1000 mots environ contenant une présentation et une comparaison des trois approches (“API sur étagère”, “Modèle sur mesure simple” et “Modèle sur mesure avancé”) :
  * Ce livrable vous servira à faire rayonner le cabinet en démontrant votre expertise technique, mais aussi à valoriser votre travail auprès de la communauté des data scientists en ligne. Et surtout, à répondre aux exigences de votre manager !
* Un support de présentation (type PowerPoint) de votre démarche méthodologique, des résultats des différents modèles élaborés et de la mise en production d’un modèle avancé.
