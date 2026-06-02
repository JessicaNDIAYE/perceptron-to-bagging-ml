# Beyond the Perceptron — Réseaux de Neurones & Apprentissage Semi-Supervisé

> De l'implémentation from scratch d'un Perceptron multi-classe jusqu'à la sélection de features en contexte semi-supervisé — un voyage complet dans les fondations du Machine Learning.

---

## Aperçu

Ce projet regroupe deux travaux pratiques en Machine Learning :

| TP | Sujet | Notebook |
|----|-------|----------|
| TP1 | Perceptron, MLP, Bagging — classification supervisée | `TP1_NN_NDIAYE_KERARMA.ipynb` |
| TP2 | Sélection de features semi-supervisée (Fisher + Laplacien) | `TP2.ipynb` |

---

## Stack technique

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-gray?logo=numpy)
![scikit-learn](https://img.shields.io/badge/scikit--learn-orange?logo=scikitlearn)
![pandas](https://img.shields.io/badge/pandas-purple?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-lightblue)

---

## TP1 — Classification Supervisée : du Perceptron au Bagging

### Datasets utilisés

| Dataset | Échantillons | Features | Classes |
|---------|-------------|----------|---------|
| Iris | 150 | 4 | 3 |
| Glass | 214 | 9 | 7 |
| Breast Cancer Wisconsin | 699 | 9 | 2 |
| Lsun | variable | 2 | 3 |
| Wave | 5 000 | 40 | 3 |

---

### 1. Perceptron Multi-Classe — From Scratch

Implémenté entièrement en NumPy, sans aucune bibliothèque de ML.

**Principe :** Pour chaque erreur de prédiction, les poids de la bonne classe sont renforcés et ceux de la classe prédite à tort sont pénalisés.

```python
class MultiClassPerceptron:
    def fit(self, X, y):
        for _ in range(self.max_iter):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(self.weights, x_i) + self.bias
                y_pred = np.argmax(linear_output)
                if y_pred != y[idx]:
                    self.weights[y[idx]] += self.lr * x_i
                    self.weights[y_pred] -= self.lr * x_i
```

**Résultats sur Iris (split stratifié 2/3 - 1/3) :**

```
Matrice de Confusion :
[[16  0  0]
 [ 0 17  0]
 [ 0  5 12]]

Accuracy : 90.00%
Classe 1 → Précision : 1.00  |  Rappel : 1.00
Classe 2 → Précision : 0.77  |  Rappel : 1.00
Classe 3 → Précision : 1.00  |  Rappel : 0.71
```

> La classe 3 est la plus difficile à discriminer — comportement attendu pour un modèle linéaire face à des classes non parfaitement séparables.

---

### 2. MLP (Perceptron Multi-Couches) — Impact de la Normalisation

Comparaison de deux configurations identiques, avec et sans `StandardScaler`.

| Configuration | Accuracy | Itérations | Loss finale |
|--------------|----------|-----------|-------------|
| Sans normalisation | **96.00%** | 247 | 0.0114 |
| Avec normalisation | 90.00% | 132 | **0.0036** |

**Architecture :** `(12, 10, 10)` — activation ReLU — solver Adam

> Normaliser n'améliore pas toujours l'accuracy, mais divise la loss par 3 et accélère la convergence de moitié. La normalisation stabilise l'apprentissage sans garantir de meilleures performances finales.

---

### 3. Bagging de Réseaux de Neurones — La Force du Collectif

K=10 MLPs entraînés sur des échantillons bootstrap distincts, agrégés par **vote majoritaire**.

```python
# Tirage avec remise pour chaque modèle
for i in range(K):
    indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
    mlp = MLPClassifier(hidden_layer_sizes=(12, 10, 10), max_iter=5000, random_state=i)
    mlp.fit(X_train[indices], y_train[indices])

# Vote majoritaire
predictions = np.array([m.predict(X_test) for m in models])
y_final, _ = mode(predictions, axis=0)
```

**Résultat sur Iris :**

```
Accuracy Bagging (K=10) : 100.00%
Accuracy MLP seul       :  96.00%
```

> Quand un modèle "dévie" sur un tirage particulier, les 9 autres le corrigent. Le bagging réduit la variance sans augmenter le biais.

---

### 4. Benchmark Comparatif — 5 Datasets

Pipeline automatisé appliqué à tous les datasets :

| Dataset | Perceptron | MLP | Bagging |
|---------|-----------|-----|---------|
| **Iris** | 90% | 92% | 96% |
| **Glass** | 47% | 80% | 61% |
| **Breast Cancer** | 95% | 94% | **96%** |
| **Lsun** | 100% | 100% | 100% |
| **Wave** | 81% | 83% | **87%** |

**Analyse par dataset :**

- **Lsun** : géométriquement simple, 100% pour tous les modèles
- **Breast Cancer** : données bien structurées, le Perceptron rivalise avec le MLP
- **Wave** : 40 features, 5 000 échantillons — le bagging compense la haute dimensionnalité
- **Glass** : la classe 4 est absente des données → déséquilibre sévère → le modèle ne l'apprend jamais, ce qui s'explique les 47% du Perceptron

---

## TP2 — Sélection de Features Semi-Supervisée

### Contexte

Sur **5 000 échantillons** (Wave, 40 features, 3 classes), seulement **10% sont étiquetés** (250 échantillons). L'objectif : identifier les features les plus discriminantes en exploitant à la fois les données labellisées et non labellisées.

### Split stratifié

```
Total : 5 000 échantillons
A (apprentissage) : 2 500  →  étiquetés : 250 (10%)  |  non étiquetés : 2 250
T (test)          : 2 500
```

### Critères de scoring

Deux scores complémentaires sont calculés pour chaque feature :

**S1 — Score supervisé (Fisher)**
Mesure la séparation inter-classes sur les données étiquetées.

```
S1(v) = Σ_k n_k · (μ_k - μ)² / Σ_k n_k · σ_k²
```

**S2 — Score non supervisé (Laplacien)**
Mesure la régularité locale de la feature sur les données non étiquetées.

```
S2(v) = Σ_i Σ_j exp(-(xi - xj)² / t) · (xi - xj)² / Var(v)
```

**Score final = S1 / S2** → features discriminantes ET géométriquement régulières.

### Top features identifiées

| Rang | Feature | S1 (Fisher) | Score final |
|------|---------|-------------|-------------|
| 1 | X14 | 0.541 | 2.03e-07 |
| 2 | X5 | 0.596 | 1.84e-07 |
| 3 | X8 | 0.676 | 1.82e-07 |
| 4 | X10 | — | 1.67e-07 |
| 5 | X15 | — | 1.55e-07 |

> X8 a le meilleur Fisher-score individuel mais arrive 3ème au classement final car S2 nuance : une feature très discriminante localement mais chaotique globalement est moins utile qu'une feature légèrement moins discriminante mais structurée.

---

## Structure du repo

```
beyond-the-perceptron/
│
├── TP1_NN_NDIAYE_KERARMA.ipynb   # Perceptron, MLP, Bagging
├── TP2.ipynb                      # Sélection semi-supervisée
├── TP2_corrected_final.ipynb      # Version corrigée TP2
│
├── iris.txt                       # Dataset Iris (150 × 4)
├── glass.txt                      # Dataset Glass (214 × 9)
├── breast-cancer-wisconsin.txt    # Dataset Breast Cancer (699 × 9)
├── Lsun.txt                       # Dataset Lsun
└── Wave.txt                       # Dataset Wave (5000 × 40)
```

---

## Résultats clés

```
Perceptron from scratch   →  90% sur Iris
MLP sans normalisation    →  96% sur Iris
Bagging (K=10)            → 100% sur Iris
Semi-supervisé            →  10% de labels suffisent pour identifier les features clés
```

---

## Auteurs

Réalisé dans le cadre d'un TP de Machine Learning — Master Data Science.

**Jessica NDIAYE** — [GitHub](https://github.com/JessicaNDIAYE)  
**Cérine KERARMA** — [GitHub](https://github.com/cerinekerarma)
