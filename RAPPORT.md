# Rapport de developpement — Architecture Doya (1999) pour jeux video

## NEUR503 — Term Paper v2

---

## Table des matieres

1. [Vue d'ensemble](#1-vue-densemble)
2. [Branche v2 : Correction en ligne et bascule automatique](#2-branche-v2--correction-en-ligne-et-bascule-automatique)
3. [Branche mario-bros : Adaptation pixel-based](#3-branche-mario-bros--adaptation-pixel-based)
4. [Fonctionnalites de production](#4-fonctionnalites-de-production)
5. [Architecture complete](#5-architecture-complete)
6. [Inventaire des fichiers](#6-inventaire-des-fichiers)
7. [Resultats experimentaux](#7-resultats-experimentaux)
8. [Guide d'utilisation](#8-guide-dutilisation)

---

## 1. Vue d'ensemble

Ce projet implemente le cadre theorique de Doya (1999, 2000) qui postule que trois structures cerebrales sont specialisees pour trois paradigmes d'apprentissage distincts :

| Structure | Paradigme | Equations |
|-----------|-----------|-----------|
| Cortex cerebral | Apprentissage non-supervise | Eq. 17-18 |
| Ganglions de la base | Apprentissage par renforcement | Eq. 10-14 |
| Cervelet | Apprentissage supervise | Eq. 5, 21 |

Le travail realise aujourd'hui couvre **deux phases majeures** :

1. **Branche `v2`** : Ajout de la correction cerebelleuse en temps reel et de la bascule automatique cervelet/ganglions de la base, avec validation sur CartPole.
2. **Branche `mario-bros`** : Adaptation complete de l'architecture pour des environnements a base de pixels (Super Mario Bros / Atari), avec justification physiologique de chaque modification.

### Historique Git

```
02bf14e  Apprentissage fonctionnel                           (main)
9e03b19  v2: correction en ligne, bascule automatique        (v2)
4ae855b  train_cartpole: ajout rapport PNG                   (v2)
7a2defe  mario-bros: architecture Doya adaptee pixel-based   (mario-bros)
```

**Total : 4 392 lignes de code Python** reparties dans 11 fichiers.

---

## 2. Branche v2 : Correction en ligne et bascule automatique

### 2.1 CerebellarCorrector (Section 4.3.2, Fig. 10)

**Fichier** : `cerebellum.py` (lignes 329-440)

Implemente la correction en temps reel par le modele predictif (forward model). Le signal d'erreur provient des fibres grimpantes de l'olive inferieure :

```
u_corrige = u_base + gain * J^T * (x_observe - x_predit)
```

- **J** : Jacobien du forward model, approxime par differences finies
- **J^T @ erreur_etat** : projette l'erreur sensorielle dans l'espace moteur
- L'erreur de prediction est accumulee pour le seuil adaptatif

### 2.2 HybridController (bascule automatique)

**Fichier** : `cerebellum.py` (lignes 443-599)

Detecteur de surprise base sur l'erreur de prediction du forward model :

```
si ||x_observe - x_predit|| > seuil_adaptatif  -->  Ganglions de la base (delibere)
sinon                                            -->  Cervelet inverse (automatique)
```

Le seuil s'adapte en ligne via une moyenne mobile exponentielle (EMA) :
```
seuil = EMA_mean + facteur * sqrt(EMA_variance)
```

### 2.3 Deux architectures selectionnables (Section 4)

**Fichier** : `train_cartpole.py` (854 lignes)

| Architecture | Section Doya | Description |
|-------------|-------------|-------------|
| Reactive (4.1) | Fig. 6 | Selection d'action par la valeur uniquement |
| Predictive (4.2.1) | Fig. 7 | Evaluation des candidats par le forward model via delta predictif (Eq. 22) |

Selection par menu interactif ou argument CLI.

### 2.4 Rapport d'entrainement PNG

`train_cartpole.py` genere un rapport multi-panneaux :
- Reward par episode (+ moyenne glissante)
- Longueur des episodes
- Erreur TD (signal dopaminergique)
- Erreur du cortex (autoencoder)
- Erreurs cerebelleuses (forward + inverse)
- Comparaison des 4 strategies de test (barres)

### 2.5 Resultats CartPole

| Strategie | Reward moyen |
|-----------|-------------|
| Ganglions de la base (4.1) | ~9.4 |
| Modele predictif (4.2.1) | **~474** |
| Modele inverse (Fig. 11) | variable |
| Hybride (v2) | variable |

Le modele predictif surpasse largement le reactif, confirmant les predictions theoriques de Doya.

---

## 3. Branche mario-bros : Adaptation pixel-based

### 3.1 Pipeline retinien (`mario_wrappers.py`, 180 lignes)

Preprocessing des frames brutes vers une representation adaptee au cortex CNN :

```
Raw 240x256x3
  -> SkipFrame(4)       : persistance motrice ~240ms (Georgopoulos et al., 1982)
  -> GrayScaleResize(84) : luminance retinienne (voie magnocellulaire)
  -> FrameStack(4)       : integration temporelle V1 (Adelson & Bergen, 1985)
  -> /255                : normalisation [0,1]
  = 84x84x4 float32
```

**Double backend** : Super Mario Bros (gym-super-mario-bros) avec fallback automatique vers Atari (ALE/MsPacman-v5) si nes-py n'est pas disponible.

### 3.2 Cortex cerebral CNN (`cerebral_cortex_mario.py`, 287 lignes)

Hierarchie visuelle ventrale implementee par autoencoder convolutionnel sparse :

#### Encodeur (V1 -> V2 -> V4 -> IT)

| Couche | Aire corticale | Dimensions | Justification |
|--------|---------------|------------|---------------|
| Conv(4,32,8,s=4)+ReLU | V1 | 84x84x4 -> 20x20x32 | Cellules simples/complexes, champs recepteurs 8x8 (Hubel & Wiesel, 1962) |
| Conv(32,64,4,s=2)+ReLU | V2 | 20x20x32 -> 9x9x64 | Detecteurs de features intermediaires |
| Conv(64,64,3,s=1)+ReLU | V4 | 9x9x64 -> 7x7x64 | Neurones selectifs aux formes |
| FC(3136,256)+ReLU | IT | 3136 -> 256 | Representations objet |
| FC(256,64) | Sortie corticale | 256 -> 64D | Vecteur compact pour modules en aval |

#### Decodeur (connexions top-down)

Architecture miroir avec ConvTranspose2d, correspondant aux connexions generatives top-down du cortex visuel (Rao & Ballard, 1999, codage predictif).

#### Fonction de perte (Eq. 17 de Doya)

```
E = ||x - W'y||^2 + lambda * sum|y_i|
    reconstruction    sparsity L1
```

- **Reconstruction MSE** : erreur de prediction (codage predictif)
- **Sparsity L1** : codage sparse cortical (Olshausen & Field, 1996)
- **Buffer replay** : 10 000 frames, entrainement asynchrone par mini-batch
- **Warmup** : 5 000 frames avec actions aleatoires avant le RL
- **`features.detach()`** : la dopamine ne module PAS les synapses intracorticales

### 3.3 Ganglions de la base MLP (`basal_ganglia_mario.py`, 256 lignes)

Acteur-critique non-lineaire pour actions discretes :

#### Critique (voie striosomale)
```
64D -> FC(128)+ReLU -> FC(64)+ReLU -> FC(1) = V(s)
```

#### Acteur (voie matricielle)
```
64D -> FC(128)+ReLU -> FC(64)+ReLU -> FC(7) -> Softmax(logits/temperature)
```

| Composant | Substrat neural | Justification |
|-----------|----------------|---------------|
| MLP hidden | MSN dendritiques | ~10 000 epines/neurone, integration non-lineaire NMDA (Wickens, 2007) |
| ReLU | Voies D1/D2 | Direct (Go) / indirect (NoGo) |
| Softmax | Competition SNr/GPi | Inhibition mutuelle selectionne le canal d'action (Eq. 13) |
| Temperature | Meta-parametre | Controle exploration (Doya Sec. 6.5), decroissance 2.0 -> 0.1 |
| Erreur TD | Dopamine SNc | delta = r + gamma*V(s') - V(s) (Eq. 10, identique) |

#### Regles d'apprentissage preservees
- **Critique** : minimiser delta^2 via backprop (generalisation multi-couche de Eq. 12)
- **Acteur** : delta * log pi(a|s) = REINFORCE (generalisation discrete de Eq. 14)

### 3.4 Cervelet sur features corticales (`cerebellum_mario.py`, 320 lignes)

Les modeles cerebelleux operent sur les features corticales 64D, pas les pixels bruts. C'est physiologiquement correct : le cervelet recoit les afferences par les fibres moussues qui ont deja ete traitees par le cortex (noyaux du pont).

#### ForwardModel (Eq. 21)
```
concat(features_64, onehot_7) = 71D -> 512 cellules granulaires -> Linear(512,64)
```
Predit features(t+1) a partir de (features(t), action(t)).

#### InverseModel (Fig. 11)
```
concat(features_64, target_64) = 128D -> 512 cellules granulaires -> Linear(512,7)
```
Calcule l'action pour passer de features(t) a features_cible.

#### CerebellarCorrectorMario (Fig. 7)
Pour les actions discretes, remplace l'approximation Jacobienne par une evaluation exhaustive : tester les 7 actions, predire l'etat resultant, selectionner la meilleure. Plus fidele a la Section 4.2.1 de Doya.

#### HybridControllerMario
Logique de surprise inchangee (erreur de prediction > seuil adaptatif EMA). Retourne un index d'action discret.

### 3.5 Pipeline d'entrainement (`train_mario.py`, 785 lignes)

```
Frame -> Wrappers(skip,gray,resize,stack) -> 84x84x4
       -> Cortex CNN encode -> 64D features (detach)
       -> BG step(features) -> action discrete (0-6)
       -> env.step(action) -> next_frame, reward, done
       -> Cortex encode+learn (autoencoder, batch=32)
       -> BG learn(reward, next_features, done) -> TD error
       -> Forward model learn(features, action, next_features)
       -> Inverse model learn(features, next_features, action)
```

---

## 4. Fonctionnalites de production

### 4.1 Sauvegarde/chargement de checkpoints

Tous les modules disposent de methodes `save()` / `load()` :

| Module | Format | Contenu |
|--------|--------|---------|
| Cortex (PyTorch) | `.pt` | Poids encodeur + decodeur |
| Ganglions de la base (PyTorch) | `.pt` | Poids critique + acteur |
| Forward model (NumPy) | `.npz` | Poids granulaires + Purkinje + stats normalisation |
| Inverse model (NumPy) | `.npz` | Idem |

Fonctions `save_checkpoint(path, ...)` et `load_checkpoint(path, ...)` gerent les 4 modules en une seule operation dans un repertoire.

### 4.2 Auto-sauvegarde du meilleur modele

Pendant l'entrainement, une moyenne glissante (fenetre=10) du reward est suivie. Quand elle depasse le meilleur score precedent, le modele complet est sauvegarde dans `checkpoints/best/`. Un checkpoint final est aussi sauvegarde dans `checkpoints/final/`.

### 4.3 Mode inference

Avec `--load PATH`, le programme charge un checkpoint sauvegarde et execute directement la phase de test (comparaison des 4 strategies), sans reentrainement.

### 4.4 Playback visuel

Avec `--load PATH --render`, le modele joue visuellement au jeu. On peut choisir une strategie specifique avec `--strategy`.

### 4.5 Rapport d'entrainement optionnel

Le rapport PNG multi-panneaux est genere uniquement si `--report` est passe en argument.

---

## 5. Architecture complete

### Flux de donnees

```
Raw 240x256x3 -> [Retine] -> 84x84x4 -> [Cortex V1->IT] -> 64D features
                                               |
                             +-----------------+------------------+
                             |                                    |
                    [Ganglions de la base]              [Cervelet]
                    Critique: 64->128->64->V(s)   Forward: (64+7)->512gc->64
                    Acteur:  64->128->64->7 act   Inverse: (64+64)->512gc->7
                             |                                    |
                             v                                    v
                    action discrete 0-6              correction / encapsulation
                             |
                             v
                        env.step(action)
```

### Correspondance Doya (1999)

| Concept theorique | Implementation | Equation/Figure |
|-------------------|----------------|-----------------|
| Cortex non-supervise | CNN sparse autoencoder | Eq. 17 |
| Erreur TD dopaminergique | delta = r + gamma*V(s') - V(s) | Eq. 10 |
| Apprentissage critique | MSE(V, r + gamma*V') | Eq. 12 |
| Apprentissage acteur | delta * log pi(a|s) | Eq. 14 |
| Temperature exploration | Decroissance lineaire 2.0->0.1 | Sec. 6.5 |
| Cellules granulaires | Expansion aleatoire fixe 512D | Sec. 3.1.3 |
| Fibres grimpantes | Erreur supervise (target - prediction) | Eq. 5 |
| Forward model | F(features, action) -> next_features | Eq. 21 |
| Inverse model | F^-1(features, target) -> action | Fig. 11 |
| Selection predictive | argmax_a gamma*V(F(s,a)) - V(s) | Eq. 22, Fig. 7 |
| Correction en ligne | J^T @ erreur_prediction | Sec. 4.3.2, Fig. 10 |
| Bascule surprise | erreur_prediction > seuil EMA | Sec. 4.3.3 |

---

## 6. Inventaire des fichiers

### Fichiers originaux (inchanges)

| Fichier | Lignes | Role |
|---------|--------|------|
| `cerebral_cortex.py` | 157 | Cortex bas-dimensionnel (bras) |
| `basal_ganglia.py` | 231 | BG bas-dimensionnel (bras) |
| `animate_arm.py` | 112 | Animation du bras articule |
| `train.py` | 610 | Entrainement bras articule |

### Fichiers v2

| Fichier | Lignes | Ajouts |
|---------|--------|--------|
| `cerebellum.py` | 600 | +CerebellarCorrector, +HybridController, +save/load |
| `train_cartpole.py` | 854 | Nouveau : 2 architectures, menu, rapport PNG, simulation |

### Fichiers mario-bros

| Fichier | Lignes | Role |
|---------|--------|------|
| `mario_wrappers.py` | 180 | Pipeline retinien + factory environnement |
| `cerebral_cortex_mario.py` | 287 | Cortex CNN autoencoder sparse |
| `basal_ganglia_mario.py` | 256 | Acteur-critique MLP discret |
| `cerebellum_mario.py` | 320 | Forward/inverse sur features + correcteur + hybride |
| `train_mario.py` | 785 | Pipeline complet + checkpoints + inference + rapport |

---

## 7. Resultats experimentaux

### Smoke test (50 episodes, MsPacman)

L'environnement ALE/MsPacman-v5 (fallback Atari) a ete valide avec succes :
- Warmup cortex (200 frames, 10 epochs) : loss 0.1089 -> 0.1027
- Entrainement 50 episodes sans erreur
- Toutes les metriques (reward, TD, cortex, forward, inverse) sont loguees correctement

### CartPole : Predictif >> Reactif

Le modele predictif (Section 4.2.1, Eq. 22) surpasse massivement le reactif (Section 4.1) :
- **Reactif** : ~9.4 reward moyen
- **Predictif** : ~474 reward moyen

Cela confirme la these de Doya : l'utilisation du forward model cerebelleux pour evaluer les consequences des actions avant de les executer est superieure a la selection purement basee sur la valeur.

---

## 8. Guide d'utilisation

### 8.1 Installation

```bash
# Dependances de base
pip install torch torchvision numpy matplotlib opencv-python gymnasium

# Pour Atari (fallback)
pip install ale-py

# Pour Super Mario Bros (optionnel, requiert compilateur C++)
pip install gym-super-mario-bros nes-py
```

### 8.2 Entrainement

#### Mode interactif (menu)
```bash
python train_mario.py
```
Un menu s'affiche pour choisir l'architecture (reactive ou predictive).

#### Mode direct
```bash
# Architecture reactive (Section 4.1, Fig. 6)
python train_mario.py reactive

# Architecture predictive (Section 4.2.1, Fig. 7) — recommandee
python train_mario.py predictive
```

#### Options d'entrainement
```bash
# Nombre d'episodes
python train_mario.py predictive -n 500

# Nombre de frames de warmup pour le cortex
python train_mario.py predictive --warmup 3000

# Repertoire de sauvegarde des checkpoints
python train_mario.py predictive --checkpoint-dir mes_modeles

# Generer le rapport PNG d'entrainement
python train_mario.py predictive --report

# Desactiver la simulation visuelle apres entrainement
python train_mario.py predictive --no-render

# Combinaison
python train_mario.py predictive -n 1000 --report --no-render
```

Le meilleur modele est **automatiquement sauvegarde** dans `checkpoints/best/` a chaque fois que la performance depasse le meilleur score precedent. Un checkpoint final est sauvegarde dans `checkpoints/final/`.

### 8.3 Inference (charger un modele sauvegarde)

#### Tester un modele (comparaison des 4 strategies)
```bash
python train_mario.py --load checkpoints/best
```

Affiche les resultats de 20 essais pour chaque strategie :
- Ganglions de la base (4.1)
- Modele predictif (4.2.1)
- Modele inverse cerebelleux (Fig. 11)
- Hybride (bascule automatique)

#### Playback visuel (regarder le modele jouer)
```bash
# Toutes les strategies, une apres l'autre
python train_mario.py --load checkpoints/best --render

# Une strategie specifique
python train_mario.py --load checkpoints/best --render --strategy bg
python train_mario.py --load checkpoints/best --render --strategy predictive
python train_mario.py --load checkpoints/best --render --strategy inverse
python train_mario.py --load checkpoints/best --render --strategy hybrid
```

### 8.4 CartPole (branche v2)

```bash
# Mode interactif
python train_cartpole.py

# Mode direct
python train_cartpole.py reactive
python train_cartpole.py predictive
```

### 8.5 Structure des checkpoints

```
checkpoints/
  best/
    cortex.pt              # Poids CNN encodeur + decodeur
    basal_ganglia.pt       # Poids critique + acteur MLP
    forward_model.npz      # Poids cellules granulaires + Purkinje
    inverse_model.npz      # Idem
  final/
    cortex.pt
    basal_ganglia.pt
    forward_model.npz
    inverse_model.npz
```

### 8.6 Resume des commandes

| Commande | Description |
|----------|-------------|
| `python train_mario.py` | Entrainement interactif |
| `python train_mario.py predictive -n 500` | 500 episodes, mode predictif |
| `python train_mario.py predictive --report` | Entrainement + rapport PNG |
| `python train_mario.py --load checkpoints/best` | Test d'un modele sauvegarde |
| `python train_mario.py --load checkpoints/best --render` | Playback visuel |
| `python train_mario.py --load checkpoints/best --render --strategy predictive` | Playback d'une strategie |

### 8.7 Troubleshooting

| Probleme | Solution |
|----------|----------|
| `nes-py` ne s'installe pas | Installer Visual C++ Build Tools 14.0+, ou utiliser le fallback Atari |
| `ModuleNotFoundError: ale_py` | `pip install ale-py` |
| `ModuleNotFoundError: cv2` | `pip install opencv-python` |
| Pas de rendu visuel | Verifier que `render_mode="human"` est supporte par l'environnement |
| Erreur CUDA | Ajouter `device="cuda"` aux constructeurs (cortex, BG) si GPU disponible |
