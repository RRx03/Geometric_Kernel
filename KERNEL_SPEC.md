# KERNEL_SPEC v1.0 — Geometric Kernel Specification

**Projet** : Geometric Kernel — Moteur SDF 3D avec Ray Marching & Marching Cubes
**Auteur** : Spécification co-rédigée, Phase 0 du refactoring complet
**Date** : Mars 2026
**Statut** : v1.0 — Validée, prête pour implémentation

---

## Table des matières

1. [Vue d'ensemble et philosophie](#1-vue-densemble-et-philosophie)
2. [Architecture modulaire](#2-architecture-modulaire)
3. [Système de coordonnées et transformations](#3-système-de-coordonnées-et-transformations)
4. [Modes d'affichage](#4-modes-daffichage)
5. [Primitives SDF](#5-primitives-sdf)
6. [Opérations CSG et transformations](#6-opérations-csg-et-transformations)
7. [Algorithme de signe — Winding Number généralisé](#7-algorithme-de-signe--winding-number-généralisé)
8. [Distance aux courbes de Bézier — Solver hybride](#8-distance-aux-courbes-de-bézier--solver-hybride)
9. [Format du buffer GPU plat (Stack Machine)](#9-format-du-buffer-gpu-plat-stack-machine)
10. [Évaluateur SDF (CPU)](#10-évaluateur-sdf-cpu)
11. [Shader Metal (GPU) — Miroir exact](#11-shader-metal-gpu--miroir-exact)
12. [Ray Marcher](#12-ray-marcher)
13. [Calcul de normales](#13-calcul-de-normales)
14. [Système de caméra](#14-système-de-caméra)
15. [Interface utilisateur et ergonomie](#15-interface-utilisateur-et-ergonomie)
16. [Configuration par JSON — render_config.json](#16-configuration-par-json--render_configjson)
17. [Marching Cubes et export STL](#17-marching-cubes-et-export-stl)
18. [Parsing de scène JSON](#18-parsing-de-scène-json)
19. [Pipeline de test et golden values](#19-pipeline-de-test-et-golden-values)
20. [Stratégie de performance](#20-stratégie-de-performance)
21. [Généralisation 3D — Chemin d'évolution](#21-généralisation-3d--chemin-dévolution)
22. [Architecture C++ cible](#22-architecture-c-cible)

---

## 1. Vue d'ensemble et philosophie

### 1.1 Rôle du Kernel

Le Geometric Kernel est la couche 3 du système de design génératif. Il reçoit un arbre CSG en JSON produit par Athena (le solveur physique) et le transforme en :

- **Rendu temps réel** via ray marching sur GPU (Metal)
- **Maillage exportable** via Marching Cubes → STL binaire

Le kernel ne participe PAS à l'optimisation. Il visualise et exporte. Le solveur est autonome.

### 1.2 Principes fondamentaux

**P1 — 3D natif, 2D comme spécialisation.**
Toute primitive, tout algorithme, toute structure de données est conçue en 3D. L'axisymétrie est un cas particulier qui utilise une réduction de dimension `float3 → float2` au point d'entrée de la primitive, jamais dans le cœur algorithmique. Quand on ajoutera des primitives purement 3D (meshes SDF, NURBS), elles s'intègreront sans modification de l'architecture.

**P2 — Miroir exact CPU/GPU.**
L'évaluateur CPU et le fragment shader Metal implémentent le même algorithme, avec les mêmes constantes, les mêmes branchements, les mêmes nombres d'itérations. La divergence maximale tolérée entre CPU et GPU est `|d_cpu - d_gpu| < 1e-5` sur tout point de l'espace. Cela garantit que le Marching Cubes (CPU) produit un maillage fidèle au rendu (GPU).

**P3 — Modularité et extensibilité.**
Chaque composant (primitive SDF, opération CSG, ray marcher, mesher) est un module indépendant avec une interface claire. Ajouter une primitive = ajouter un type dans l'enum + implémenter `flatten()` côté C++ + implémenter `eval()` côté CPU et GPU. Zéro modification ailleurs.

**P4 — Robustesse numérique.**
Aucune division par zéro possible. Aucun edge case non traité. Chaque fonction a des préconditions documentées et des comportements de fallback définis. Les epsilons ne sont pas magiques — ils sont dérivés de la physique du problème (taille caractéristique de la pièce, précision machine float32).

**P5 — Performance prévisible.**
Le coût d'évaluation de chaque primitive est borné et connu. Le ray marcher a un budget de pas fixe. Le Marching Cubes a un budget de voxels explicite. Aucune boucle infinie possible.

### 1.3 Unités

Le kernel travaille en **mètres** (SI), conformément à la sortie d'Athena. L'export STL convertit en millimètres (×1000). Toutes les constantes numériques (epsilons, bornes de ray marching, résolution MC) sont exprimées en mètres.

---

## 2. Architecture modulaire

### 2.1 Couches du kernel

```
┌─────────────────────────────────────────────────────┐
│                    Application                       │
│        main.cpp — SDL2 event loop, CLI              │
├─────────────────────────────────────────────────────┤
│                   SceneParser                        │
│        JSON → Arbre SDFNode (AST géométrique)       │
├──────────────────┬──────────────────────────────────┤
│   Renderer       │         Mesher                    │
│   Metal pipeline │    Marching Cubes → STL          │
│   Ray marcher    │    Utilise SDFEvaluator (CPU)    │
│   Fragment shader│                                   │
├──────────────────┴──────────────────────────────────┤
│              SDFEvaluator (CPU)                       │
│     Stack machine miroir du shader GPU              │
├─────────────────────────────────────────────────────┤
│              SDFNode (AST)                            │
│    Arbre C++ → flatten() → buffer GPU plat          │
├─────────────────────────────────────────────────────┤
│              SDFShared.h                              │
│    Types, enum, struct SDFNodeGPU (64 bytes)        │
│    Partagé entre C++ et Metal                       │
└─────────────────────────────────────────────────────┘
```

### 2.2 Flux de données

```
scene.json ──→ SceneParser ──→ SDFNode tree ──→ flatten() ──→ SDFNodeGPU buffer[]
                                                                 │
                                              ┌──────────────────┴──────────────────┐
                                              ▼                                      ▼
                                        Metal buffer                          SDFEvaluator
                                        Fragment shader                       Marching Cubes
                                        Ray marching                          → STL export
                                        → Écran                               → Fichier
```

### 2.3 Contrat inter-modules

| Producteur | Consommateur | Interface | Invariant |
|---|---|---|---|
| SceneParser | SDFNode tree | `parseFile() → shared_ptr<SDFNode>` | Arbre valide, pas de nœud null |
| SDFNode tree | GPU buffer | `flatten(vector<SDFNodeGPU>&) → int` | Buffer linéaire, indices valides |
| GPU buffer | SDFEvaluator | `evaluate(float3) → float` | Même résultat que le shader |
| GPU buffer | Fragment shader | `map(float3) → float` | Même résultat que l'évaluateur |
| SDFEvaluator | Mesher | `evaluate(float3) → float` | SDF continu, lipschitz ≤ 1 |

---

## 3. Système de coordonnées et transformations

### 3.1 Repère monde

Le repère monde est **main droite** :

- **Y** = haut (vertical)
- **X** = droite
- **Z** = vers l'observateur

C'est le repère standard d'OpenGL/Metal. L'axe de révolution pour l'axisymétrique est **Y**.

### 3.2 Réduction axisymétrique

Pour les primitives 2D axisymétriques, la réduction de dimension se fait au point d'entrée de l'évaluation :

```
float3 pos3d → float2 pos2d = (length(pos3d.xz), pos3d.y)
                                    ↑ r (radial)    ↑ y (axial)
```

Cette réduction est effectuée **une seule fois**, dans le dispatch de la stack machine, AVANT l'appel à la fonction d'évaluation de la primitive. La primitive elle-même ne connaît que `float2`.

### 3.3 Transformations

Le nœud `SDF_OP_TRANSFORM` applique une transformation affine (translation, rotation, scale uniforme) au point avant évaluation de l'enfant. Voir Section 6.4 pour le détail de l'encoding GPU. En v1, les primitives simples restent positionnées via leur champ `position`, mais `SDF_OP_TRANSFORM` est disponible pour les assemblages et les instanciations multiples.

### 3.4 Systèmes de coordonnées supportés (à terme)

| Système | Variables | Réduction vers 3D cartésien |
|---|---|---|
| `axisymmetric` | `(r, y)` | `(r·cos(θ), y, r·sin(θ))` — implicite par symétrie |
| `cartesian` | `(x, y, z)` | Identité |
| `cylindrical` | `(r, θ, z)` | `(r·cos(θ), z, r·sin(θ))` |
| `spherical` | `(ρ, θ, φ)` | `(ρ·sin(φ)·cos(θ), ρ·cos(φ), ρ·sin(φ)·sin(θ))` |

En v1, seul `axisymmetric` et `cartesian` sont implémentés. Les autres s'ajouteront comme de simples fonctions de conversion `coords → float3` dans le SceneParser.

---

## 4. Modes d'affichage

### 4.1 Trois modes

Le kernel doit savoir dans quel mode il affiche la géométrie. Le mode est spécifié dans le `scene.json` ou le `render_config.json` :

| Mode | Description | Réduction | Usage |
|---|---|---|---|
| `"3d"` | Rendu 3D complet | Aucune — les primitives sont évaluées directement en float3 | Pièces 3D natives (futures) |
| `"2d_axisymmetric"` | Profil 2D en révolution autour de l'axe Y | `float3 → float2(length(p.xz), p.y)` | Tuyères, chambres, corps de révolution |
| `"2d"` | Profil 2D plat (plan XY) | `float3 → float2(p.x, p.y)` — le rendu montre une coupe | Profils d'aile, sections, débogage |

### 4.2 Détection automatique

Si le mode n'est pas spécifié explicitement, le kernel le déduit des primitives présentes dans la scène :
- Si la scène contient au moins une primitive de la famille `*_2D` et aucune primitive 3D pure → `"2d_axisymmetric"` (défaut historique)
- Si la scène ne contient que des primitives 3D → `"3d"`
- Si le JSON spécifie `"display_mode"` → c'est celui-là qui prime

### 4.3 Impact sur l'évaluation

Le mode d'affichage détermine la fonction de réduction utilisée par le dispatch de la stack machine. Chaque primitive 2D a une réduction par défaut (axisymétrique) mais le mode `"2d"` utilise une réduction plane à la place. Les primitives 3D ne sont pas affectées par le mode.

### 4.4 Impact sur l'export

- En mode `"2d_axisymmetric"` : le Marching Cubes produit un solide de révolution complet (360°)
- En mode `"2d"` : le Marching Cubes produit une extrusion du profil (épaisseur configurable)
- En mode `"3d"` : le Marching Cubes fonctionne normalement

---

## 5. Primitives SDF

### 5.1 Taxonomie

Les primitives sont classées en deux familles, mais partagent la même interface :

**Famille 3D native** — évaluation directe en `float3` :

| Type | Enum | Paramètres |
|---|---|---|
| Sphère | `SDF_TYPE_SPHERE` | `position: float3`, `radius: float` |
| Boîte | `SDF_TYPE_BOX` | `position: float3`, `halfExtents: float3` |
| Cylindre | `SDF_TYPE_CYLINDER` | `position: float3`, `radius: float`, `halfHeight: float` |
| Tore | `SDF_TYPE_TORUS` | `position: float3`, `majorRadius: float`, `minorRadius: float` |
| Capsule | `SDF_TYPE_CAPSULE` | `pointA: float3`, `pointB: float3`, `radius: float` |

**Famille 2D axisymétrique** — réduction `float3 → float2` puis évaluation en `float2` :

| Type | Enum | Paramètres |
|---|---|---|
| Cercle 2D | `SDF_TYPE_CIRCLE_2D` | `center: float2`, `radius: float` |
| Rectangle 2D | `SDF_TYPE_RECT_2D` | `center: float2`, `halfExtents: float2` |
| Bézier quadratique 2D | `SDF_TYPE_BEZIER2D` | `p0, p1, p2: float2`, `thickness: float` |
| Bézier cubique 2D | `SDF_TYPE_CUBIC_BEZIER2D` | `p0, p1, p2, p3: float2`, `thickness: float` |
| Composite Spline 2D | `SDF_TYPE_COMPOSITE_SPLINE2D` | `points[N]: float2`, `thickness: float` |

### 5.2 Sphère

```
SDF(p) = length(p - center) - radius
```

Exacte, aucun edge case.

### 5.3 Boîte (alignée aux axes)

```
d = abs(p - center) - halfExtents
SDF(p) = length(max(d, 0)) + min(max(d.x, d.y, d.z), 0)
```

Distance euclidienne exacte. Le terme `min(max(...), 0)` gère l'intérieur.

### 5.4 Cylindre (axe Y)

```
d_radial = length(p.xz - center.xz) - radius
d_axial = abs(p.y - center.y) - halfHeight
d = float2(d_radial, d_axial)
SDF(p) = length(max(d, 0)) + min(max(d.x, d.y), 0)
```

### 5.5 Tore (axe Y)

```
q = float2(length(p.xz - center.xz) - majorRadius, p.y - center.y)
SDF(p) = length(q) - minorRadius
```

### 5.6 Capsule

```
pa = p - pointA
ba = pointB - pointA
t = clamp(dot(pa, ba) / dot(ba, ba), 0, 1)
SDF(p) = length(pa - ba * t) - radius
```

### 5.7 Cercle 2D (axisymétrique → Tore 3D)

Après réduction : `p2d = (length(p.xz), p.y)`

```
SDF(p2d) = length(p2d - center) - radius
```

### 5.8 Rectangle 2D (axisymétrique → Cylindre creux)

Après réduction : `p2d = (length(p.xz), p.y)`

```
d = abs(p2d - center) - halfExtents
SDF(p2d) = length(max(d, 0)) + min(max(d.x, d.y), 0)
```

### 5.9 Bézier quadratique 2D

Courbe définie par 3 points de contrôle `(A, B, C)`. La distance est calculée par le solver hybride (Section 7). Le signe est déterminé par le winding number (Section 6) quand `thickness = 0`.

```
d_unsigned = hybridBezierDistance(p2d, A, B, C)  // Section 7
if thickness > 0:
    SDF(p2d) = d_unsigned - thickness
else:
    sign = windingSign(p2d, profile)              // Section 6
    SDF(p2d) = sign * d_unsigned
```

### 5.10 Bézier cubique 2D

Même logique que la quadratique, avec 4 points de contrôle `(A, B, C, D)`. Le solver hybride (Section 7) supporte les deux ordres.

### 5.11 Composite Spline 2D

C'est la primitive critique — celle qui encode les profils de tuyère d'Athena.

**Définition** : une B-spline ouverte (open uniform B-spline) de degré 2, définie par N points de contrôle. Elle se décompose en `N-2` segments de Bézier quadratique via l'algorithme standard de décomposition B-spline :

```
Si N = 2 : un seul segment dégénéré A → midpoint(A,B) → B
Si N = 3 : un segment (pts[0], pts[1], pts[2])
Si N ≥ 4 :
    Segment 0     : (pts[0], pts[1], mid(pts[1], pts[2]))
    Segment k     : (mid(pts[k], pts[k+1]), pts[k+1], mid(pts[k+1], pts[k+2]))   pour 1 ≤ k ≤ N-4
    Segment N-3   : (mid(pts[N-3], pts[N-2]), pts[N-2], pts[N-1])
```

**Distance non-signée** : minimum sur tous les segments via le solver hybride (Section 7).

**Distance signée** : déterminée par le winding number généralisé (Section 6). C'est le changement architectural majeur par rapport à l'ancien kernel.

---

## 6. Opérations CSG et transformations

### 6.1 Opérations binaires

| Opération | Enum | Formule |
|---|---|---|
| Union | `SDF_OP_UNION` | `min(d1, d2)` |
| Soustraction | `SDF_OP_SUBTRACT` | `max(d1, -d2)` |
| Intersection | `SDF_OP_INTERSECT` | `max(d1, d2)` |
| Union lisse | `SDF_OP_SMOOTH_UNION` | Polynomial smooth min (voir ci-dessous) |

### 6.2 Union lisse (Smooth Union)

```
h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0, 1)
SDF = d2 * (1 - h) + d1 * h - k * h * (1 - h)
```

Où `k` est le facteur de lissage. `k = 0` équivaut à `min(d1, d2)`.

### 6.3 Opérations n-aires (futur, prévu)

L'architecture actuelle est binaire (left/right). Pour le futur, un nœud `SDF_OP_N_UNION` prendra un nombre variable d'enfants. Le buffer plat encodera `childCount` dans le header et les indices des enfants dans des DATA_CARRIERs, exactement comme CompositeSpline2D encode ses points. Ce n'est pas implémenté en v1 mais la structure de données le permet déjà.

### 6.4 Transformation géométrique — SDF_OP_TRANSFORM

Le nœud `SDF_OP_TRANSFORM` applique une transformation affine (translation, rotation, scale uniforme) à son unique enfant. La transformation est encodée comme une matrice inverse 4×3 dans des DATA_CARRIERs.

**Principe** : pour évaluer `Transform(T, child)` au point `p`, on applique l'inverse de la transformation au point, puis on évalue l'enfant :

```
SDF_transform(p) = SDF_child(T⁻¹ · p) * scale_factor
```

Le `scale_factor` corrige la distance en cas de scale non-unitaire : `scale_factor = 1.0` pour rotation + translation pure, `scale_factor = s` pour un scale uniforme `s`.

**Encoding GPU** :

```
nodes[i]   : type = SDF_OP_TRANSFORM
             leftChildIndex = index de l'enfant
             params = (scale_factor, 0, 0, 0)

nodes[i+1] : type = SDF_DATA_CARRIER   // Lignes 0-1 de la matrice inverse
             position = (m00, m01, m02)
             params   = (m03, m10, m11, m12)

nodes[i+2] : type = SDF_DATA_CARRIER   // Ligne 2 de la matrice inverse
             position = (m13, m20, m21)
             params   = (m22, m23, 0, 0)
```

On n'encode que les 3 premières lignes (la 4ème est toujours `[0, 0, 0, 1]` pour une transformation affine).

**Parsing JSON** :

```json
{
    "type": "Transform",
    "translate": [0.1, 0.0, 0.0],
    "rotate": { "axis": [0, 1, 0], "angle_deg": 45 },
    "scale": 1.0,
    "child": { "type": "Sphere", "radius": 0.05 }
}
```

**Cas d'usage** : positionnement de sous-assemblages, rotation de pales, symétries, instances multiples d'une même géométrie avec des orientations différentes.

### 6.5 Propriété de Lipschitz

Les opérations CSG préservent la propriété de Lipschitz-1 des SDF sous-jacents : si `|∇SDF₁| ≤ 1` et `|∇SDF₂| ≤ 1`, alors `|∇(op(SDF₁, SDF₂))| ≤ 1`. Cela garantit que le ray marcher peut avancer en toute sécurité de `d` à chaque pas. Exception : SmoothUnion avec un grand `k` peut localement dépasser 1 — le ray marcher applique un facteur de sécurité de `0.8` dans les zones de blending. `SDF_OP_TRANSFORM` préserve Lipschitz-1 si le scale est uniforme.

---

## 7. Algorithme de signe — Winding Number généralisé

### 7.1 Pourquoi changer

L'ancien kernel déterminait le signe par `r_point < r_curve` (le point est-il entre l'axe et la courbe ?). Cette heuristique échoue aux jonctions de segments B-spline, là où `r_curve` fait un saut discret d'un segment à l'autre. Résultat : des encoches visuelles aux jonctions.

### 7.2 Principe du winding number

Le winding number `w(p)` d'un point `p` par rapport à une courbe fermée `C` compte le nombre de fois que `C` tourne autour de `p` :

```
w(p) = (1 / 2π) ∮_C dθ
```

Si `w(p) ≠ 0`, le point est à l'intérieur. Si `w(p) = 0`, il est à l'extérieur.

### 7.3 Application à un profil axisymétrique ouvert

Un profil de tuyère n'est PAS une courbe fermée — c'est un arc ouvert du plan `(r, y)`. Pour obtenir une région fermée qui représente le solide (tout ce qui est entre l'axe et le profil), on ferme le profil en ajoutant virtuellement :

1. Un segment vertical du dernier point `pts[N-1]` vers l'axe `(0, pts[N-1].y)`
2. Un segment horizontal le long de l'axe de `(0, pts[N-1].y)` à `(0, pts[0].y)`
3. Un segment vertical de `(0, pts[0].y)` vers le premier point `pts[0]`

Cela forme un polygone fermé (les segments de fermeture sont des segments droits, pas des courbes).

### 7.4 Calcul du winding number sur les segments B-spline

Pour chaque segment de Bézier quadratique `(A, B, C)`, la contribution au winding number est l'angle signé `Δθ` subtenu par l'arc vu depuis le point `p` :

```
Δθ = atan2(cross(A-p, C-p), dot(A-p, C-p))
```

C'est une approximation linéaire. Pour plus de précision (courbes très incurvées), on subdivise le segment en `K` sous-arcs et on somme les `Δθ` :

```
Δθ_total = Σ_{k=0}^{K-1} atan2(cross(Q_k - p, Q_{k+1} - p), dot(Q_k - p, Q_{k+1} - p))
```

Où `Q_k = evalQuad(A, B, C, k/K)`.

**Choix de K — adaptatif selon la courbure** : le nombre de subdivisions par segment B-spline est déterminé dynamiquement :

```
K(segment) = clamp(ceil(curvature(A, B, C) / threshold), K_MIN, K_MAX)
```

Où `curvature(A, B, C) = length(A - 2B + C)` est une estimation de la courbure (distance du point de contrôle médian à la corde). Les constantes :

| Paramètre | Valeur | Rôle |
|---|---|---|
| `K_MIN` | 2 | Segments quasi-droits |
| `K_MAX` | 16 | Segments très courbés (hélices, impellers) |
| `threshold` | 0.01 m | Seuil de courbure pour augmenter K |

Cela couvre les profils simples (tuyères : K≈2-4) comme les géométries complexes (pales de turbine, impellers : K≈8-16). Le coût additionnel est `K × atan2` par segment, adapté au besoin réel.

### 7.5 Calcul sur les segments de fermeture

Les 3 segments de fermeture (droites) ont une contribution exacte au winding number :

```
Pour un segment droit (P₁, P₂) :
Δθ = atan2(cross(P₁ - p, P₂ - p), dot(P₁ - p, P₂ - p))
```

### 7.6 Signe final

```
winding = Σ(Δθ_spline_segments) + Σ(Δθ_closure_segments)
sign = (winding > π) ? -1.0 : 1.0    // |w| ≥ 1 → intérieur
```

Note : on compare à `π` et non `0` parce que la somme des angles donne `2π·w(p)` et on veut `|w| ≥ 1`.

### 7.7 Avantages

- **Continu** : pas de discontinuité aux jonctions de segments
- **Robuste** : fonctionne pour n'importe quelle forme de profil (convexe, concave, avec inflexions)
- **Extensible** : fonctionne identiquement pour des profils 3D fermés (surfaces)
- **Parallélisable** : chaque segment est indépendant

### 7.8 Optimisation : early-out par bounding box

Avant de calculer le winding number complet, on teste si le point est clairement à l'extérieur :

```
if p2d.r < 0 :                          toujours extérieur (impossible physiquement, mais défense)
if p2d.y < yMin - margin || p2d.y > yMax + margin :  extérieur (au-delà du profil)
if p2d.r > rMax + margin :               extérieur (au-delà du rayon max)
```

Où `yMin, yMax, rMax` sont pré-calculés sur les points de contrôle, et `margin` est un petit facteur de sécurité.

---

## 8. Distance aux courbes de Bézier — Solver hybride

### 8.1 Problème

Trouver le point `C(t*)` le plus proche de `p` sur une courbe de Bézier `C(t)`, où `t ∈ [0, 1]`.

### 8.2 Approche analytique (seed)

Pour une Bézier quadratique `C(t) = (1-t)²A + 2(1-t)tB + t²C`, la condition d'optimalité est :

```
d/dt |p - C(t)|² = 0
⟺ dot(C(t) - p, C'(t)) = 0
```

C'est un polynôme de degré 3 en `t`. Les racines réelles dans `[0, 1]` donnent les candidats. On ajoute les bornes `t = 0` et `t = 1`.

Pour une Bézier cubique, c'est un polynôme de degré 5 — on passe directement au sampling.

### 8.3 Approche Newton-Raphson (raffinement)

À partir de chaque seed `t₀` (racine analytique ou sample), on itère :

```
f(t) = dot(C(t) - p, C'(t))
f'(t) = dot(C'(t), C'(t)) + dot(C(t) - p, C''(t))
t_{n+1} = t_n - f(t_n) / f'(t_n)
t_{n+1} = clamp(t_{n+1}, 0, 1)
```

**Nombre d'itérations** : 3 suffisent pour converger à `< 1e-6` depuis un bon seed.

**Guard contre division par zéro** : si `|f'(t)| < 1e-10`, on ne met pas à jour `t`.

### 8.4 Algorithme complet

```
function hybridBezierDistance(p, A, B, C) → float:
    // Phase 1 : Seeds analytiques
    coeffs = computeCubicCoeffs(p, A, B, C)      // coefficients du polynôme degré 3
    roots = solveCubic(coeffs)                     // 1 à 3 racines réelles
    candidates = filter(roots, t ∈ [0, 1]) ∪ {0, 1}

    // Phase 2 : Raffinement Newton
    bestDist = +∞
    for t0 in candidates:
        t = t0
        for i in 0..2:                             // 3 itérations de Newton
            ft = dot(C(t) - p, C'(t))
            fpt = dot(C'(t), C'(t)) + dot(C(t) - p, C''(t))
            if |fpt| > 1e-10:
                t = clamp(t - ft / fpt, 0, 1)
        d = length(p - C(t))
        bestDist = min(bestDist, d)

    return bestDist
```

### 8.5 Fallback pour Bézier cubique

Pas de solution analytique simple. On utilise un sampling uniforme comme seed :

```
function hybridCubicBezierDistance(p, A, B, C, D) → float:
    // Phase 1 : Sampling uniforme (8 échantillons)
    bestT = 0, bestDist = +∞
    for j in 0..8:
        t = j / 8.0
        d = length(p - evalCubic(A, B, C, D, t))
        if d < bestDist: bestDist = d; bestT = t

    // Phase 2 : Newton depuis le meilleur sample
    t = bestT
    for i in 0..4:                                  // 4 itérations
        ft = dot(evalCubic(t) - p, evalCubicDeriv(t))
        fpt = dot(evalCubicDeriv(t), evalCubicDeriv(t)) + dot(evalCubic(t) - p, evalCubicDeriv2(t))
        if |fpt| > 1e-10:
            t = clamp(t - ft / fpt, 0, 1)

    return length(p - evalCubic(A, B, C, D, t))
```

### 8.6 Résolution de cubique (pour les seeds analytiques)

On utilise la formule de Cardano, avec la réduction de Tschirnhaus pour éliminer le terme quadratique. L'implémentation doit gérer le cas discriminant `Δ < 0` (3 racines réelles, formule trigonométrique) et `Δ ≥ 0` (1 racine réelle). Référence : Numerical Recipes, §5.6.

```
function solveCubic(a, b, c, d) → float[3], int count:
    // Normaliser : t³ + pt + q = 0 (après Tschirnhaus)
    p = (3ac - b²) / (3a²)
    q = (2b³ - 9abc + 27a²d) / (27a³)
    Δ = -(4p³ + 27q²)

    if Δ > 0:    // 3 racines réelles
        θ = acos(-q/2 · sqrt(-27/p³)) / 3
        m = 2 · sqrt(-p/3)
        return [m·cos(θ), m·cos(θ - 2π/3), m·cos(θ - 4π/3)] - b/(3a), count=3
    else:         // 1 racine réelle + 2 complexes
        A = cbrt(-q/2 + sqrt(-Δ/108))
        B = cbrt(-q/2 - sqrt(-Δ/108))
        return [A + B] - b/(3a), count=1
```

### 8.7 Performance comparée

| Méthode | Coût par segment | Précision | Robustesse |
|---|---|---|---|
| Ternary search (ancien) | 6 + 12 = 18 evals | ~1e-4 | Bonne |
| Analytique pure (IQ) | 1 solve cubique | ~1e-7 | Fragile (edge cases) |
| **Hybride (cette spec)** | **1 solve + 3 Newton = ~8 evals** | **~1e-6** | **Robuste** |

Le gain est d'environ **2× sur l'ancien** en coût et **100× en précision**.

---

## 9. Format du buffer GPU plat (Stack Machine)

### 9.1 Struct SDFNodeGPU

```c
struct SDFNodeGPU {
    int   type;              // SDFNodeType enum
    int   leftChildIndex;    // Index dans le buffer (-1 si feuille)
    int   rightChildIndex;   // Index dans le buffer (-1 si feuille)
    int   _pad0;             // Padding pour alignement 16 bytes

    float3 position;         // 12 bytes — position ou données
    float  _pad_pos;         // Padding pour alignement 16 bytes

    float4 params;           // 16 bytes — paramètres spécifiques

    float  smoothFactor;     // Pour SmoothUnion
    float  _pad1;
    float  _pad2;
    float  _pad3;
};
// sizeof(SDFNodeGPU) = 64 bytes — alignement cache-friendly
```

**Modification vs ancien kernel** : on ajoute `_pad_pos` après `position` pour garantir un alignement 16 bytes de `position` ET de `params`. L'ancien kernel n'avait pas ce padding, ce qui causait des problèmes d'alignement subtils sur certains GPU.

### 9.2 Encoding par type

| Type | position | params | Notes |
|---|---|---|---|
| SPHERE | `center.xyz` | `(radius, 0, 0, 0)` | |
| BOX | `center.xyz` | `(halfX, halfY, halfZ, 0)` | |
| CYLINDER | `center.xyz` | `(radius, halfHeight, 0, 0)` | |
| TORUS | `center.xyz` | `(majorR, minorR, 0, 0)` | |
| CAPSULE | `pointA.xyz` | `(pointB.x, pointB.y, pointB.z, radius)` | |
| CIRCLE_2D | `(center.r, center.y, 0)` | `(radius, 0, 0, 0)` | |
| RECT_2D | `(center.r, center.y, 0)` | `(halfW, halfH, 0, 0)` | |
| BEZIER2D | `(p0.r, p0.y, thickness)` | `(p1.r, p1.y, p2.r, p2.y)` | |
| CUBIC_BEZIER2D | `(p0.r, p0.y, thickness)` | `(p1.r, p1.y, p2.r, p2.y)` | + 1 DATA_CARRIER |
| COMPOSITE_SPLINE2D | `(0, 0, 0)` | `(N, thickness, 0, 0)` | + ceil(N/3) DATA_CARRIERs |
| DATA_CARRIER | `(pt_k.r, pt_k.y, 0)` | `(pt_{k+1}.r, pt_{k+1}.y, pt_{k+2}.r, pt_{k+2}.y)` | 3 points par carrier |

### 9.3 Encoding des opérations CSG

| Type | leftChildIndex | rightChildIndex | smoothFactor |
|---|---|---|---|
| OP_UNION | index gauche | index droite | — |
| OP_SUBTRACT | index base | index à soustraire | — |
| OP_INTERSECT | index gauche | index droite | — |
| OP_SMOOTH_UNION | index gauche | index droite | `k` |

### 9.4 Ordre de linéarisation

L'arbre CSG est linéarisé en **post-ordre** (enfants avant parent). Cela permet à la stack machine de simplement itérer le buffer de gauche à droite : quand elle rencontre une feuille, elle pousse le résultat ; quand elle rencontre un opérateur, elle dépile deux valeurs et pousse le résultat.

```
Exemple : Subtract(CompositeSpline_ext, CompositeSpline_int)

Buffer :
  [0]  CompositeSpline2D header (ext)    → évalue, push
  [1]  DATA_CARRIER (pts ext)
  [2]  DATA_CARRIER (pts ext)
  ...
  [k]  CompositeSpline2D header (int)    → évalue, push
  [k+1] DATA_CARRIER (pts int)
  ...
  [m]  OP_SUBTRACT                       → pop 2, push max(d1, -d2)
```

### 9.5 Enum SDFNodeType

```c
enum SDFNodeType {
    SDF_DATA_CARRIER          = -1,

    // Primitives 3D
    SDF_TYPE_SPHERE           = 0,
    SDF_TYPE_BOX              = 1,
    SDF_TYPE_CYLINDER         = 2,
    SDF_TYPE_TORUS            = 3,
    SDF_TYPE_CAPSULE          = 4,

    // Primitives 2D axisymétriques
    SDF_TYPE_CIRCLE_2D        = 10,
    SDF_TYPE_RECT_2D          = 11,
    SDF_TYPE_BEZIER2D         = 12,
    SDF_TYPE_CUBIC_BEZIER2D   = 13,
    SDF_TYPE_COMPOSITE_SPLINE2D = 14,

    // Opérations CSG
    SDF_OP_UNION              = 100,
    SDF_OP_SUBTRACT           = 101,
    SDF_OP_INTERSECT          = 102,
    SDF_OP_SMOOTH_UNION       = 103,

    // Transformations
    SDF_OP_TRANSFORM          = 200,
};
```

**Modification vs ancien kernel** : les valeurs enum sont espacées par famille (0-9 = 3D, 10-19 = 2D, 100-109 = CSG, 200+ = transforms). L'ancien kernel avait un comptage séquentiel (0,1,2,3,4,5,6,7,10,11,12,13) qui ne laissait pas de place pour les ajouts futurs.

---

## 10. Évaluateur SDF (CPU)

### 10.1 Interface

```cpp
class SDFEvaluator {
public:
    explicit SDFEvaluator(const std::vector<SDFNodeGPU>& nodes);

    // Évaluation unique
    float evaluate(simd::float3 pos) const;

    // Évaluation batch (pour Marching Cubes)
    void evaluateBatch(const simd::float3* positions, float* results, size_t count) const;

    // Bounding box du SDF (calculée à la construction)
    simd::float3 boundsMin() const;
    simd::float3 boundsMax() const;
};
```

### 10.2 Stack machine

L'évaluateur itère le buffer linéairement. Une stack locale de taille 64 suffit pour des arbres CSG de profondeur raisonnable.

```
pour chaque nœud i dans buffer:
    si type == DATA_CARRIER : skip
    si type ∈ primitives :
        d = evalPrimitive(pos, node)
        push(d)
    si type ∈ opérations :
        d2 = pop(), d1 = pop()
        push(op(d1, d2))
retourner top()
```

### 10.3 Dispatch des primitives

Le dispatch se fait par un `switch` sur le type. Les primitives 2D font la réduction `float3 → float2` avant d'appeler la fonction d'évaluation :

```cpp
case SDF_TYPE_COMPOSITE_SPLINE2D: {
    float2 p2d = float2(sqrt(pos.x*pos.x + pos.z*pos.z), pos.y);
    stack[sp++] = evalCompositeSpline2D(p2d, i);
    i += numDataCarriers(node);
    break;
}
```

### 10.4 Calcul de bounding box

À la construction, l'évaluateur parcourt tous les nœuds feuilles pour calculer la bounding box englobante. Pour les primitives axisymétriques, la bounding box 3D est déduite du profil 2D :

```
Pour CompositeSpline2D avec points (r_i, y_i) :
    rMax = max(r_i) + margin
    yMin = min(y_i) - margin
    yMax = max(y_i) + margin
    bbox3D = [(-rMax, yMin, -rMax), (rMax, yMax, rMax)]
```

---

## 11. Shader Metal (GPU) — Miroir exact

### 11.1 Principe

Le fragment shader implémente exactement les mêmes algorithmes que l'évaluateur CPU. Le fichier `SDFShared.h` est inclus par les DEUX — il contient l'enum, la struct, et les macros conditionnelles `__METAL_VERSION__`.

### 11.2 Struct partagée

```c
#ifdef __METAL_VERSION__
    #define SDF_FLOAT3 float3
    #define SDF_FLOAT4 float4
    #define SDF_PAD_POS  // Metal aligne automatiquement
#else
    #include <simd/simd.h>
    #define SDF_FLOAT3 simd::float3
    #define SDF_FLOAT4 simd::float4
    #define SDF_PAD_POS float _pad_pos;
#endif
```

### 11.3 Fonctions partagées

Les fonctions mathématiques pures (evalQuad, solveCubic, hybridBezierDistance, windingNumber) seront idéalement dans un fichier `.h` partagé incluable des deux côtés. Si Metal ne le permet pas directement, on utilise un système de génération : un fichier source unique `.sdf.inl` avec des macros pour les types, compilé en C++ et en MSL.

### 11.4 Contraintes Metal

- Pas d'allocation dynamique → tableaux de taille fixe (`float2 pts[64]`)
- Pas de récursion → la stack machine est déjà itérative
- `atan2` est disponible en MSL
- `clamp`, `min`, `max`, `length`, `dot`, `cross` sont disponibles

---

## 12. Ray Marcher

### 12.1 Algorithme — Sphere tracing

```
pour chaque pixel (u, v) :
    ro = camPos
    rd = normalize(u * camRight + v * camUp + camForward)
    t = 0
    pour i in 0..MAX_STEPS :
        p = ro + rd * t
        d = map(p)                       // évalue le SDF complet
        hitEps = max(MIN_HIT_EPS, t * RELATIVE_HIT_EPS)
        si d < hitEps :
            → HIT : calculer normale, shading, retourner couleur
        si t > MAX_DISTANCE :
            → MISS : retourner couleur de fond
        t += d * STEP_SAFETY_FACTOR
    → MISS (budget épuisé)
```

### 12.2 Constantes

| Constante | Valeur | Justification |
|---|---|---|
| `MAX_STEPS` | 256 | Budget fixe, suffisant pour les scènes CSG |
| `MAX_DISTANCE` | 100.0 m | Au-delà, rien d'intéressant |
| `MIN_HIT_EPS` | 5e-5 m = 50 μm | Résolution minimale (épaisseur de paroi ~0.2mm) |
| `RELATIVE_HIT_EPS` | 3e-4 | Epsilon proportionnel à la distance (artefacts au loin) |
| `STEP_SAFETY_FACTOR` | 0.8 | Marge de sécurité pour les SDF non-exactes (smooth union) |

### 12.3 Optimisations futures

- **Over-relaxation** : `t += d * 1.5` quand la SDF est connue exacte (pas de smooth ops). Réduit le nombre de pas d'environ 30%.
- **Cone tracing** : pour l'AO et les ombres douces, marcher un cône plutôt qu'un rayon.

---

## 13. Calcul de normales

### 13.1 Différences finies centrales

```
eps = max(MIN_NORMAL_EPS, t * RELATIVE_NORMAL_EPS)
n = normalize(
    map(p + (eps,0,0)) - map(p - (eps,0,0)),
    map(p + (0,eps,0)) - map(p - (0,eps,0)),
    map(p + (0,0,eps)) - map(p - (0,0,eps))
)
```

### 13.2 Constantes

| Constante | Valeur | Justification |
|---|---|---|
| `MIN_NORMAL_EPS` | 1e-4 m | Éviter les artefacts numériques |
| `RELATIVE_NORMAL_EPS` | 3e-4 | Adapter au loin |

### 13.3 Coût

6 évaluations SDF par pixel touché. C'est le coût dominant après le ray marching. Pour optimiser, on pourrait utiliser le gradient analytique des primitives simples (sphère, boîte) mais le gain ne justifie pas la complexité en v1.

---

## 14. Système de caméra

### 14.1 Modèle — Caméra orbitale sphérique

La caméra orbite autour d'un point cible avec 3 degrés de liberté :

```
position = target + distance * (
    cos(elevation) * sin(azimuth),
    sin(elevation),
    cos(elevation) * cos(azimuth)
)
forward = normalize(target - position)
right = normalize(cross(worldUp, forward))
up = cross(forward, right)
```

### 14.2 Contrôles

| Action | Input | Effet |
|---|---|---|
| Orbiter | Clic gauche + drag | Modifie azimuth et elevation |
| Pan | Clic droit + drag | Translate target dans le plan caméra |
| Zoom | Scroll | Modifie distance (proportionnel) |

### 14.3 Paramètres par défaut

```
distance   = 0.5 m    (pièces SI, typiquement 0.1-1 m)
azimuth    = 0.5 rad  (léger angle pour voir le 3D)
elevation  = 0.3 rad  (légère élévation)
target     = (0, 0, 0)
```

### 14.4 Auto-framing (futur)

À terme, la caméra pourra se positionner automatiquement pour cadrer la bounding box de la scène. En v1, le positionnement est manuel.

### 14.5 Struct Uniforms (CPU → GPU)

```c
struct Uniforms {
    SDF_FLOAT3 camPos;     float _p1;
    SDF_FLOAT3 camForward; float _p2;
    SDF_FLOAT3 camRight;   float _p3;
    SDF_FLOAT3 camUp;      float _p4;
};
```

Chaque `float3` est paddé à 16 bytes pour l'alignement Metal. sizeof(Uniforms) = 64 bytes.

---

## 15. Interface utilisateur et ergonomie

### 15.1 Philosophie

Le kernel est une interface homme-machine. Il doit être ergonomique : navigation 3D fluide, fenêtre redimensionnable, éclairage lisible, contrôles intuitifs. L'objectif est que l'ingénieur puisse inspecter sa pièce sans friction.

### 15.2 Fenêtre et redimensionnement

- Fenêtre SDL2 avec flag `SDL_WINDOW_RESIZABLE`
- Support plein écran via `F11` ou double-clic sur la barre de titre
- Le viewport et les textures MSAA/depth sont recréés à chaque resize
- Ratio de pixels retina (HiDPI) géré via `SDL_GetWindowSizeInPixels` sur macOS
- Taille minimale : 400×300. Pas de taille maximale.

### 15.3 Navigation 3D

| Action | Input souris | Input trackpad | Effet |
|---|---|---|---|
| Orbiter | Clic gauche + drag | Deux doigts + drag | Rotation azimuth/élévation |
| Pan | Clic droit + drag | Shift + deux doigts | Translation du target |
| Zoom | Molette scroll | Pinch | Distance proportionnelle |
| Recentrer | Double-clic gauche | Double-tap | Recentre sur l'objet (auto-framing) |
| Reset vue | Touche `R` | Touche `R` | Retour à la vue par défaut |

Sensibilités :
- Orbite : `0.005 rad/pixel` (fluide, pas trop rapide)
- Pan : proportionnel à la distance caméra (`distance * 0.002 / pixel`)
- Zoom : proportionnel (`distance * 0.1 / cran de scroll`)
- Élévation clampée à `[-π/2 + 0.01, π/2 - 0.01]` (pas de flip au pôle)

### 15.4 Auto-framing

Sur pression de `F` ou au double-clic, la caméra se positionne pour cadrer la bounding box de la scène :

```
target = center(bbox)
distance = max(bbox_diagonal) * 1.5 / tan(fov/2)
azimuth = π/4     (vue 3/4)
elevation = π/6   (30° d'élévation)
```

Transition animée sur 0.3 secondes (interpolation linéaire de distance, azimuth, elevation, target).

### 15.5 Éclairage

Éclairage Phong basique avec 3 composantes :

```
ambient  = 0.15
diffuse  = max(dot(normal, lightDir), 0.0) * 0.7
specular = pow(max(dot(reflect(-lightDir, normal), viewDir), 0.0), 32.0) * 0.15
color    = baseColor * (ambient + diffuse) + specular
```

**Direction de la lumière** : fixée en espace caméra (la lumière suit la caméra). Vecteur par défaut : `normalize(camRight + 2*camUp + camForward)` — éclairage depuis le haut-droite.

**Couleur de base** : `(0.75, 0.78, 0.82)` — gris métallique neutre.

**Fond** : gradient vertical de `(0.12, 0.12, 0.14)` (bas) à `(0.22, 0.22, 0.25)` (haut) — fond sombre professionnel.

### 15.6 Ambient Occlusion (AO) approximé

AO basé sur le nombre de pas du ray marcher :

```
ao = 1.0 - (steps / MAX_STEPS) * 0.5
```

Simple et gratuit (pas de calcul supplémentaire). Donne une impression de profondeur dans les cavités.

### 15.7 Raccourcis clavier

| Touche | Action |
|---|---|
| `R` | Reset vue |
| `F` | Auto-frame (cadrer l'objet) |
| `F11` | Plein écran toggle |
| `W` | Toggle wireframe / solid (futur) |
| `E` | Exporter STL |
| `1` | Vue de face (azimuth=0, elevation=0) |
| `2` | Vue de droite (azimuth=π/2) |
| `3` | Vue de dessus (elevation=π/2-ε) |
| `4` | Vue 3/4 (azimuth=π/4, elevation=π/6) |
| `Esc` | Quitter |

---

## 16. Configuration par JSON — render_config.json

### 16.1 Principe

Tous les paramètres du kernel qui ne sont pas des données géométriques résident dans un fichier `render_config.json`. Cela permet de les coupler avec une interface Python sans recompiler le kernel.

### 16.2 Format

```json
{
    "display": {
        "width": 1280,
        "height": 720,
        "fullscreen": false,
        "msaa_samples": 4,
        "vsync": true
    },
    "ray_marcher": {
        "max_steps": 256,
        "max_distance": 100.0,
        "min_hit_eps": 5e-5,
        "relative_hit_eps": 3e-4,
        "step_safety_factor": 0.8
    },
    "normals": {
        "min_eps": 1e-4,
        "relative_eps": 3e-4
    },
    "lighting": {
        "ambient": 0.15,
        "diffuse_strength": 0.7,
        "specular_strength": 0.15,
        "specular_power": 32.0,
        "base_color": [0.75, 0.78, 0.82],
        "background_bottom": [0.12, 0.12, 0.14],
        "background_top": [0.22, 0.22, 0.25]
    },
    "camera": {
        "initial_distance": 0.5,
        "initial_azimuth": 0.5,
        "initial_elevation": 0.3,
        "initial_target": [0.0, 0.0, 0.0],
        "auto_frame": true
    },
    "export": {
        "format": "stl_binary",
        "scale_factor": 1000.0,
        "max_voxels_per_dim": 512,
        "min_voxel_size": 0.0,
        "auto_resolution": true
    },
    "winding": {
        "k_min": 2,
        "k_max": 16,
        "curvature_threshold": 0.01
    },
    "scene": {
        "file": "scene.json",
        "display_mode": "auto"
    }
}
```

### 16.3 Valeurs par défaut

Si `render_config.json` est absent, le kernel utilise les valeurs par défaut codées en dur (celles documentées dans les sections précédentes). Si un champ est absent du JSON, sa valeur par défaut est utilisée. Aucun champ n'est obligatoire.

### 16.4 Rechargement à chaud (futur)

À terme, le kernel pourra surveiller `render_config.json` via `kqueue`/`FSEvents` et recharger les paramètres en temps réel. En v1, le fichier est lu une seule fois au démarrage.

### 16.5 Couplage avec Python

L'interface Python (front-end) peut écrire ce fichier avant de lancer le kernel, ou le modifier pendant l'exécution (quand le rechargement à chaud sera implémenté). Les paramètres d'export sont particulièrement utiles pour le pipeline automatisé `Athena → Kernel → STL → slicer`.

---

## 17. Marching Cubes et export STL

### 17.1 Algorithme

Marching Cubes classique sur une grille 3D régulière. Pour chaque voxel, les 8 coins sont évalués par le SDFEvaluator CPU. L'iso-surface `SDF = 0` est extraite par interpolation linéaire sur les arêtes.

### 17.2 Résolution adaptative

La résolution doit être adaptée à l'épaisseur minimale de la pièce. Règle :

```
voxelSize = min(épaisseur_min / 3, dimension_max / MAX_VOXELS_PER_DIM)
```

Pour une tuyère avec paroi de 0.2 mm : `voxelSize ≤ 0.066 mm = 6.6e-5 m`.

**Budget** : `MAX_VOXELS_PER_DIM = 512`. Pour une pièce de 0.15 m, ça donne `voxelSize = 0.15/512 ≈ 3e-4 m = 0.3 mm`. Si la paroi est plus fine, il faudra soit augmenter le budget, soit utiliser un MC adaptatif (octree).

### 17.3 Parallélisation

```cpp
// Évaluation parallèle de la grille
std::for_each(std::execution::par, zSlices.begin(), zSlices.end(),
    [&](int z) {
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++)
                grid[z][y][x] = evaluator.evaluate(gridPoint(x, y, z));
    });
```

L'extraction des triangles est ensuite séquentielle par tranche Z (peu coûteux comparé à l'évaluation).

### 17.4 Export STL binaire

```
Header      : 80 bytes (description)
Num triangles : uint32
Pour chaque triangle :
    Normal    : float3 (12 bytes)
    Vertex 1  : float3 (12 bytes)
    Vertex 2  : float3 (12 bytes)
    Vertex 3  : float3 (12 bytes)
    Attribute : uint16 (0)
```

Les coordonnées sont converties en millimètres (`× 1000`) à l'écriture.

### 17.5 Validation watertight

Après extraction, on vérifie que le maillage est fermé (watertight) en comptant que chaque arête est partagée par exactement 2 triangles. Si ce n'est pas le cas, on log un warning mais on exporte quand même.

---

## 18. Parsing de scène JSON

### 18.1 Format d'entrée

Le JSON est produit par `Athena::GeometryBuilder`. Structure :

```json
{
    "type": "Intersect",
    "left": {
        "type": "Subtract",
        "base": {
            "type": "CompositeSpline2D",
            "points": [[r0, y0], [r1, y1], ...],
            "thickness": 0
        },
        "subtract": {
            "type": "CompositeSpline2D",
            "points": [[r0, y0], [r1, y1], ...],
            "thickness": 0
        }
    },
    "right": {
        "type": "Box",
        "position": [px, py, pz],
        "bounds": [bx, by, bz]
    }
}
```

### 18.2 Types supportés en v1

| Type JSON | Champs requis | Classe C++ |
|---|---|---|
| `"Sphere"` | `position`, `radius` | `Sphere` |
| `"Box"` | `position`, `bounds` | `Box` |
| `"Cylinder"` | `position`, `radius`, `height` | `Cylinder` |
| `"CompositeSpline2D"` | `points`, `thickness` | `CompositeSpline2D` |
| `"Bezier2D"` | `p0`, `p1`, `p2`, `thickness` | `Bezier2D` |
| `"CubicBezier2D"` | `p0`, `p1`, `p2`, `p3`, `thickness` | `CubicBezier2D` |
| `"Union"` | `left`, `right` | `Union` |
| `"Subtract"` | `base`, `subtract` | `Subtract` |
| `"Intersect"` | `left`, `right` | `Intersect` |
| `"SmoothUnion"` | `left`, `right`, `smoothFactor` | `SmoothUnion` |

### 18.3 Gestion d'erreurs

Le parseur doit :
- Rejeter les nœuds sans `type` avec un message clair
- Rejeter les champs manquants avec le nom du type et du champ
- Tronquer les messages d'erreur JSON à 200 chars pour la lisibilité
- Ne jamais retourner un nœud `nullptr` (exception si erreur)

### 18.4 Extensions prévues (non implémentées en v1)

- `"children"` : tableau pour opérations n-aires
- `"transform"` : matrice 4×4 ou `{"translate": ..., "rotate": ..., "scale": ...}`
- `"Mesh3D"` : référence à un fichier de maillage SDF
- `"coordinate_system"` : override du système de coordonnées par nœud

---

## 19. Pipeline de test et golden values

### 19.1 Principe

Chaque primitive et chaque opérateur a un jeu de **golden values** : des paires `(point, distance_attendue)` calculées analytiquement. Le pipeline de test est :

```
Golden Values → SDFEvaluator CPU → comparaison → PASS/FAIL
                                                      ↓
                                                 Shader GPU → comparaison CPU/GPU → PASS/FAIL
```

### 19.2 Golden values par primitive

**Sphère** (center=(0,0,0), radius=1) :

| Point | Distance | Justification |
|---|---|---|
| (0,0,0) | -1.0 | Centre |
| (1,0,0) | 0.0 | Surface |
| (2,0,0) | 1.0 | Extérieur |
| (0.5,0.5,0.5) | sqrt(0.75)-1 ≈ -0.134 | Intérieur |

**Boîte** (center=(0,0,0), halfExtents=(1,1,1)) :

| Point | Distance | Justification |
|---|---|---|
| (0,0,0) | -1.0 | Centre |
| (1,0,0) | 0.0 | Surface (face) |
| (1,1,0) | 0.0 | Surface (arête) |
| (2,0,0) | 1.0 | Extérieur (face) |
| (2,2,0) | sqrt(2)-0 ≈ 1.414 | Extérieur (coin) |

**CompositeSpline2D** (profil linéaire simple) :
Points : `[(0.05, -0.05), (0.05, 0.05)]` (cylindre de rayon 5cm, hauteur 10cm)

| Point 3D | Point 2D (r,y) | Distance | Justification |
|---|---|---|---|
| (0, 0, 0) | (0, 0) | -0.05 | Sur l'axe, intérieur |
| (0.05, 0, 0) | (0.05, 0) | 0.0 | Sur la paroi |
| (0.1, 0, 0) | (0.1, 0) | 0.05 | Extérieur |
| (0, 0.1, 0) | (0, 0.1) | >0 | Au-dessus, extérieur |

### 19.3 Tolérance

- CPU golden values : `|d_computed - d_expected| < 1e-5`
- GPU vs CPU : `|d_gpu - d_cpu| < 1e-5`

### 19.4 Automatisation

Un exécutable de test (`test_kernel`) charge un jeu de golden values, évalue via CPU et GPU, et retourne un code de sortie 0 (tout passe) ou 1 (échec avec détails).

---

## 20. Stratégie de performance

### 20.1 Budget cible

- **60 fps** minimum pour les scènes typiques (1 tuyère, ~20 nœuds SDF)
- **Résolution** : 800×600 minimum, idéalement fenêtre redimensionnable
- **Latence d'export STL** : < 30 secondes pour résolution 256³, < 5 minutes pour 512³

### 20.2 Profil de coût (ray marching)

Pour chaque pixel :
- Ray marching : ~100 pas en moyenne × 1 évaluation SDF = 100 evals
- Chaque eval SDF : ~20 nœuds × coût par nœud
- Nœuds coûteux : CompositeSpline2D = ~11 segments × 8 evals (hybride) ≈ 88 evals Bézier
- Normales : 6 evals SDF supplémentaires

**Total par pixel touché** : ~100 × (coût_map) + 6 × (coût_map) ≈ 106 × coût_map

### 20.3 Optimisations v1

1. **Early-out dans le ray marcher** : `MAX_DISTANCE` et budget de pas
2. **Epsilon adaptatif** : `hitEps` proportionnel à `t` (pas besoin de précision au loin)
3. **Skip DATA_CARRIER** : le compteur `i` saute les carriers (pas de branch inutile)
4. **Bézier hybride** : 2× plus rapide que le ternary search (Section 7)

### 20.4 Optimisations futures

1. **BVH sur les nœuds** : hiérarchie de bounding boxes pour court-circuiter les sous-arbres lointains
2. **Over-relaxation** : sphere tracing avec pas > d quand le SDF est exact
3. **LOD temporel** : réduire MAX_STEPS pendant les mouvements de caméra, augmenter au repos
4. **Compute shader MC** : paralléliser le Marching Cubes sur GPU plutôt que CPU

---

## 21. Généralisation 3D — Chemin d'évolution

### 21.1 Ce qui est déjà 3D dans cette spec

- Les primitives 3D natives (Sphere, Box, Cylinder, Torus, Capsule)
- Le ray marcher (travaille en float3)
- Le Marching Cubes (grille 3D)
- Le système de caméra (orbite 3D)
- Les opérations CSG (agnostiques à la dimension)
- Le buffer GPU plat (structure identique pour 2D et 3D)

### 21.2 Ce qui est 2D par choix d'implémentation (pas par limitation)

- Les profils de tuyère sont axisymétriques (premier cas d'usage)
- Le winding number est calculé en 2D (dans le plan r,y)
- Les golden values 2D testent la réduction axisymétrique

### 21.3 Ajouts pour le 3D complet

Pour passer au 3D complet, il faudra ajouter :

1. **Primitives 3D avancées** :
   - `Mesh3D` : SDF discrétisée sur une grille 3D, interpolation trilinéaire
   - `NURBS3D` : surface NURBS, distance par projection itérative
   - `ExtrudedProfile` : profil 2D extrudé le long d'un path 3D
   - `RevolvedProfile` : profil 2D avec angle de révolution partiel

2. **Winding number 3D** :
   Pour les surfaces fermées, le winding number se généralise en 3D via l'angle solide. L'algorithme est le même principe (somme des contributions) mais sur des triangles plutôt que des segments.

3. **Transformations** :
   `SDF_OP_TRANSFORM` avec une matrice 4×4 pour positionner/orienter des sous-assemblages.

4. **BVH** :
   Indispensable pour les scènes multi-pièces (>100 nœuds).

### 21.4 Ce qui ne changera PAS

- La struct SDFNodeGPU (64 bytes)
- La stack machine
- Le ray marcher
- Le Marching Cubes
- Le système de caméra
- Le format STL
- L'interface SDFEvaluator

C'est le but de cette architecture : les fondations sont 3D, seul le contenu (les primitives) évolue.

---

## 22. Architecture C++ cible

```
Geometric_Kernel/
├── Makefile
├── SDFShared.h                    # Enum, struct SDFNodeGPU — partagé C++/Metal
├── shaders/
│   └── kernel.metal               # Fragment shader : ray marcher + SDF eval
├── src/
│   ├── main.cpp                   # SDL2 event loop, CLI, orchestration
│   ├── Renderer.hpp/.cpp          # Metal pipeline, caméra orbitale, uniforms
│   ├── Camera.hpp/.cpp            # Logique de caméra (séparée du renderer)
│   ├── SDFNode.hpp                # Arbre AST : classes Sphere, Box, CompositeSpline2D...
│   ├── SDFEvaluator.hpp/.cpp      # Stack machine CPU, évaluation batch
│   ├── SDFMath.hpp                # Fonctions partagées : Bézier, winding, solveCubic
│   ├── SceneParser.hpp/.cpp       # JSON → arbre SDFNode
│   ├── Mesher.hpp/.cpp            # Marching Cubes → STL binaire
│   ├── MCTables.h                 # Tables du Marching Cubes (edge, tri)
│   └── MathUtils.h                # Utilitaires (clamp, lerp, etc.)
├── tests/
│   ├── test_primitives.cpp        # Golden values pour chaque primitive
│   ├── test_csg.cpp               # Golden values pour les opérations CSG
│   ├── test_winding.cpp           # Tests du winding number
│   ├── test_bezier.cpp            # Tests du solver hybride Bézier
│   └── test_cpu_gpu_mirror.cpp    # Comparaison CPU/GPU sur grille
├── metal-cpp/                     # Bindings Metal C++ (inchangé)
└── scenes/
    ├── test_sphere.json           # Scènes de test simples
    ├── test_box.json
    ├── test_nozzle.json           # Tuyère Athena
    └── scene.json                 # Scène courante (symlink ou copie)
```

### 22.1 Modifications structurelles vs ancien kernel

| Ancien | Nouveau | Raison |
|---|---|---|
| `Shared.h` (Uniforms) | Intégré dans `SDFShared.h` | Un seul header partagé CPU/GPU |
| Pas de `Camera.hpp` | `Camera.hpp` séparé | Découplage caméra/renderer |
| `SDFEvaluator` header-only | `.hpp + .cpp` | Compilation séparée, temps de build |
| Pas de `SDFMath.hpp` | Fonctions math partagées | Code commun CPU/GPU en un seul endroit |
| Pas de tests | `tests/` complet | Golden values, regression |
| `scene.json` en racine | `scenes/` répertoire | Organisation |
| Enum séquentielle | Enum espacée par famille | Extensibilité |

---

## Annexe A — Glossaire

| Terme | Définition |
|---|---|
| **SDF** | Signed Distance Field — fonction qui retourne la distance signée à une surface |
| **CSG** | Constructive Solid Geometry — combinaison de solides par opérations booléennes |
| **Ray marching** | Algorithme de rendu itératif utilisant le SDF pour avancer le long d'un rayon |
| **Sphere tracing** | Variante de ray marching où le pas = valeur du SDF (garanti sûr) |
| **Marching Cubes** | Algorithme d'extraction d'iso-surface sur grille 3D |
| **Winding number** | Nombre de tours d'une courbe autour d'un point — détermine intérieur/extérieur |
| **B-spline** | Courbe définie par N points de contrôle, décomposable en segments de Bézier |
| **Stack machine** | Modèle d'exécution où les opérandes sont empilés/dépilés (pas de registres nommés) |
| **Lipschitz** | Propriété d'une SDF : `|SDF(a) - SDF(b)| ≤ |a - b|` — le gradient est borné par 1 |
| **Golden value** | Valeur de référence calculée analytiquement pour les tests |

---

## Annexe B — Références

1. Inigo Quilez — "Distance functions" — https://iquilezles.org/articles/distfunctions/
2. Inigo Quilez — "Distance to quadratic Bezier" — https://iquilezles.org/articles/distfunctions2d/
3. Hart, J.C. — "Sphere tracing: A geometric method for the antialiased ray tracing of implicit surfaces" (1996)
4. Lorensen & Cline — "Marching Cubes: A high resolution 3D surface construction algorithm" (1987)
5. Hormann & Agathos — "The point in polygon problem for arbitrary polygons" (2001) — winding number
6. Press et al. — "Numerical Recipes" — §5.6 (résolution de cubiques)
7. Metal Shading Language Specification — Apple Developer Documentation

---

**Fin de la KERNEL_SPEC v1.0**

*Ce document doit être validé section par section avant le passage à la Phase 1 (implémentation).*
*Toute modification architecturale future doit être reflétée ici AVANT d'être codée.*
