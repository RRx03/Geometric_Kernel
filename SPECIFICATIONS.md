# Architecture du Solveur — Spécification v3

*Référence unique. Toute implémentation doit être compatible avec ce document.*

---

## 0. Principes

1. **L'utilisateur décrit un problème physique, pas une géométrie.**
2. **Tout est paramétrique.** Toute valeur peut être fixe, variable, dérivée, ou une expression dépendant de n'importe quoi.
3. **Le solveur gère l'espace.** Recentrage automatique, détection d'axe, l'utilisateur pose ses BC où il veut.
4. **Contraintes incompatibles = erreur explicite** avec diagnostic et suggestions.
5. **Modulaire à l'extrême.** Nouveau type de pièce = nouveau PhysicsModule, rien d'autre à toucher.

---

## 1. Structure du `problem.json`

```
problem.json
├── meta                  (type, unités SI)
├── materials             (propriétés mécaniques/thermiques)
├── named_points          (points nommés : fixes, variables, mobiles)
├── boundary_conditions   (surfaces + distributions physiques + anchors)
├── constraints           (exigences globales scalaires)
├── field_constraints     (relations continues ∀z, ∀(r,z), ∀(r,θ,z)...)
├── parameters            (fixed | variable | derived | expression)
└── optimization          (méthode, poids, tolérance)
```

---

## 2. Expressions — Le Langage Unifié

Toute valeur numérique dans le JSON peut être une expression.

| Forme | Exemple | Signification |
|-------|---------|---------------|
| Nombre | `0.04` | Valeur fixe |
| `"param:X"` | `"param:r_exit"` | Référence à un paramètre |
| `"material:X.Y"` | `"material:wall.yield_strength"` | Propriété matériau |
| `"point:X.Y"` | `"point:ENDTUYERE.z"` | Coordonnée d'un point nommé |
| `"field:X@Y"` | `"field:velocity@THROAT"` | Champ physique en un point nommé |
| `"field:X@[...]"` | `"field:velocity@[0, z]"` | Champ au point construit `[...]` |
| `"domain:X"` | `"domain:z"` | Variable paramétrique courante (dans field_constraints) |
| `"expr:..."` | `"expr:2 * field:velocity@[0, z]"` | Expression arbitraire |

**Grammaire :**
```
expr := number | "param:" id | "material:" id.prop | "point:" id.comp
      | "field:" quantity "@" (id | "[" expr, ... "]")
      | "domain:" id
      | expr op expr          // + - * / ^
      | func(expr, ...)       // sin cos sqrt min max abs exp log
      | "(" expr ")"
```

```cpp
class ExpressionEvaluator {
public:
    struct Context {
        std::map<std::string, float> parameters;
        std::map<std::string, MaterialProperties> materials;
        std::map<std::string, simd::float3> namedPoints;
        std::map<std::string, float> domainVariables; // z, r, theta...
        std::function<float(const std::string& field, simd::float3 point)> fieldEvaluator;
    };
    static float evaluate(const std::string& expr, const Context& ctx);
};
```

---

## 3. Named Points

```json
"named_points": {
    "INLET_CENTER": { "position": [0, 0, 0], "type": "fixed" },
    "ENDTUYERE":    { "position": [0, 0, "param:L_nozzle"], "type": "constrained" },
    "THROAT":       { "position": [0, 0, "param:z_throat"], "type": "variable" },
    "WALL_PROBE":   { "position": ["expr:profile_r(domain:z)", 0, "domain:z"], "type": "mobile", "domain_variable": "z" }
}
```

- `fixed` : coordonnées constantes
- `variable` : coordonnées optimisées par le solveur
- `constrained` : lié à une autre BC ou un paramètre
- `mobile` : glisse sur un domaine (n'existe que dans le contexte d'une `field_constraint`)

---

## 4. Boundary Conditions

```json
"boundary_conditions": [
    {
        "id": "inlet",
        "geometry": { "type": "Disk2D", "center": "point:INLET_CENTER", "normal_axis": "z", "radius": 0.04 },
        "distributions": {
            "total_pressure":    { "type": "uniform", "value": 7e6 },
            "total_temperature": { "type": "uniform", "value": 3500.0 },
            "velocity":          { "type": "function", "coordinate_system": "axisymmetric", "expression": "expr:param:V_max * (1 - (r / 0.04)^2)" }
        },
        "anchors": { "axis_alignment": "z" }
    },
    {
        "id": "outlet",
        "geometry": { "type": "Disk2D", "center": "point:ENDTUYERE", "normal_axis": "z", "radius": "param:r_exit" },
        "distributions": {
            "static_pressure": { "type": "target", "value": 101325.0, "tolerance": 0.05 }
        },
        "anchors": { "concentric_with": "inlet", "distance_along_axis": "param:L_nozzle" }
    }
]
```

**Types de distributions :**

| Type | Description |
|------|-------------|
| `uniform` | Constante sur la surface |
| `radial` | f(r) axisymétrique |
| `integrated` | Intégrale sur la surface (ex: débit total) |
| `target` | Objectif que le solveur doit atteindre |
| `function` | Expression arbitraire de (r, θ, x, y, z) selon le `coordinate_system` |

Pour le type `function`, l'utilisateur choisit son système de coordonnées :
- `"axisymmetric"` → variable `r` disponible (distance radiale), implicite ∀θ
- `"cylindrical"` → variables `r`, `theta` disponibles
- `"cartesian"` → variables `x`, `y`, `z` disponibles

---

## 5. Constraints (Scalaires Globales)

```json
"constraints": [
    { "id": "thrust",     "type": "performance", "quantity": "thrust", "op": ">=", "value": 50000.0 },
    { "id": "isp",        "type": "performance", "quantity": "specific_impulse", "op": "maximize" },
    { "id": "envelope",   "type": "geometric", "max_radius": 0.15, "max_length": 0.5 },
    { "id": "structural", "type": "structural", "quantity": "von_mises_max", "op": "<", "value": "expr:material:wall.yield_strength / material:wall.safety_factor" },
    { "id": "thermal",    "type": "thermal", "quantity": "wall_temp_max", "op": "<", "value": "material:wall.max_service_temperature" },
    { "id": "mass",       "type": "mass", "quantity": "total_mass", "op": "minimize" }
]
```

---

## 6. Field Constraints — Relations sur Domaines Continus

### 6.1 Principe

Une field constraint est une relation qui doit être satisfaite **pour chaque point** d'un domaine paramétré. C'est le cœur de la puissance du solveur.

### 6.2 Systèmes de Coordonnées — Convention `@[...]`

**Règle fondamentale : on écrit TOUJOURS toutes les coordonnées du système choisi.**

Les variables du domaine paramétrique **écrasent** les valeurs écrites
à leur position. Les coordonnées non-paramétriques sont évaluées
telles quelles. Il n'y a jamais d'ambiguïté sur l'ordre.

**Systèmes disponibles :**

| Système | Composantes dans `@[...]` | Dimension |
|---------|---------------------------|-----------|
| `axisymmetric` | `@[r, z]` | 2D (∀θ implicite) |
| `cylindrical` | `@[r, θ, z]` | 3D |
| `cartesian` | `@[x, y, z]` | 3D |
| `spherical` | `@[r, θ, φ]` | 3D |

**Comment ça fonctionne avec les domaines paramétriques :**

Prenons `cylindrical` (toujours 3 composantes `@[r, θ, z]`) :

| Domaine | On écrit | Effet |
|---------|----------|-------|
| `parametric_1d(z)` | `@[0.5, 0, z]` | `z` écrasé par le domaine. `r=0.5`, `θ=0` fixes. ∀z ∈ [lo, hi]. |
| `parametric_1d(theta)` | `@[0.5, theta, 0.1]` | `theta` écrasé. `r=0.5`, `z=0.1` fixes. ∀θ ∈ [lo, hi]. |
| `parametric_1d(r)` | `@[r, 0, 0.1]` | `r` écrasé. `θ=0`, `z=0.1` fixes. ∀r ∈ [lo, hi]. |
| `parametric_2d(r, z)` | `@[r, 0, z]` | `r` et `z` écrasés. `θ=0` fixe. |
| `parametric_3d(r, theta, z)` | `@[r, theta, z]` | Tout écrasé. Volume complet. |

Prenons `axisymmetric` (toujours 2 composantes `@[r, z]`) :

| Domaine | On écrit | Effet |
|---------|----------|-------|
| `parametric_1d(z)` | `@[0, z]` | `z` écrasé. `r=0` fixe (sur l'axe). ∀z. |
| `parametric_1d(r)` | `@[r, 0.1]` | `r` écrasé. `z=0.1` fixe. ∀r. |
| `parametric_2d(r, z)` | `@[r, z]` | Tout écrasé. Plan (r,z) complet. |

Prenons `cartesian` (toujours 3 composantes `@[x, y, z]`) :

| Domaine | On écrit | Effet |
|---------|----------|-------|
| `parametric_1d(z)` | `@[0, 0, z]` | `z` écrasé. `x=0`, `y=0` fixes. |
| `parametric_2d(x, z)` | `@[x, 0, z]` | `x` et `z` écrasés. `y=0` fixe. |

**Pour le front-end :** les coordonnées paramétriques sont affichées
en rouge ou remplacées par le nom de la variable (ex: `@[0.5, ?, z]`
où `?` est affiché en rouge car θ est paramétrique). La valeur écrite
à cette position est ignorée — le front-end peut la remplacer par
le nom de la variable pour plus de clarté.

**Dimension spatiale réelle = dimension paramétrique + dimensions implicites :**

| `coordinate_system` | Dim paramétrique | Dim implicite (∀θ) | Dim spatiale réelle |
|---------------------|------------------|--------------------|---------------------|
| `axisymmetric` + 0D (point) | 0 | +1 | **1** (cercle) |
| `axisymmetric` + 1D(z) | 1 | +1 | **2** (surface de révolution) |
| `axisymmetric` + 2D(r,z) | 2 | +1 | **3** (volume de révolution) |
| `cylindrical` + 1D(z) | 1 | 0 | **1** (ligne) |
| `cylindrical` + 2D(θ,z) | 2 | 0 | **2** (surface) |
| `cylindrical` + 3D(r,θ,z) | 3 | 0 | **3** (volume) |
| `cartesian` + 1D(z) | 1 | 0 | **1** (ligne) |
| `cartesian` + 3D(x,y,z) | 3 | 0 | **3** (volume) |

Le solveur n'échantillonne jamais les dimensions implicites — elles
sont traitées analytiquement par les PhysicsModules.

### 6.3 Syntaxe

```json
"field_constraints": [
    {
        "id": "velocity_ratio",
        "description": "∀z: vitesse sur l'axe = 2× vitesse à r=0.5m",
        "coordinate_system": "axisymmetric",
        "domain": {
            "type": "parametric_1d",
            "variables": ["z"],
            "ranges": { "z": [0, "point:ENDTUYERE.z"] },
            "samples": [50]
        },
        "relation": { "lhs": "field:velocity@[0, z]", "op": "=", "rhs": "expr:2 * field:velocity@[0.5, z]" },
        "tolerance": 0.05,
        "priority": 2
    },
    {
        "id": "monotonic_mach",
        "description": "∀z dans le divergent: Mach croissant",
        "coordinate_system": "axisymmetric",
        "domain": {
            "type": "parametric_1d",
            "variables": ["z"],
            "ranges": { "z": ["point:THROAT.z", "point:ENDTUYERE.z"] },
            "samples": [30]
        },
        "relation": { "lhs": "field:mach@[0, z + 0.001]", "op": ">=", "rhs": "field:mach@[0, z]" },
        "tolerance": 0.0,
        "priority": 1
    },
    {
        "id": "exit_pressure_nonaxisym",
        "description": "Pression en sortie, profil complet (r,θ) — PAS axisymétrique",
        "coordinate_system": "cylindrical",
        "domain": {
            "type": "parametric_2d",
            "variables": ["r", "theta"],
            "ranges": { "r": [0, "param:r_exit"], "theta": [0, "expr:2*3.14159"] },
            "samples": [20, 16]
        },
        "relation": { "lhs": "field:static_pressure@[r, theta, point:ENDTUYERE.z]", "op": "=", "rhs": "101325.0" },
        "tolerance": 0.05,
        "priority": 1
    },
    {
        "id": "wall_stress",
        "description": "σ_VM < σ_yield/SF partout dans la paroi",
        "coordinate_system": "axisymmetric",
        "domain": { "type": "surface", "surface": "wall_internal", "samples": [100] },
        "relation": { "lhs": "field:von_mises@_sample", "op": "<", "rhs": "expr:material:wall.yield_strength / material:wall.safety_factor" },
        "tolerance": 0.0,
        "priority": 1
    },
    {
        "id": "pressure_at_exit_center",
        "description": "Contrainte ponctuelle (cas dégénéré)",
        "coordinate_system": "cartesian",
        "domain": { "type": "point", "location": "point:ENDTUYERE" },
        "relation": { "lhs": "field:static_pressure@ENDTUYERE", "op": "=", "rhs": "101325.0" },
        "tolerance": 0.05,
        "priority": 1
    }
]
```

### 6.4 Évaluation

À chaque itération du solveur :

1. Le domaine est discrétisé (N₁ × N₂ × ... échantillons).
2. Pour chaque échantillon, les variables paramétriques sont injectées dans `Context.domainVariables`.
3. `@[r, z]` est converti en point 3D selon le `coordinate_system` :
   - `axisymmetric` : `(r, 0, z)` — le solveur sait que ∀θ est implicite
   - `cylindrical` : `(r·cos(θ), r·sin(θ), z)`
   - `cartesian` : `(x, y, z)` tel quel
4. `lhs` et `rhs` sont évalués via `ExpressionEvaluator`.
5. La violation est agrégée : **max** pour priorité 1, **L2** pour priorité 2+.

### 6.5 Points Mobiles

```json
"WALL_PROBE": {
    "position": ["expr:profile_r(domain:z)", 0, "domain:z"],
    "type": "mobile",
    "domain_variable": "z"
}
```

`domain:z` = valeur courante de la variable z du domaine englobant. Ce point n'existe que dans le contexte d'une `field_constraint`.

---

## 7. Parameters

```json
"parameters": {
    "r_exit":          { "type": "variable", "initial": 0.035, "bounds": [0.01, 0.15] },
    "r_throat":        { "type": "variable", "initial": 0.015, "bounds": [0.005, 0.04] },
    "L_nozzle":        { "type": "variable", "initial": 0.3,   "bounds": [0.1, 0.5] },
    "wall_thickness":  { "type": "derived",  "equation": "pressure_vessel" },
    "n_ctrl":          { "type": "fixed",    "value": 7 },
    "profile_r":       { "type": "variable_array", "size": "param:n_ctrl", "bounds": [0.005, 0.15] }
}
```

---

## 8. Stratégie d'Optimisation — Multi-Fidélité

### 8.1 Deux boucles imbriquées

Le solveur n'a PAS une seule boucle. Il a deux boucles imbriquées :

```
Boucle EXTERNE : raffinement de fidélité (rough → medium → fine)
│
├─ Fidélité k : choisir le modèle physique + sample size
│  │
│  └─ Boucle INTERNE : optimisation à fidélité fixe
│     ├─ Paramètres → PhysicsModule(fidélité k) → Champs
│     ├─ Champs → FieldConstraints + Constraints → Coût
│     ├─ Coût → Gradient → Nouveaux paramètres
│     └─ Convergence ? → Passer à fidélité k+1
│
├─ Fidélité k+1 : modèle plus précis, démarrer depuis les
│  meilleurs paramètres de la fidélité k
│  └─ ...
│
└─ Fidélité finale : résultat envoyé au GeometryBuilder → geometry.json
```

L'idée : la fidélité basse converge vite (physique simple, peu
d'échantillons) et donne un bon point de départ. La fidélité haute
part de ce bon point et n'a besoin que de quelques itérations pour
affiner. Le temps total est beaucoup plus court que de tout calculer
en haute fidélité dès le départ.

### 8.2 Niveaux de Fidélité (pour la tuyère)

| Niveau | Modèle physique | Samples | Coût/itération | Usage |
|--------|----------------|---------|----------------|-------|
| 1 (rough) | Quasi-1D isentropique | 10-20 | ~1 ms | Dimensionnement initial |
| 2 (medium) | Quasi-1D + corrections de couche limite | 50-100 | ~10 ms | Optimisation du profil |
| 3 (fine) | Euler axisymétrique | 200-500 | ~1-10 s | Validation et affinage |
| 4 (CFD) | Navier-Stokes axisym. | 1000+ | ~1 min | Validation finale (futur) |

Les niveaux 1-2 sont implémentés en premier. Les niveaux 3-4 sont
des PhysicsModules futurs — l'architecture les supporte sans modification.

### 8.3 Trois Modes Utilisateur

L'utilisateur choisit son mode dans la section `optimization` du JSON :

```json
"optimization": {
    "mode": "dynamic",
    "method": "nelder_mead",

    "fixed_params": {
        "max_iterations": 500,
        "sample_size": 100
    },

    "dynamic_params": {
        "fidelity_levels": [
            { "samples": 10,  "max_iterations": 100, "physics": "isentropic_1d" },
            { "samples": 50,  "max_iterations": 50,  "physics": "isentropic_1d_corrected" },
            { "samples": 200, "max_iterations": 20,  "physics": "euler_axisym" }
        ]
    },

    "convergence_params": {
        "max_total_iterations": 2000,
        "criteria_tolerances": {
            "thrust": 0.01,
            "isp": 0.005,
            "structural": 0.0,
            "thermal": 0.0
        },
        "min_samples": 10,
        "max_samples": 500,
        "sample_growth_factor": 2.0
    }
}
```

**Mode `fixed`** — Fidélité unique.
Le sample size et le nombre d'itérations sont fixés. Simple, prédictible.
Bon pour le debug et quand la physique est peu coûteuse (quasi-1D).
Le solveur exécute une seule boucle interne.

**Mode `dynamic`** — Multi-fidélité explicite.
L'utilisateur définit les niveaux de fidélité avec leur sample size,
nombre max d'itérations, et modèle physique. Le solveur les parcourt
dans l'ordre, en passant les meilleurs paramètres au niveau suivant.
C'est le mode par défaut pour la production.

**Mode `convergence`** — Adaptatif par critères.
Le solveur commence avec `min_samples` et itère. Si tous les critères
sont dans leur tolérance, il multiplie les samples par `sample_growth_factor`
et recommence. Il s'arrête quand les critères restent satisfaits après
raffinement OU quand `max_samples` ou `max_total_iterations` est atteint.
C'est le mode le plus intelligent mais le plus coûteux.

### 8.4 Évaluation des Champs

**Les champs sont recalculés à chaque itération.**

À chaque itération de la boucle interne, les paramètres définissent
une géométrie courante. Le PhysicsModule évalue les champs (vitesse,
pression, température, contrainte mécanique) SUR cette géométrie.

```
Itération i :
  1. Paramètres P_i → Géométrie G_i (profil de la tuyère)
  2. PhysicsModule(G_i) → Champs F_i (vitesse, pression, T, σ)
  3. FieldConstraints(F_i) → Violations V_i
  4. CostFunction(V_i) → Coût C_i
  5. Optimizer(C_i, ∇C_i) → Paramètres P_{i+1}
```

Les champs ne sont PAS calculés une fois puis réutilisés — ils
DÉPENDENT de la géométrie qui change. `field:velocity@[0.5, z]`
est un appel au PhysicsModule qui recalcule la vitesse pour la
section locale de la tuyère à la station z, étant donné le rayon
du profil à cette station (qui vient des paramètres courants).

En quasi-1D, ce recalcul est instantané (une évaluation analytique).
En CFD, c'est un solveur numérique complet — d'où l'importance
du multi-fidélité.

### 8.5 Envoi au Kernel Géométrique

À chaque changement de fidélité (fin de boucle interne), le solveur
peut optionnellement envoyer la géométrie courante au kernel pour
affichage temps réel. L'utilisateur voit la tuyère évoluer :

```
Fidélité 1 terminée → geometry_rough.json → Kernel affiche (forme grossière)
Fidélité 2 terminée → geometry_medium.json → Kernel met à jour (forme affinée)
Fidélité 3 terminée → geometry_final.json → Kernel affiche la forme finale
```

C'est purement optionnel — le kernel ne participe PAS à l'optimisation.
Il ne fait que visualiser. Le solveur est autonome.

---

## 9. Recentrage Automatique

Le solveur déduit le repère à partir des anchors (disques concentriques → axe commun), construit une matrice de transformation, et travaille dans le repère recentré.

---

## 10. Détection d'Incompatibilité

Quatre types d'erreurs structurées : `INCOMPATIBLE_CONSTRAINTS`, `UNDERDETERMINED`, `NO_CONVERGENCE`, `PHYSICAL_VIOLATION`. Chacune avec message, éléments en conflit, et suggestion de correction.

---

## 11. Stratégie de Performance

### 11.1 Principe

Chaque milliseconde compte dans la boucle d'optimisation. Une itération
typique sera exécutée des centaines de fois. Les gains de performance
sont multiplicatifs : 2ms gagnés × 500 itérations × 3 niveaux de
fidélité = 3 secondes. Sur des runs de production, ça devient des minutes.

### 11.2 Cache de Champs

Le PhysicsModule recalcule les champs à chaque itération. MAIS :
si les paramètres n'ont changé que de ε, les champs n'ont changé
que localement. On cache les résultats et on invalide sélectivement.

```cpp
class FieldCache {
    // Clé = hash des paramètres (arrondi à une précision configurable)
    // Valeur = champ évalué à chaque point d'échantillonnage
    std::unordered_map<size_t, std::vector<float>> cache;

    // Invalidation partielle : si seul param[3] a changé,
    // seuls les échantillons dépendant de param[3] sont recalculés.
    // Le graphe de dépendances est construit au parsing.
    DependencyGraph dependencies;
};
```

### 11.3 Parallélisation

Les évaluations de field constraints sont **embarrassingly parallel** :
chaque échantillon est indépendant. On utilise `std::execution::par`
sur les boucles d'échantillonnage.

```cpp
// Évaluation parallèle des N échantillons d'une field constraint
std::vector<float> violations(N);
std::for_each(std::execution::par, indices.begin(), indices.end(),
    [&](int i) {
        // Injecter les variables paramétriques pour l'échantillon i
        Context ctx = baseContext;
        ctx.domainVariables[varName] = lo + i * step;
        violations[i] = evaluateRelation(fc, ctx);
    });
```

Le gradient numérique est aussi parallélisable : chaque perturbation
de paramètre est indépendante.

### 11.4 Simplifications Numériques

L'utilisateur choisit le niveau de simplification via `optimization.precision` :

| Précision | Intégration | Gradient | Usage |
|-----------|-------------|----------|-------|
| `fast` | Rectangles (ordre 1) | Forward differences (N+1 évals) | Exploration rapide |
| `standard` | Simpson (ordre 4) | Central differences (2N évals) | Production |
| `high` | Gauss-Legendre (ordre 8) | Central + Richardson (4N évals) | Validation |

```json
"optimization": {
    "precision": "standard",
    ...
}
```

### 11.5 Évaluation Paresseuse

Les contraintes de priorité basse ne sont pas évaluées si une
contrainte de priorité haute est déjà violée massivement.
Inutile de calculer la masse optimale si la tuyère explose.

```
Si constraint["structural"].violation > 10× tolérance :
    skip constraints de priorité > 1
    retourner coût = +∞
```

---

## 12. Généralisation 3D

### 12.1 Ce qui est déjà 3D dans la spec

- `coordinate_system: "cartesian"` avec `@[x, y, z]`
- `parametric_3d` avec 3 variables
- Le Geometric Kernel (SDF, Ray Marching, Marching Cubes) est déjà 3D
- Les PhysicsModules sont des interfaces abstraites — aucune hypothèse de dimension

### 12.2 Ce qui est 2D "par choix d'implémentation" (pas par limitation)

- Le PhysicsModule `IsentropicNozzle` est quasi-1D (premier module)
- Le profil de tuyère est axisymétrique (CompositeSpline2D dans le plan r,z)
- L'export STL est déjà 3D (le Marching Cubes travaille en 3D)

### 12.3 Chemin vers le 3D complet

Pour passer au 3D, il suffit d'ajouter :
1. Un PhysicsModule 3D (ex: `EulerSolver3D`) — le framework l'appelle exactement comme le quasi-1D
2. Des primitives géométriques 3D dans le GeometryBuilder (le kernel les supporte déjà : Box, Sphere)
3. Des field constraints en `cartesian` + `parametric_3d`

RIEN dans le framework (ExpressionEvaluator, Optimizer, CostEvaluator,
FieldConstraintEvaluator, MultiFidelityScheduler) ne change.

---

## 13. Architecture C++

```
src/solver/
├── ExpressionEvaluator.hpp     // Parse "expr:", "param:", "field:@[r,z]", "domain:"
├── ProblemParser.hpp            // problem.json → ProblemDefinition
├── ProblemDefinition.hpp        // Structs : BC, Constraint, FieldConstraint, Parameter...
├── CoordinateResolver.hpp       // Détection d'axe, recentrage
├── ParameterSpace.hpp           // Vecteur d'optim, bornes, dérivés
├── ConstraintChecker.hpp        // Détection sur/sous-contrainte
├── FieldConstraintEvaluator.hpp // Discrétise domaines, évalue relations ∀
├── FieldCache.hpp               // Cache de champs avec invalidation sélective
├── CostEvaluator.hpp            // Coût pondéré (scalaire + champ)
├── MultiFidelityScheduler.hpp   // Gère les niveaux de fidélité et les transitions
├── Optimizer.hpp                // Nelder-Mead, gradient, BFGS
├── Solver.hpp                   // Point d'entrée
└── physics/
     ├── PhysicsModule.hpp       // Interface abstraite
     ├── IsentropicNozzle.hpp    // Quasi-1D (premier module)
     ├── PressureVessel.hpp      // Épaisseur paroi
     ├── ThermalWall.hpp         // Flux thermique
     └── ...                     // EulerSolver2D, NavierStokes3D (futur)
```

---

## 14. Ordre d'Implémentation

1. **ExpressionEvaluator** — Parse et évalue toutes les formes d'expression
2. **ProblemParser + ProblemDefinition** — Lecture JSON, résolution des refs
3. **CoordinateResolver** — Détection d'axe, recentrage
4. **ParameterSpace + ConstraintChecker** — Vecteur d'optim, détection incompatibilités
5. **PhysicsModules** — IsentropicNozzle, PressureVessel, ThermalWall
6. **FieldConstraintEvaluator + FieldCache** — Discrétisation, évaluation, cache
7. **CostEvaluator + Optimizer** — Fonction de coût, Nelder-Mead, parallélisation gradient
8. **MultiFidelityScheduler** — Boucle externe, transitions entre niveaux
9. **GeometryBuilder** — Paramètres → arbre CSG JSON
10. **Diagnostics** — Messages d'erreur structurés
