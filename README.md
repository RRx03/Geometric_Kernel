Geometric Kernel Based on my Metal Cpp Template.

Converts SDF into visible meshes, export SDF trees to STL using Marching Cubes Algorithm.

**Make it run : **
Open projet's folder in your Teminal,
type : make init
then : make run

Overview of the current version :

https://github.com/user-attachments/assets/325a1199-f630-4936-a09a-2076040ce13d

Améliorations à venir :

- Paralléliser le Marching Cubes lui-même avec std::for_each(std::execution::par, ...) sur les tranches Z.
- Ajouter des primitives de base supplémentaires (torus, cône, etc.) et des transformations (rotation, mise à l'échelle).
- Implémenter des optimisations pour les SDF complexes, comme la hiérarchisation spatiale (BVH) pour accélérer les évaluations de distance.
- Ajouter une interface utilisateur pour éditer les SDF en temps réel, avec des outils de dessin et de manipulation.
- Supporter l'exportation vers d'autres formats de maillage (OBJ, PLY) et l'importation de SDF à partir de fichiers.
- Opérations CSG limitées à des arbres binaires : Actuellement, Union, Subtract, et Intersect prennent exactement 2 enfants (left/right). Permettre des opérations n-aires (ex: Union de 3+ enfants) pour plus de flexibilité dans la construction d'arbres SDF. C'est verbeux et profond. Pour être solver-friendly, il serait préférable d'accepter un tableau children :
  json{"type": "Union", "children": [S1, S2, S3, S4, S5]}
- Augmenter la résolution d'export, ou permettre a l'utilisateur de le choisir, pour des maillages plus détaillés. Actuellement, la résolution est fixe à 64^3, ce qui peut être insuffisant pour des SDF complexes.
