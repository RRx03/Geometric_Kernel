#ifndef SDFShared_h
#define SDFShared_h

#ifdef __METAL_VERSION__
#define SDF_FLOAT3 float3
#define SDF_FLOAT4 float4
#else
#include <simd/simd.h>
#define SDF_FLOAT3 simd::float3
#define SDF_FLOAT4 simd::float4
#endif

enum SDFNodeType {
  SDF_DATA_CARRIER = -1,

  // Primitives 3D
  SDF_TYPE_SPHERE = 0,
  SDF_TYPE_BOX = 1,
  SDF_TYPE_CYLINDER = 2,

  // Primitives 2D axisymétriques
  SDF_TYPE_CIRCLE_2D = 3,
  SDF_TYPE_RECT_2D = 4,
  SDF_TYPE_BEZIER2D = 5,
  SDF_TYPE_CUBIC_BEZIER2D = 6,

  // ─────────────────────────────────────────────────────────
  // Composite Spline 2D — Profil signé à N points
  //
  // Un seul nœud GPU logique qui encode un profil composite.
  //
  // Layout dans le buffer plat :
  //   nodes[i]   : type = SDF_TYPE_COMPOSITE_SPLINE2D
  //                params.x = nombre de points N (cast en float)
  //                params.y = thickness (0 = demi-plan signé)
  //                position = (0, 0, 0) réservé
  //
  //   nodes[i+1] : type = SDF_DATA_CARRIER
  //                position = (point0.r, point0.y, 0)
  //                params   = (point1.r, point1.y, point2.r, point2.y)
  //
  //   nodes[i+2] : type = SDF_DATA_CARRIER
  //                position = (point3.r, point3.y, 0)
  //                params   = (point4.r, point4.y, point5.r, point5.y)
  //
  //   ... (on pack 5 points dans le premier DATA_CARRIER,
  //        puis 5 par DATA_CARRIER suivant)
  //
  // Packing : chaque DATA_CARRIER stocke jusqu'à 5 points :
  //   position.x, position.y = point[k+0]
  //   params.x, params.y     = point[k+1]  (si existe)
  //   params.z, params.w     = point[k+2]  (si existe)
  //   → Non, simplifions : 3 points par DATA_CARRIER :
  //     position = (p0.r, p0.y, 0)
  //     params   = (p1.r, p1.y, p2.r, p2.y)
  //   → 3 points par nœud. Pour N points, ceil(N/3) DATA_CARRIERs.
  //
  // L'évaluateur (CPU et GPU) lit le header, puis parcourt
  // les DATA_CARRIERs pour reconstruire les points, décompose
  // en segments de Bézier quadratique (B-spline ouverte),
  // et calcule la distance signée au profil complet.
  // ─────────────────────────────────────────────────────────
  SDF_TYPE_COMPOSITE_SPLINE2D = 7,

  // Opérations booléennes CSG
  SDF_OP_UNION = 10,
  SDF_OP_SUBTRACT = 11,
  SDF_OP_SMOOTH_UNION = 12,
  SDF_OP_INTERSECT = 13
};

struct SDFNodeGPU {
  int type;
  int leftChildIndex;
  int rightChildIndex;
  int _pad0;

  SDF_FLOAT3 position;
  SDF_FLOAT4 params;
  float smoothFactor;
  float _pad1;
  float _pad2;
  float _pad3;
};

#ifndef __METAL_VERSION__
static_assert(sizeof(SDFNodeGPU) == 64, "SDFNodeGPU doit faire 64 bytes");
static_assert(offsetof(SDFNodeGPU, position) == 16,
              "position doit etre a l'offset 16");
static_assert(offsetof(SDFNodeGPU, params) == 32,
              "params doit etre a l'offset 32");
static_assert(offsetof(SDFNodeGPU, smoothFactor) == 48,
              "smoothFactor doit etre a l'offset 48");
#endif

#endif /* SDFShared_h */