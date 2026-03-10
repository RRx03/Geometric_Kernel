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