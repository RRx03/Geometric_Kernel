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
  SDF_TYPE_SPHERE = 0,
  SDF_TYPE_BOX = 1,
  SDF_TYPE_CYLINDER = 2,
  SDF_TYPE_CIRCLE_2D = 3,
  SDF_TYPE_RECT_2D = 4,
  SDF_TYPE_BEZIER2D = 5,

  SDF_OP_UNION = 10,
  SDF_OP_SUBTRACT = 11,
  SDF_OP_SMOOTH_UNION = 12,
  SDF_OP_INTERSECT = 13
};

struct SDFNodeGPU {
  int type;
  int leftChildIndex;
  int rightChildIndex;

  SDF_FLOAT3 position;
  SDF_FLOAT4 params;

  float smoothFactor;
};

#endif /* SDFShared_h */