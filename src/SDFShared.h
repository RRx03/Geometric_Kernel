#ifndef SDFShared_h
#define SDFShared_h

#include <simd/simd.h>

enum SDFNodeType {
  SDF_TYPE_SPHERE = 0,
  SDF_TYPE_BOX = 1,
  SDF_TYPE_CYLINDER = 2,

  SDF_OP_UNION = 10,
  SDF_OP_SUBTRACT = 11,
  SDF_OP_SMOOTH_UNION = 12
};

struct SDFNodeGPU {
  int type;
  int leftChildIndex;
  int rightChildIndex;

  simd::float3 position;
  simd::float4 params;

  float smoothFactor;
};

#endif /* SDFShared_h */