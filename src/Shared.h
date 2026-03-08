
#pragma once
#include <simd/simd.h>

struct VertexData {
  vector_float3 position;
  vector_float3 color;
};

struct Uniforms {
  simd::float3 camPos;
  float padding1;
  simd::float3 camForward;
  float padding2;
  simd::float3 camRight;
  float padding3;
  simd::float3 camUp;
  float padding4;
};