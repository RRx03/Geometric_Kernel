
#pragma once
#include <simd/simd.h>

struct VertexData {
  vector_float3 position;
  vector_float3 color;
};

struct Uniforms {
  matrix_float4x4 modelMatrix;
  matrix_float4x4 viewMatrix;
  matrix_float4x4 projectionMatrix;
};