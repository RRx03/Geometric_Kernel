
#pragma once
#include <cmath>
#include <simd/simd.h>

namespace Math {

inline float radians(float degrees) { return degrees * M_PI / 180.0f; }

inline matrix_float4x4 makePerspective(float fovRadians, float aspect,
                                       float znear, float zfar) {
  float ys = 1.0f / std::tan(fovRadians * 0.5f);
  float xs = ys / aspect;
  float zs = zfar / (znear - zfar);

  return (matrix_float4x4){{{xs, 0.0f, 0.0f, 0.0f},
                            {0.0f, ys, 0.0f, 0.0f},
                            {0.0f, 0.0f, zs, -1.0f},
                            {0.0f, 0.0f, zs * znear, 0.0f}}};
}

inline matrix_float4x4 makeTranslate(vector_float3 v) {
  matrix_float4x4 m = matrix_identity_float4x4;
  m.columns[3].x = v.x;
  m.columns[3].y = v.y;
  m.columns[3].z = v.z;
  return m;
}

inline matrix_float4x4 makeYRotation(float angleRadians) {
  float c = std::cos(angleRadians);
  float s = std::sin(angleRadians);
  matrix_float4x4 m = matrix_identity_float4x4;
  m.columns[0].x = c;
  m.columns[0].z = -s;
  m.columns[2].x = s;
  m.columns[2].z = c;
  return m;
}
inline matrix_float4x4 makeIdentity() { return matrix_identity_float4x4; }
} // namespace Math