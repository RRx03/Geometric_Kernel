#pragma once
#include "SDFEvaluator.hpp"
#include <string>
#include <vector>

struct Triangle {
  simd::float3 v1, v2, v3;
};

class Mesher {
public:
  static void generateSTL(const SDFEvaluator &evaluator, simd::float3 minBounds,
                          simd::float3 maxBounds, float resolution,
                          const std::string &filename,
                          float exportScale = 1000.0f);
};