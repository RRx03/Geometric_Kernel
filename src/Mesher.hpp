#pragma once
// ═══════════════════════════════════════════════════════════════
// Mesher.hpp — Two-Pass Adaptive Marching Cubes → STL  (v5)
//
// No Laplacian smoothing — the sign artifacts are fixed at the
// source in SDFMath.hpp (analytical ray crossing).
// ═══════════════════════════════════════════════════════════════

#include "SDFEvaluator.hpp"
#include <algorithm>
#include <string>
#include <vector>

struct Triangle {
    simd::float3 v1, v2, v3;
    simd::float3 normal;
};

class Mesher {
public:
  static void exportSTL(const SDFEvaluator &evaluator,
                        const std::string &filename, int coarseDiv = 128,
                        int fineFactor = 8, float exportScale = 1000.0f);

  static std::vector<Triangle> extractMesh(const SDFEvaluator &evaluator,
                                           int coarseDiv = 128,
                                           int fineFactor = 8);

  // Legacy API compatibility
  static void exportSTL(const SDFEvaluator &evaluator,
                        const std::string &filename, float resolution,
                        int maxVoxelsPerDim, float exportScale) {
    int coarse = std::max(32, maxVoxelsPerDim / 8);
    exportSTL(evaluator, filename, coarse, 8, exportScale);
  }
};