#pragma once
// ═══════════════════════════════════════════════════════════════
// Mesher.hpp — Marching Cubes → STL export
// ═══════════════════════════════════════════════════════════════

#include "SDFEvaluator.hpp"
#include <string>
#include <vector>

struct Triangle {
    simd::float3 v1, v2, v3;
};

class Mesher {
public:
    // Export STL binaire
    // resolution : taille de voxel en mètres (0 = auto)
    // maxVoxelsPerDim : budget max par dimension
    // exportScale : multiplicateur (1000 = m→mm)
    static void exportSTL(const SDFEvaluator& evaluator,
                          const std::string& filename,
                          float resolution = 0.0f,
                          int maxVoxelsPerDim = 256,
                          float exportScale = 1000.0f);

    // Extraction de triangles (pour usage programmatique)
    static std::vector<Triangle> extractMesh(const SDFEvaluator& evaluator,
                                              float resolution = 0.0f,
                                              int maxVoxelsPerDim = 256);
};
