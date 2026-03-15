// ═══════════════════════════════════════════════════════════════
// Mesher.cpp — Marching Cubes → STL Export
// ═══════════════════════════════════════════════════════════════

#include "Mesher.hpp"
#include "MCTables.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>
#include <thread>

static simd::float3 vertexInterp(float iso, simd::float3 p1, simd::float3 p2,
                                  float v1, float v2) {
    if (std::abs(v1 - v2) < 1e-10f) return p1;
    float t = (iso - v1) / (v2 - v1);
    return p1 + t * (p2 - p1);
}

static const int cornerOff[8][3] = {
    {0,0,0},{1,0,0},{1,1,0},{0,1,0},
    {0,0,1},{1,0,1},{1,1,1},{0,1,1}
};

std::vector<Triangle> Mesher::extractMesh(const SDFEvaluator& eval,
                                           float resolution, int maxVPD) {
    simd::float3 bMin = eval.boundsMin();
    simd::float3 bMax = eval.boundsMax();
    simd::float3 sz = bMax - bMin;
    float maxDim = std::max({sz.x, sz.y, sz.z});

    float res = resolution;
    if (res <= 0.0f) res = maxDim / (float)maxVPD;
    res = std::max(res, maxDim / (float)maxVPD);

    int nx = std::max(2, std::min(maxVPD, (int)std::ceil(sz.x / res) + 1));
    int ny = std::max(2, std::min(maxVPD, (int)std::ceil(sz.y / res) + 1));
    int nz = std::max(2, std::min(maxVPD, (int)std::ceil(sz.z / res) + 1));

    std::cout << "[Mesher] Grid: " << nx << "x" << ny << "x" << nz
              << " (" << (long long)nx*ny*nz << " pts), voxel="
              << (res*1000.0f) << "mm\n";

    // Evaluate SDF on grid — parallel by Z slices
    size_t totalPts = (size_t)nx * ny * nz;
    std::vector<float> grid(totalPts);

    auto idx = [&](int x, int y, int z) -> size_t {
        return (size_t)z * ny * nx + (size_t)y * nx + x;
    };

    // Parallel evaluation
    int nThreads = std::max(1, (int)std::thread::hardware_concurrency());
    auto evalSlice = [&](int zStart, int zEnd) {
        for (int z = zStart; z < zEnd; z++) {
            for (int y = 0; y < ny; y++) {
                for (int x = 0; x < nx; x++) {
                    simd::float3 p = bMin + simd_make_float3(
                        x * res, y * res, z * res);
                    grid[idx(x, y, z)] = eval.evaluate(p);
                }
            }
        }
    };

    // Split Z slices across threads
    std::vector<std::thread> threads;
    int slicesPerThread = std::max(1, nz / nThreads);
    for (int t = 0; t < nThreads; t++) {
        int zStart = t * slicesPerThread;
        int zEnd = (t == nThreads - 1) ? nz : std::min(zStart + slicesPerThread, nz);
        if (zStart < zEnd) {
            threads.emplace_back(evalSlice, zStart, zEnd);
        }
    }
    for (auto& t : threads) t.join();

    std::cout << "[Mesher] Grid evaluated. Extracting triangles...\n";

    // Extract triangles
    std::vector<Triangle> triangles;
    float iso = 0.0f;

    for (int z = 0; z < nz - 1; z++) {
        for (int y = 0; y < ny - 1; y++) {
            for (int x = 0; x < nx - 1; x++) {
                simd::float3 p[8];
                float val[8];
                for (int c = 0; c < 8; c++) {
                    int cx = x + cornerOff[c][0];
                    int cy = y + cornerOff[c][1];
                    int cz = z + cornerOff[c][2];
                    p[c] = bMin + simd_make_float3(cx * res, cy * res, cz * res);
                    val[c] = grid[idx(cx, cy, cz)];
                }

                int cubeIdx = 0;
                for (int c = 0; c < 8; c++)
                    if (val[c] < iso) cubeIdx |= (1 << c);

                if (edgeTable[cubeIdx] == 0) continue;

                simd::float3 vertlist[12];
                int edges[12][2] = {
                    {0,1},{1,2},{2,3},{3,0},
                    {4,5},{5,6},{6,7},{7,4},
                    {0,4},{1,5},{2,6},{3,7}
                };
                for (int e = 0; e < 12; e++) {
                    if (edgeTable[cubeIdx] & (1 << e)) {
                        int a = edges[e][0], b = edges[e][1];
                        vertlist[e] = vertexInterp(iso, p[a], p[b], val[a], val[b]);
                    }
                }

                for (int i = 0; triTable[cubeIdx][i] != -1; i += 3) {
                    Triangle tri;
                    tri.v1 = vertlist[triTable[cubeIdx][i]];
                    tri.v2 = vertlist[triTable[cubeIdx][i + 1]];
                    tri.v3 = vertlist[triTable[cubeIdx][i + 2]];
                    triangles.push_back(tri);
                }
            }
        }
    }

    std::cout << "[Mesher] " << triangles.size() << " triangles extracted.\n";
    return triangles;
}

void Mesher::exportSTL(const SDFEvaluator& eval, const std::string& filename,
                        float resolution, int maxVPD, float exportScale) {
    auto triangles = extractMesh(eval, resolution, maxVPD);

    std::ofstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "[Mesher] Cannot open '" << filename << "' for writing.\n";
        return;
    }

    // STL binary header
    char header[80] = {};
    std::snprintf(header, 80, "Geometric Kernel STL Export — %zu triangles",
                  triangles.size());
    f.write(header, 80);

    uint32_t numTris = (uint32_t)triangles.size();
    f.write(reinterpret_cast<char*>(&numTris), 4);

    for (const auto& tri : triangles) {
        // Compute normal
        simd::float3 e1 = tri.v2 - tri.v1;
        simd::float3 e2 = tri.v3 - tri.v1;
        simd::float3 n = simd_normalize(simd_cross(e1, e2));

        // Scale to export units (m → mm)
        auto write3 = [&](simd::float3 v) {
            float buf[3] = {v.x * exportScale, v.y * exportScale, v.z * exportScale};
            f.write(reinterpret_cast<char*>(buf), 12);
        };

        write3(n);
        write3(tri.v1);
        write3(tri.v2);
        write3(tri.v3);

        uint16_t attr = 0;
        f.write(reinterpret_cast<char*>(&attr), 2);
    }

    f.close();
    std::cout << "[Mesher] Exported " << numTris << " triangles to '"
              << filename << "' (scale=" << exportScale << "x)\n";
}
