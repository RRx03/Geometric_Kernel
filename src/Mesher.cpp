// ═══════════════════════════════════════════════════════════════
// Mesher.cpp — Two-Pass Adaptive Marching Cubes  (v5)
//
// Clean implementation: no Laplacian smoothing.
// The sign artifacts (horizontal bands) are fixed at the source
// in SDFMath.hpp via analytical ray crossing.
//
// Features:
//   - Two-pass: coarse 128³ + fine 8× near surface = effective 1024³
//   - Flipped winding (outward normals for SDF negative=inside)
//   - SDF gradient normals (smooth shading)
//   - Parallel work-stealing for fine pass
//   - Correct STL binary format
// ═══════════════════════════════════════════════════════════════

#include "Mesher.hpp"
#include "MCTables.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

static const int cOff[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                               {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
static const int mcEdges[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0},
                                   {4, 5}, {5, 6}, {6, 7}, {7, 4},
                                   {0, 4}, {1, 5}, {2, 6}, {3, 7}};

static simd::float3 vInterp(float iso, simd::float3 p1, simd::float3 p2,
                            float v1, float v2) {
  if (std::abs(v1 - v2) < 1e-10f)
    return p1;
  float t = (iso - v1) / (v2 - v1);
  return p1 + t * (p2 - p1);
}

static simd::float3 sdfNormal(const SDFEvaluator &eval, simd::float3 p,
                              float eps) {
  float dx = eval.evaluate(p + simd_make_float3(eps, 0, 0)) -
             eval.evaluate(p - simd_make_float3(eps, 0, 0));
  float dy = eval.evaluate(p + simd_make_float3(0, eps, 0)) -
             eval.evaluate(p - simd_make_float3(0, eps, 0));
  float dz = eval.evaluate(p + simd_make_float3(0, 0, eps)) -
             eval.evaluate(p - simd_make_float3(0, 0, eps));
  simd::float3 n = simd_make_float3(dx, dy, dz);
  float len = simd_length(n);
  return (len > 1e-10f) ? n / len : simd_make_float3(0, 1, 0);
}

// ── Run MC on a local fine grid, append triangles ──
static void marchLocalGrid(const float *grid, int fN, simd::float3 origin,
                           float fRes, const SDFEvaluator &eval, float gradEps,
                           std::vector<Triangle> &out) {
  auto idx = [&](int x, int y, int z) -> int {
    return z * (fN + 1) * (fN + 1) + y * (fN + 1) + x;
  };
  for (int z = 0; z < fN; z++)
    for (int y = 0; y < fN; y++)
      for (int x = 0; x < fN; x++) {
        simd::float3 p[8];
        float val[8];
        for (int c = 0; c < 8; c++) {
          int cx = x + cOff[c][0], cy = y + cOff[c][1], cz = z + cOff[c][2];
          p[c] = origin + simd_make_float3(cx * fRes, cy * fRes, cz * fRes);
          val[c] = grid[idx(cx, cy, cz)];
        }
        int ci = 0;
        for (int c = 0; c < 8; c++)
          if (val[c] < 0.0f)
            ci |= (1 << c);
        if (edgeTable[ci] == 0)
          continue;

        simd::float3 vl[12];
        for (int e = 0; e < 12; e++)
          if (edgeTable[ci] & (1 << e))
            vl[e] = vInterp(0.0f, p[mcEdges[e][0]], p[mcEdges[e][1]],
                            val[mcEdges[e][0]], val[mcEdges[e][1]]);

        for (int i = 0; triTable[ci][i] != -1; i += 3) {
          Triangle tri;
          tri.v1 = vl[triTable[ci][i]];
          tri.v2 = vl[triTable[ci][i + 2]]; // SWAPPED for outward normals
          tri.v3 = vl[triTable[ci][i + 1]]; // SWAPPED

          simd::float3 centroid = (tri.v1 + tri.v2 + tri.v3) / 3.0f;
          tri.normal = sdfNormal(eval, centroid, gradEps);

          out.push_back(tri);
        }
      }
}

std::vector<Triangle> Mesher::extractMesh(const SDFEvaluator &eval,
                                          int coarseDiv, int fineFactor) {
  coarseDiv = std::clamp(coarseDiv, 8, 512);
  fineFactor = std::clamp(fineFactor, 1, 16);

  simd::float3 bMin = eval.boundsMin();
  simd::float3 bMax = eval.boundsMax();
  simd::float3 sz = bMax - bMin;
  float maxDim = std::max({sz.x, sz.y, sz.z});
  float cRes = maxDim / (float)coarseDiv;
  int cnx = std::clamp((int)std::ceil(sz.x / cRes) + 1, 2, coarseDiv + 1);
  int cny = std::clamp((int)std::ceil(sz.y / cRes) + 1, 2, coarseDiv + 1);
  int cnz = std::clamp((int)std::ceil(sz.z / cRes) + 1, 2, coarseDiv + 1);
  float fRes = cRes / (float)fineFactor;
  float gradEps = fRes * 0.5f;

  std::cout << "[Mesher] Two-pass adaptive MC\n"
            << "  Coarse: " << cnx << "x" << cny << "x" << cnz
            << "  voxel=" << (cRes * 1000.0f) << "mm\n"
            << "  Fine: " << fineFactor << "x  effective=" << (fRes * 1000.0f)
            << "mm\n";

  // ══ PASS 1: Coarse grid (parallel) ══
  std::vector<float> coarseGrid((size_t)cnx * cny * cnz);
  auto cIdx = [&](int x, int y, int z) -> size_t {
    return (size_t)z * cny * cnx + (size_t)y * cnx + x;
  };
  {
    int nT = std::max(1, (int)std::thread::hardware_concurrency());
    std::vector<std::thread> threads;
    int spt = std::max(1, cnz / nT);
    for (int t = 0; t < nT; t++) {
      int zs = t * spt, ze = (t == nT - 1) ? cnz : std::min(zs + spt, cnz);
      if (zs < ze)
        threads.emplace_back([&, zs, ze]() {
          for (int z = zs; z < ze; z++)
            for (int y = 0; y < cny; y++)
              for (int x = 0; x < cnx; x++)
                coarseGrid[cIdx(x, y, z)] = eval.evaluate(
                    bMin + simd_make_float3(x * cRes, y * cRes, z * cRes));
        });
    }
    for (auto &t : threads)
      t.join();
  }
  std::cout << "  Coarse grid evaluated.\n";

  // ══ PASS 1b: Find active cells ══
  struct CellCoord {
    int x, y, z;
  };
  std::vector<CellCoord> activeCells;
  float surfTh = cRes * 2.0f;

  for (int z = 0; z < cnz - 1; z++)
    for (int y = 0; y < cny - 1; y++)
      for (int x = 0; x < cnx - 1; x++) {
        bool hasPos = false, hasNeg = false, hasNear = false;
        for (int c = 0; c < 8; c++) {
          float v =
              coarseGrid[cIdx(x + cOff[c][0], y + cOff[c][1], z + cOff[c][2])];
          if (v < 0)
            hasNeg = true;
          if (v >= 0)
            hasPos = true;
          if (std::abs(v) < surfTh)
            hasNear = true;
        }
        if ((hasPos && hasNeg) || hasNear)
          activeCells.push_back({x, y, z});
      }

  size_t totalCoarse = (size_t)(cnx - 1) * (cny - 1) * (cnz - 1);
  std::cout << "  Active: " << activeCells.size() << " / " << totalCoarse
            << " ("
            << (100.0f * activeCells.size() / std::max((size_t)1, totalCoarse))
            << "%)\n";

  // ══ PASS 2: Fine MC on active cells (parallel work-stealing) ══
  int nT = std::max(1, (int)std::thread::hardware_concurrency());
  int fN = fineFactor;
  int fGridSize = (fN + 1) * (fN + 1) * (fN + 1);
  std::vector<std::vector<Triangle>> threadTris(nT);
  std::atomic<int> counter(0);
  int totalActive = (int)activeCells.size();

  std::vector<std::thread> threads;
  for (int t = 0; t < nT; t++) {
    threads.emplace_back([&, t]() {
      std::vector<float> lg(fGridSize);
      auto fi = [&](int x, int y, int z) -> int {
        return z * (fN + 1) * (fN + 1) + y * (fN + 1) + x;
      };
      while (true) {
        int ci = counter.fetch_add(1);
        if (ci >= totalActive)
          break;

        auto &cc = activeCells[ci];
        simd::float3 org =
            bMin + simd_make_float3(cc.x * cRes, cc.y * cRes, cc.z * cRes);

        for (int z = 0; z <= fN; z++)
          for (int y = 0; y <= fN; y++)
            for (int x = 0; x <= fN; x++)
              lg[fi(x, y, z)] = eval.evaluate(
                  org + simd_make_float3(x * fRes, y * fRes, z * fRes));

        marchLocalGrid(lg.data(), fN, org, fRes, eval, gradEps, threadTris[t]);

        // Progress every 5%
        if (totalActive > 20 && ci % (totalActive / 20) == 0)
          std::cout << "  [" << (100 * ci / totalActive) << "%]\n";
      }
    });
  }
  for (auto &t : threads)
    t.join();

  // ── Merge ──
  size_t totalTris = 0;
  for (auto &tv : threadTris)
    totalTris += tv.size();

  std::vector<Triangle> allTris;
  allTris.reserve(totalTris);
  for (auto &tv : threadTris) {
    allTris.insert(allTris.end(), tv.begin(), tv.end());
    tv.clear();
    tv.shrink_to_fit();
  }

  std::cout << "  Done: " << allTris.size() << " triangles\n";
  return allTris;
}

void Mesher::exportSTL(const SDFEvaluator &eval, const std::string &filename,
                       int coarseDiv, int fineFactor, float exportScale) {
  auto tris = extractMesh(eval, coarseDiv, fineFactor);

  std::ofstream f(filename, std::ios::binary);
  if (!f.is_open()) {
    std::cerr << "[Mesher] Cannot open " << filename << "\n";
    return;
  }

  // ── STL binary: 80-byte header + 4-byte count + N × 50 bytes ──
  char hdr[80];
  memset(hdr, 0, 80);
  std::snprintf(hdr, 80, "Geometric Kernel STL");
  f.write(hdr, 80);

  uint32_t nt = (uint32_t)tris.size();
  f.write(reinterpret_cast<const char *>(&nt), 4);

  for (const auto &tri : tris) {
    // Normal (3 floats = 12 bytes)
    float nb[3] = {tri.normal.x, tri.normal.y, tri.normal.z};
    f.write(reinterpret_cast<const char *>(nb), 12);

    // Vertex 1 (3 floats = 12 bytes)
    float v1[3] = {tri.v1.x * exportScale, tri.v1.y * exportScale,
                   tri.v1.z * exportScale};
    f.write(reinterpret_cast<const char *>(v1), 12);

    // Vertex 2 (3 floats = 12 bytes)
    float v2[3] = {tri.v2.x * exportScale, tri.v2.y * exportScale,
                   tri.v2.z * exportScale};
    f.write(reinterpret_cast<const char *>(v2), 12);

    // Vertex 3 (3 floats = 12 bytes)
    float v3[3] = {tri.v3.x * exportScale, tri.v3.y * exportScale,
                   tri.v3.z * exportScale};
    f.write(reinterpret_cast<const char *>(v3), 12);

    // Attribute byte count (2 bytes)
    uint16_t attr = 0;
    f.write(reinterpret_cast<const char *>(&attr), 2);
  }

  f.close();

  // ── Verify file size matches expected ──
  size_t expected = 80 + 4 + (size_t)nt * 50;
  std::ifstream check(filename, std::ios::binary | std::ios::ate);
  size_t actual = check.tellg();
  check.close();

  if (actual == expected) {
    std::cout << "[Mesher] Exported " << nt << " tris to " << filename << " ("
              << (actual / 1024) << " KB)\n";
  } else {
    std::cerr << "[Mesher] WARNING: STL size mismatch! Expected " << expected
              << " bytes, got " << actual << " bytes\n";
  }
}