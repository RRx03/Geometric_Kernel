#include "Mesher.hpp"
#include "MCTables.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

static simd::float3 vInterp(float iso, simd::float3 p1, simd::float3 p2,
                            float v1, float v2) {
  if (std::abs(v1 - v2) < 1e-10f)
    return p1;
  float t = (iso - v1) / (v2 - v1);
  return p1 + t * (p2 - p1);
}
static const int cOff[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                               {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
static const int edges[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0},
                                 {4, 5}, {5, 6}, {6, 7}, {7, 4},
                                 {0, 4}, {1, 5}, {2, 6}, {3, 7}};

std::vector<Triangle> Mesher::extractMesh(const SDFEvaluator &eval,
                                          float resolution, int maxVPD) {
  simd::float3 bMin = eval.boundsMin(), bMax = eval.boundsMax(),
               sz = bMax - bMin;
  float maxDim = std::max({sz.x, sz.y, sz.z});
  float res = resolution;
  if (res <= 0)
    res = maxDim / (float)maxVPD;
  res = std::max(res, maxDim / (float)maxVPD);
  int nx = std::clamp((int)std::ceil(sz.x / res) + 1, 2, maxVPD);
  int ny = std::clamp((int)std::ceil(sz.y / res) + 1, 2, maxVPD);
  int nz = std::clamp((int)std::ceil(sz.z / res) + 1, 2, maxVPD);
  std::cout << "[Mesher] Grid: " << nx << "x" << ny << "x" << nz
            << " voxel=" << (res * 1000) << "mm\n";

  std::vector<float> grid((size_t)nx * ny * nz);
  auto idx = [&](int x, int y, int z) -> size_t {
    return (size_t)z * ny * nx + (size_t)y * nx + x;
  };

  int nT = std::max(1, (int)std::thread::hardware_concurrency());
  std::vector<std::thread> threads;
  int spt = std::max(1, nz / nT);
  for (int t = 0; t < nT; t++) {
    int zs = t * spt, ze = (t == nT - 1) ? nz : std::min(zs + spt, nz);
    if (zs < ze)
      threads.emplace_back([&, zs, ze]() {
        for (int z = zs; z < ze; z++)
          for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++)
              grid[idx(x, y, z)] = eval.evaluate(
                  bMin + simd_make_float3(x * res, y * res, z * res));
      });
  }
  for (auto &t : threads)
    t.join();
  std::cout << "[Mesher] Grid evaluated.\n";

  std::vector<Triangle> tris;
  float iso = 0;
  for (int z = 0; z < nz - 1; z++)
    for (int y = 0; y < ny - 1; y++)
      for (int x = 0; x < nx - 1; x++) {
        simd::float3 p[8];
        float val[8];
        for (int c = 0; c < 8; c++) {
          int cx = x + cOff[c][0], cy = y + cOff[c][1], cz = z + cOff[c][2];
          p[c] = bMin + simd_make_float3(cx * res, cy * res, cz * res);
          val[c] = grid[idx(cx, cy, cz)];
        }
        int ci = 0;
        for (int c = 0; c < 8; c++)
          if (val[c] < iso)
            ci |= (1 << c);
        if (edgeTable[ci] == 0)
          continue;
        simd::float3 vl[12];
        for (int e = 0; e < 12; e++)
          if (edgeTable[ci] & (1 << e))
            vl[e] = vInterp(iso, p[edges[e][0]], p[edges[e][1]],
                            val[edges[e][0]], val[edges[e][1]]);
        for (int i = 0; triTable[ci][i] != -1; i += 3) {
          Triangle tri;
          tri.v1 = vl[triTable[ci][i]];
          tri.v2 = vl[triTable[ci][i + 1]];
          tri.v3 = vl[triTable[ci][i + 2]];
          tris.push_back(tri);
        }
      }
  std::cout << "[Mesher] " << tris.size() << " triangles.\n";
  return tris;
}

void Mesher::exportSTL(const SDFEvaluator &eval, const std::string &filename,
                       float resolution, int maxVPD, float exportScale) {
  auto tris = extractMesh(eval, resolution, maxVPD);
  std::ofstream f(filename, std::ios::binary);
  if (!f.is_open()) {
    std::cerr << "[Mesher] Cannot open " << filename << "\n";
    return;
  }
  char hdr[80] = {};
  std::snprintf(hdr, 80, "Geometric Kernel STL");
  f.write(hdr, 80);
  uint32_t nt = (uint32_t)tris.size();
  f.write((char *)&nt, 4);
  for (const auto &tri : tris) {
    simd::float3 e1 = tri.v2 - tri.v1, e2 = tri.v3 - tri.v1;
    simd::float3 n = simd_cross(e1, e2);
    float len = simd_length(n);
    if (len > 1e-10f)
      n = n / len;
    else
      n = simd_make_float3(0, 1, 0);
    float nb[3] = {n.x, n.y, n.z};
    f.write((char *)nb, 12);
    for (int v = 0; v < 3; v++) {
      simd::float3 vt = (v == 0) ? tri.v1 : (v == 1) ? tri.v2 : tri.v3;
      float vb[3] = {vt.x * exportScale, vt.y * exportScale,
                     vt.z * exportScale};
      f.write((char *)vb, 12);
    }
    uint16_t attr = 0;
    f.write((char *)&attr, 2);
  }
  f.close();
  std::cout << "[Mesher] Exported " << nt << " tris to " << filename << "\n";
}