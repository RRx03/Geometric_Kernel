#include "Mesher.hpp"
#include <fstream>
#include <iostream>

#include "MCTables.h"

#pragma pack(push, 1)
struct STLTriangle {
  float normal[3];
  float v1[3];
  float v2[3];
  float v3[3];
  uint16_t attributeByteCount;
};
#pragma pack(pop)

static simd::float3 vertexInterp(float isolevel, simd::float3 p1,
                                 simd::float3 p2, float valp1, float valp2) {
  if (std::abs(valp1 - valp2) < 1e-5f)
    return p1;
  float mu = (isolevel - valp1) / (valp2 - valp1);
  return p1 + mu * (p2 - p1);
}

void Mesher::generateSTL(const SDFEvaluator &evaluator, simd::float3 minB,
                         simd::float3 maxB, float res,
                         const std::string &filename, float exportScale) {
  std::vector<Triangle> triangles;
  const float isoLevel = 0.0f;

  const int gridX = static_cast<int>(std::ceil((maxB.x - minB.x) / res));
  const int gridY = static_cast<int>(std::ceil((maxB.y - minB.y) / res));
  const int gridZ = static_cast<int>(std::ceil((maxB.z - minB.z) / res));

  const int numX = gridX + 1;
  const int numY = gridY + 1;
  const int numZ = gridZ + 1;

  std::cout << "Demarrage du maillage." << std::endl;
  std::cout << "  Unites internes: metres (SI)" << std::endl;
  std::cout << "  Export scale: x" << exportScale
            << (exportScale == 1000.0f ? " (millimetres)" : "") << std::endl;
  std::cout << "  Bounding box: [" << minB.x << ", " << minB.y << ", " << minB.z
            << "] → [" << maxB.x << ", " << maxB.y << ", " << maxB.z << "] m"
            << std::endl;
  std::cout << "  Resolution: " << res * 1000.0f << " mm/voxel" << std::endl;
  std::cout << "  Voxels: " << gridX << " x " << gridY << " x " << gridZ
            << " = " << (long)gridX * gridY * gridZ << std::endl;

  auto idx = [&](int x, int y, int z) -> size_t {
    return static_cast<size_t>(z) * (numY * numX) +
           static_cast<size_t>(y) * numX + x;
  };

  std::vector<float> valCache(static_cast<size_t>(numX) * numY * numZ);

  for (int z = 0; z < numZ; z++) {
    if (z % std::max(1, numZ / 10) == 0)
      std::cout << "  Cache SDF: " << (100 * z / numZ) << "%" << std::endl;
    for (int y = 0; y < numY; y++) {
      for (int x = 0; x < numX; x++) {
        simd::float3 p = minB + simd_make_float3(x * res, y * res, z * res);
        valCache[idx(x, y, z)] = evaluator.evaluate(p);
      }
    }
  }
  std::cout << "  Cache SDF: 100%" << std::endl;

  static constexpr int cornerOffsets[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0},
                                              {0, 1, 0}, {0, 0, 1}, {1, 0, 1},
                                              {1, 1, 1}, {0, 1, 1}};

  for (int z = 0; z < gridZ; z++) {
    for (int y = 0; y < gridY; y++) {
      for (int x = 0; x < gridX; x++) {
        simd::float3 p[8];
        float val[8];

        for (int c = 0; c < 8; c++) {
          int cx = x + cornerOffsets[c][0];
          int cy = y + cornerOffsets[c][1];
          int cz = z + cornerOffsets[c][2];
          p[c] = minB + simd_make_float3(cx * res, cy * res, cz * res);
          val[c] = valCache[idx(cx, cy, cz)];
        }

        int cubeIndex = 0;
        if (val[0] < isoLevel)
          cubeIndex |= 1;
        if (val[1] < isoLevel)
          cubeIndex |= 2;
        if (val[2] < isoLevel)
          cubeIndex |= 4;
        if (val[3] < isoLevel)
          cubeIndex |= 8;
        if (val[4] < isoLevel)
          cubeIndex |= 16;
        if (val[5] < isoLevel)
          cubeIndex |= 32;
        if (val[6] < isoLevel)
          cubeIndex |= 64;
        if (val[7] < isoLevel)
          cubeIndex |= 128;

        if (edgeTable[cubeIndex] == 0)
          continue;

        simd::float3 vertlist[12];
        if (edgeTable[cubeIndex] & 1)
          vertlist[0] = vertexInterp(isoLevel, p[0], p[1], val[0], val[1]);
        if (edgeTable[cubeIndex] & 2)
          vertlist[1] = vertexInterp(isoLevel, p[1], p[2], val[1], val[2]);
        if (edgeTable[cubeIndex] & 4)
          vertlist[2] = vertexInterp(isoLevel, p[2], p[3], val[2], val[3]);
        if (edgeTable[cubeIndex] & 8)
          vertlist[3] = vertexInterp(isoLevel, p[3], p[0], val[3], val[0]);
        if (edgeTable[cubeIndex] & 16)
          vertlist[4] = vertexInterp(isoLevel, p[4], p[5], val[4], val[5]);
        if (edgeTable[cubeIndex] & 32)
          vertlist[5] = vertexInterp(isoLevel, p[5], p[6], val[5], val[6]);
        if (edgeTable[cubeIndex] & 64)
          vertlist[6] = vertexInterp(isoLevel, p[6], p[7], val[6], val[7]);
        if (edgeTable[cubeIndex] & 128)
          vertlist[7] = vertexInterp(isoLevel, p[7], p[4], val[7], val[4]);
        if (edgeTable[cubeIndex] & 256)
          vertlist[8] = vertexInterp(isoLevel, p[0], p[4], val[0], val[4]);
        if (edgeTable[cubeIndex] & 512)
          vertlist[9] = vertexInterp(isoLevel, p[1], p[5], val[1], val[5]);
        if (edgeTable[cubeIndex] & 1024)
          vertlist[10] = vertexInterp(isoLevel, p[2], p[6], val[2], val[6]);
        if (edgeTable[cubeIndex] & 2048)
          vertlist[11] = vertexInterp(isoLevel, p[3], p[7], val[3], val[7]);

        for (int i = 0; triTable[cubeIndex][i] != -1; i += 3) {
          Triangle tri;
          tri.v1 = vertlist[triTable[cubeIndex][i]];
          tri.v2 = vertlist[triTable[cubeIndex][i + 2]];
          tri.v3 = vertlist[triTable[cubeIndex][i + 1]];
          triangles.push_back(tri);
        }
      }
    }
  }

  std::cout << "Ecriture de " << triangles.size() << " triangles dans "
            << filename << std::endl;
  std::ofstream out(filename, std::ios::binary);

  char header[80] = {0};
  snprintf(header, 80, "Generative Kernel STL - SI metres x%.0f", exportScale);
  out.write(header, 80);

  uint32_t triCount = static_cast<uint32_t>(triangles.size());
  out.write(reinterpret_cast<const char *>(&triCount), sizeof(triCount));

  for (const auto &t : triangles) {
    simd::float3 edge1 = t.v2 - t.v1;
    simd::float3 edge2 = t.v3 - t.v1;
    simd::float3 cross = simd_cross(edge1, edge2);
    float len = simd_length(cross);
    simd::float3 normal =
        (len > 1e-8f) ? (cross / len) : simd_make_float3(0.0f, 1.0f, 0.0f);

    STLTriangle stlTri;
    stlTri.normal[0] = normal.x;
    stlTri.normal[1] = normal.y;
    stlTri.normal[2] = normal.z;

    stlTri.v1[0] = t.v1.x * exportScale;
    stlTri.v1[1] = t.v1.y * exportScale;
    stlTri.v1[2] = t.v1.z * exportScale;
    stlTri.v2[0] = t.v2.x * exportScale;
    stlTri.v2[1] = t.v2.y * exportScale;
    stlTri.v2[2] = t.v2.z * exportScale;
    stlTri.v3[0] = t.v3.x * exportScale;
    stlTri.v3[1] = t.v3.y * exportScale;
    stlTri.v3[2] = t.v3.z * exportScale;

    stlTri.attributeByteCount = 0;
    out.write(reinterpret_cast<const char *>(&stlTri), sizeof(STLTriangle));
  }
  out.close();

  float dimX = (maxB.x - minB.x) * exportScale;
  float dimY = (maxB.y - minB.y) * exportScale;
  float dimZ = (maxB.z - minB.z) * exportScale;
  std::cout << "Export termine ! Dimensions dans le STL: " << dimX << " x "
            << dimY << " x " << dimZ
            << (exportScale == 1000.0f ? " mm" : " unites") << std::endl;
}