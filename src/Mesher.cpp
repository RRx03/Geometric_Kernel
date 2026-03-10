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

simd::float3 vertexInterp(float isolevel, simd::float3 p1, simd::float3 p2,
                          float valp1, float valp2) {
  if (std::abs(valp1 - valp2) < 0.00001f)
    return p1;
  float mu = (isolevel - valp1) / (valp2 - valp1);
  return p1 + mu * (p2 - p1);
}

void Mesher::generateSTL(const SDFEvaluator &evaluator, simd::float3 minB,
                         simd::float3 maxB, float res,
                         const std::string &filename) {
  std::vector<Triangle> triangles;
  float isoLevel = 0.0f;

  int gridX = std::ceil((maxB.x - minB.x) / res);
  int gridY = std::ceil((maxB.y - minB.y) / res);
  int gridZ = std::ceil((maxB.z - minB.z) / res);

  std::cout << "Demarrage du maillage. Voxels: " << gridX * gridY * gridZ
            << std::endl;

  for (int z = 0; z < gridZ; z++) {
    for (int y = 0; y < gridY; y++) {
      for (int x = 0; x < gridX; x++) {

        simd::float3 p[8];
        float val[8];

        p[0] = minB + simd_make_float3(x * res, y * res, z * res);
        p[1] = p[0] + simd_make_float3(res, 0, 0);
        p[2] = p[0] + simd_make_float3(res, res, 0);
        p[3] = p[0] + simd_make_float3(0, res, 0);
        p[4] = p[0] + simd_make_float3(0, 0, res);
        p[5] = p[1] + simd_make_float3(0, 0, res);
        p[6] = p[2] + simd_make_float3(0, 0, res);
        p[7] = p[3] + simd_make_float3(0, 0, res);

        for (int i = 0; i < 8; i++)
          val[i] = evaluator.evaluate(p[i]);

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
  snprintf(header, 80, "Generative Kernel STL Export");
  out.write(header, 80);

  uint32_t triCount = triangles.size();
  out.write(reinterpret_cast<const char *>(&triCount), sizeof(triCount));

  for (const auto &t : triangles) {
    simd::float3 normal = simd_normalize(simd_cross(t.v2 - t.v1, t.v3 - t.v1));

    STLTriangle stlTri;
    stlTri.normal[0] = normal.x;
    stlTri.normal[1] = normal.y;
    stlTri.normal[2] = normal.z;

    stlTri.v1[0] = t.v1.x;
    stlTri.v1[1] = t.v1.y;
    stlTri.v1[2] = t.v1.z;
    stlTri.v2[0] = t.v2.x;
    stlTri.v2[1] = t.v2.y;
    stlTri.v2[2] = t.v2.z;
    stlTri.v3[0] = t.v3.x;
    stlTri.v3[1] = t.v3.y;
    stlTri.v3[2] = t.v3.z;

    stlTri.attributeByteCount = 0;

    out.write(reinterpret_cast<const char *>(&stlTri), sizeof(STLTriangle));
  }
  out.close();
  std::cout << "Export termine !" << std::endl;
}