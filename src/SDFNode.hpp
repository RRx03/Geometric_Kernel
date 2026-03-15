#pragma once
#include "SDFShared.h"
#include <memory>
#include <simd/simd.h>
#include <vector>

namespace Geometry {

class SDFNode {
public:
  virtual ~SDFNode() = default;
  virtual int flatten(std::vector<SDFNodeGPU> &buffer) const = 0;
};

// ══════════════════════════════════════════════════════════
// Primitives 3D
// ══════════════════════════════════════════════════════════

class Sphere : public SDFNode {
public:
  simd::float3 position;
  float radius;
  Sphere(simd::float3 pos, float r) : position(pos), radius(r) {}
  int flatten(std::vector<SDFNodeGPU> &buffer) const override {
    SDFNodeGPU n = {};
    n.type = SDF_TYPE_SPHERE;
    n.leftChildIndex = -1;
    n.rightChildIndex = -1;
    n.position = position;
    n.params = {radius, 0, 0, 0};
    buffer.push_back(n);
    return (int)buffer.size() - 1;
  }
};

class Box : public SDFNode {
public:
  simd::float3 position;
  simd::float3 bounds;
  Box(simd::float3 pos, simd::float3 b) : position(pos), bounds(b) {}
  int flatten(std::vector<SDFNodeGPU> &buffer) const override {
    SDFNodeGPU n = {};
    n.type = SDF_TYPE_BOX;
    n.leftChildIndex = -1;
    n.rightChildIndex = -1;
    n.position = position;
    n.params = {bounds.x, bounds.y, bounds.z, 0};
    buffer.push_back(n);
    return (int)buffer.size() - 1;
  }
};

// ══════════════════════════════════════════════════════════
// Primitives 2D axisymétriques
// ══════════════════════════════════════════════════════════

class Circle2D : public SDFNode {
public:
  simd::float3 position;
  float radius;
  Circle2D(simd::float3 pos, float r) : position(pos), radius(r) {}
  int flatten(std::vector<SDFNodeGPU> &buffer) const override {
    SDFNodeGPU n = {};
    n.type = SDF_TYPE_CIRCLE_2D;
    n.leftChildIndex = -1;
    n.rightChildIndex = -1;
    n.position = position;
    n.params = {radius, 0, 0, 0};
    buffer.push_back(n);
    return (int)buffer.size() - 1;
  }
};

class Rect2D : public SDFNode {
public:
  simd::float3 position;
  simd::float3 bounds;
  Rect2D(simd::float3 pos, simd::float3 b) : position(pos), bounds(b) {}
  int flatten(std::vector<SDFNodeGPU> &buffer) const override {
    SDFNodeGPU n = {};
    n.type = SDF_TYPE_RECT_2D;
    n.leftChildIndex = -1;
    n.rightChildIndex = -1;
    n.position = position;
    n.params = {bounds.x, bounds.y, 0, 0};
    buffer.push_back(n);
    return (int)buffer.size() - 1;
  }
};

class Bezier2D : public SDFNode {
public:
  simd::float2 p0, p1, p2;
  float thickness;
  Bezier2D(simd::float2 a, simd::float2 b, simd::float2 c, float t)
      : p0(a), p1(b), p2(c), thickness(t) {}
  int flatten(std::vector<SDFNodeGPU> &buffer) const override {
    SDFNodeGPU n = {};
    n.type = SDF_TYPE_BEZIER2D;
    n.leftChildIndex = -1;
    n.rightChildIndex = -1;
    n.position = {p0.x, p0.y, thickness};
    n.params = {p1.x, p1.y, p2.x, p2.y};
    buffer.push_back(n);
    return (int)buffer.size() - 1;
  }
};

class CubicBezier2D : public SDFNode {
public:
  simd::float2 p0, p1, p2, p3;
  float thickness;
  CubicBezier2D(simd::float2 a, simd::float2 b, simd::float2 c, simd::float2 d,
                float t)
      : p0(a), p1(b), p2(c), p3(d), thickness(t) {}
  int flatten(std::vector<SDFNodeGPU> &buffer) const override {
    SDFNodeGPU n0 = {};
    n0.type = SDF_TYPE_CUBIC_BEZIER2D;
    n0.leftChildIndex = -1;
    n0.rightChildIndex = -1;
    n0.position = {p0.x, p0.y, thickness};
    n0.params = {p1.x, p1.y, p2.x, p2.y};
    SDFNodeGPU n1 = {};
    n1.type = SDF_DATA_CARRIER;
    n1.leftChildIndex = -1;
    n1.rightChildIndex = -1;
    n1.position = {p3.x, p3.y, 0};
    n1.params = {0, 0, 0, 0};
    buffer.push_back(n0);
    buffer.push_back(n1);
    return (int)buffer.size() - 2;
  }
};

// ─────────────────────────────────────────────────────────
// CompositeSpline2D — Profil à N points, distance SIGNÉE
//
// Émet un header node (SDF_TYPE_COMPOSITE_SPLINE2D) suivi de
// ceil(N/3) nœuds DATA_CARRIER contenant les points packés.
//
// Le header stocke :
//   params.x = N (nombre de points, cast en float)
//   params.y = thickness (0 = mode demi-plan signé)
//
// Chaque DATA_CARRIER stocke 3 points :
//   position = (p[k].r, p[k].y, 0)
//   params   = (p[k+1].r, p[k+1].y, p[k+2].r, p[k+2].y)
//
// L'évaluateur (CPU et GPU) :
//   1. Lit le header pour obtenir N et thickness
//   2. Parcourt les DATA_CARRIERs pour reconstruire les points
//   3. Décompose en segments de Bézier quadratique (B-spline)
//   4. Calcule la distance non-signée au segment le plus proche
//   5. Détermine le signe : r_point < r_courbe → négatif (solide)
//   6. Retourne sign * distance - thickness
//
// Le résultat est un SDF signé : négatif = entre l'axe et le profil.
// Subtract(externe, interne) fonctionne alors correctement.
// ─────────────────────────────────────────────────────────
class CompositeSpline2D : public SDFNode {
public:
  std::vector<simd::float2> points;
  float thickness;

  CompositeSpline2D(std::vector<simd::float2> pts, float t)
      : points(std::move(pts)), thickness(t) {}

  int flatten(std::vector<SDFNodeGPU> &buffer) const override {
    int N = (int)points.size();
    if (N < 2) {
      // Dégénéré : un cercle
      SDFNodeGPU n = {};
      n.type = SDF_TYPE_CIRCLE_2D;
      n.leftChildIndex = -1;
      n.rightChildIndex = -1;
      n.position = {points[0].x, points[0].y, 0};
      n.params = {0.01f, 0, 0, 0};
      buffer.push_back(n);
      return (int)buffer.size() - 1;
    }

    int headerIdx = (int)buffer.size();

    // Header node
    SDFNodeGPU header = {};
    header.type = SDF_TYPE_COMPOSITE_SPLINE2D;
    header.leftChildIndex = -1;
    header.rightChildIndex = -1;
    header.position = {0, 0, 0};
    header.params = {(float)N, thickness, 0, 0};
    buffer.push_back(header);

    // Pack points into DATA_CARRIER nodes, 3 points per node
    for (int k = 0; k < N; k += 3) {
      SDFNodeGPU dc = {};
      dc.type = SDF_DATA_CARRIER;
      dc.leftChildIndex = -1;
      dc.rightChildIndex = -1;

      // Point k+0 → position.xy
      dc.position = {points[k].x, points[k].y, 0};

      // Point k+1 → params.xy (if exists)
      if (k + 1 < N) {
        dc.params.x = points[k + 1].x;
        dc.params.y = points[k + 1].y;
      }
      // Point k+2 → params.zw (if exists)
      if (k + 2 < N) {
        dc.params.z = points[k + 2].x;
        dc.params.w = points[k + 2].y;
      }

      buffer.push_back(dc);
    }

    return headerIdx;
  }
};

// ══════════════════════════════════════════════════════════
// Opérations CSG
// ══════════════════════════════════════════════════════════

class Union : public SDFNode {
public:
  std::shared_ptr<SDFNode> left, right;
  Union(std::shared_ptr<SDFNode> l, std::shared_ptr<SDFNode> r)
      : left(l), right(r) {}
  int flatten(std::vector<SDFNodeGPU> &buffer) const override {
    int li = left->flatten(buffer);
    int ri = right->flatten(buffer);
    SDFNodeGPU n = {};
    n.type = SDF_OP_UNION;
    n.leftChildIndex = li;
    n.rightChildIndex = ri;
    buffer.push_back(n);
    return (int)buffer.size() - 1;
  }
};

class SmoothUnion : public SDFNode {
public:
  std::shared_ptr<SDFNode> left, right;
  float smoothFactor;
  SmoothUnion(std::shared_ptr<SDFNode> l, std::shared_ptr<SDFNode> r, float k)
      : left(l), right(r), smoothFactor(k) {}
  int flatten(std::vector<SDFNodeGPU> &buffer) const override {
    int li = left->flatten(buffer);
    int ri = right->flatten(buffer);
    SDFNodeGPU n = {};
    n.type = SDF_OP_SMOOTH_UNION;
    n.leftChildIndex = li;
    n.rightChildIndex = ri;
    n.smoothFactor = smoothFactor;
    buffer.push_back(n);
    return (int)buffer.size() - 1;
  }
};

class Subtract : public SDFNode {
public:
  std::shared_ptr<SDFNode> baseShape, subtractShape;
  Subtract(std::shared_ptr<SDFNode> b, std::shared_ptr<SDFNode> s)
      : baseShape(b), subtractShape(s) {}
  int flatten(std::vector<SDFNodeGPU> &buffer) const override {
    int li = baseShape->flatten(buffer);
    int ri = subtractShape->flatten(buffer);
    SDFNodeGPU n = {};
    n.type = SDF_OP_SUBTRACT;
    n.leftChildIndex = li;
    n.rightChildIndex = ri;
    buffer.push_back(n);
    return (int)buffer.size() - 1;
  }
};

class Intersect : public SDFNode {
public:
  std::shared_ptr<SDFNode> left, right;
  Intersect(std::shared_ptr<SDFNode> l, std::shared_ptr<SDFNode> r)
      : left(l), right(r) {}
  int flatten(std::vector<SDFNodeGPU> &buffer) const override {
    int li = left->flatten(buffer);
    int ri = right->flatten(buffer);
    SDFNodeGPU n = {};
    n.type = SDF_OP_INTERSECT;
    n.leftChildIndex = li;
    n.rightChildIndex = ri;
    buffer.push_back(n);
    return (int)buffer.size() - 1;
  }
};

} // namespace Geometry