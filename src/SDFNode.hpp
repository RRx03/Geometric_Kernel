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

class CompositeSpline2D : public SDFNode {
public:
  std::vector<simd::float2> points;
  float thickness;

  CompositeSpline2D(std::vector<simd::float2> pts, float t)
      : points(std::move(pts)), thickness(t) {}

  int flatten(std::vector<SDFNodeGPU> &buffer) const override {
    int N = (int)points.size();

    if (N < 2) {
      SDFNodeGPU n = {};
      n.type = SDF_TYPE_CIRCLE_2D;
      n.leftChildIndex = -1;
      n.rightChildIndex = -1;
      n.position = {points[0].x, points[0].y, 0};
      n.params = {0.01f, 0, 0, 0};
      buffer.push_back(n);
      return (int)buffer.size() - 1;
    }

    if (N == 2) {
      simd::float2 mid = (points[0] + points[1]) * 0.5f;
      return emitSegment(buffer, points[0], mid, points[1]);
    }

    if (N == 3) {
      return emitSegment(buffer, points[0], points[1], points[2]);
    }

    int numSegments = N - 2;

    int prevIdx =
        emitSegment(buffer, points[0], points[1], mid(points[1], points[2]));

    for (int i = 1; i < numSegments - 1; i++) {
      int segIdx =
          emitSegment(buffer, mid(points[i], points[i + 1]), points[i + 1],
                      mid(points[i + 1], points[i + 2]));
      prevIdx = emitUnion(buffer, prevIdx, segIdx);
    }

    int lastIdx = emitSegment(buffer, mid(points[N - 3], points[N - 2]),
                              points[N - 2], points[N - 1]);
    prevIdx = emitUnion(buffer, prevIdx, lastIdx);

    return prevIdx;
  }

private:
  static simd::float2 mid(simd::float2 a, simd::float2 b) {
    return (a + b) * 0.5f;
  }

  int emitSegment(std::vector<SDFNodeGPU> &buffer, simd::float2 start,
                  simd::float2 control, simd::float2 end) const {
    SDFNodeGPU n = {};
    n.type = SDF_TYPE_BEZIER2D;
    n.leftChildIndex = -1;
    n.rightChildIndex = -1;
    n.position = {start.x, start.y, thickness};
    n.params = {control.x, control.y, end.x, end.y};
    buffer.push_back(n);
    return (int)buffer.size() - 1;
  }

  int emitUnion(std::vector<SDFNodeGPU> &buffer, int leftIdx,
                int rightIdx) const {
    SDFNodeGPU n = {};
    n.type = SDF_OP_UNION;
    n.leftChildIndex = leftIdx;
    n.rightChildIndex = rightIdx;
    buffer.push_back(n);
    return (int)buffer.size() - 1;
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