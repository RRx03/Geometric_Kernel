#pragma once
#include "../SDFShared.h"
#include <cmath>
#include <memory>
#include <simd/simd.h>
#include <vector>

namespace Geometry {

class SDFNode {
public:
    virtual ~SDFNode() = default;
    virtual int flatten(std::vector<SDFNodeGPU>& buffer) const = 0;
};

inline SDFNodeGPU makeNode() {
  SDFNodeGPU n{};
  n.type = SDF_DATA_CARRIER;
  n.leftChildIndex = -1;
  n.rightChildIndex = -1;
  return n;
}

// ═══ Primitives 3D ═══

class Sphere : public SDFNode {
public:
  simd::float3 center;
  float radius;
  Sphere(simd::float3 c, float r) : center(c), radius(r) {}
  int flatten(std::vector<SDFNodeGPU> &buf) const override {
    SDFNodeGPU n = makeNode();
    n.type = SDF_TYPE_SPHERE;
    n.pos_x = center.x;
    n.pos_y = center.y;
    n.pos_z = center.z;
    n.param_x = radius;
    buf.push_back(n);
    return (int)buf.size() - 1;
  }
};
class Box : public SDFNode {
public:
  simd::float3 center, halfExtents;
  Box(simd::float3 c, simd::float3 h) : center(c), halfExtents(h) {}
  int flatten(std::vector<SDFNodeGPU> &buf) const override {
    SDFNodeGPU n = makeNode();
    n.type = SDF_TYPE_BOX;
    n.pos_x = center.x;
    n.pos_y = center.y;
    n.pos_z = center.z;
    n.param_x = halfExtents.x;
    n.param_y = halfExtents.y;
    n.param_z = halfExtents.z;
    buf.push_back(n);
    return (int)buf.size() - 1;
  }
};
class Cylinder : public SDFNode {
public:
  simd::float3 center;
  float radius, halfHeight;
  Cylinder(simd::float3 c, float r, float hh)
      : center(c), radius(r), halfHeight(hh) {}
  int flatten(std::vector<SDFNodeGPU> &buf) const override {
    SDFNodeGPU n = makeNode();
    n.type = SDF_TYPE_CYLINDER;
    n.pos_x = center.x;
    n.pos_y = center.y;
    n.pos_z = center.z;
    n.param_x = radius;
    n.param_y = halfHeight;
    buf.push_back(n);
    return (int)buf.size() - 1;
  }
};
class Torus : public SDFNode {
public:
  simd::float3 center;
  float majorR, minorR;
  Torus(simd::float3 c, float R, float r) : center(c), majorR(R), minorR(r) {}
  int flatten(std::vector<SDFNodeGPU> &buf) const override {
    SDFNodeGPU n = makeNode();
    n.type = SDF_TYPE_TORUS;
    n.pos_x = center.x;
    n.pos_y = center.y;
    n.pos_z = center.z;
    n.param_x = majorR;
    n.param_y = minorR;
    buf.push_back(n);
    return (int)buf.size() - 1;
  }
};
class Capsule : public SDFNode {
public:
  simd::float3 pointA, pointB;
  float radius;
  Capsule(simd::float3 a, simd::float3 b, float r)
      : pointA(a), pointB(b), radius(r) {}
  int flatten(std::vector<SDFNodeGPU> &buf) const override {
    SDFNodeGPU n = makeNode();
    n.type = SDF_TYPE_CAPSULE;
    n.pos_x = pointA.x;
    n.pos_y = pointA.y;
    n.pos_z = pointA.z;
    n.param_x = pointB.x;
    n.param_y = pointB.y;
    n.param_z = pointB.z;
    n.param_w = radius;
    buf.push_back(n);
    return (int)buf.size() - 1;
  }
};

// ═══ Primitives 2D ═══

class Circle2D : public SDFNode {
public:
  simd::float2 center;
  float radius;
  Circle2D(simd::float2 c, float r) : center(c), radius(r) {}
  int flatten(std::vector<SDFNodeGPU> &buf) const override {
    SDFNodeGPU n = makeNode();
    n.type = SDF_TYPE_CIRCLE_2D;
    n.pos_x = center.x;
    n.pos_y = center.y;
    n.param_x = radius;
    buf.push_back(n);
    return (int)buf.size() - 1;
  }
};
class Rect2D : public SDFNode {
public:
  simd::float2 center, halfExtents;
  Rect2D(simd::float2 c, simd::float2 h) : center(c), halfExtents(h) {}
  int flatten(std::vector<SDFNodeGPU> &buf) const override {
    SDFNodeGPU n = makeNode();
    n.type = SDF_TYPE_RECT_2D;
    n.pos_x = center.x;
    n.pos_y = center.y;
    n.param_x = halfExtents.x;
    n.param_y = halfExtents.y;
    buf.push_back(n);
    return (int)buf.size() - 1;
  }
};
class Bezier2D : public SDFNode {
public:
  simd::float2 p0, p1, p2;
  float thickness;
  Bezier2D(simd::float2 a, simd::float2 b, simd::float2 c, float t)
      : p0(a), p1(b), p2(c), thickness(t) {}
  int flatten(std::vector<SDFNodeGPU> &buf) const override {
    SDFNodeGPU n = makeNode();
    n.type = SDF_TYPE_BEZIER2D;
    n.pos_x = p0.x;
    n.pos_y = p0.y;
    n.pos_z = thickness;
    n.param_x = p1.x;
    n.param_y = p1.y;
    n.param_z = p2.x;
    n.param_w = p2.y;
    buf.push_back(n);
    return (int)buf.size() - 1;
  }
};
class CubicBezier2D : public SDFNode {
public:
  simd::float2 p0, p1, p2, p3;
  float thickness;
  CubicBezier2D(simd::float2 a, simd::float2 b, simd::float2 c, simd::float2 d,
                float t)
      : p0(a), p1(b), p2(c), p3(d), thickness(t) {}
  int flatten(std::vector<SDFNodeGPU> &buf) const override {
    SDFNodeGPU n0 = makeNode();
    n0.type = SDF_TYPE_CUBIC_BEZIER2D;
    n0.pos_x = p0.x;
    n0.pos_y = p0.y;
    n0.pos_z = thickness;
    n0.param_x = p1.x;
    n0.param_y = p1.y;
    n0.param_z = p2.x;
    n0.param_w = p2.y;
    SDFNodeGPU n1 = makeNode();
    n1.pos_x = p3.x;
    n1.pos_y = p3.y;
    buf.push_back(n0);
    buf.push_back(n1);
    return (int)buf.size() - 2;
  }
};

// CompositeSpline2D — stores ORIGINAL B-spline control points
class CompositeSpline2D : public SDFNode {
public:
  std::vector<simd::float2> points;
  float thickness;
  CompositeSpline2D(std::vector<simd::float2> pts, float t)
      : points(std::move(pts)), thickness(t) {}
  int flatten(std::vector<SDFNodeGPU> &buf) const override {
    int N = (int)points.size();
    if (N < 2) {
      SDFNodeGPU n = makeNode();
      n.type = SDF_TYPE_CIRCLE_2D;
      if (N == 1) {
        n.pos_x = points[0].x;
        n.pos_y = points[0].y;
      }
      n.param_x = 0.001f;
      buf.push_back(n);
      return (int)buf.size() - 1;
    }
    int hIdx = (int)buf.size();
    SDFNodeGPU hdr = makeNode();
    hdr.type = SDF_TYPE_COMPOSITE_SPLINE2D;
    hdr.param_x = (float)N;
    hdr.param_y = thickness;
    buf.push_back(hdr);
    for (int k = 0; k < N; k += 3) {
      SDFNodeGPU dc = makeNode();
      dc.pos_x = points[k].x;
      dc.pos_y = points[k].y;
      if (k + 1 < N) {
        dc.param_x = points[k + 1].x;
        dc.param_y = points[k + 1].y;
      }
      if (k + 2 < N) {
        dc.param_z = points[k + 2].x;
        dc.param_w = points[k + 2].y;
      }
      buf.push_back(dc);
    }
    return hIdx;
  }
};

// ═══ CSG ═══

class Union : public SDFNode {
public:
  std::shared_ptr<SDFNode> left, right;
  Union(std::shared_ptr<SDFNode> l, std::shared_ptr<SDFNode> r)
      : left(std::move(l)), right(std::move(r)) {}
  int flatten(std::vector<SDFNodeGPU> &buf) const override {
    int li = left->flatten(buf), ri = right->flatten(buf);
    SDFNodeGPU n = makeNode();
    n.type = SDF_OP_UNION;
    n.leftChildIndex = li;
    n.rightChildIndex = ri;
    buf.push_back(n);
    return (int)buf.size() - 1;
  }
};
class Subtract : public SDFNode {
public:
  std::shared_ptr<SDFNode> base, subtracted;
  Subtract(std::shared_ptr<SDFNode> b, std::shared_ptr<SDFNode> s)
      : base(std::move(b)), subtracted(std::move(s)) {}
  int flatten(std::vector<SDFNodeGPU> &buf) const override {
    int li = base->flatten(buf), ri = subtracted->flatten(buf);
    SDFNodeGPU n = makeNode();
    n.type = SDF_OP_SUBTRACT;
    n.leftChildIndex = li;
    n.rightChildIndex = ri;
    buf.push_back(n);
    return (int)buf.size() - 1;
  }
};
class Intersect : public SDFNode {
public:
  std::shared_ptr<SDFNode> left, right;
  Intersect(std::shared_ptr<SDFNode> l, std::shared_ptr<SDFNode> r)
      : left(std::move(l)), right(std::move(r)) {}
  int flatten(std::vector<SDFNodeGPU> &buf) const override {
    int li = left->flatten(buf), ri = right->flatten(buf);
    SDFNodeGPU n = makeNode();
    n.type = SDF_OP_INTERSECT;
    n.leftChildIndex = li;
    n.rightChildIndex = ri;
    buf.push_back(n);
    return (int)buf.size() - 1;
  }
};
class SmoothUnion : public SDFNode {
public:
  std::shared_ptr<SDFNode> left, right;
  float k;
  SmoothUnion(std::shared_ptr<SDFNode> l, std::shared_ptr<SDFNode> r, float k_)
      : left(std::move(l)), right(std::move(r)), k(k_) {}
  int flatten(std::vector<SDFNodeGPU> &buf) const override {
    int li = left->flatten(buf), ri = right->flatten(buf);
    SDFNodeGPU n = makeNode();
    n.type = SDF_OP_SMOOTH_UNION;
    n.leftChildIndex = li;
    n.rightChildIndex = ri;
    n.smoothFactor = k;
    buf.push_back(n);
    return (int)buf.size() - 1;
  }
};
class Transform : public SDFNode {
public:
  std::shared_ptr<SDFNode> child;
  simd::float3 translate, rotateAxis;
  float rotateAngleRad, scale;
  Transform(std::shared_ptr<SDFNode> c, simd::float3 trans = {0, 0, 0},
            simd::float3 axis = {0, 1, 0}, float angle = 0, float s = 1)
      : child(std::move(c)), translate(trans), rotateAxis(axis),
        rotateAngleRad(angle), scale(s) {}
  int flatten(std::vector<SDFNodeGPU> &buf) const override {
    int ci = child->flatten(buf);
    float is = std::abs(scale) > 1e-10f ? 1.0f / scale : 1.0f;
    float c_ = std::cos(-rotateAngleRad), s_ = std::sin(-rotateAngleRad),
          t_ = 1 - c_;
    float nx = rotateAxis.x, ny = rotateAxis.y, nz = rotateAxis.z;
    float m00 = is * (t_ * nx * nx + c_), m01 = is * (t_ * nx * ny - s_ * nz),
          m02 = is * (t_ * nx * nz + s_ * ny);
    float m10 = is * (t_ * nx * ny + s_ * nz), m11 = is * (t_ * ny * ny + c_),
          m12 = is * (t_ * ny * nz - s_ * nx);
    float m20 = is * (t_ * nx * nz - s_ * ny),
          m21 = is * (t_ * ny * nz + s_ * nx), m22 = is * (t_ * nz * nz + c_);
    float ox = -(m00 * translate.x + m01 * translate.y + m02 * translate.z);
    float oy = -(m10 * translate.x + m11 * translate.y + m12 * translate.z);
    float oz = -(m20 * translate.x + m21 * translate.y + m22 * translate.z);
    SDFNodeGPU n = makeNode();
    n.type = SDF_OP_TRANSFORM;
    n.leftChildIndex = ci;
    n.param_x = scale;
    SDFNodeGPU d0 = makeNode();
    d0.pos_x = m00;
    d0.pos_y = m01;
    d0.pos_z = m02;
    d0.param_x = ox;
    d0.param_y = m10;
    d0.param_z = m11;
    d0.param_w = m12;
    SDFNodeGPU d1 = makeNode();
    d1.pos_x = oy;
    d1.pos_y = m20;
    d1.pos_z = m21;
    d1.param_x = m22;
    d1.param_y = oz;
    buf.push_back(n);
    buf.push_back(d0);
    buf.push_back(d1);
    return (int)buf.size() - 3;
  }
};

} // namespace Geometry