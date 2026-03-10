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

class Sphere : public SDFNode {
public:
  simd::float3 position;
  float radius;

  Sphere(simd::float3 pos, float r) : position(pos), radius(r) {}

  int flatten(std::vector<SDFNodeGPU> &buffer) const override {
    SDFNodeGPU gpuNode = {};
    gpuNode.type = SDF_TYPE_SPHERE;
    gpuNode.leftChildIndex = -1;
    gpuNode.rightChildIndex = -1;
    gpuNode.position = position;
    gpuNode.params = {radius, 0.0f, 0.0f, 0.0f};

    buffer.push_back(gpuNode);
    return (int)buffer.size() - 1;
  }
};

class Box : public SDFNode {
public:
  simd::float3 position;
  simd::float3 bounds;

  Box(simd::float3 pos, simd::float3 b) : position(pos), bounds(b) {}

  int flatten(std::vector<SDFNodeGPU> &buffer) const override {
    SDFNodeGPU gpuNode = {};
    gpuNode.type = SDF_TYPE_BOX;
    gpuNode.leftChildIndex = -1;
    gpuNode.rightChildIndex = -1;
    gpuNode.position = position;
    gpuNode.params = {bounds.x, bounds.y, bounds.z, 0.0f};

    buffer.push_back(gpuNode);
    return (int)buffer.size() - 1;
  }
};

class Union : public SDFNode {
public:
  std::shared_ptr<SDFNode> left;
  std::shared_ptr<SDFNode> right;

  Union(std::shared_ptr<SDFNode> l, std::shared_ptr<SDFNode> r)
      : left(l), right(r) {}

  int flatten(std::vector<SDFNodeGPU> &buffer) const override {
    int leftIdx = left->flatten(buffer);
    int rightIdx = right->flatten(buffer);

    SDFNodeGPU gpuNode = {};
    gpuNode.type = SDF_OP_UNION;
    gpuNode.leftChildIndex = leftIdx;
    gpuNode.rightChildIndex = rightIdx;

    buffer.push_back(gpuNode);
    return (int)buffer.size() - 1;
  }
};

class Circle2D : public SDFNode {
public:
  simd::float3 position;
  float radius;

  Circle2D(simd::float3 pos, float r) : position(pos), radius(r) {}

  int flatten(std::vector<SDFNodeGPU> &buffer) const override {
    SDFNodeGPU gpuNode = {};
    gpuNode.type = SDF_TYPE_CIRCLE_2D;
    gpuNode.leftChildIndex = -1;
    gpuNode.rightChildIndex = -1;
    gpuNode.position = position;
    gpuNode.params = {radius, 0.0f, 0.0f, 0.0f};

    buffer.push_back(gpuNode);
    return (int)buffer.size() - 1;
  }
};

class Rect2D : public SDFNode {
public:
  simd::float3 position;
  simd::float3 bounds;

  Rect2D(simd::float3 pos, simd::float3 b) : position(pos), bounds(b) {}

  int flatten(std::vector<SDFNodeGPU> &buffer) const override {
    SDFNodeGPU gpuNode = {};
    gpuNode.type = SDF_TYPE_RECT_2D;
    gpuNode.leftChildIndex = -1;
    gpuNode.rightChildIndex = -1;
    gpuNode.position = position;
    gpuNode.params = {bounds.x, bounds.y, 0.0f, 0.0f};

    buffer.push_back(gpuNode);
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
    SDFNodeGPU gpuNode = {};
    gpuNode.type = SDF_TYPE_BEZIER2D;
    gpuNode.leftChildIndex = -1;
    gpuNode.rightChildIndex = -1;
    gpuNode.position = {p0.x, p0.y, thickness};
    gpuNode.params = {p1.x, p1.y, p2.x, p2.y};

    buffer.push_back(gpuNode);
    return (int)buffer.size() - 1;
  }
};

class Subtract : public SDFNode {
public:
  std::shared_ptr<SDFNode> baseShape;
  std::shared_ptr<SDFNode> subtractShape;

  Subtract(std::shared_ptr<SDFNode> base, std::shared_ptr<SDFNode> sub)
      : baseShape(base), subtractShape(sub) {}

  int flatten(std::vector<SDFNodeGPU> &buffer) const override {
    int leftIdx = baseShape->flatten(buffer);
    int rightIdx = subtractShape->flatten(buffer);

    SDFNodeGPU gpuNode = {};
    gpuNode.type = SDF_OP_SUBTRACT;
    gpuNode.leftChildIndex = leftIdx;
    gpuNode.rightChildIndex = rightIdx;

    buffer.push_back(gpuNode);
    return (int)buffer.size() - 1;
  }
};

class Intersect : public SDFNode {
public:
  std::shared_ptr<SDFNode> left;
  std::shared_ptr<SDFNode> right;

  Intersect(std::shared_ptr<SDFNode> l, std::shared_ptr<SDFNode> r)
      : left(l), right(r) {}

  int flatten(std::vector<SDFNodeGPU> &buffer) const override {
    int leftIdx = left->flatten(buffer);
    int rightIdx = right->flatten(buffer);

    SDFNodeGPU gpuNode = {};
    gpuNode.type = SDF_OP_INTERSECT;
    gpuNode.leftChildIndex = leftIdx;
    gpuNode.rightChildIndex = rightIdx;

    buffer.push_back(gpuNode);
    return (int)buffer.size() - 1;
  }
};

} // namespace Geometry