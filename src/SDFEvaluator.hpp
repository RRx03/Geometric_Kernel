#pragma once
#include "SDFShared.h"
#include <algorithm>
#include <cmath>
#include <simd/simd.h>
#include <vector>

class SDFEvaluator {
public:
  SDFEvaluator(const std::vector<SDFNodeGPU> &nodes) : _nodes(nodes) {}

  float evaluate(simd::float3 pos) const {
    float stack[32];
    int sp = 0;

    for (size_t i = 0; i < _nodes.size(); ++i) {
      SDFNodeGPU node = _nodes[i];
      float dist = 0.0f;

      switch (node.type) {
      case SDF_TYPE_SPHERE:
        dist = simd_length(pos - node.position) - node.params.x;
        stack[sp++] = dist;
        break;
      case SDF_TYPE_BOX: {
        simd::float3 d =
            simd_abs(pos - node.position) -
            simd_make_float3(node.params.x, node.params.y, node.params.z);
        dist = simd_length(simd_max(d, simd_make_float3(0.0f))) +
               std::min(std::max(d.x, std::max(d.y, d.z)), 0.0f);
        stack[sp++] = dist;
        break;
      }
      case SDF_TYPE_CIRCLE_2D: {
        simd::float2 p2d =
            simd_make_float2(std::sqrt(pos.x * pos.x + pos.z * pos.z), pos.y);
        dist = simd_length(p2d -
                           simd_make_float2(node.position.x, node.position.y)) -
               node.params.x;
        stack[sp++] = dist;
        break;
      }
      case SDF_TYPE_RECT_2D: {
        simd::float2 p2d =
            simd_make_float2(std::sqrt(pos.x * pos.x + pos.z * pos.z), pos.y);
        simd::float2 d =
            simd_abs(p2d - simd_make_float2(node.position.x, node.position.y)) -
            simd_make_float2(node.params.x, node.params.y);
        dist = simd_length(simd_max(d, simd_make_float2(0.0f))) +
               std::min(std::max(d.x, d.y), 0.0f);
        stack[sp++] = dist;
        break;
      }
      case SDF_TYPE_BEZIER2D: {
        simd::float2 p2d =
            simd_make_float2(std::sqrt(pos.x * pos.x + pos.z * pos.z), pos.y);
        simd::float2 p0 = simd_make_float2(node.position.x, node.position.y);
        simd::float2 p1 = simd_make_float2(node.params.x, node.params.y);
        simd::float2 p2 = simd_make_float2(node.params.z, node.params.w);
        float thickness = node.position.z;
        dist = sdBezier(p2d, p0, p1, p2) - thickness;
        stack[sp++] = dist;
        break;
      }
      case SDF_OP_UNION: {
        float d2 = stack[--sp];
        float d1 = stack[--sp];
        dist = std::min(d1, d2);
        stack[sp++] = dist;
        break;
      }
      case SDF_OP_SUBTRACT: {
        float d2 = stack[--sp];
        float d1 = stack[--sp];
        dist = std::max(d1, -d2);
        stack[sp++] = dist;
        break;
      }
      case SDF_OP_INTERSECT: {
        float d2 = stack[--sp];
        float d1 = stack[--sp];
        dist = std::max(d1, d2);
        stack[sp++] = dist;
        break;
      }
      }
    }
    return stack[0];
  }

private:
  std::vector<SDFNodeGPU> _nodes;

  static float sdBezier(simd::float2 pos, simd::float2 A, simd::float2 B,
                        simd::float2 C) {
    simd::float2 a = B - A;
    simd::float2 b = A - 2.0f * B + C;
    simd::float2 c = a * 2.0f;
    simd::float2 d = A - pos;

    float kk = 1.0f / simd_dot(b, b);
    float kx = kk * simd_dot(a, b);
    float ky = kk * (2.0f * simd_dot(a, a) + simd_dot(d, b)) / 3.0f;
    float kz = kk * simd_dot(d, a);

    float res = 0.0f;
    float p = ky - kx * kx;
    float p3 = p * p * p;
    float q = kx * (2.0f * kx * kx - 3.0f * ky) + kz;
    float h = q * q + 4.0f * p3;

    if (h >= 0.0f) {
      h = std::sqrt(h);
      simd::float2 x =
          (simd_make_float2(h, -h) - simd_make_float2(q, q)) / 2.0f;
      simd::float2 uv = simd_make_float2(
          std::copysign(std::pow(std::abs(x.x), 1.0f / 3.0f), x.x),
          std::copysign(std::pow(std::abs(x.y), 1.0f / 3.0f), x.y));
      float t = std::clamp(uv.x + uv.y - kx, 0.0f, 1.0f);
      res = simd_length(d + (c + b * t) * t);
    } else {
      float z = std::sqrt(-p);
      float v = std::acos(q / (p * z * 2.0f)) / 3.0f;
      float m = std::cos(v);
      float n = std::sin(v) * 1.732050808f;
      float tx = std::clamp((m + m) * z - kx, 0.0f, 1.0f);
      float ty = std::clamp((-n - m) * z - kx, 0.0f, 1.0f);
      res = std::min(simd_length(d + (c + b * tx) * tx),
                     simd_length(d + (c + b * ty) * ty));
    }
    return res;
  }
};