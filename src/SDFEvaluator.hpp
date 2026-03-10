#pragma once
#include "SDFShared.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <simd/simd.h>
#include <vector>

class SDFEvaluator {
public:
  SDFEvaluator(const std::vector<SDFNodeGPU> &nodes) : _nodes(nodes) {}

  float evaluate(simd::float3 pos) const {
    constexpr int MAX_STACK = 64;
    float stack[MAX_STACK];
    int sp = 0;

    for (size_t i = 0; i < _nodes.size(); ++i) {
      const SDFNodeGPU &node = _nodes[i];
      float dist = 0.0f;

      switch (node.type) {

      case SDF_DATA_CARRIER:
        continue;

      case SDF_TYPE_SPHERE:
        dist = simd_length(pos - node.position) - node.params.x;
        assert(sp < MAX_STACK && "SDF stack overflow");
        stack[sp++] = dist;
        break;

      case SDF_TYPE_BOX: {
        simd::float3 d =
            simd_abs(pos - node.position) -
            simd_make_float3(node.params.x, node.params.y, node.params.z);
        dist = simd_length(simd_max(d, simd_make_float3(0.0f))) +
               std::min(std::max(d.x, std::max(d.y, d.z)), 0.0f);
        assert(sp < MAX_STACK && "SDF stack overflow");
        stack[sp++] = dist;
        break;
      }

      case SDF_TYPE_CIRCLE_2D: {
        simd::float2 p2d =
            simd_make_float2(std::sqrt(pos.x * pos.x + pos.z * pos.z), pos.y);
        dist = simd_length(p2d -
                           simd_make_float2(node.position.x, node.position.y)) -
               node.params.x;
        assert(sp < MAX_STACK && "SDF stack overflow");
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
        assert(sp < MAX_STACK && "SDF stack overflow");
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
        dist = sdBezierQuadratic(p2d, p0, p1, p2) - thickness;
        assert(sp < MAX_STACK && "SDF stack overflow");
        stack[sp++] = dist;
        break;
      }

      case SDF_TYPE_CUBIC_BEZIER2D: {
        assert(i + 1 < _nodes.size() && "CubicBezier2D: DATA_CARRIER manquant");
        assert(_nodes[i + 1].type == SDF_DATA_CARRIER);
        const SDFNodeGPU &extra = _nodes[i + 1];
        simd::float2 p2d =
            simd_make_float2(std::sqrt(pos.x * pos.x + pos.z * pos.z), pos.y);
        simd::float2 p0 = simd_make_float2(node.position.x, node.position.y);
        simd::float2 p1 = simd_make_float2(node.params.x, node.params.y);
        simd::float2 p2 = simd_make_float2(node.params.z, node.params.w);
        simd::float2 p3 = simd_make_float2(extra.position.x, extra.position.y);
        float thickness = node.position.z;
        dist = sdBezierCubic(p2d, p0, p1, p2, p3) - thickness;
        i++;
        assert(sp < MAX_STACK && "SDF stack overflow");
        stack[sp++] = dist;
        break;
      }

      case SDF_OP_UNION: {
        assert(sp >= 2 && "SDF stack underflow");
        float d2 = stack[--sp], d1 = stack[--sp];
        stack[sp++] = std::min(d1, d2);
        break;
      }
      case SDF_OP_SMOOTH_UNION: {
        assert(sp >= 2 && "SDF stack underflow");
        float d2 = stack[--sp], d1 = stack[--sp];
        float k = node.smoothFactor;
        float h = std::clamp(0.5f + 0.5f * (d2 - d1) / k, 0.0f, 1.0f);
        stack[sp++] = d2 * (1.0f - h) + d1 * h - k * h * (1.0f - h);
        break;
      }
      case SDF_OP_SUBTRACT: {
        assert(sp >= 2 && "SDF stack underflow");
        float d2 = stack[--sp], d1 = stack[--sp];
        stack[sp++] = std::max(d1, -d2);
        break;
      }
      case SDF_OP_INTERSECT: {
        assert(sp >= 2 && "SDF stack underflow");
        float d2 = stack[--sp], d1 = stack[--sp];
        stack[sp++] = std::max(d1, d2);
        break;
      }
      default:
        assert(false && "SDFEvaluator: type inconnu");
        break;
      }
    }

    assert(sp == 1 && "SDF stack: resultat != 1");
    return stack[0];
  }

private:
  std::vector<SDFNodeGPU> _nodes;

  static float sdBezierQuadratic(simd::float2 pos, simd::float2 A,
                                 simd::float2 B, simd::float2 C) {
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

  static simd::float2 evalCubic(simd::float2 p0, simd::float2 p1,
                                simd::float2 p2, simd::float2 p3, float t) {
    float u = 1.0f - t;
    float uu = u * u;
    float tt = t * t;
    return uu * u * p0 + 3.0f * uu * t * p1 + 3.0f * u * tt * p2 + tt * t * p3;
  }

  static float sdBezierCubic(simd::float2 pos, simd::float2 p0, simd::float2 p1,
                             simd::float2 p2, simd::float2 p3) {
    constexpr int N = 16;
    float minDist = 1e10f;
    simd::float2 prev = p0;
    for (int i = 1; i <= N; i++) {
      float t = static_cast<float>(i) / N;
      simd::float2 curr = evalCubic(p0, p1, p2, p3, t);
      simd::float2 seg = curr - prev;
      float segLen2 = simd_dot(seg, seg);
      float proj = 0.0f;
      if (segLen2 > 1e-10f)
        proj = std::clamp(simd_dot(pos - prev, seg) / segLen2, 0.0f, 1.0f);
      simd::float2 closest = prev + proj * seg;
      minDist = std::min(minDist, simd_length(pos - closest));
      prev = curr;
    }
    float lo = 0.0f, hi = 1.0f;
    for (int iter = 0; iter < 24; iter++) {
      float m1 = lo + (hi - lo) / 3.0f;
      float m2 = hi - (hi - lo) / 3.0f;
      float d1 = simd_length(pos - evalCubic(p0, p1, p2, p3, m1));
      float d2 = simd_length(pos - evalCubic(p0, p1, p2, p3, m2));
      if (d1 < d2)
        hi = m2;
      else
        lo = m1;
    }
    float tBest = (lo + hi) * 0.5f;
    return std::min(minDist,
                    simd_length(pos - evalCubic(p0, p1, p2, p3, tBest)));
  }
};