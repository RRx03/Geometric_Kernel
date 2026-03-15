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
        stack[sp++] = dist;
        break;

      case SDF_TYPE_BOX: {
        simd::float3 d =
            simd_abs(pos - node.position) -
            simd_make_float3(node.params.x, node.params.y, node.params.z);
        stack[sp++] = simd_length(simd_max(d, simd_make_float3(0.0f))) +
                      std::min(std::max(d.x, std::max(d.y, d.z)), 0.0f);
        break;
      }

      case SDF_TYPE_CIRCLE_2D: {
        simd::float2 p2d =
            simd_make_float2(std::sqrt(pos.x * pos.x + pos.z * pos.z), pos.y);
        stack[sp++] = simd_length(p2d - simd_make_float2(node.position.x,
                                                         node.position.y)) -
                      node.params.x;
        break;
      }

      case SDF_TYPE_RECT_2D: {
        simd::float2 p2d =
            simd_make_float2(std::sqrt(pos.x * pos.x + pos.z * pos.z), pos.y);
        simd::float2 d =
            simd_abs(p2d - simd_make_float2(node.position.x, node.position.y)) -
            simd_make_float2(node.params.x, node.params.y);
        stack[sp++] = simd_length(simd_max(d, simd_make_float2(0.0f))) +
                      std::min(std::max(d.x, d.y), 0.0f);
        break;
      }

      case SDF_TYPE_BEZIER2D: {
        simd::float2 p2d =
            simd_make_float2(std::sqrt(pos.x * pos.x + pos.z * pos.z), pos.y);
        float rC;
        stack[sp++] =
            sdBezierQuadraticRobust(
                p2d, simd_make_float2(node.position.x, node.position.y),
                simd_make_float2(node.params.x, node.params.y),
                simd_make_float2(node.params.z, node.params.w), rC) -
            node.position.z;
        break;
      }

      case SDF_TYPE_CUBIC_BEZIER2D: {
        const SDFNodeGPU &extra = _nodes[i + 1];
        simd::float2 p2d =
            simd_make_float2(std::sqrt(pos.x * pos.x + pos.z * pos.z), pos.y);
        stack[sp++] =
            sdBezierCubic(
                p2d, simd_make_float2(node.position.x, node.position.y),
                simd_make_float2(node.params.x, node.params.y),
                simd_make_float2(node.params.z, node.params.w),
                simd_make_float2(extra.position.x, extra.position.y)) -
            node.position.z;
        i++;
        break;
      }

      case SDF_TYPE_COMPOSITE_SPLINE2D: {
        stack[sp++] = evalCompositeSpline2D(pos, i);
        i += ((int)node.params.x + 2) / 3;
        break;
      }

      case SDF_OP_UNION: {
        float d2 = stack[--sp], d1 = stack[--sp];
        stack[sp++] = std::min(d1, d2);
        break;
      }
      case SDF_OP_SMOOTH_UNION: {
        float d2 = stack[--sp], d1 = stack[--sp];
        float k = node.smoothFactor;
        float h = std::clamp(0.5f + 0.5f * (d2 - d1) / k, 0.0f, 1.0f);
        stack[sp++] = d2 * (1 - h) + d1 * h - k * h * (1 - h);
        break;
      }
      case SDF_OP_SUBTRACT: {
        float d2 = stack[--sp], d1 = stack[--sp];
        stack[sp++] = std::max(d1, -d2);
        break;
      }
      case SDF_OP_INTERSECT: {
        float d2 = stack[--sp], d1 = stack[--sp];
        stack[sp++] = std::max(d1, d2);
        break;
      }
      default:
        break;
      }
    }
    return (sp >= 1) ? stack[0] : 1e10f;
  }

private:
  std::vector<SDFNodeGPU> _nodes;

  static simd::float2 evalQuadratic(simd::float2 A, simd::float2 B,
                                    simd::float2 C, float t) {
    float u = 1.0f - t;
    return u * u * A + 2.0f * u * t * B + t * t * C;
  }

  static float sdBezierQuadraticRobust(simd::float2 pos, simd::float2 A,
                                       simd::float2 B, simd::float2 C,
                                       float &rCurve) {
    float bestT = 0.0f, bestDist = 1e10f;
    for (int j = 0; j <= 6; j++) {
      float t = (float)j / 6.0f;
      float d = simd_length(pos - evalQuadratic(A, B, C, t));
      if (d < bestDist) {
        bestDist = d;
        bestT = t;
      }
    }
    float lo = std::max(0.0f, bestT - 0.17f),
          hi = std::min(1.0f, bestT + 0.17f);
    for (int iter = 0; iter < 12; iter++) {
      float m1 = lo + (hi - lo) / 3.0f, m2 = hi - (hi - lo) / 3.0f;
      if (simd_length(pos - evalQuadratic(A, B, C, m1)) <
          simd_length(pos - evalQuadratic(A, B, C, m2)))
        hi = m2;
      else
        lo = m1;
    }
    bestT = (lo + hi) * 0.5f;
    simd::float2 cp = evalQuadratic(A, B, C, bestT);
    rCurve = cp.x;
    return simd_length(pos - cp);
  }

  static simd::float2 evalCubic(simd::float2 p0, simd::float2 p1,
                                simd::float2 p2, simd::float2 p3, float t) {
    float u = 1.0f - t;
    return u * u * u * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 +
           t * t * t * p3;
  }

  static float sdBezierCubic(simd::float2 pos, simd::float2 p0, simd::float2 p1,
                             simd::float2 p2, simd::float2 p3) {
    float minDist = 1e10f;
    simd::float2 prev = p0;
    for (int j = 1; j <= 12; j++) {
      float t = (float)j / 12.0f;
      simd::float2 curr = evalCubic(p0, p1, p2, p3, t);
      simd::float2 seg = curr - prev;
      float sl2 = simd_dot(seg, seg);
      float proj = (sl2 > 1e-10f)
                       ? std::clamp(simd_dot(pos - prev, seg) / sl2, 0.0f, 1.0f)
                       : 0.0f;
      minDist = std::min(minDist, simd_length(pos - (prev + proj * seg)));
      prev = curr;
    }
    float lo = 0, hi = 1;
    for (int iter = 0; iter < 16; iter++) {
      float m1 = lo + (hi - lo) / 3, m2 = hi - (hi - lo) / 3;
      if (simd_length(pos - evalCubic(p0, p1, p2, p3, m1)) <
          simd_length(pos - evalCubic(p0, p1, p2, p3, m2)))
        hi = m2;
      else
        lo = m1;
    }
    return std::min(minDist, simd_length(pos - evalCubic(p0, p1, p2, p3,
                                                         (lo + hi) * 0.5f)));
  }

  float evalCompositeSpline2D(simd::float3 pos3d, size_t headerIdx) const {
    const SDFNodeGPU &header = _nodes[headerIdx];
    int N = std::min((int)header.params.x, 64);
    float thickness = header.params.y;

    simd::float2 pts[64];
    int ptIdx = 0;
    size_t ci = headerIdx + 1;
    while (ptIdx < N && ci < _nodes.size() &&
           _nodes[ci].type == SDF_DATA_CARRIER) {
      const SDFNodeGPU &dc = _nodes[ci];
      if (ptIdx < N)
        pts[ptIdx++] = simd_make_float2(dc.position.x, dc.position.y);
      if (ptIdx < N)
        pts[ptIdx++] = simd_make_float2(dc.params.x, dc.params.y);
      if (ptIdx < N)
        pts[ptIdx++] = simd_make_float2(dc.params.z, dc.params.w);
      ci++;
    }
    N = ptIdx;
    if (N < 2)
      return 1e10f;

    simd::float2 p2d = simd_make_float2(
        std::sqrt(pos3d.x * pos3d.x + pos3d.z * pos3d.z), pos3d.y);

    float minDist = 1e10f, rAt = 0.0f;
    auto evalSeg = [&](simd::float2 A, simd::float2 B, simd::float2 C) {
      float rC;
      float d = sdBezierQuadraticRobust(p2d, A, B, C, rC);
      if (d < minDist) {
        minDist = d;
        rAt = rC;
      }
    };

    if (N == 2) {
      evalSeg(pts[0], (pts[0] + pts[1]) * 0.5f, pts[1]);
    } else if (N == 3) {
      evalSeg(pts[0], pts[1], pts[2]);
    } else {
      evalSeg(pts[0], pts[1], (pts[1] + pts[2]) * 0.5f);
      for (int s = 1; s < N - 3; s++)
        evalSeg((pts[s] + pts[s + 1]) * 0.5f, pts[s + 1],
                (pts[s + 1] + pts[s + 2]) * 0.5f);
      evalSeg((pts[N - 3] + pts[N - 2]) * 0.5f, pts[N - 2], pts[N - 1]);
    }

    // Y-clamping: outside the profile's Y-range → always exterior
    float yMin = std::min(pts[0].y, pts[N - 1].y);
    float yMax = std::max(pts[0].y, pts[N - 1].y);
    bool outsideY = (p2d.y < yMin - 0.001f) || (p2d.y > yMax + 0.001f);

    float sign = (p2d.x < rAt && !outsideY) ? -1.0f : 1.0f;
    if (thickness > 1e-6f)
      return minDist - thickness;
    return sign * minDist;
  }
};