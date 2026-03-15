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
        float rC;
        dist = sdBezierQuadraticRobust(p2d, p0, p1, p2, rC) - thickness;
        assert(sp < MAX_STACK && "SDF stack overflow");
        stack[sp++] = dist;
        break;
      }

      case SDF_TYPE_CUBIC_BEZIER2D: {
        assert(i + 1 < _nodes.size());
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

      case SDF_TYPE_COMPOSITE_SPLINE2D: {
        dist = evalCompositeSpline2D(pos, i);
        assert(sp < MAX_STACK && "SDF stack overflow");
        stack[sp++] = dist;
        int N = (int)node.params.x;
        int numCarriers = (N + 2) / 3;
        i += numCarriers;
        break;
      }

      case SDF_OP_UNION: {
        assert(sp >= 2);
        float d2 = stack[--sp], d1 = stack[--sp];
        stack[sp++] = std::min(d1, d2);
        break;
      }
      case SDF_OP_SMOOTH_UNION: {
        assert(sp >= 2);
        float d2 = stack[--sp], d1 = stack[--sp];
        float k = node.smoothFactor;
        float h = std::clamp(0.5f + 0.5f * (d2 - d1) / k, 0.0f, 1.0f);
        stack[sp++] = d2 * (1.0f - h) + d1 * h - k * h * (1.0f - h);
        break;
      }
      case SDF_OP_SUBTRACT: {
        assert(sp >= 2);
        float d2 = stack[--sp], d1 = stack[--sp];
        stack[sp++] = std::max(d1, -d2);
        break;
      }
      case SDF_OP_INTERSECT: {
        assert(sp >= 2);
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

  // ═══════════════════════════════════════════════════════════════
  // Evaluate a quadratic Bezier point at parameter t
  // ═══════════════════════════════════════════════════════════════
  static simd::float2 evalQuadratic(simd::float2 A, simd::float2 B,
                                    simd::float2 C, float t) {
    float u = 1.0f - t;
    return u * u * A + 2.0f * u * t * B + t * t * C;
  }

  // ═══════════════════════════════════════════════════════════════
  // Robust quadratic Bezier distance + rCurve via ternary search
  // No edge cases, always converges.
  // ═══════════════════════════════════════════════════════════════
  static float sdBezierQuadraticRobust(simd::float2 pos, simd::float2 A,
                                       simd::float2 B, simd::float2 C,
                                       float &rCurve) {
    // Coarse subdivision first
    constexpr int SUBDIV = 8;
    float bestT = 0.0f;
    float bestDist = 1e10f;

    for (int j = 0; j <= SUBDIV; j++) {
      float t = (float)j / SUBDIV;
      simd::float2 pt = evalQuadratic(A, B, C, t);
      float d = simd_length(pos - pt);
      if (d < bestDist) {
        bestDist = d;
        bestT = t;
      }
    }

    // Ternary search refinement around bestT
    float lo = std::max(0.0f, bestT - 1.0f / SUBDIV);
    float hi = std::min(1.0f, bestT + 1.0f / SUBDIV);

    for (int iter = 0; iter < 20; iter++) {
      float m1 = lo + (hi - lo) / 3.0f;
      float m2 = hi - (hi - lo) / 3.0f;
      float d1 = simd_length(pos - evalQuadratic(A, B, C, m1));
      float d2 = simd_length(pos - evalQuadratic(A, B, C, m2));
      if (d1 < d2)
        hi = m2;
      else
        lo = m1;
    }

    bestT = (lo + hi) * 0.5f;
    simd::float2 closestPt = evalQuadratic(A, B, C, bestT);
    rCurve = closestPt.x; // x = r (radial coordinate)
    return simd_length(pos - closestPt);
  }

  // ═══════════════════════════════════════════════════════════════
  // Cubic Bezier distance (ternary search)
  // ═══════════════════════════════════════════════════════════════
  static simd::float2 evalCubic(simd::float2 p0, simd::float2 p1,
                                simd::float2 p2, simd::float2 p3, float t) {
    float u = 1.0f - t;
    return u * u * u * p0 + 3.0f * u * u * t * p1 + 3.0f * u * t * t * p2 +
           t * t * t * p3;
  }

  static float sdBezierCubic(simd::float2 pos, simd::float2 p0, simd::float2 p1,
                             simd::float2 p2, simd::float2 p3) {
    constexpr int N = 16;
    float minDist = 1e10f;
    simd::float2 prev = p0;
    for (int j = 1; j <= N; j++) {
      float t = (float)j / N;
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

  // ═══════════════════════════════════════════════════════════════
  // CompositeSpline2D — Signed distance
  // ═══════════════════════════════════════════════════════════════
  float evalCompositeSpline2D(simd::float3 pos3d, size_t headerIdx) const {
    const SDFNodeGPU &header = _nodes[headerIdx];
    int N = (int)header.params.x;
    float thickness = header.params.y;

    // Unpack points
    simd::float2 pts[64];
    N = std::min(N, 64);
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

    // Project to 2D
    simd::float2 p2d = simd_make_float2(
        std::sqrt(pos3d.x * pos3d.x + pos3d.z * pos3d.z), pos3d.y);

    // Evaluate all B-spline segments
    float minDist = 1e10f;
    float rAtClosest = 0.0f;

    auto evalSeg = [&](simd::float2 A, simd::float2 B, simd::float2 C) {
      float rC;
      float d = sdBezierQuadraticRobust(p2d, A, B, C, rC);
      if (d < minDist) {
        minDist = d;
        rAtClosest = rC;
      }
    };

    if (N == 2) {
      simd::float2 mid = (pts[0] + pts[1]) * 0.5f;
      evalSeg(pts[0], mid, pts[1]);
    } else if (N == 3) {
      evalSeg(pts[0], pts[1], pts[2]);
    } else {
      // First segment
      evalSeg(pts[0], pts[1], (pts[1] + pts[2]) * 0.5f);
      // Middle segments
      for (int s = 1; s < N - 3; s++) {
        evalSeg((pts[s] + pts[s + 1]) * 0.5f, pts[s + 1],
                (pts[s + 1] + pts[s + 2]) * 0.5f);
      }
      // Last segment
      evalSeg((pts[N - 3] + pts[N - 2]) * 0.5f, pts[N - 2], pts[N - 1]);
    }

    // Sign: r_point < r_curve → inside (negative)
    float sign = (p2d.x < rAtClosest) ? -1.0f : 1.0f;

    if (thickness > 1e-6f)
      return minDist - thickness; // Tube mode
    return sign * minDist;        // Half-plane signed mode
  }
};