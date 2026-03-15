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

      case SDF_TYPE_COMPOSITE_SPLINE2D: {
        dist = evalCompositeSpline2D(pos, i);
        assert(sp < MAX_STACK && "SDF stack overflow");
        stack[sp++] = dist;
        // Skip the DATA_CARRIER nodes
        int N = (int)node.params.x;
        int numCarriers = (N + 2) / 3;
        i += numCarriers;
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

  // ═══════════════════════════════════════════════════════════════
  // Distance non-signée au Bézier quadratique (IQ's method)
  // ═══════════════════════════════════════════════════════════════
  static float sdBezierQuadratic(simd::float2 pos, simd::float2 A,
                                 simd::float2 B, simd::float2 C) {
    simd::float2 a = B - A;
    simd::float2 b = A - 2.0f * B + C;
    simd::float2 c = a * 2.0f;
    simd::float2 d = A - pos;
    float kk = 1.0f / std::max(simd_dot(b, b), 1e-10f);
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
      float t1 = std::clamp(-(m + n) * z - kx, 0.0f, 1.0f);
      float t2 = std::clamp((m - n) * z * 0.5f - kx, 0.0f, 1.0f);
      // Note: third root omitted (always worst)
      float d1 = simd_length_squared(d + (c + b * t1) * t1);
      float d2 = simd_length_squared(d + (c + b * t2) * t2);
      res = std::sqrt(std::min(d1, d2));
    }
    return res;
  }

  // ═══════════════════════════════════════════════════════════════
  // Distance non-signée au Bézier cubique (subdivision + ternary search)
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
    // Ternary search refinement
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
  // CompositeSpline2D — Distance SIGNÉE au profil composite
  //
  // 1. Unpack les points depuis les DATA_CARRIERs
  // 2. Décompose en segments de Bézier quadratique (B-spline)
  // 3. Pour chaque segment, calcule la distance non-signée
  //    ET le r_courbe au point le plus proche sur la courbe
  // 4. Signe : r_point < r_courbe → négatif (solide)
  // 5. Retourne sign * minDist - thickness
  // ═══════════════════════════════════════════════════════════════
  float evalCompositeSpline2D(simd::float3 pos3d, size_t headerIdx) const {
    const SDFNodeGPU &header = _nodes[headerIdx];
    int N = (int)header.params.x;
    float thickness = header.params.y;

    // ── 1. Unpack points ──
    simd::float2 pts[64]; // Max 64 points
    N = std::min(N, 64);
    int ptIdx = 0;
    size_t carrierIdx = headerIdx + 1;
    while (ptIdx < N && carrierIdx < _nodes.size() &&
           _nodes[carrierIdx].type == SDF_DATA_CARRIER) {
      const SDFNodeGPU &dc = _nodes[carrierIdx];
      if (ptIdx < N)
        pts[ptIdx++] = simd_make_float2(dc.position.x, dc.position.y);
      if (ptIdx < N)
        pts[ptIdx++] = simd_make_float2(dc.params.x, dc.params.y);
      if (ptIdx < N)
        pts[ptIdx++] = simd_make_float2(dc.params.z, dc.params.w);
      carrierIdx++;
    }
    N = ptIdx; // Actual number of points unpacked

    if (N < 2)
      return 1e10f;

    // ── 2. Project 3D → 2D axisymétrique ──
    simd::float2 p2d = simd_make_float2(
        std::sqrt(pos3d.x * pos3d.x + pos3d.z * pos3d.z), pos3d.y);

    // ── 3. Decompose into quadratic Bézier segments (B-spline) ──
    // For N points, we have N-2 segments (same as CompositeSpline2D::flatten
    // in the old Union approach, but now we evaluate them all in one pass).
    //
    // Segment 0: P0, P1, mid(P1,P2)
    // Segment i: mid(P[i], P[i+1]), P[i+1], mid(P[i+1], P[i+2])
    // Segment last: mid(P[N-3], P[N-2]), P[N-2], P[N-1]

    float minDist = 1e10f;
    float rCurveAtClosest = 0.0f;

    if (N == 2) {
      // Degenerate: single line segment treated as quadratic with mid control
      simd::float2 mid = (pts[0] + pts[1]) * 0.5f;
      float d =
          sdBezierQuadraticWithR(p2d, pts[0], mid, pts[1], rCurveAtClosest);
      minDist = d;
    } else if (N == 3) {
      float rC;
      float d = sdBezierQuadraticWithR(p2d, pts[0], pts[1], pts[2], rC);
      if (d < minDist) {
        minDist = d;
        rCurveAtClosest = rC;
      }
    } else {
      int numSeg = N - 2;

      // First segment: P0, P1, mid(P1, P2)
      {
        simd::float2 A = pts[0];
        simd::float2 B = pts[1];
        simd::float2 C = (pts[1] + pts[2]) * 0.5f;
        float rC;
        float d = sdBezierQuadraticWithR(p2d, A, B, C, rC);
        if (d < minDist) {
          minDist = d;
          rCurveAtClosest = rC;
        }
      }

      // Middle segments
      for (int s = 1; s < numSeg - 1; s++) {
        simd::float2 A = (pts[s] + pts[s + 1]) * 0.5f;
        simd::float2 B = pts[s + 1];
        simd::float2 C = (pts[s + 1] + pts[s + 2]) * 0.5f;
        float rC;
        float d = sdBezierQuadraticWithR(p2d, A, B, C, rC);
        if (d < minDist) {
          minDist = d;
          rCurveAtClosest = rC;
        }
      }

      // Last segment: mid(P[N-3], P[N-2]), P[N-2], P[N-1]
      {
        simd::float2 A = (pts[N - 3] + pts[N - 2]) * 0.5f;
        simd::float2 B = pts[N - 2];
        simd::float2 C = pts[N - 1];
        float rC;
        float d = sdBezierQuadraticWithR(p2d, A, B, C, rC);
        if (d < minDist) {
          minDist = d;
          rCurveAtClosest = rC;
        }
      }
    }

    // ── 4. Determine sign ──
    // r_point < r_curve → point is between axis and wall → inside (negative)
    float sign = (p2d.x < rCurveAtClosest) ? -1.0f : 1.0f;

    // ── 5. Return signed distance ──
    if (thickness > 1e-6f) {
      // Tube mode: wall of given thickness around the profile
      return minDist - thickness;
    }
    // Half-plane mode: everything between axis and profile is solid
    return sign * minDist;
  }

  // ═══════════════════════════════════════════════════════════════
  // sdBezierQuadraticWithR — Distance + r_curve at closest point
  // Same algorithm as sdBezierQuadratic but also returns the r
  // coordinate of the closest point on the curve.
  // ═══════════════════════════════════════════════════════════════
  static float sdBezierQuadraticWithR(simd::float2 pos, simd::float2 A,
                                      simd::float2 B, simd::float2 C,
                                      float &rCurve) {
    simd::float2 a = B - A;
    simd::float2 b = A - 2.0f * B + C;
    simd::float2 c = a * 2.0f;
    simd::float2 d = A - pos;
    float kk = 1.0f / std::max(simd_dot(b, b), 1e-10f);
    float kx = kk * simd_dot(a, b);
    float ky = kk * (2.0f * simd_dot(a, a) + simd_dot(d, b)) / 3.0f;
    float kz = kk * simd_dot(d, a);
    float res = 0.0f;
    float bestT = 0.0f;
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
      bestT = std::clamp(uv.x + uv.y - kx, 0.0f, 1.0f);
      res = simd_length(d + (c + b * bestT) * bestT);
    } else {
      float z = std::sqrt(-p);
      float v = std::acos(std::clamp(q / (p * z * 2.0f), -1.0f, 1.0f)) / 3.0f;
      float m = std::cos(v);
      float n = std::sin(v) * 1.732050808f;
      float t1 = std::clamp(-(m + n) * z - kx, 0.0f, 1.0f);
      float t2 = std::clamp((m - n) * z * 0.5f - kx, 0.0f, 1.0f);
      float d1 = simd_length_squared(d + (c + b * t1) * t1);
      float d2 = simd_length_squared(d + (c + b * t2) * t2);
      if (d1 < d2) {
        bestT = t1;
        res = std::sqrt(d1);
      } else {
        bestT = t2;
        res = std::sqrt(d2);
      }
    }
    // Evaluate the point on the curve at bestT to get its r coordinate
    simd::float2 curvePoint = A + (c + b * bestT) * bestT;
    rCurve = curvePoint.x; // x = r (radial coordinate)
    return res;
  }
};