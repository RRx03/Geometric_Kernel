#pragma once
// ═══════════════════════════════════════════════════════════════
// SDFMath.hpp — Fonctions mathématiques partagées CPU/GPU
//
// Contient :
//   - Évaluation de Bézier (quadratique, cubique) + dérivées
//   - Résolution de cubique (Cardano/trigonométrique)
//   - Solver hybride de distance (analytique + Newton-Raphson)
//   - Analytical ray crossing pour le signe des profils 2D
//     (remplace l'ancien winding number par atan2)
//
// Ce fichier est inclus UNIQUEMENT côté C++.
// Les mêmes algorithmes sont réimplémentés en MSL dans kernel.metal
// avec les mêmes constantes et le même nombre d'itérations (P2).
// ═══════════════════════════════════════════════════════════════

#include <algorithm>
#include <cmath>
#include <simd/simd.h>

namespace SDFMath {

// ─────────────────────────────────────────────────────────
// Constantes
// ─────────────────────────────────────────────────────────
constexpr float PI          = 3.14159265358979323846f;
constexpr float TWO_PI      = 6.28318530717958647692f;
constexpr float EPS_ZERO = 1e-10f;
constexpr float EPS_NEWTON = 1e-10f;
constexpr int NEWTON_ITER_QUAD = 3;
constexpr int NEWTON_ITER_CUBIC = 4;
constexpr int CUBIC_SAMPLES = 8;

// ─────────────────────────────────────────────────────────
// Utilitaires SIMD
// ─────────────────────────────────────────────────────────
inline float cross2D(simd::float2 a, simd::float2 b) {
    return a.x * b.y - a.y * b.x;
}

inline float dot2D(simd::float2 a, simd::float2 b) {
    return a.x * b.x + a.y * b.y;
}

inline float length2D(simd::float2 v) {
    return std::sqrt(v.x * v.x + v.y * v.y);
}

inline simd::float2 make_f2(float x, float y) {
    return simd_make_float2(x, y);
}

// ─────────────────────────────────────────────────────────
// Bézier quadratique : C(t) = (1-t)²A + 2(1-t)tB + t²C
// ─────────────────────────────────────────────────────────
inline simd::float2 evalQuadratic(simd::float2 A, simd::float2 B,
                                   simd::float2 C, float t) {
    float u = 1.0f - t;
    return u * u * A + 2.0f * u * t * B + t * t * C;
}

inline simd::float2 evalQuadraticDeriv(simd::float2 A, simd::float2 B,
                                        simd::float2 C, float t) {
    return 2.0f * ((1.0f - t) * (B - A) + t * (C - B));
}

inline simd::float2 evalQuadraticDeriv2(simd::float2 A, simd::float2 B,
                                         simd::float2 C) {
    return 2.0f * (A - 2.0f * B + C);
}

// ─────────────────────────────────────────────────────────
// Bézier cubique
// ─────────────────────────────────────────────────────────
inline simd::float2 evalCubic(simd::float2 P0, simd::float2 P1,
                               simd::float2 P2, simd::float2 P3, float t) {
    float u = 1.0f - t;
    float uu = u * u;
    float tt = t * t;
    return u * uu * P0 + 3.0f * uu * t * P1 + 3.0f * u * tt * P2 + t * tt * P3;
}

inline simd::float2 evalCubicDeriv(simd::float2 P0, simd::float2 P1,
                                    simd::float2 P2, simd::float2 P3, float t) {
    float u = 1.0f - t;
    simd::float2 d01 = P1 - P0;
    simd::float2 d12 = P2 - P1;
    simd::float2 d23 = P3 - P2;
    return 3.0f * (u * u * d01 + 2.0f * u * t * d12 + t * t * d23);
}

inline simd::float2 evalCubicDeriv2(simd::float2 P0, simd::float2 P1,
                                     simd::float2 P2, simd::float2 P3, float t) {
    simd::float2 a = P2 - 2.0f * P1 + P0;
    simd::float2 b = P3 - 2.0f * P2 + P1;
    return 6.0f * ((1.0f - t) * a + t * b);
}

// ─────────────────────────────────────────────────────────
// Résolution de cubique — Cardano / Trigonométrique
// ─────────────────────────────────────────────────────────
inline int solveCubic(float a, float b, float c, float d, float roots[3]) {
  if (std::abs(a) < EPS_ZERO) {
    if (std::abs(b) < EPS_ZERO) {
      if (std::abs(c) < EPS_ZERO)
        return 0;
      roots[0] = -d / c;
      return 1;
    }
    float disc = c * c - 4.0f * b * d;
    if (disc < 0.0f)
      return 0;
    float sq = std::sqrt(disc);
    roots[0] = (-c + sq) / (2.0f * b);
    roots[1] = (-c - sq) / (2.0f * b);
    return (disc < EPS_ZERO) ? 1 : 2;
  }

    float p_coeff = (3.0f * a * c - b * b) / (3.0f * a * a);
    float q_coeff = (2.0f * b * b * b - 9.0f * a * b * c + 27.0f * a * a * d)
                    / (27.0f * a * a * a);
    float offset = -b / (3.0f * a);
    float disc = -(4.0f * p_coeff * p_coeff * p_coeff + 27.0f * q_coeff * q_coeff);

    if (disc > EPS_ZERO) {
      float m = 2.0f * std::sqrt(-p_coeff / 3.0f);
      float arg = 3.0f * q_coeff / (p_coeff * m);
      arg = std::clamp(arg, -1.0f, 1.0f);
      float theta = std::acos(arg) / 3.0f;
      roots[0] = m * std::cos(theta) + offset;
      roots[1] = m * std::cos(theta - TWO_PI / 3.0f) + offset;
      roots[2] = m * std::cos(theta - 2.0f * TWO_PI / 3.0f) + offset;
      return 3;
    } else {
      float sq =
          std::sqrt(std::max(0.0f, q_coeff * q_coeff / 4.0f +
                                       p_coeff * p_coeff * p_coeff / 27.0f));
      float u = std::cbrt(-q_coeff / 2.0f + sq);
      float v = std::cbrt(-q_coeff / 2.0f - sq);
      roots[0] = u + v + offset;
      return 1;
    }
}

// ─────────────────────────────────────────────────────────
// Newton-Raphson
// ─────────────────────────────────────────────────────────
inline float newtonRefineQuadratic(simd::float2 p, simd::float2 A,
                                    simd::float2 B, simd::float2 C, float t0) {
    float t = t0;
    simd::float2 d2 = evalQuadraticDeriv2(A, B, C);
    for (int i = 0; i < NEWTON_ITER_QUAD; ++i) {
        simd::float2 ct = evalQuadratic(A, B, C, t);
        simd::float2 dt = evalQuadraticDeriv(A, B, C, t);
        simd::float2 diff = ct - p;
        float f  = dot2D(diff, dt);
        float fp = dot2D(dt, dt) + dot2D(diff, d2);
        if (std::abs(fp) > EPS_NEWTON) {
            t -= f / fp;
            t = std::clamp(t, 0.0f, 1.0f);
        }
    }
    return t;
}

inline float newtonRefineCubic(simd::float2 p, simd::float2 P0, simd::float2 P1,
                                simd::float2 P2, simd::float2 P3, float t0) {
    float t = t0;
    for (int i = 0; i < NEWTON_ITER_CUBIC; ++i) {
        simd::float2 ct  = evalCubic(P0, P1, P2, P3, t);
        simd::float2 dt  = evalCubicDeriv(P0, P1, P2, P3, t);
        simd::float2 d2t = evalCubicDeriv2(P0, P1, P2, P3, t);
        simd::float2 diff = ct - p;
        float f  = dot2D(diff, dt);
        float fp = dot2D(dt, dt) + dot2D(diff, d2t);
        if (std::abs(fp) > EPS_NEWTON) {
            t -= f / fp;
            t = std::clamp(t, 0.0f, 1.0f);
        }
    }
    return t;
}

// ─────────────────────────────────────────────────────────
// Distance hybride — Bézier quadratique
// ─────────────────────────────────────────────────────────
inline float distanceToQuadraticBezier(simd::float2 p, simd::float2 A,
                                       simd::float2 B, simd::float2 C,
                                       float &bestT_out) {
  simd::float2 u = B - A;
  simd::float2 v = A - 2.0f * B + C;
  simd::float2 w = A - p;

  float coeff_a = 2.0f * dot2D(v, v);
  float coeff_b = 6.0f * dot2D(u, v);
  float coeff_c = 2.0f * (2.0f * dot2D(u, u) + dot2D(w, v));
  float coeff_d = 2.0f * dot2D(w, u);

  float roots[3];
  int nRoots = solveCubic(coeff_a, coeff_b, coeff_c, coeff_d, roots);

  float candidates[5];
  int nCand = 0;
  candidates[nCand++] = 0.0f;
  candidates[nCand++] = 1.0f;
  for (int i = 0; i < nRoots; ++i) {
    if (roots[i] >= -0.01f && roots[i] <= 1.01f) {
      candidates[nCand++] = std::clamp(roots[i], 0.0f, 1.0f);
    }
  }

  float bestDist = 1e10f;
  float bestT = 0.0f;
  for (int i = 0; i < nCand; ++i) {
    float t = newtonRefineQuadratic(p, A, B, C, candidates[i]);
    simd::float2 cp = evalQuadratic(A, B, C, t);
    float d = length2D(p - cp);
    if (d < bestDist) {
      bestDist = d;
      bestT = t;
    }
  }

  bestT_out = bestT;
  return bestDist;
}

inline float distanceToQuadraticBezier(simd::float2 p, simd::float2 A,
                                        simd::float2 B, simd::float2 C) {
    float t;
    return distanceToQuadraticBezier(p, A, B, C, t);
}

// ─────────────────────────────────────────────────────────
// Distance hybride — Bézier cubique
// ─────────────────────────────────────────────────────────
inline float distanceToCubicBezier(simd::float2 p, simd::float2 P0,
                                   simd::float2 P1, simd::float2 P2,
                                   simd::float2 P3, float &bestT_out) {
  float bestT = 0.0f;
  float bestDist = 1e10f;
  for (int j = 0; j <= CUBIC_SAMPLES; ++j) {
    float t = (float)j / (float)CUBIC_SAMPLES;
    float d = length2D(p - evalCubic(P0, P1, P2, P3, t));
    if (d < bestDist) {
      bestDist = d;
      bestT = t;
    }
  }
  bestT = newtonRefineCubic(p, P0, P1, P2, P3, bestT);
  bestDist = length2D(p - evalCubic(P0, P1, P2, P3, bestT));
  bestT_out = bestT;
  return bestDist;
}

inline float distanceToCubicBezier(simd::float2 p, simd::float2 P0,
                                    simd::float2 P1, simd::float2 P2,
                                    simd::float2 P3) {
    float t;
    return distanceToCubicBezier(p, P0, P1, P2, P3, t);
}

// ═══════════════════════════════════════════════════════════════
// SIGN DETERMINATION — Analytical Ray Crossing
//
// Replaces the old atan2-based winding number which was
// numerically unstable at B-spline segment boundaries,
// causing horizontal band artifacts in STL exports.
//
// This is the CPU mirror of bezCross/lineCross in kernel.metal v4.
// ═══════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────
// Analytical ray crossing for a quadratic Bézier segment
//
// Counts crossings of a horizontal ray from p in +x direction
// with the Bézier curve C(t) = (1-t)²A + 2(1-t)tB + t²C.
//
// Solves C_y(t) = p.y exactly:
//   (Ay - 2By + Cy)t² + 2(By - Ay)t + (Ay - py) = 0
//
// For each real root t ∈ (0,1), checks if C_x(t) > p.x.
// EXACT — no tessellation, no atan2, no edge cases at junctions.
// ─────────────────────────────────────────────────────────
inline int bezierCrossing(simd::float2 p, simd::float2 A, simd::float2 B,
                          simd::float2 C) {
  float qa = A.y - 2.0f * B.y + C.y;
  float qb = 2.0f * (B.y - A.y);
  float qc = A.y - p.y;

  int cnt = 0;

  if (std::abs(qa) < 1e-8f) {
    // Degenerate: linear in y
    if (std::abs(qb) > 1e-8f) {
      float t = -qc / qb;
      if (t >= 0.0f && t < 1.0f) {
        float u = 1.0f - t;
        float rx = u * u * A.x + 2.0f * u * t * B.x + t * t * C.x;
        if (rx > p.x)
          cnt++;
      }
    }
    return cnt;
  }

  float disc = qb * qb - 4.0f * qa * qc;
  if (disc < 0.0f)
    return 0;

  float sq = std::sqrt(disc);
  float inv2a = 0.5f / qa;
  float t1 = (-qb + sq) * inv2a;
  float t2 = (-qb - sq) * inv2a;

  if (t1 >= 0.0f && t1 < 1.0f) {
    float u = 1.0f - t1;
    float rx = u * u * A.x + 2.0f * u * t1 * B.x + t1 * t1 * C.x;
    if (rx > p.x)
      cnt++;
  }
  if (t2 >= 0.0f && t2 < 1.0f) {
    float u = 1.0f - t2;
    float rx = u * u * A.x + 2.0f * u * t2 * B.x + t2 * t2 * C.x;
    if (rx > p.x)
      cnt++;
  }

  return cnt;
}

// ─────────────────────────────────────────────────────────
// Ray crossing for a straight line segment (closure edges)
// ─────────────────────────────────────────────────────────
inline int lineCrossing(simd::float2 p, simd::float2 a, simd::float2 b) {
  if ((a.y > p.y) != (b.y > p.y)) {
    float rx = a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x);
    if (rx > p.x)
      return 1;
  }
  return 0;
}

// ─────────────────────────────────────────────────────────
// windingNumberInside — Analytical ray crossing
//
// Same signature as the old atan2 version. Same semantics.
// Used by compositeSplineDistance() below — no other changes needed.
// ─────────────────────────────────────────────────────────
inline bool windingNumberInside(simd::float2 p, const simd::float2* pts, int N) {
    if (N < 2) return false;

    // Perturb query Y to avoid exact alignment with B-spline knot
    // Y-coordinates. At knots, floating point makes the quadratic
    // root land at t ≈ -1e-9 or t ≈ 1+1e-9, losing the crossing
    // in BOTH adjacent segments. A 0.7µm offset is 2800× smaller
    // than typical wall thickness (2mm) — negligible geometrically
    // but sufficient to avoid the degeneracy.
    p.y += 7.31e-7f;

    int cx = 0;

    // ── Crossings on B-spline Bézier segments ──
    if (N == 2) {
      cx += bezierCrossing(p, pts[0], (pts[0] + pts[1]) * 0.5f, pts[1]);
    } else if (N == 3) {
      cx += bezierCrossing(p, pts[0], pts[1], pts[2]);
    } else {
      // First segment
      cx += bezierCrossing(p, pts[0], pts[1], (pts[1] + pts[2]) * 0.5f);
      // Internal segments
      for (int s = 1; s < N - 3; ++s) {
        cx += bezierCrossing(p, (pts[s] + pts[s + 1]) * 0.5f, pts[s + 1],
                             (pts[s + 1] + pts[s + 2]) * 0.5f);
      }
        // Last segment
        cx += bezierCrossing(p, (pts[N - 3] + pts[N - 2]) * 0.5f, pts[N - 2],
                             pts[N - 1]);
    }

    // ── Closure through axis at r = -1e-4 (matching GPU) ──
    simd::float2 lastPt  = pts[N - 1];
    simd::float2 firstPt = pts[0];
    constexpr float AXIS_R = -1e-4f;
    simd::float2 axisEnd   = make_f2(AXIS_R, lastPt.y);
    simd::float2 axisStart = make_f2(AXIS_R, firstPt.y);

    cx += lineCrossing(p, lastPt, axisEnd);
    cx += lineCrossing(p, axisEnd, axisStart);
    cx += lineCrossing(p, axisStart, firstPt);

    return (cx & 1) == 1;
}

// ─────────────────────────────────────────────────────────
// Distance signée complète pour CompositeSpline2D
//
// 1. Calcule la distance non-signée (min sur tous les segments)
// 2. Détermine le signe via ray crossing analytique
// 3. Applique thickness si > 0
// ─────────────────────────────────────────────────────────
inline float compositeSplineDistance(simd::float2 p, const simd::float2* pts,
                                     int N, float thickness) {
    if (N < 2) return 1e10f;

    float minDist = 1e10f;

    auto evalSeg = [&](simd::float2 A, simd::float2 B, simd::float2 C) {
        float d = distanceToQuadraticBezier(p, A, B, C);
        minDist = std::min(minDist, d);
    };

    if (N == 2) {
        evalSeg(pts[0], (pts[0] + pts[1]) * 0.5f, pts[1]);
    } else if (N == 3) {
        evalSeg(pts[0], pts[1], pts[2]);
    } else {
        evalSeg(pts[0], pts[1], (pts[1] + pts[2]) * 0.5f);
        for (int s = 1; s < N - 3; ++s) {
            evalSeg((pts[s] + pts[s + 1]) * 0.5f, pts[s + 1],
                    (pts[s + 1] + pts[s + 2]) * 0.5f);
        }
        evalSeg((pts[N - 3] + pts[N - 2]) * 0.5f, pts[N - 2], pts[N - 1]);
    }

    if (thickness > 1e-6f) {
        return minDist - thickness;
    }

    bool inside = windingNumberInside(p, pts, N);
    return inside ? -minDist : minDist;
}

} // namespace SDFMath