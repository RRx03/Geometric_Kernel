#pragma once
// ═══════════════════════════════════════════════════════════════
// SDFMath.hpp — Fonctions mathématiques partagées CPU/GPU
//
// Contient :
//   - Évaluation de Bézier (quadratique, cubique) + dérivées
//   - Résolution de cubique (Cardano/trigonométrique)
//   - Solver hybride de distance (analytique + Newton-Raphson)
//   - Winding number adaptatif pour le signe des profils 2D
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
constexpr float EPS_ZERO    = 1e-10f;   // Guard contre division par zéro
constexpr float EPS_NEWTON  = 1e-10f;   // Guard Newton denominator
constexpr int   NEWTON_ITER_QUAD = 3;   // Itérations Newton pour quadratique
constexpr int   NEWTON_ITER_CUBIC = 4;  // Itérations Newton pour cubique
constexpr int   CUBIC_SAMPLES = 8;      // Échantillons pour seed cubique

// Winding number adaptatif
constexpr int   WINDING_K_MIN = 2;
constexpr int   WINDING_K_MAX = 16;
constexpr float WINDING_CURVATURE_THRESHOLD = 0.01f;

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

// C'(t) = 2[(1-t)(B-A) + t(C-B)]
inline simd::float2 evalQuadraticDeriv(simd::float2 A, simd::float2 B,
                                        simd::float2 C, float t) {
    return 2.0f * ((1.0f - t) * (B - A) + t * (C - B));
}

// C''(t) = 2(A - 2B + C)  — constant
inline simd::float2 evalQuadraticDeriv2(simd::float2 A, simd::float2 B,
                                         simd::float2 C) {
    return 2.0f * (A - 2.0f * B + C);
}

// ─────────────────────────────────────────────────────────
// Bézier cubique : C(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
// ─────────────────────────────────────────────────────────
inline simd::float2 evalCubic(simd::float2 P0, simd::float2 P1,
                               simd::float2 P2, simd::float2 P3, float t) {
    float u = 1.0f - t;
    float uu = u * u;
    float tt = t * t;
    return u * uu * P0 + 3.0f * uu * t * P1 + 3.0f * u * tt * P2 + t * tt * P3;
}

// C'(t) = 3[(1-t)²(P1-P0) + 2(1-t)t(P2-P1) + t²(P3-P2)]
inline simd::float2 evalCubicDeriv(simd::float2 P0, simd::float2 P1,
                                    simd::float2 P2, simd::float2 P3, float t) {
    float u = 1.0f - t;
    simd::float2 d01 = P1 - P0;
    simd::float2 d12 = P2 - P1;
    simd::float2 d23 = P3 - P2;
    return 3.0f * (u * u * d01 + 2.0f * u * t * d12 + t * t * d23);
}

// C''(t) = 6[(1-t)(P2 - 2P1 + P0) + t(P3 - 2P2 + P1)]
inline simd::float2 evalCubicDeriv2(simd::float2 P0, simd::float2 P1,
                                     simd::float2 P2, simd::float2 P3, float t) {
    simd::float2 a = P2 - 2.0f * P1 + P0;
    simd::float2 b = P3 - 2.0f * P2 + P1;
    return 6.0f * ((1.0f - t) * a + t * b);
}

// ─────────────────────────────────────────────────────────
// Résolution de cubique — Cardano / Trigonométrique
//
// Résout ax³ + bx² + cx + d = 0
// Retourne le nombre de racines réelles (1 ou 3) dans roots[]
// ─────────────────────────────────────────────────────────
inline int solveCubic(float a, float b, float c, float d, float roots[3]) {
    if (std::abs(a) < EPS_ZERO) {
        // Dégénère en quadratique bx² + cx + d = 0
        if (std::abs(b) < EPS_ZERO) {
            // Dégénère en linéaire cx + d = 0
            if (std::abs(c) < EPS_ZERO) return 0;
            roots[0] = -d / c;
            return 1;
        }
        float disc = c * c - 4.0f * b * d;
        if (disc < 0.0f) return 0;
        float sq = std::sqrt(disc);
        roots[0] = (-c + sq) / (2.0f * b);
        roots[1] = (-c - sq) / (2.0f * b);
        return (disc < EPS_ZERO) ? 1 : 2;
    }

    // Normaliser : x³ + px + q = 0 (Tschirnhaus)
    float p_coeff = (3.0f * a * c - b * b) / (3.0f * a * a);
    float q_coeff = (2.0f * b * b * b - 9.0f * a * b * c + 27.0f * a * a * d)
                    / (27.0f * a * a * a);
    float offset = -b / (3.0f * a);

    float disc = -(4.0f * p_coeff * p_coeff * p_coeff + 27.0f * q_coeff * q_coeff);

    if (disc > EPS_ZERO) {
        // 3 racines réelles (cas trigonométrique)
        float m = 2.0f * std::sqrt(-p_coeff / 3.0f);
        float arg = 3.0f * q_coeff / (p_coeff * m);
        arg = std::clamp(arg, -1.0f, 1.0f);
        float theta = std::acos(arg) / 3.0f;

        roots[0] = m * std::cos(theta) + offset;
        roots[1] = m * std::cos(theta - TWO_PI / 3.0f) + offset;
        roots[2] = m * std::cos(theta - 2.0f * TWO_PI / 3.0f) + offset;
        return 3;
    } else {
        // 1 racine réelle (Cardano)
        float sq = std::sqrt(std::max(0.0f, q_coeff * q_coeff / 4.0f +
                                              p_coeff * p_coeff * p_coeff / 27.0f));
        float u = std::cbrt(-q_coeff / 2.0f + sq);
        float v = std::cbrt(-q_coeff / 2.0f - sq);
        roots[0] = u + v + offset;
        return 1;
    }
}

// ─────────────────────────────────────────────────────────
// Newton-Raphson sur f(t) = dot(C(t) - p, C'(t))
//
// Affine le paramètre t pour minimiser |p - C(t)|
// ─────────────────────────────────────────────────────────
inline float newtonRefineQuadratic(simd::float2 p, simd::float2 A,
                                    simd::float2 B, simd::float2 C, float t0) {
    float t = t0;
    simd::float2 d2 = evalQuadraticDeriv2(A, B, C); // constant pour quadratique
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
//
// Phase 1 : Seeds analytiques (résolution cubique)
// Phase 2 : Raffinement Newton-Raphson (3 itérations)
//
// Retourne la distance non-signée ET le paramètre t* (via bestT_out)
// ─────────────────────────────────────────────────────────
inline float distanceToQuadraticBezier(simd::float2 p, simd::float2 A,
                                        simd::float2 B, simd::float2 C,
                                        float& bestT_out) {
    // Construire le polynôme f(t) = dot(C(t) - p, C'(t)) = 0
    // C(t) = (1-t)²A + 2(1-t)tB + t²C
    // C'(t) = 2[(B-A) + t(A - 2B + C)]
    //
    // Soit: u = B - A,  v = A - 2B + C
    // C(t) - p = A - p + 2tu + t²v  (en réarrangeant)
    // C'(t) = 2(u + tv)
    //
    // f(t) = dot(A - p + 2tu + t²v, 2(u + tv))
    //      = polynomial degré 3 en t

    simd::float2 u = B - A;
    simd::float2 v = A - 2.0f * B + C;
    simd::float2 w = A - p;

    // f(t) = at³ + bt² + ct + d = 0
    // a = 2·dot(v,v)
    // b = 6·dot(u,v)
    // c = 2·(2·dot(u,u) + dot(w,v))
    // d = 2·dot(w,u)
    float coeff_a = 2.0f * dot2D(v, v);
    float coeff_b = 6.0f * dot2D(u, v);
    float coeff_c = 2.0f * (2.0f * dot2D(u, u) + dot2D(w, v));
    float coeff_d = 2.0f * dot2D(w, u);

    // Phase 1 : Seeds analytiques
    float roots[3];
    int nRoots = solveCubic(coeff_a, coeff_b, coeff_c, coeff_d, roots);

    // Candidats : racines dans [0,1] + bornes
    float candidates[5];
    int nCand = 0;
    candidates[nCand++] = 0.0f;
    candidates[nCand++] = 1.0f;
    for (int i = 0; i < nRoots; ++i) {
        if (roots[i] >= -0.01f && roots[i] <= 1.01f) {
            candidates[nCand++] = std::clamp(roots[i], 0.0f, 1.0f);
        }
    }

    // Phase 2 : Raffinement Newton + sélection du meilleur
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

// Surcharge sans paramètre t de sortie
inline float distanceToQuadraticBezier(simd::float2 p, simd::float2 A,
                                        simd::float2 B, simd::float2 C) {
    float t;
    return distanceToQuadraticBezier(p, A, B, C, t);
}

// ─────────────────────────────────────────────────────────
// Distance hybride — Bézier cubique
//
// Phase 1 : Sampling uniforme (8 échantillons) comme seed
// Phase 2 : Newton-Raphson (4 itérations)
// ─────────────────────────────────────────────────────────
inline float distanceToCubicBezier(simd::float2 p, simd::float2 P0,
                                    simd::float2 P1, simd::float2 P2,
                                    simd::float2 P3, float& bestT_out) {
    // Phase 1 : Sampling
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

    // Phase 2 : Newton
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

// ─────────────────────────────────────────────────────────
// Winding number adaptatif
//
// Calcule la contribution d'un segment au winding number.
// K est adapté à la courbure du segment.
// ─────────────────────────────────────────────────────────

// Courbure estimée d'un segment quadratique = |A - 2B + C|
inline float estimateCurvature(simd::float2 A, simd::float2 B, simd::float2 C) {
    return length2D(A - 2.0f * B + C);
}

// Nombre de subdivisions adaptatif
inline int adaptiveK(simd::float2 A, simd::float2 B, simd::float2 C,
                     float threshold = WINDING_CURVATURE_THRESHOLD) {
    float curv = estimateCurvature(A, B, C);
    int k = (int)std::ceil(curv / threshold);
    return std::clamp(k, WINDING_K_MIN, WINDING_K_MAX);
}

// Contribution angulaire d'un segment droit (P1 → P2) vu depuis p
inline float windingAngleSegment(simd::float2 p, simd::float2 P1, simd::float2 P2) {
    simd::float2 v1 = P1 - p;
    simd::float2 v2 = P2 - p;
    float cr = cross2D(v1, v2);
    float dt = dot2D(v1, v2);
    return std::atan2(cr, dt);
}

// Contribution au winding d'un segment Bézier quadratique (A,B,C) vu depuis p
// Subdivise en K sous-arcs et somme les angles
inline float windingAngleBezierQuad(simd::float2 p, simd::float2 A,
                                     simd::float2 B, simd::float2 C, int K) {
    float totalAngle = 0.0f;
    simd::float2 prev = A;
    for (int k = 1; k <= K; ++k) {
        float t = (float)k / (float)K;
        simd::float2 curr = evalQuadratic(A, B, C, t);
        totalAngle += windingAngleSegment(p, prev, curr);
        prev = curr;
    }
    return totalAngle;
}

// ─────────────────────────────────────────────────────────
// Winding number complet pour un profil CompositeSpline2D
//
// Le profil est défini par N points de contrôle (B-spline ouverte).
// On le ferme virtuellement en ajoutant 3 segments droits :
//   pts[N-1] → (0, pts[N-1].y)  — vers l'axe
//   (0, pts[N-1].y) → (0, pts[0].y)  — le long de l'axe
//   (0, pts[0].y) → pts[0]  — retour au profil
//
// Retourne true si le point est à l'intérieur du solide.
// ─────────────────────────────────────────────────────────
inline bool windingNumberInside(simd::float2 p, const simd::float2* pts, int N) {
    if (N < 2) return false;

    float totalAngle = 0.0f;

    // ── Segments B-spline ──
    if (N == 2) {
        simd::float2 mid = (pts[0] + pts[1]) * 0.5f;
        int K = adaptiveK(pts[0], mid, pts[1]);
        totalAngle += windingAngleBezierQuad(p, pts[0], mid, pts[1], K);
    } else if (N == 3) {
        int K = adaptiveK(pts[0], pts[1], pts[2]);
        totalAngle += windingAngleBezierQuad(p, pts[0], pts[1], pts[2], K);
    } else {
        // Premier segment : (pts[0], pts[1], mid(pts[1], pts[2]))
        {
            simd::float2 A = pts[0];
            simd::float2 B = pts[1];
            simd::float2 C = (pts[1] + pts[2]) * 0.5f;
            int K = adaptiveK(A, B, C);
            totalAngle += windingAngleBezierQuad(p, A, B, C, K);
        }
        // Segments internes
        for (int s = 1; s < N - 3; ++s) {
            simd::float2 A = (pts[s] + pts[s + 1]) * 0.5f;
            simd::float2 B = pts[s + 1];
            simd::float2 C = (pts[s + 1] + pts[s + 2]) * 0.5f;
            int K = adaptiveK(A, B, C);
            totalAngle += windingAngleBezierQuad(p, A, B, C, K);
        }
        // Dernier segment : (mid(pts[N-3], pts[N-2]), pts[N-2], pts[N-1])
        {
            simd::float2 A = (pts[N - 3] + pts[N - 2]) * 0.5f;
            simd::float2 B = pts[N - 2];
            simd::float2 C = pts[N - 1];
            int K = adaptiveK(A, B, C);
            totalAngle += windingAngleBezierQuad(p, A, B, C, K);
        }
    }

    // ── Segments de fermeture (droites) ──
    simd::float2 lastPt  = pts[N - 1];
    simd::float2 firstPt = pts[0];
    // Close to r = -epsilon (not r = 0) to avoid atan2 ambiguity
    // when the test point lies exactly on the axis (r = 0).
    // The closure polygon passes just behind the axis, so all
    // points with r >= 0 inside the profile are correctly classified.
    constexpr float AXIS_R = -1e-4f;
    simd::float2 axisEnd   = make_f2(AXIS_R, lastPt.y);
    simd::float2 axisStart = make_f2(AXIS_R, firstPt.y);

    totalAngle += windingAngleSegment(p, lastPt, axisEnd);
    totalAngle += windingAngleSegment(p, axisEnd, axisStart);
    totalAngle += windingAngleSegment(p, axisStart, firstPt);

    // |w| ≥ 1 → intérieur. totalAngle = 2π·w(p)
    return std::abs(totalAngle) > PI;
}

// ─────────────────────────────────────────────────────────
// Distance signée complète pour CompositeSpline2D
//
// 1. Calcule la distance non-signée (min sur tous les segments)
// 2. Détermine le signe via winding number
// 3. Applique thickness si > 0
// ─────────────────────────────────────────────────────────
inline float compositeSplineDistance(simd::float2 p, const simd::float2* pts,
                                     int N, float thickness) {
    if (N < 2) return 1e10f;

    float minDist = 1e10f;

    // Lambda pour évaluer un segment et mettre à jour minDist
    auto evalSeg = [&](simd::float2 A, simd::float2 B, simd::float2 C) {
        float d = distanceToQuadraticBezier(p, A, B, C);
        minDist = std::min(minDist, d);
    };

    // Décomposition B-spline → segments Bézier quadratiques
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

    // Mode épaisseur : pas besoin de signe
    if (thickness > 1e-6f) {
        return minDist - thickness;
    }

    // Mode demi-plan signé : winding number
    bool inside = windingNumberInside(p, pts, N);
    return inside ? -minDist : minDist;
}

} // namespace SDFMath