// ═══════════════════════════════════════════════════════════════
// test_primitives.cpp — Golden Values Test Suite
//
// Valide chaque primitive SDF, chaque opération CSG, et le
// winding number avec des valeurs calculées analytiquement.
//
// Compilation (macOS) :
//   clang++ -std=c++17 -O2 -I.. tests/test_primitives.cpp -o test_primitives
//
// Exécution :
//   ./test_primitives
//   → retourne 0 si tout passe, 1 si échec
// ═══════════════════════════════════════════════════════════════

#include "../SDFShared.h"
#include "../src/SDFNode.hpp"
#include "../src/SDFEvaluator.hpp"
#include "../src/SDFMath.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────
// Framework de test minimal
// ─────────────────────────────────────────────────────────
static int g_passed = 0;
static int g_failed = 0;
static int g_total  = 0;

constexpr float TOLERANCE = 1e-4f; // Tolérance pour les golden values

void check(const std::string& name, float got, float expected, float tol = TOLERANCE) {
    g_total++;
    float err = std::abs(got - expected);
    if (err < tol) {
        g_passed++;
        printf("  [PASS] %-45s  got=%.6f  exp=%.6f  err=%.1e\n",
               name.c_str(), got, expected, err);
    } else {
        g_failed++;
        printf("  [FAIL] %-45s  got=%.6f  exp=%.6f  err=%.1e  !!!\n",
               name.c_str(), got, expected, err);
    }
}

// Helper : évalue un SDFNode à un point 3D
float eval(const Geometry::SDFNode& node, simd::float3 p) {
    std::vector<SDFNodeGPU> buffer;
    node.flatten(buffer);
    SDFEvaluator evaluator(buffer);
    return evaluator.evaluate(p);
}

simd::float3 p3(float x, float y, float z) {
    return simd_make_float3(x, y, z);
}

simd::float2 p2(float x, float y) {
    return simd_make_float2(x, y);
}

// ═══════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════

void testSphere() {
    printf("\n── Sphere (center=origin, radius=1) ──\n");
    Geometry::Sphere s(p3(0,0,0), 1.0f);

    check("center",             eval(s, p3(0,0,0)),       -1.0f);
    check("surface +X",        eval(s, p3(1,0,0)),        0.0f);
    check("surface +Y",        eval(s, p3(0,1,0)),        0.0f);
    check("surface +Z",        eval(s, p3(0,0,1)),        0.0f);
    check("outside +X",        eval(s, p3(2,0,0)),        1.0f);
    check("inside diagonal",   eval(s, p3(0.5f,0.5f,0.5f)),
          std::sqrt(0.75f) - 1.0f);
    check("outside diagonal",  eval(s, p3(1,1,1)),
          std::sqrt(3.0f) - 1.0f);

    printf("── Sphere (center=(1,2,3), radius=0.5) ──\n");
    Geometry::Sphere s2(p3(1,2,3), 0.5f);
    check("center",            eval(s2, p3(1,2,3)),       -0.5f);
    check("surface",           eval(s2, p3(1.5f,2,3)),    0.0f);
}

void testBox() {
    printf("\n── Box (center=origin, halfExtents=(1,1,1)) ──\n");
    Geometry::Box b(p3(0,0,0), p3(1,1,1));

    check("center",            eval(b, p3(0,0,0)),        -1.0f);
    check("surface face +X",   eval(b, p3(1,0,0)),        0.0f);
    check("surface edge XY",   eval(b, p3(1,1,0)),        0.0f);
    check("outside face +X",   eval(b, p3(2,0,0)),        1.0f);
    check("outside corner",    eval(b, p3(2,2,0)),
          std::sqrt(2.0f));

    printf("── Box (center=(0,0,0), halfExtents=(0.5, 0.5, 0.5)) ──\n");
    Geometry::Box b2(p3(0,0,0), p3(0.5f,0.5f,0.5f));
    check("center",            eval(b2, p3(0,0,0)),       -0.5f);
    check("surface",           eval(b2, p3(0.5f,0,0)),    0.0f);
}

void testCylinder() {
    printf("\n── Cylinder (center=origin, r=1, halfHeight=1) ──\n");
    Geometry::Cylinder c(p3(0,0,0), 1.0f, 1.0f);

    check("center",            eval(c, p3(0,0,0)),        -1.0f);
    check("on side surface",   eval(c, p3(1,0,0)),        0.0f);
    check("on top",            eval(c, p3(0,1,0)),        0.0f);
    check("outside radial",    eval(c, p3(2,0,0)),        1.0f);
    check("outside axial",     eval(c, p3(0,2,0)),        1.0f);
}

void testTorus() {
    printf("\n── Torus (center=origin, R=1, r=0.25) ──\n");
    Geometry::Torus t(p3(0,0,0), 1.0f, 0.25f);

    // Point sur le cercle central du tore (distance = -r)
    check("on tube center",    eval(t, p3(1,0,0)),        -0.25f);
    // Point sur la surface extérieure
    check("outer surface",     eval(t, p3(1.25f,0,0)),    0.0f);
    // Point sur la surface intérieure
    check("inner surface",     eval(t, p3(0.75f,0,0)),    0.0f);
    // Centre de l'espace (loin du tore)
    check("origin",            eval(t, p3(0,0,0)),        1.0f - 0.25f);
}

void testCapsule() {
    printf("\n── Capsule (A=(0,-1,0), B=(0,1,0), r=0.5) ──\n");
    Geometry::Capsule c(p3(0,-1,0), p3(0,1,0), 0.5f);

    check("center",            eval(c, p3(0,0,0)),        -0.5f);
    check("on side surface",   eval(c, p3(0.5f,0,0)),     0.0f);
    check("on end cap +Y",     eval(c, p3(0,1.5f,0)),     0.0f);
    check("outside radial",    eval(c, p3(1.5f,0,0)),     1.0f);
}

void testCircle2D() {
    printf("\n── Circle2D (center=(0.5, 0), radius=0.2) → Tore en 3D ──\n");
    Geometry::Circle2D c(p2(0.5f, 0), 0.2f);

    // Point sur l'axe Y → r = 0, distance au cercle = |0 - 0.5| - 0.2 = 0.3
    // Mais wait, c'est length(p2d - center) - radius
    // p2d = (r=0, y=0), center = (0.5, 0)
    // distance = sqrt(0.25) - 0.2 = 0.5 - 0.2 = 0.3
    check("on axis",           eval(c, p3(0,0,0)),        0.3f);
    // Point à r=0.5, y=0 → p2d=(0.5,0), d = 0 - 0.2 = -0.2
    check("on ring center",    eval(c, p3(0.5f,0,0)),     -0.2f);
    // Point à r=0.7, y=0 → d = 0.2 - 0.2 = 0
    check("on surface",        eval(c, p3(0.7f,0,0)),     0.0f);
}

void testRect2D() {
    printf("\n── Rect2D (center=(0.5, 0), halfExtents=(0.2, 0.3)) ──\n");
    Geometry::Rect2D r(p2(0.5f, 0), p2(0.2f, 0.3f));

    // p3(0.5, 0, 0) → p2d=(0.5, 0) → d=max(|0|-0.2, |0|-0.3) = max(-0.2,-0.3) = -0.2
    check("center of rect",    eval(r, p3(0.5f,0,0)),     -0.2f);
    // p3(0.7, 0, 0) → p2d=(0.7, 0) → d = max(0.2-0.2, -0.3) = max(0, -0.3) = 0
    check("on surface r-edge", eval(r, p3(0.7f,0,0)),     0.0f);
}

void testCompositeSpline2D_Simple() {
    printf("\n── CompositeSpline2D : cylindre simple (2 pts) ──\n");
    // Profil vertical droit : r=0.05 de y=-0.05 à y=0.05
    // C'est un cylindre de rayon 5cm, hauteur 10cm
    std::vector<simd::float2> pts = {p2(0.05f, -0.05f), p2(0.05f, 0.05f)};
    Geometry::CompositeSpline2D spline(pts, 0.0f);

    // Point sur l'axe (r=0, y=0) → intérieur (winding), distance ~0.05
    check("on axis, inside",    eval(spline, p3(0,0,0)),
          -0.05f, 2e-3f);

    // Point à r=0.05, y=0 → sur la surface, distance ~0
    check("on wall surface",    eval(spline, p3(0.05f,0,0)),
          0.0f, 2e-3f);

    // Point à r=0.1, y=0 → extérieur, distance ~0.05
    check("outside radial",     eval(spline, p3(0.1f,0,0)),
          0.05f, 2e-3f);

    // Point au-dessus (r=0, y=0.1) → extérieur
    float d_above = eval(spline, p3(0,0.1f,0));
    check("above, exterior",    (d_above > 0) ? 1.0f : -1.0f, 1.0f);
}

void testCompositeSpline2D_Nozzle() {
    printf("\n── CompositeSpline2D : profil tuyère simplifié (5 pts) ──\n");
    // Profil convergent-divergent simplifié
    // (r, y) : entrée large → col étroit → sortie large
    std::vector<simd::float2> pts = {
        p2(0.04f, -0.06f),   // Entrée (r=4cm)
        p2(0.03f, -0.03f),   // Convergent
        p2(0.02f,  0.00f),   // Col (r=2cm)
        p2(0.03f,  0.03f),   // Divergent
        p2(0.04f,  0.06f),   // Sortie (r=4cm)
    };
    Geometry::CompositeSpline2D spline(pts, 0.0f);

    // Centre du col (r=0, y=0) → intérieur, distance ~0.02
    check("throat center",      eval(spline, p3(0,0,0)),
          -0.02f, 3e-3f);

    // Point loin dehors → extérieur
    float d_far = eval(spline, p3(0.1f, 0, 0));
    check("far outside, +sign", (d_far > 0) ? 1.0f : -1.0f, 1.0f);

    // Point inside near entrance (r=0.02, y=-0.05) — clearly inside
    float d_entry = eval(spline, p3(0.02f, -0.05f, 0));
    check("entry mid, inside",  (d_entry < 0) ? 1.0f : -1.0f, 1.0f);
}

void testCSG_Subtract() {
    printf("\n── CSG : Subtract(sphere r=1, sphere r=0.5 offset) ──\n");
    auto outer = std::make_shared<Geometry::Sphere>(p3(0,0,0), 1.0f);
    auto inner = std::make_shared<Geometry::Sphere>(p3(0,0,0), 0.5f);
    Geometry::Subtract sub(outer, inner);

    // Centre : outer=-1, inner=-0.5 → max(-1, -(-0.5)) = max(-1, 0.5) = 0.5
    check("center (hollow)",    eval(sub, p3(0,0,0)),      0.5f);

    // Point à r=0.75 : outer=-0.25, inner=0.25 → max(-0.25, -0.25) = -0.25
    check("in shell",          eval(sub, p3(0.75f,0,0)),   -0.25f);

    // Surface extérieure inchangée
    check("outer surface",     eval(sub, p3(1,0,0)),       0.0f);

    // Surface intérieure
    check("inner surface",     eval(sub, p3(0.5f,0,0)),    0.0f);
}

void testCSG_Union() {
    printf("\n── CSG : Union(sphere1, sphere2 décalée) ──\n");
    auto s1 = std::make_shared<Geometry::Sphere>(p3(-0.5f,0,0), 1.0f);
    auto s2 = std::make_shared<Geometry::Sphere>(p3( 0.5f,0,0), 1.0f);
    Geometry::Union u(s1, s2);

    // Centre : min(d1, d2) = min(-0.5, -0.5) = -0.5
    check("center",            eval(u, p3(0,0,0)),         -0.5f);

    // Point au milieu entre les deux : min(0, 0) = 0... non
    // s1 à (1.5,0,0) : d=length((1.5+0.5))-1 = 2-1 = 1
    // s2 à (1.5,0,0) : d=length((1.5-0.5))-1 = 1-1 = 0
    check("surface s2 side",   eval(u, p3(1.5f,0,0)),      0.0f);
}

void testCSG_Intersect() {
    printf("\n── CSG : Intersect(sphere r=1, box 0.5) ──\n");
    auto s = std::make_shared<Geometry::Sphere>(p3(0,0,0), 1.0f);
    auto b = std::make_shared<Geometry::Box>(p3(0,0,0), p3(0.5f,0.5f,0.5f));
    Geometry::Intersect inter(s, b);

    // Centre : max(-1, -0.5) = -0.5
    check("center",            eval(inter, p3(0,0,0)),     -0.5f);

    // Face de la boîte à x=0.5 : sphere=-0.5, box=0 → max(-0.5, 0) = 0
    check("box surface",       eval(inter, p3(0.5f,0,0)),  0.0f);
}

void testCSG_SmoothUnion() {
    printf("\n── CSG : SmoothUnion(sphere, sphere, k=0.1) ──\n");
    auto s1 = std::make_shared<Geometry::Sphere>(p3(-0.3f,0,0), 0.5f);
    auto s2 = std::make_shared<Geometry::Sphere>(p3( 0.3f,0,0), 0.5f);
    Geometry::SmoothUnion su(s1, s2, 0.1f);

    // Au centre, le smooth union doit être plus négatif que min(d1, d2)
    float d_smooth = eval(su, p3(0,0,0));
    float d_hard   = std::min(
        simd_length(p3(0.3f,0,0)) - 0.5f,
        simd_length(p3(0.3f,0,0)) - 0.5f
    );
    check("smoother than hard", (d_smooth < d_hard) ? 1.0f : -1.0f, 1.0f);
    check("still negative",     (d_smooth < 0) ? 1.0f : -1.0f, 1.0f);
}

void testWindingNumber() {
    printf("\n── Winding Number : carré fermé ──\n");
    // Profil carré : (r=1, y=-1) → (r=1, y=1) (bord droit)
    // Fermé virtuellement vers l'axe
    simd::float2 pts[] = {p2(1.0f, -1.0f), p2(1.0f, 1.0f)};

    // Point intérieur (r=0.5, y=0) → inside
    check("inside square",
          SDFMath::windingNumberInside(p2(0.5f, 0), pts, 2) ? 1.0f : 0.0f, 1.0f);

    // Point extérieur (r=1.5, y=0) → outside
    check("outside square",
          SDFMath::windingNumberInside(p2(1.5f, 0), pts, 2) ? 1.0f : 0.0f, 0.0f);

    // Point sur l'axe (r=0, y=0) → inside
    check("on axis, inside",
          SDFMath::windingNumberInside(p2(0.001f, 0), pts, 2) ? 1.0f : 0.0f, 1.0f);

    // Point au-dessus (r=0.5, y=1.5) → outside
    check("above, outside",
          SDFMath::windingNumberInside(p2(0.5f, 1.5f), pts, 2) ? 1.0f : 0.0f, 0.0f);
}

void testBezierSolver() {
    printf("\n── Bézier Solver : distance à quadratique ──\n");
    // Segment droit A=(0,0), B=(0.5,0), C=(1,0)
    // Distance de (0.5, 1) au segment = 1.0 (perpendiculaire)
    simd::float2 A = p2(0, 0), B = p2(0.5f, 0), C = p2(1, 0);

    float d1 = SDFMath::distanceToQuadraticBezier(p2(0.5f, 1.0f), A, B, C);
    check("perpendicular distance", d1, 1.0f, 1e-4f);

    // Distance du point (0, 0) au segment = 0
    float d2 = SDFMath::distanceToQuadraticBezier(p2(0, 0), A, B, C);
    check("on endpoint A",         d2, 0.0f, 1e-4f);

    // Distance du point (-1, 0) au segment = 1 (prolongement)
    float d3 = SDFMath::distanceToQuadraticBezier(p2(-1, 0), A, B, C);
    check("before endpoint A",     d3, 1.0f, 1e-4f);

    // Arc courbé : A=(0,0), B=(0.5,1), C=(1,0) — parabole
    simd::float2 Bc = p2(0.5f, 1.0f);
    float d4 = SDFMath::distanceToQuadraticBezier(p2(0.5f, 0.5f), A, Bc, C);
    // Le sommet de la parabole est à (0.5, 0.5), donc distance = 0
    check("at parabola vertex",     d4, 0.0f, 1e-3f);
}

void testCubicSolver() {
    printf("\n── Cubic Solver : résolution de cubique ──\n");
    // x³ - 6x² + 11x - 6 = 0 → racines x=1, x=2, x=3
    float roots[3];
    int n = SDFMath::solveCubic(1.0f, -6.0f, 11.0f, -6.0f, roots);
    check("3 real roots",          (float)n, 3.0f);

    // Trier les racines
    if (n == 3) {
        if (roots[0] > roots[1]) std::swap(roots[0], roots[1]);
        if (roots[1] > roots[2]) std::swap(roots[1], roots[2]);
        if (roots[0] > roots[1]) std::swap(roots[0], roots[1]);
        check("root 1 = 1",       roots[0], 1.0f, 1e-4f);
        check("root 2 = 2",       roots[1], 2.0f, 1e-4f);
        check("root 3 = 3",       roots[2], 3.0f, 1e-4f);
    }
}

// ═══════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════
int main() {
    printf("═══════════════════════════════════════════════════\n");
    printf("  Geometric Kernel — Golden Values Test Suite\n");
    printf("═══════════════════════════════════════════════════\n");

    testSphere();
    testBox();
    testCylinder();
    testTorus();
    testCapsule();
    testCircle2D();
    testRect2D();
    testCompositeSpline2D_Simple();
    testCompositeSpline2D_Nozzle();
    testCSG_Subtract();
    testCSG_Union();
    testCSG_Intersect();
    testCSG_SmoothUnion();
    testWindingNumber();
    testBezierSolver();
    testCubicSolver();

    printf("\n═══════════════════════════════════════════════════\n");
    printf("  Results: %d/%d passed", g_passed, g_total);
    if (g_failed > 0)
        printf(", %d FAILED", g_failed);
    printf("\n═══════════════════════════════════════════════════\n");

    return (g_failed > 0) ? 1 : 0;
}