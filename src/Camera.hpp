#pragma once
// ═══════════════════════════════════════════════════════════════
// Camera.hpp — Caméra orbitale sphérique
//
// Séparée du Renderer pour modularité.
// Gère : orbite, pan, zoom, auto-framing, vues prédéfinies,
//         transitions animées.
// ═══════════════════════════════════════════════════════════════

#include "../SDFShared.h"
#include <algorithm>
#include <cmath>
#include <simd/simd.h>

class Camera {
public:
    // ── State ──
    float distance   = 0.5f;
    float azimuth    = 0.5f;
    float elevation  = 0.3f;
    simd::float3 target = {0.0f, 0.0f, 0.0f};

    // ── Animation ──
    bool  animating = false;
    float animTime  = 0.0f;
    float animDuration = 0.3f;

    float anim_startDist, anim_endDist;
    float anim_startAz,   anim_endAz;
    float anim_startEl,   anim_endEl;
    simd::float3 anim_startTarget, anim_endTarget;

    // ── Sensitivities (from spec §15.3) ──
    static constexpr float ORBIT_SPEED = 0.005f;   // rad/pixel
    static constexpr float PAN_FACTOR  = 0.002f;    // distance-proportional
    static constexpr float ZOOM_FACTOR = 0.1f;      // distance-proportional
    static constexpr float ELEV_MIN    = -1.5608f;  // -π/2 + 0.01
    static constexpr float ELEV_MAX    =  1.5608f;  // π/2 - 0.01
    static constexpr float MIN_DIST    = 0.0001f;

    // ─────────────────────────────────────────────────────
    // Controls
    // ─────────────────────────────────────────────────────

    void orbit(float dx, float dy) {
        azimuth   -= dx * ORBIT_SPEED;
        elevation += dy * ORBIT_SPEED;
        elevation  = std::clamp(elevation, ELEV_MIN, ELEV_MAX);
    }

    void pan(float dx, float dy) {
        float speed = distance * PAN_FACTOR;

        // Pan in camera-local XY plane
        float ca = std::cos(azimuth), sa = std::sin(azimuth);

        target.x -= (ca * dx + sa * std::sin(elevation) * dy) * speed;
        target.y += std::cos(elevation) * dy * speed;
        target.z -= (-sa * dx + ca * std::sin(elevation) * dy) * speed;
    }

    void zoom(float delta) {
        distance -= delta * distance * ZOOM_FACTOR;
        distance = std::max(MIN_DIST, distance);
    }

    // ─────────────────────────────────────────────────────
    // Preset views
    // ─────────────────────────────────────────────────────

    void viewFront()    { animateTo(distance, 0.0f, 0.0f, target); }
    void viewRight()    { animateTo(distance, 1.5708f, 0.0f, target); }
    void viewTop()      { animateTo(distance, azimuth, 1.56f, target); }
    void viewThreeQtr() { animateTo(distance, 0.7854f, 0.5236f, target); }

    void resetView(float dist = 0.5f) {
        animateTo(dist, 0.5f, 0.3f, simd_make_float3(0, 0, 0));
    }

    // ─────────────────────────────────────────────────────
    // Auto-frame : fit bounding box
    // ─────────────────────────────────────────────────────

    void autoFrame(simd::float3 bMin, simd::float3 bMax) {
        simd::float3 center = (bMin + bMax) * 0.5f;
        simd::float3 size = bMax - bMin;
        float maxDim = std::max({size.x, size.y, size.z});

        if (maxDim < 1e-6f) maxDim = 1.0f;

        // FOV ~90° pour un fullscreen quad → tan(45°) = 1
        float newDist = maxDim * 1.5f;

        animateTo(newDist, 0.7854f, 0.5236f, center); // vue 3/4
    }

    // ─────────────────────────────────────────────────────
    // Animation update (call each frame with dt)
    // ─────────────────────────────────────────────────────

    void update(float dt) {
        if (!animating) return;

        animTime += dt;
        float t = std::min(animTime / animDuration, 1.0f);

        // Smooth step
        t = t * t * (3.0f - 2.0f * t);

        distance  = anim_startDist + (anim_endDist - anim_startDist) * t;
        azimuth   = anim_startAz   + (anim_endAz   - anim_startAz)   * t;
        elevation = anim_startEl   + (anim_endEl   - anim_startEl)   * t;
        target    = anim_startTarget + (anim_endTarget - anim_startTarget) * t;

        if (animTime >= animDuration) {
            animating = false;
        }
    }

    // ─────────────────────────────────────────────────────
    // Compute camera vectors → Uniforms
    // ─────────────────────────────────────────────────────

    Uniforms computeUniforms() const {
        Uniforms u{};

        float ce = std::cos(elevation);
        float se = std::sin(elevation);
        float ca = std::cos(azimuth);
        float sa = std::sin(azimuth);

        float cpx = target.x + distance * ce * sa;
        float cpy = target.y + distance * se;
        float cpz = target.z + distance * ce * ca;
        u.camPosX = cpx; u.camPosY = cpy; u.camPosZ = cpz;

        // Forward = normalize(target - camPos)
        simd::float3 camPos = simd_make_float3(cpx, cpy, cpz);
        simd::float3 fwd = simd_normalize(target - camPos);
        u.camFwdX = fwd.x; u.camFwdY = fwd.y; u.camFwdZ = fwd.z;

        simd::float3 worldUp = {0.0f, 1.0f, 0.0f};
        simd::float3 right = simd_normalize(simd_cross(worldUp, fwd));

        // Handle near-pole degeneration
        if (simd_length(right) < 0.001f) {
            right = simd_make_float3(1.0f, 0.0f, 0.0f);
        }
        u.camRightX = right.x; u.camRightY = right.y; u.camRightZ = right.z;

        simd::float3 up = simd_cross(fwd, right);
        u.camUpX = up.x; u.camUpY = up.y; u.camUpZ = up.z;

        return u;
    }

private:
    void animateTo(float dist, float az, float el, simd::float3 tgt) {
        anim_startDist   = distance;
        anim_startAz     = azimuth;
        anim_startEl     = elevation;
        anim_startTarget = target;

        anim_endDist   = dist;
        anim_endAz     = az;
        anim_endEl     = el;
        anim_endTarget = tgt;

        animTime  = 0.0f;
        animating = true;
    }
};
