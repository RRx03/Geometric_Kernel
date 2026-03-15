#pragma once
#include "../SDFShared.h"
#include "SDFMath.hpp"
#include <algorithm>
#include <cmath>
#include <simd/simd.h>
#include <vector>

class SDFEvaluator {
public:
    explicit SDFEvaluator(const std::vector<SDFNodeGPU>& nodes)
        : _nodes(nodes) { computeBounds(); }

    float evaluate(simd::float3 pos) const {
        constexpr int MAX_STACK = 64;
        float stack[MAX_STACK]; int sp = 0;

        for (size_t i = 0; i < _nodes.size(); ++i) {
            const SDFNodeGPU& nd = _nodes[i];
            if (nd.type == SDF_DATA_CARRIER) continue;

            switch (nd.type) {

            case SDF_TYPE_SPHERE: {
                simd::float3 c = simd_make_float3(nd.pos_x, nd.pos_y, nd.pos_z);
                stack[sp++] = simd_length(pos - c) - nd.param_x;
                break;
            }
            case SDF_TYPE_BOX: {
                simd::float3 c = simd_make_float3(nd.pos_x, nd.pos_y, nd.pos_z);
                simd::float3 h = simd_make_float3(nd.param_x, nd.param_y, nd.param_z);
                simd::float3 d = simd_abs(pos - c) - h;
                stack[sp++] = simd_length(simd_max(d, simd_make_float3(0.0f)))
                    + std::min(std::max(d.x, std::max(d.y, d.z)), 0.0f);
                break;
            }
            case SDF_TYPE_CYLINDER: {
                simd::float3 lp = pos - simd_make_float3(nd.pos_x, nd.pos_y, nd.pos_z);
                float dR = std::sqrt(lp.x*lp.x + lp.z*lp.z) - nd.param_x;
                float dA = std::abs(lp.y) - nd.param_y;
                float oR = std::max(dR, 0.0f), oA = std::max(dA, 0.0f);
                stack[sp++] = std::sqrt(oR*oR + oA*oA) + std::min(std::max(dR, dA), 0.0f);
                break;
            }
            case SDF_TYPE_TORUS: {
                simd::float3 lp = pos - simd_make_float3(nd.pos_x, nd.pos_y, nd.pos_z);
                float qr = std::sqrt(lp.x*lp.x + lp.z*lp.z) - nd.param_x;
                stack[sp++] = std::sqrt(qr*qr + lp.y*lp.y) - nd.param_y;
                break;
            }
            case SDF_TYPE_CAPSULE: {
                simd::float3 A = simd_make_float3(nd.pos_x, nd.pos_y, nd.pos_z);
                simd::float3 B = simd_make_float3(nd.param_x, nd.param_y, nd.param_z);
                float radius = nd.param_w;
                simd::float3 pa = pos - A, ba = B - A;
                float baba = simd_dot(ba, ba);
                float t = (baba > 1e-10f) ? std::clamp(simd_dot(pa, ba) / baba, 0.0f, 1.0f) : 0.0f;
                stack[sp++] = simd_length(pa - ba * t) - radius;
                break;
            }

            // ── Primitives 2D ──

            case SDF_TYPE_CIRCLE_2D: {
                simd::float2 p2d = simd_make_float2(std::sqrt(pos.x*pos.x+pos.z*pos.z), pos.y);
                simd::float2 c = simd_make_float2(nd.pos_x, nd.pos_y);
                stack[sp++] = simd_length(p2d - c) - nd.param_x;
                break;
            }
            case SDF_TYPE_RECT_2D: {
                simd::float2 p2d = simd_make_float2(std::sqrt(pos.x*pos.x+pos.z*pos.z), pos.y);
                simd::float2 c = simd_make_float2(nd.pos_x, nd.pos_y);
                simd::float2 h = simd_make_float2(nd.param_x, nd.param_y);
                simd::float2 d = simd_abs(p2d - c) - h;
                stack[sp++] = simd_length(simd_max(d, simd_make_float2(0.0f)))
                    + std::min(std::max(d.x, d.y), 0.0f);
                break;
            }
            case SDF_TYPE_BEZIER2D: {
                simd::float2 p2d = simd_make_float2(std::sqrt(pos.x*pos.x+pos.z*pos.z), pos.y);
                simd::float2 A = simd_make_float2(nd.pos_x, nd.pos_y);
                simd::float2 B = simd_make_float2(nd.param_x, nd.param_y);
                simd::float2 C = simd_make_float2(nd.param_z, nd.param_w);
                stack[sp++] = SDFMath::distanceToQuadraticBezier(p2d, A, B, C) - nd.pos_z;
                break;
            }
            case SDF_TYPE_CUBIC_BEZIER2D: {
                simd::float2 p2d = simd_make_float2(std::sqrt(pos.x*pos.x+pos.z*pos.z), pos.y);
                const SDFNodeGPU& ex = _nodes[i+1];
                simd::float2 P0 = simd_make_float2(nd.pos_x, nd.pos_y);
                simd::float2 P1 = simd_make_float2(nd.param_x, nd.param_y);
                simd::float2 P2 = simd_make_float2(nd.param_z, nd.param_w);
                simd::float2 P3 = simd_make_float2(ex.pos_x, ex.pos_y);
                stack[sp++] = SDFMath::distanceToCubicBezier(p2d, P0, P1, P2, P3) - nd.pos_z;
                i++;
                break;
            }
            case SDF_TYPE_COMPOSITE_SPLINE2D: {
                simd::float2 p2d = simd_make_float2(std::sqrt(pos.x*pos.x+pos.z*pos.z), pos.y);
                stack[sp++] = evalCompositeSpline2D(p2d, i);
                int N = (int)nd.param_x;
                i += (N + 2) / 3;
                break;
            }

            // ── CSG ──

            case SDF_OP_UNION: {
                float d2 = stack[--sp], d1 = stack[--sp];
                stack[sp++] = std::min(d1, d2); break;
            }
            case SDF_OP_SUBTRACT: {
                float d2 = stack[--sp], d1 = stack[--sp];
                stack[sp++] = std::max(d1, -d2); break;
            }
            case SDF_OP_INTERSECT: {
                float d2 = stack[--sp], d1 = stack[--sp];
                stack[sp++] = std::max(d1, d2); break;
            }
            case SDF_OP_SMOOTH_UNION: {
                float d2 = stack[--sp], d1 = stack[--sp];
                float k = nd.smoothFactor;
                if (k < 1e-10f) { stack[sp++] = std::min(d1, d2); }
                else {
                    float h = std::clamp(0.5f + 0.5f*(d2-d1)/k, 0.0f, 1.0f);
                    stack[sp++] = d2*(1-h) + d1*h - k*h*(1-h);
                }
                break;
            }
            case SDF_OP_TRANSFORM: {
                // Post-order limitation: child already evaluated with untransformed point
                // Only scale correction applies correctly in this model
                float scaleFactor = nd.param_x;
                float childDist = stack[--sp];
                stack[sp++] = childDist * scaleFactor;
                i += 2; // skip DATA_CARRIERs
                break;
            }
            default: break;
            }
        }
        return (sp >= 1) ? stack[0] : 1e10f;
    }

    void evaluateBatch(const simd::float3* positions, float* results, size_t count) const {
        for (size_t i = 0; i < count; ++i) results[i] = evaluate(positions[i]);
    }

    simd::float3 boundsMin() const { return _boundsMin; }
    simd::float3 boundsMax() const { return _boundsMax; }

private:
    std::vector<SDFNodeGPU> _nodes;
    simd::float3 _boundsMin = {-1,-1,-1}, _boundsMax = {1,1,1};

    float evalCompositeSpline2D(simd::float2 p2d, size_t headerIdx) const {
        const SDFNodeGPU& hdr = _nodes[headerIdx];
        int N = std::min((int)hdr.param_x, 64);
        float thickness = hdr.param_y;
        simd::float2 pts[64]; int ptIdx = 0;
        size_t ci = headerIdx + 1;
        while (ptIdx < N && ci < _nodes.size() && _nodes[ci].type == SDF_DATA_CARRIER) {
            const SDFNodeGPU& dc = _nodes[ci];
            if (ptIdx < N) pts[ptIdx++] = simd_make_float2(dc.pos_x, dc.pos_y);
            if (ptIdx < N) pts[ptIdx++] = simd_make_float2(dc.param_x, dc.param_y);
            if (ptIdx < N) pts[ptIdx++] = simd_make_float2(dc.param_z, dc.param_w);
            ci++;
        }
        N = ptIdx;
        return SDFMath::compositeSplineDistance(p2d, pts, N, thickness);
    }

    void computeBounds() {
        float margin = 0.01f; bool has = false;
        simd::float3 bMin = {1e10f,1e10f,1e10f}, bMax = {-1e10f,-1e10f,-1e10f};
        for (size_t i = 0; i < _nodes.size(); ++i) {
            const SDFNodeGPU& nd = _nodes[i];
            simd::float3 lo, hi; bool valid = false;
            switch (nd.type) {
            case SDF_TYPE_SPHERE: {
                simd::float3 c = simd_make_float3(nd.pos_x,nd.pos_y,nd.pos_z);
                float r = nd.param_x;
                lo = c - simd_make_float3(r); hi = c + simd_make_float3(r); valid=true; break;
            }
            case SDF_TYPE_BOX: {
                simd::float3 c = simd_make_float3(nd.pos_x,nd.pos_y,nd.pos_z);
                simd::float3 h = simd_make_float3(nd.param_x,nd.param_y,nd.param_z);
                lo = c - h; hi = c + h; valid=true; break;
            }
            case SDF_TYPE_CYLINDER: {
                simd::float3 c = simd_make_float3(nd.pos_x,nd.pos_y,nd.pos_z);
                float r=nd.param_x, hh=nd.param_y;
                lo = c - simd_make_float3(r,hh,r); hi = c + simd_make_float3(r,hh,r); valid=true; break;
            }
            case SDF_TYPE_TORUS: {
                simd::float3 c = simd_make_float3(nd.pos_x,nd.pos_y,nd.pos_z);
                float R=nd.param_x+nd.param_y, h=nd.param_y;
                lo = c - simd_make_float3(R,h,R); hi = c + simd_make_float3(R,h,R); valid=true; break;
            }
            case SDF_TYPE_COMPOSITE_SPLINE2D: {
                int N = std::min((int)nd.param_x, 64);
                float rMax=0, yMin=1e10f, yMax=-1e10f; int ptIdx=0;
                size_t ci = i + 1;
                while (ptIdx<N && ci<_nodes.size() && _nodes[ci].type==SDF_DATA_CARRIER) {
                    const SDFNodeGPU& dc = _nodes[ci];
                    auto addPt = [&](float r, float y) { rMax=std::max(rMax,r); yMin=std::min(yMin,y); yMax=std::max(yMax,y); };
                    if (ptIdx<N) { addPt(dc.pos_x, dc.pos_y); ptIdx++; }
                    if (ptIdx<N) { addPt(dc.param_x, dc.param_y); ptIdx++; }
                    if (ptIdx<N) { addPt(dc.param_z, dc.param_w); ptIdx++; }
                    ci++;
                }
                lo = simd_make_float3(-rMax,yMin,-rMax);
                hi = simd_make_float3(rMax,yMax,rMax); valid=true; break;
            }
            default: break;
            }
            if (valid) { has=true; bMin=simd_min(bMin,lo); bMax=simd_max(bMax,hi); }
        }
        if (has) { simd::float3 m=simd_make_float3(margin); _boundsMin=bMin-m; _boundsMax=bMax+m; }
    }
};
