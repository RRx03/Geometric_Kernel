#include <metal_stdlib>
using namespace metal;

#include "../SDFShared.h"

constant int C_DC=-1, C_SPHERE=0, C_BOX=1, C_CYL=2, C_TORUS=3, C_CAPSULE=4;
constant int C_CIRCLE2D=10, C_RECT2D=11, C_BEZ2D=12, C_CBEZ2D=13, C_CSPLINE2D=14;
constant int C_UNION=100, C_SUB=101, C_INTER=102, C_SMOOTH=103, C_XFORM=200;

struct RasterizerData { float4 position [[position]]; float2 uv; };

inline float3 ndPos(SDFNodeGPU n) { return float3(n.pos_x, n.pos_y, n.pos_z); }
inline float2 ndPos2(SDFNodeGPU n) { return float2(n.pos_x, n.pos_y); }
inline float4 ndPar(SDFNodeGPU n) { return float4(n.param_x, n.param_y, n.param_z, n.param_w); }

// ─────────────────────────────────────────────────────────
// Bézier — lightweight ternary search (6 coarse + 10 refine)
// ─────────────────────────────────────────────────────────
float2 evalQ(float2 A, float2 B, float2 C, float t) {
    float u = 1.0-t;
    return u*u*A + 2.0*u*t*B + t*t*C;
}

// Returns distance AND the r-coordinate of the closest curve point (via rCurve)
float distQBez(float2 pos, float2 A, float2 B, float2 C, thread float& rCurve) {
    float bestT = 0.0, bestD = 1e10;
    for (int j = 0; j <= 6; j++) {
        float t = float(j) / 6.0;
        float d = length(pos - evalQ(A, B, C, t));
        if (d < bestD) { bestD = d; bestT = t; }
    }
    float lo = max(0.0, bestT - 1.0/6.0);
    float hi = min(1.0, bestT + 1.0/6.0);
    for (int i = 0; i < 10; i++) {
        float m1 = lo + (hi-lo) * 0.333333;
        float m2 = hi - (hi-lo) * 0.333333;
        if (length(pos - evalQ(A, B, C, m1)) < length(pos - evalQ(A, B, C, m2)))
            hi = m2;
        else
            lo = m1;
    }
    bestT = (lo+hi) * 0.5;
    float2 cp = evalQ(A, B, C, bestT);
    rCurve = cp.x;
    return length(pos - cp);
}

// Overload without rCurve output
float distQBezSimple(float2 pos, float2 A, float2 B, float2 C) {
    float r; return distQBez(pos, A, B, C, r);
}

float2 evalCub(float2 P0, float2 P1, float2 P2, float2 P3, float t) {
    float u = 1.0-t;
    return u*u*u*P0 + 3.0*u*u*t*P1 + 3.0*u*t*t*P2 + t*t*t*P3;
}

float distCBez(float2 pos, float2 P0, float2 P1, float2 P2, float2 P3) {
    float bestT = 0.0, bestD = 1e10;
    for (int j = 0; j <= 8; j++) {
        float t = float(j) / 8.0;
        float d = length(pos - evalCub(P0, P1, P2, P3, t));
        if (d < bestD) { bestD = d; bestT = t; }
    }
    float lo = max(0.0, bestT - 0.125);
    float hi = min(1.0, bestT + 0.125);
    for (int i = 0; i < 10; i++) {
        float m1 = lo + (hi-lo)*0.333333;
        float m2 = hi - (hi-lo)*0.333333;
        if (length(pos - evalCub(P0,P1,P2,P3,m1)) < length(pos - evalCub(P0,P1,P2,P3,m2)))
            hi = m2;
        else
            lo = m1;
    }
    return length(pos - evalCub(P0, P1, P2, P3, (lo+hi)*0.5));
}

// ─────────────────────────────────────────────────────────
// CompositeSpline2D — GPU version
//
// Sign method: r_point < r_curve (fast, O(1) per eval)
// This works because axisymmetric profiles are monotonic
// in r for a given y-slice near the closest point.
// The sign is: inside if point is between axis and curve.
// Y-clamping: outside the y-range → always exterior.
// ─────────────────────────────────────────────────────────
float sdCSpline(float2 p2d, constant SDFNodeGPU* nd, int hIdx, int N, float th) {
    float2 pts[64]; int pi = 0;
    int ci = hIdx + 1, mc = (N+2)/3;
    for (int c = 0; c < mc && pi < N; c++) {
        SDFNodeGPU dc = nd[ci+c];
        if (pi < N) pts[pi++] = float2(dc.pos_x, dc.pos_y);
        if (pi < N) pts[pi++] = float2(dc.param_x, dc.param_y);
        if (pi < N) pts[pi++] = float2(dc.param_z, dc.param_w);
    }
    N = pi; if (N < 2) return 1e10;

    float minDist = 1e10;
    float rAtClosest = 0.0;

    // Evaluate all B-spline segments, track closest point's r
    if (N == 2) {
        float rC;
        float d = distQBez(p2d, pts[0], (pts[0]+pts[1])*0.5, pts[1], rC);
        if (d < minDist) { minDist = d; rAtClosest = rC; }
    } else if (N == 3) {
        float rC;
        float d = distQBez(p2d, pts[0], pts[1], pts[2], rC);
        if (d < minDist) { minDist = d; rAtClosest = rC; }
    } else {
        {   float rC;
            float d = distQBez(p2d, pts[0], pts[1], (pts[1]+pts[2])*0.5, rC);
            if (d < minDist) { minDist = d; rAtClosest = rC; } }
        for (int s = 1; s < N-3; s++) {
            float rC;
            float d = distQBez(p2d, (pts[s]+pts[s+1])*0.5, pts[s+1], (pts[s+1]+pts[s+2])*0.5, rC);
            if (d < minDist) { minDist = d; rAtClosest = rC; } }
        {   float rC;
            float d = distQBez(p2d, (pts[N-3]+pts[N-2])*0.5, pts[N-2], pts[N-1], rC);
            if (d < minDist) { minDist = d; rAtClosest = rC; } }
    }

    if (th > 1e-6) return minDist - th;

    // Sign: inside if r_point < r_curve AND within y-range
    float yMin = min(pts[0].y, pts[N-1].y);
    float yMax = max(pts[0].y, pts[N-1].y);
    bool outsideY = (p2d.y < yMin - 0.0005) || (p2d.y > yMax + 0.0005);
    float sgn = (p2d.x < rAtClosest && !outsideY) ? -1.0 : 1.0;

    return sgn * minDist;
}

// ─────────────────────────────────────────────────────────
// map() — SDF stack machine
// ─────────────────────────────────────────────────────────
float map(float3 pos, constant SDFNodeGPU* nd, int nc) {
    float stk[64]; int sp = 0;
    for (int i = 0; i < nc; i++) {
        SDFNodeGPU n = nd[i];
        if (n.type == C_DC) continue;
        float3 np = ndPos(n);
        float4 pm = ndPar(n);

        if      (n.type == C_SPHERE)   { stk[sp++] = length(pos-np) - pm.x; }
        else if (n.type == C_BOX)      { float3 d=abs(pos-np)-pm.xyz; stk[sp++]=length(max(d,0.0))+min(max(d.x,max(d.y,d.z)),0.0); }
        else if (n.type == C_CYL)      { float3 lp=pos-np; float dR=sqrt(lp.x*lp.x+lp.z*lp.z)-pm.x; float dA=abs(lp.y)-pm.y;
                                          stk[sp++]=sqrt(max(dR,0.0)*max(dR,0.0)+max(dA,0.0)*max(dA,0.0))+min(max(dR,dA),0.0); }
        else if (n.type == C_TORUS)    { float3 lp=pos-np; float qr=sqrt(lp.x*lp.x+lp.z*lp.z)-pm.x; stk[sp++]=sqrt(qr*qr+lp.y*lp.y)-pm.y; }
        else if (n.type == C_CAPSULE)  { float3 A=np,B=pm.xyz; float rad=pm.w; float3 pa=pos-A,ba=B-A; float bb=dot(ba,ba);
                                          float t=bb>1e-10?clamp(dot(pa,ba)/bb,0.0,1.0):0.0; stk[sp++]=length(pa-ba*t)-rad; }
        else if (n.type == C_CIRCLE2D) { float2 p2=float2(length(pos.xz),pos.y); stk[sp++]=length(p2-ndPos2(n))-pm.x; }
        else if (n.type == C_RECT2D)   { float2 p2=float2(length(pos.xz),pos.y); float2 d=abs(p2-ndPos2(n))-pm.xy;
                                          stk[sp++]=length(max(d,0.0))+min(max(d.x,d.y),0.0); }
        else if (n.type == C_BEZ2D)    { float2 p2=float2(length(pos.xz),pos.y); stk[sp++]=distQBezSimple(p2,ndPos2(n),pm.xy,pm.zw)-n.pos_z; }
        else if (n.type == C_CBEZ2D)   { SDFNodeGPU ex=nd[i+1]; float2 p2=float2(length(pos.xz),pos.y);
                                          stk[sp++]=distCBez(p2,ndPos2(n),pm.xy,pm.zw,float2(ex.pos_x,ex.pos_y))-n.pos_z; i++; }
        else if (n.type == C_CSPLINE2D){ float2 p2=float2(length(pos.xz),pos.y); int N=int(pm.x);
                                          stk[sp++]=sdCSpline(p2,nd,i,N,pm.y); i+=(N+2)/3; }
        else if (n.type == C_UNION)    { float d2=stk[--sp],d1=stk[--sp]; stk[sp++]=min(d1,d2); }
        else if (n.type == C_SUB)      { float d2=stk[--sp],d1=stk[--sp]; stk[sp++]=max(d1,-d2); }
        else if (n.type == C_INTER)    { float d2=stk[--sp],d1=stk[--sp]; stk[sp++]=max(d1,d2); }
        else if (n.type == C_SMOOTH)   { float d2=stk[--sp],d1=stk[--sp]; float k=n.smoothFactor;
                                          if(k<1e-10){stk[sp++]=min(d1,d2);}else{
                                          float h=clamp(0.5+0.5*(d2-d1)/k,0.0,1.0); stk[sp++]=d2*(1.0-h)+d1*h-k*h*(1.0-h);} }
        else if (n.type == C_XFORM)    { float sf=pm.x; float cd=stk[--sp]; stk[sp++]=cd*sf; i+=2; }
    }
    return sp >= 1 ? stk[0] : 1e10;
}

// ─────────────────────────────────────────────────────────
// Normal
// ─────────────────────────────────────────────────────────
float3 calcNormal(float3 p, float rt, constant SDFNodeGPU* nd, int nc, constant RenderParams& rp) {
    float eps = max(rp.minNormalEps, rt * rp.relativeNormalEps);
    float2 e = float2(eps, 0.0);
    return normalize(float3(
        map(p+e.xyy,nd,nc) - map(p-e.xyy,nd,nc),
        map(p+e.yxy,nd,nc) - map(p-e.yxy,nd,nc),
        map(p+e.yyx,nd,nc) - map(p-e.yyx,nd,nc)
    ));
}

// ─────────────────────────────────────────────────────────
// Vertex
// ─────────────────────────────────────────────────────────
vertex RasterizerData vertex_main(uint vid [[vertex_id]]) {
    RasterizerData out;
    float2 g[4] = { float2(-1,-1), float2(1,-1), float2(-1,1), float2(1,1) };
    out.position = float4(g[vid], 0.0, 1.0);
    out.uv = g[vid];
    return out;
}

// ─────────────────────────────────────────────────────────
// Fragment
// ─────────────────────────────────────────────────────────
fragment float4 fragment_main(
    RasterizerData in [[stage_in]],
    constant Uniforms& u [[buffer(1)]],
    constant SDFNodeGPU* sdf [[buffer(2)]],
    constant int& cnt [[buffer(3)]],
    constant RenderParams& rp [[buffer(4)]]
) {
    float3 ro = float3(u.camPosX, u.camPosY, u.camPosZ);
    float3 fwd = float3(u.camFwdX, u.camFwdY, u.camFwdZ);
    float3 rt = float3(u.camRightX, u.camRightY, u.camRightZ);
    float3 up = float3(u.camUpX, u.camUpY, u.camUpZ);
    float3 rd = normalize(in.uv.x * rt + in.uv.y * up + fwd);

    float bgMix = in.uv.y * 0.5 + 0.5;
    float3 bgCol = mix(float3(rp.bgBottomR, rp.bgBottomG, rp.bgBottomB),
                       float3(rp.bgTopR, rp.bgTopG, rp.bgTopB), bgMix);

    float t = 0.0;
    int steps = 0;
    for (int i = 0; i < rp.maxSteps; i++) {
        float3 p = ro + rd * t;
        float d = map(p, sdf, cnt);
        float hitEps = max(rp.minHitEps, t * rp.relativeHitEps);

        if (d < hitEps) {
            float3 n = calcNormal(p, t, sdf, cnt, rp);
            float3 ld = normalize(rt + 2.0*up + fwd);
            float3 vd = -rd;
            float3 ref = reflect(-ld, n);
            float diff = max(dot(n, ld), 0.0);
            float spec = pow(max(dot(ref, vd), 0.0), rp.specularPower);
            float ao = 1.0 - float(steps) / float(rp.maxSteps) * 0.5;
            float3 bc = float3(rp.baseColorR, rp.baseColorG, rp.baseColorB);
            float3 col = bc * (rp.ambient + rp.diffuseStrength * diff) * ao + rp.specularStrength * spec;
            return float4(col, 1.0);
        }
        if (t > rp.maxDistance) break;
        t += d * rp.stepSafetyFactor;
        steps++;
    }
    return float4(bgCol, 1.0);
}
