#include <metal_stdlib>
using namespace metal;

// ═══ Shared struct (must match SDFShared.h) ═══
struct SDFNodeGPU {
    int type;
    int leftChildIndex;
    int rightChildIndex;
    int _pad0;
    float3 position;
    float4 params;
    float smoothFactor;
    float _pad1;
    float _pad2;
    float _pad3;
};

struct Uniforms {
    float3 camPos;
    float3 camForward;
    float3 camUp;
    float3 camRight;
};

struct RasterizerData {
    float4 position [[position]];
    float2 uv;
};

// ═══ SDF Type constants ═══
constant int SDF_DATA_CARRIER = -1;
constant int SDF_TYPE_SPHERE = 0;
constant int SDF_TYPE_BOX = 1;
constant int SDF_TYPE_CIRCLE_2D = 3;
constant int SDF_TYPE_RECT_2D = 4;
constant int SDF_TYPE_BEZIER2D = 5;
constant int SDF_TYPE_CUBIC_BEZIER2D = 6;
constant int SDF_TYPE_COMPOSITE_SPLINE2D = 7;
constant int SDF_OP_UNION = 10;
constant int SDF_OP_SUBTRACT = 11;
constant int SDF_OP_SMOOTH_UNION = 12;
constant int SDF_OP_INTERSECT = 13;

// ═══ Quadratic Bézier unsigned distance (IQ's method) ═══
float sdBezierQuadratic(float2 pos, float2 A, float2 B, float2 C) {
    float2 a = B - A;
    float2 b = A - 2.0 * B + C;
    float2 c = a * 2.0;
    float2 d = A - pos;
    float kk = 1.0 / max(dot(b, b), 1e-10);
    float kx = kk * dot(a, b);
    float ky = kk * (2.0 * dot(a, a) + dot(d, b)) / 3.0;
    float kz = kk * dot(d, a);
    float p = ky - kx * kx;
    float p3 = p * p * p;
    float q = kx * (2.0 * kx * kx - 3.0 * ky) + kz;
    float h = q * q + 4.0 * p3;
    float res = 0.0;
    if (h >= 0.0) {
        h = sqrt(h);
        float2 x = (float2(h, -h) - float2(q, q)) / 2.0;
        float2 uv = float2(sign(x.x) * pow(abs(x.x), 1.0/3.0),
                           sign(x.y) * pow(abs(x.y), 1.0/3.0));
        float t = clamp(uv.x + uv.y - kx, 0.0, 1.0);
        res = length(d + (c + b * t) * t);
    } else {
        float z = sqrt(-p);
        float v = acos(clamp(q / (p * z * 2.0), -1.0, 1.0)) / 3.0;
        float m = cos(v);
        float n = sin(v) * 1.732050808;
        float t1 = clamp(-(m + n) * z - kx, 0.0, 1.0);
        float t2 = clamp((m - n) * z * 0.5 - kx, 0.0, 1.0);
        float d1 = length_squared(d + (c + b * t1) * t1);
        float d2 = length_squared(d + (c + b * t2) * t2);
        res = sqrt(min(d1, d2));
    }
    return res;
}

// ═══ Quadratic Bézier with r_curve output ═══
// Returns unsigned distance and writes r coordinate of closest curve point
float sdBezierQuadraticWithR(float2 pos, float2 A, float2 B, float2 C,
                              thread float &rCurve) {
    float2 a = B - A;
    float2 b = A - 2.0 * B + C;
    float2 c = a * 2.0;
    float2 d = A - pos;
    float kk = 1.0 / max(dot(b, b), 1e-10);
    float kx = kk * dot(a, b);
    float ky = kk * (2.0 * dot(a, a) + dot(d, b)) / 3.0;
    float kz = kk * dot(d, a);
    float p = ky - kx * kx;
    float p3 = p * p * p;
    float q = kx * (2.0 * kx * kx - 3.0 * ky) + kz;
    float h = q * q + 4.0 * p3;
    float res = 0.0;
    float bestT = 0.0;
    if (h >= 0.0) {
        h = sqrt(h);
        float2 x = (float2(h, -h) - float2(q, q)) / 2.0;
        float2 uv = float2(sign(x.x) * pow(abs(x.x), 1.0/3.0),
                           sign(x.y) * pow(abs(x.y), 1.0/3.0));
        bestT = clamp(uv.x + uv.y - kx, 0.0, 1.0);
        res = length(d + (c + b * bestT) * bestT);
    } else {
        float z = sqrt(-p);
        float v = acos(clamp(q / (p * z * 2.0), -1.0, 1.0)) / 3.0;
        float m = cos(v);
        float n = sin(v) * 1.732050808;
        float t1 = clamp(-(m + n) * z - kx, 0.0, 1.0);
        float t2 = clamp((m - n) * z * 0.5 - kx, 0.0, 1.0);
        float d1 = length_squared(d + (c + b * t1) * t1);
        float d2 = length_squared(d + (c + b * t2) * t2);
        if (d1 < d2) { bestT = t1; res = sqrt(d1); }
        else         { bestT = t2; res = sqrt(d2); }
    }
    float2 curvePoint = A + (c + b * bestT) * bestT;
    rCurve = curvePoint.x;
    return res;
}

// ═══ Cubic Bézier (subdivision + ternary search) ═══
float2 evalCubic(float2 p0, float2 p1, float2 p2, float2 p3, float t) {
    float u = 1.0 - t;
    return u*u*u*p0 + 3.0*u*u*t*p1 + 3.0*u*t*t*p2 + t*t*t*p3;
}

float sdBezierCubic(float2 pos, float2 p0, float2 p1, float2 p2, float2 p3) {
    constexpr int N = 16;
    float minDist = 1e10;
    float2 prev = p0;
    for (int i = 1; i <= N; i++) {
        float t = float(i) / float(N);
        float2 curr = evalCubic(p0, p1, p2, p3, t);
        float2 seg = curr - prev;
        float segLen2 = dot(seg, seg);
        float proj = 0.0;
        if (segLen2 > 1e-10)
            proj = clamp(dot(pos - prev, seg) / segLen2, 0.0, 1.0);
        float2 closest = prev + proj * seg;
        minDist = min(minDist, length(pos - closest));
        prev = curr;
    }
    float lo = 0.0, hi = 1.0;
    for (int iter = 0; iter < 24; iter++) {
        float m1 = lo + (hi - lo) / 3.0;
        float m2 = hi - (hi - lo) / 3.0;
        float d1 = length(pos - evalCubic(p0, p1, p2, p3, m1));
        float d2 = length(pos - evalCubic(p0, p1, p2, p3, m2));
        if (d1 < d2) hi = m2; else lo = m1;
    }
    float tBest = (lo + hi) * 0.5;
    return min(minDist, length(pos - evalCubic(p0, p1, p2, p3, tBest)));
}

// ═══ CompositeSpline2D — Signed distance to N-point profile ═══
float sdCompositeSpline2D(float2 p2d, constant SDFNodeGPU* nodes,
                          int headerIdx, int N, float thickness) {
    // Unpack points from DATA_CARRIER nodes (3 points per carrier)
    float2 pts[64];
    int ptIdx = 0;
    int carrierIdx = headerIdx + 1;
    int maxCarriers = (N + 2) / 3;

    for (int c = 0; c < maxCarriers && ptIdx < N; c++) {
        SDFNodeGPU dc = nodes[carrierIdx + c];
        if (ptIdx < N) pts[ptIdx++] = float2(dc.position.x, dc.position.y);
        if (ptIdx < N) pts[ptIdx++] = float2(dc.params.x, dc.params.y);
        if (ptIdx < N) pts[ptIdx++] = float2(dc.params.z, dc.params.w);
    }

    if (ptIdx < 2) return 1e10;
    N = ptIdx;

    // Find minimum distance across all B-spline segments
    float minDist = 1e10;
    float rCurveAtClosest = 0.0;

    if (N == 2) {
        float2 mid = (pts[0] + pts[1]) * 0.5;
        float rC;
        minDist = sdBezierQuadraticWithR(p2d, pts[0], mid, pts[1], rC);
        rCurveAtClosest = rC;
    } else if (N == 3) {
        float rC;
        minDist = sdBezierQuadraticWithR(p2d, pts[0], pts[1], pts[2], rC);
        rCurveAtClosest = rC;
    } else {
        int numSeg = N - 2;

        // First segment
        {
            float2 A = pts[0];
            float2 B = pts[1];
            float2 C = (pts[1] + pts[2]) * 0.5;
            float rC;
            float d = sdBezierQuadraticWithR(p2d, A, B, C, rC);
            if (d < minDist) { minDist = d; rCurveAtClosest = rC; }
        }

        // Middle segments
        for (int s = 1; s < numSeg - 1; s++) {
            float2 A = (pts[s] + pts[s+1]) * 0.5;
            float2 B = pts[s+1];
            float2 C = (pts[s+1] + pts[s+2]) * 0.5;
            float rC;
            float d = sdBezierQuadraticWithR(p2d, A, B, C, rC);
            if (d < minDist) { minDist = d; rCurveAtClosest = rC; }
        }

        // Last segment
        {
            float2 A = (pts[N-3] + pts[N-2]) * 0.5;
            float2 B = pts[N-2];
            float2 C = pts[N-1];
            float rC;
            float d = sdBezierQuadraticWithR(p2d, A, B, C, rC);
            if (d < minDist) { minDist = d; rCurveAtClosest = rC; }
        }
    }

    // Determine sign: r_point < r_curve → inside (negative)
    float sgn = (p2d.x < rCurveAtClosest) ? -1.0 : 1.0;

    if (thickness > 1e-6) {
        // Tube mode
        return minDist - thickness;
    }
    // Half-plane signed mode
    return sgn * minDist;
}

// ═══ Stack Machine SDF ═══
float map(float3 pos, constant SDFNodeGPU* nodes, int nodeCount) {
    float stack[64];
    int sp = 0;
    for (int i = 0; i < nodeCount; ++i) {
        SDFNodeGPU node = nodes[i];
        if (node.type == SDF_DATA_CARRIER) continue;

        if (node.type == SDF_TYPE_SPHERE) {
            stack[sp++] = length(pos - node.position) - node.params.x;
        }
        else if (node.type == SDF_TYPE_BOX) {
            float3 d = abs(pos - node.position) - node.params.xyz;
            stack[sp++] = length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0);
        }
        else if (node.type == SDF_TYPE_CIRCLE_2D) {
            float2 p2d = float2(length(pos.xz), pos.y);
            stack[sp++] = length(p2d - node.position.xy) - node.params.x;
        }
        else if (node.type == SDF_TYPE_RECT_2D) {
            float2 p2d = float2(length(pos.xz), pos.y);
            float2 d = abs(p2d - node.position.xy) - node.params.xy;
            stack[sp++] = length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
        }
        else if (node.type == SDF_TYPE_BEZIER2D) {
            float2 p2d = float2(length(pos.xz), pos.y);
            stack[sp++] = sdBezierQuadratic(p2d, node.position.xy, node.params.xy, node.params.zw) - node.position.z;
        }
        else if (node.type == SDF_TYPE_CUBIC_BEZIER2D) {
            SDFNodeGPU extra = nodes[i + 1];
            float2 p2d = float2(length(pos.xz), pos.y);
            stack[sp++] = sdBezierCubic(p2d, node.position.xy, node.params.xy, node.params.zw, extra.position.xy) - node.position.z;
            i++;
        }
        else if (node.type == SDF_TYPE_COMPOSITE_SPLINE2D) {
            float2 p2d = float2(length(pos.xz), pos.y);
            int N = int(node.params.x);
            float thickness = node.params.y;
            stack[sp++] = sdCompositeSpline2D(p2d, nodes, i, N, thickness);
            // Skip DATA_CARRIER nodes
            int numCarriers = (N + 2) / 3;
            i += numCarriers;
        }
        else if (node.type == SDF_OP_UNION) {
            float d2 = stack[--sp], d1 = stack[--sp];
            stack[sp++] = min(d1, d2);
        }
        else if (node.type == SDF_OP_SMOOTH_UNION) {
            float d2 = stack[--sp], d1 = stack[--sp];
            float k = node.smoothFactor;
            float h = clamp(0.5 + 0.5*(d2 - d1)/k, 0.0, 1.0);
            stack[sp++] = d2*(1.0 - h) + d1*h - k*h*(1.0 - h);
        }
        else if (node.type == SDF_OP_SUBTRACT) {
            float d2 = stack[--sp], d1 = stack[--sp];
            stack[sp++] = max(d1, -d2);
        }
        else if (node.type == SDF_OP_INTERSECT) {
            float d2 = stack[--sp], d1 = stack[--sp];
            stack[sp++] = max(d1, d2);
        }
    }
    return stack[0];
}

vertex RasterizerData vertex_main(uint vertexID [[vertex_id]]) {
    RasterizerData out;
    float2 grid[4] = { float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1) };
    out.position = float4(grid[vertexID], 0.0, 1.0);
    out.uv = grid[vertexID];
    return out;
}

float3 calcNormal(float3 p, float rayT, constant SDFNodeGPU* nodes, int nodeCount) {
    float eps = max(0.001, rayT * 0.0005);
    float2 e = float2(eps, 0.0);
    return normalize(float3(
        map(p + e.xyy, nodes, nodeCount) - map(p - e.xyy, nodes, nodeCount),
        map(p + e.yxy, nodes, nodeCount) - map(p - e.yxy, nodes, nodeCount),
        map(p + e.yyx, nodes, nodeCount) - map(p - e.yyx, nodes, nodeCount)
    ));
}

fragment float4 fragment_main(
    RasterizerData in [[stage_in]],
    constant Uniforms& uniforms [[buffer(1)]],
    constant SDFNodeGPU* sdfNodes [[buffer(2)]],
    constant int& sdfNodeCount [[buffer(3)]]
) {
    float3 ro = uniforms.camPos;
    float3 rd = normalize(in.uv.x * uniforms.camRight + in.uv.y * uniforms.camUp + uniforms.camForward);

    float t = 0.0;
    for(int i = 0; i < 128; i++) {
        float3 p = ro + rd * t;
        float d = map(p, sdfNodes, sdfNodeCount);
        float hitEps = max(0.0005 * t, 0.0001);
        if(d < hitEps) {
            float3 n = calcNormal(p, t, sdfNodes, sdfNodeCount);
            float3 lightDir = normalize(float3(1.0, 1.0, 1.0));
            float diff = max(dot(n, lightDir), 0.1);
            float ao = 1.0 - (float(i) / 128.0);
            float3 baseColor = float3(0.8, 0.8, 0.85);
            return float4(baseColor * diff * ao, 1.0);
        }
        if(t > 100.0) break;
        t += d;
    }
    return float4(0.15, 0.15, 0.15, 1.0);
}
