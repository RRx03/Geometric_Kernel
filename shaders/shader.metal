#include <metal_stdlib>
using namespace metal;

struct SDFNodeGPU {
    int type; int leftChildIndex; int rightChildIndex; int _pad0;
    float3 position; float4 params;
    float smoothFactor; float _pad1; float _pad2; float _pad3;
};

struct Uniforms {
    float3 camPos; float padding1;
    float3 camForward; float padding2;
    float3 camRight; float padding3;
    float3 camUp; float padding4;
};

struct RasterizerData { float4 position [[position]]; float2 uv; };

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

float2 evalQuad(float2 A, float2 B, float2 C, float t) {
    float u = 1.0 - t;
    return u*u*A + 2.0*u*t*B + t*t*C;
}

// Optimized: 6 subdivisions + 12 ternary iterations (was 8+20)
float sdBezierQuadraticWithR(float2 pos, float2 A, float2 B, float2 C, thread float &rCurve) {
    float bestT = 0.0, bestDist = 1e10;
    for (int j = 0; j <= 6; j++) {
        float t = float(j) / 6.0;
        float d = length(pos - evalQuad(A, B, C, t));
        if (d < bestDist) { bestDist = d; bestT = t; }
    }
    float lo = max(0.0, bestT - 0.17), hi = min(1.0, bestT + 0.17);
    for (int iter = 0; iter < 12; iter++) {
        float m1 = lo + (hi-lo)/3.0, m2 = hi - (hi-lo)/3.0;
        if (length(pos - evalQuad(A, B, C, m1)) < length(pos - evalQuad(A, B, C, m2))) hi = m2; else lo = m1;
    }
    bestT = (lo+hi)*0.5;
    float2 cp = evalQuad(A, B, C, bestT);
    rCurve = cp.x;
    return length(pos - cp);
}

float sdBezierQuadratic(float2 pos, float2 A, float2 B, float2 C) {
    float rC; return sdBezierQuadraticWithR(pos, A, B, C, rC);
}

float2 evalCubic(float2 p0, float2 p1, float2 p2, float2 p3, float t) {
    float u = 1.0-t;
    return u*u*u*p0 + 3.0*u*u*t*p1 + 3.0*u*t*t*p2 + t*t*t*p3;
}

float sdBezierCubic(float2 pos, float2 p0, float2 p1, float2 p2, float2 p3) {
    float minDist = 1e10; float2 prev = p0;
    for (int i = 1; i <= 12; i++) {
        float t = float(i)/12.0;
        float2 curr = evalCubic(p0,p1,p2,p3,t);
        float2 seg = curr-prev; float sl2 = dot(seg,seg);
        float proj = (sl2>1e-10) ? clamp(dot(pos-prev,seg)/sl2, 0.0, 1.0) : 0.0;
        minDist = min(minDist, length(pos-(prev+proj*seg)));
        prev = curr;
    }
    float lo=0.0, hi=1.0;
    for (int iter=0; iter<16; iter++) {
        float m1=lo+(hi-lo)/3.0, m2=hi-(hi-lo)/3.0;
        if (length(pos-evalCubic(p0,p1,p2,p3,m1)) < length(pos-evalCubic(p0,p1,p2,p3,m2))) hi=m2; else lo=m1;
    }
    return min(minDist, length(pos-evalCubic(p0,p1,p2,p3,(lo+hi)*0.5)));
}

float sdCompositeSpline2D(float2 p2d, constant SDFNodeGPU* nodes, int headerIdx, int N, float thickness) {
    float2 pts[64]; int ptIdx = 0;
    int ci = headerIdx + 1, maxC = (N+2)/3;
    for (int c = 0; c < maxC && ptIdx < N; c++) {
        SDFNodeGPU dc = nodes[ci+c];
        if (ptIdx<N) pts[ptIdx++] = float2(dc.position.x, dc.position.y);
        if (ptIdx<N) pts[ptIdx++] = float2(dc.params.x, dc.params.y);
        if (ptIdx<N) pts[ptIdx++] = float2(dc.params.z, dc.params.w);
    }
    N = ptIdx; if (N<2) return 1e10;

    float minDist = 1e10, rAt = 0.0;

    if (N == 2) {
        float rC; float d = sdBezierQuadraticWithR(p2d, pts[0], (pts[0]+pts[1])*0.5, pts[1], rC);
        if (d<minDist) { minDist=d; rAt=rC; }
    } else if (N == 3) {
        float rC; float d = sdBezierQuadraticWithR(p2d, pts[0], pts[1], pts[2], rC);
        if (d<minDist) { minDist=d; rAt=rC; }
    } else {
        { float rC; float d=sdBezierQuadraticWithR(p2d, pts[0], pts[1], (pts[1]+pts[2])*0.5, rC);
          if(d<minDist){minDist=d;rAt=rC;} }
        for (int s=1; s<N-3; s++) {
            float rC; float d=sdBezierQuadraticWithR(p2d, (pts[s]+pts[s+1])*0.5, pts[s+1], (pts[s+1]+pts[s+2])*0.5, rC);
            if(d<minDist){minDist=d;rAt=rC;}
        }
        { float rC; float d=sdBezierQuadraticWithR(p2d, (pts[N-3]+pts[N-2])*0.5, pts[N-2], pts[N-1], rC);
          if(d<minDist){minDist=d;rAt=rC;} }
    }

    // Sign: inside if r_point < r_curve (between axis and wall)
    // Also clamp: if point is outside the Y-range of the profile,
    // it's definitely outside (positive distance)
    float yMin = pts[0].y, yMax = pts[N-1].y;
    if (yMin > yMax) { float tmp=yMin; yMin=yMax; yMax=tmp; }
    bool outsideY = (p2d.y < yMin - 0.001) || (p2d.y > yMax + 0.001);

    float sgn = (p2d.x < rAt && !outsideY) ? -1.0 : 1.0;
    if (thickness > 1e-6) return minDist - thickness;
    return sgn * minDist;
}

float map(float3 pos, constant SDFNodeGPU* nodes, int nodeCount) {
    float stack[64]; int sp = 0;
    for (int i = 0; i < nodeCount; ++i) {
        SDFNodeGPU node = nodes[i];
        if (node.type == SDF_DATA_CARRIER) continue;

        if (node.type == SDF_TYPE_SPHERE) { stack[sp++] = length(pos-node.position)-node.params.x; }
        else if (node.type == SDF_TYPE_BOX) {
            float3 d = abs(pos-node.position)-node.params.xyz;
            stack[sp++] = length(max(d,0.0))+min(max(d.x,max(d.y,d.z)),0.0);
        }
        else if (node.type == SDF_TYPE_CIRCLE_2D) {
            float2 p2d = float2(length(pos.xz), pos.y);
            stack[sp++] = length(p2d-node.position.xy)-node.params.x;
        }
        else if (node.type == SDF_TYPE_RECT_2D) {
            float2 p2d = float2(length(pos.xz), pos.y);
            float2 d = abs(p2d-node.position.xy)-node.params.xy;
            stack[sp++] = length(max(d,0.0))+min(max(d.x,d.y),0.0);
        }
        else if (node.type == SDF_TYPE_BEZIER2D) {
            float2 p2d = float2(length(pos.xz), pos.y);
            stack[sp++] = sdBezierQuadratic(p2d, node.position.xy, node.params.xy, node.params.zw)-node.position.z;
        }
        else if (node.type == SDF_TYPE_CUBIC_BEZIER2D) {
            SDFNodeGPU extra = nodes[i+1];
            float2 p2d = float2(length(pos.xz), pos.y);
            stack[sp++] = sdBezierCubic(p2d, node.position.xy, node.params.xy, node.params.zw, extra.position.xy)-node.position.z;
            i++;
        }
        else if (node.type == SDF_TYPE_COMPOSITE_SPLINE2D) {
            float2 p2d = float2(length(pos.xz), pos.y);
            int N = int(node.params.x); float thickness = node.params.y;
            stack[sp++] = sdCompositeSpline2D(p2d, nodes, i, N, thickness);
            i += (N+2)/3;
        }
        else if (node.type == SDF_OP_UNION) { float d2=stack[--sp],d1=stack[--sp]; stack[sp++]=min(d1,d2); }
        else if (node.type == SDF_OP_SMOOTH_UNION) {
            float d2=stack[--sp],d1=stack[--sp]; float k=node.smoothFactor;
            float h=clamp(0.5+0.5*(d2-d1)/k,0.0,1.0);
            stack[sp++]=d2*(1.0-h)+d1*h-k*h*(1.0-h);
        }
        else if (node.type == SDF_OP_SUBTRACT) { float d2=stack[--sp],d1=stack[--sp]; stack[sp++]=max(d1,-d2); }
        else if (node.type == SDF_OP_INTERSECT) { float d2=stack[--sp],d1=stack[--sp]; stack[sp++]=max(d1,d2); }
    }
    return (sp>=1) ? stack[0] : 1e10;
}

vertex RasterizerData vertex_main(uint vertexID [[vertex_id]]) {
    RasterizerData out;
    float2 grid[4] = { float2(-1,-1), float2(1,-1), float2(-1,1), float2(1,1) };
    out.position = float4(grid[vertexID], 0.0, 1.0);
    out.uv = grid[vertexID];
    return out;
}

float3 calcNormal(float3 p, float rayT, constant SDFNodeGPU* nodes, int nodeCount) {
    float eps = max(0.0005, rayT * 0.0003);
    float2 e = float2(eps, 0.0);
    return normalize(float3(
        map(p+e.xyy, nodes, nodeCount)-map(p-e.xyy, nodes, nodeCount),
        map(p+e.yxy, nodes, nodeCount)-map(p-e.yxy, nodes, nodeCount),
        map(p+e.yyx, nodes, nodeCount)-map(p-e.yyx, nodes, nodeCount)
    ));
}

fragment float4 fragment_main(
    RasterizerData in [[stage_in]],
    constant Uniforms& uniforms [[buffer(1)]],
    constant SDFNodeGPU* sdfNodes [[buffer(2)]],
    constant int& sdfNodeCount [[buffer(3)]]
) {
    float3 ro = uniforms.camPos;
    float3 rd = normalize(in.uv.x*uniforms.camRight + in.uv.y*uniforms.camUp + uniforms.camForward);

    float t = 0.0;
    for(int i = 0; i < 200; i++) {
        float3 p = ro + rd*t;
        float d = map(p, sdfNodes, sdfNodeCount);
        float hitEps = max(0.0002*t, 0.00005);
        if(d < hitEps) {
            float3 n = calcNormal(p, t, sdfNodes, sdfNodeCount);
            float3 lightDir = normalize(float3(1,1,1));
            float diff = max(dot(n,lightDir), 0.1);
            float ao = 1.0 - (float(i)/200.0) * 0.5;
            return float4(float3(0.8, 0.8, 0.85)*diff*ao, 1.0);
        }
        if(t > 50.0) break;
        t += d;
    }
    return float4(0.15, 0.15, 0.15, 1.0);
}
