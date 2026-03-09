#include <metal_stdlib>
using namespace metal;

#include "../src/Shared.h"
#include "../src/SDFShared.h"

struct RasterizerData {
    float4 position [[position]];
    float2 uv;
};

// (Inigo Quilez)
float sdBezier(float2 pos, float2 A, float2 B, float2 C) {
    float2 a = B - A;
    float2 b = A - 2.0*B + C;
    float2 c = a * 2.0;
    float2 d = A - pos;

    float kk = 1.0 / dot(b,b);
    float kx = kk * dot(a,b);
    float ky = kk * (2.0*dot(a,a)+dot(d,b)) / 3.0;
    float kz = kk * dot(d,a);

    float res = 0.0;
    float p = ky - kx*kx;
    float p3 = p*p*p;
    float q = kx*(2.0*kx*kx - 3.0*ky) + kz;
    float h = q*q + 4.0*p3;

    if(h >= 0.0) { 
        h = sqrt(h);
        float2 x = (float2(h, -h) - q) / 2.0;
        float2 uv = sign(x)*pow(abs(x), float2(1.0/3.0));
        float t = clamp(uv.x+uv.y-kx, 0.0, 1.0);
        res = length(d + (c + b*t)*t);
    } else {
        float z = sqrt(-p);
        float v = acos( q/(p*z*2.0) ) / 3.0;
        float m = cos(v);
        float n = sin(v)*1.732050808;
        float3 t = clamp(float3(m+m, -n-m, n-m)*z-kx, 0.0, 1.0);
        res = min(length(d+(c+b*t.x)*t.x), length(d+(c+b*t.y)*t.y));
    }
    return res;
}
float map(float3 pos, constant SDFNodeGPU* nodes, int nodeCount) {
    float stack[32];
    int sp = 0;

    
    for (int i = 0; i < nodeCount; ++i) {
        SDFNodeGPU node = nodes[i];
        float dist = 0.0;

        if (node.type == SDF_TYPE_SPHERE) {
            dist = length(pos - node.position) - node.params.x;
            stack[sp++] = dist;
        } 
        else if (node.type == SDF_TYPE_BOX) {
            float3 d = abs(pos - node.position) - node.params.xyz;
            dist = length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0);
            stack[sp++] = dist;
        } 
        else if (node.type == SDF_OP_UNION) {
            float d2 = stack[--sp];
            float d1 = stack[--sp];
            dist = min(d1, d2);
            stack[sp++] = dist;
        }
        else if (node.type == SDF_TYPE_CIRCLE_2D) {
            float2 p2d = float2(length(pos.xz), pos.y);
            dist = length(p2d - node.position.xy) - node.params.x;
            stack[sp++] = dist;
        }
        else if (node.type == SDF_TYPE_RECT_2D) {
            float2 p2d = float2(length(pos.xz), pos.y);
            float2 d = abs(p2d - node.position.xy) - node.params.xy;
            dist = length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
            stack[sp++] = dist;
        }
        else if (node.type == SDF_TYPE_BEZIER2D) {
            float2 p2d = float2(length(pos.xz), pos.y);
            float2 p0 = node.position.xy;
            float2 p1 = node.params.xy;
            float2 p2 = node.params.zw;
            float thickness = node.position.z;
            
            dist = sdBezier(p2d, p0, p1, p2) - thickness;
            stack[sp++] = dist;
        }
        else if (node.type == SDF_OP_UNION) {
            float d2 = stack[--sp]; float d1 = stack[--sp];
            dist = min(d1, d2); stack[sp++] = dist;
        }
        else if (node.type == SDF_OP_SUBTRACT) {
            float d2 = stack[--sp]; 
            float d1 = stack[--sp]; 
            dist = max(d1, -d2);    
            stack[sp++] = dist;
        }
        else if (node.type == SDF_OP_INTERSECT) {
            float d2 = stack[--sp];
            float d1 = stack[--sp];
            dist = max(d1, d2);
            stack[sp++] = dist;
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

float3 calcNormal(float3 p, constant SDFNodeGPU* nodes, int nodeCount) {
    float2 e = float2(0.001, 0.0);
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
    
    // Ray Marching
    float t = 0.0;
    for(int i = 0; i < 100; i++) {
        float3 p = ro + rd * t;
        float d = map(p, sdfNodes, sdfNodeCount);
        
        if(d < 0.001) {
            // --- Touched Surface ---
            float3 n = calcNormal(p, sdfNodes, sdfNodeCount); // Normale
            float3 lightDir = normalize(float3(1.0, 1.0, 1.0)); // Light coming from above-right-front
            
            // Lambert Law
            float diff = max(dot(n, lightDir), 0.1); // 0.1 = ambient minimum
            
            // Simple Ambient Occlusion based on number of steps taken (more steps = more occluded)
            float ao = 1.0 - (float(i) / 100.0);
            
            // Base color (light gray) modulated by diffuse and ambient occlusion
            float3 baseColor = float3(0.8, 0.8, 0.85); 
            float3 finalColor = baseColor * diff * ao;
            
            return float4(finalColor, 1.0);
        }
        if(t > 100.0) break;
        t += d;
    }
    
    // Background color
    return float4(0.15, 0.15, 0.15, 1.0); 
}