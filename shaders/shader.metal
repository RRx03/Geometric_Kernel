#include <metal_stdlib>
using namespace metal;

#include "../src/Shared.h"
#include "../src/SDFShared.h"

struct RasterizerData {
    float4 position [[position]];
    float2 uv;
};


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


fragment float4 fragment_main(
    RasterizerData in [[stage_in]],
    constant Uniforms& uniforms [[buffer(1)]],
    constant SDFNodeGPU* sdfNodes [[buffer(2)]], // Notre arbre C++ arrive ici !
    constant int& sdfNodeCount [[buffer(3)]]
) {

    float3 ro = float3(0.0, 0.0, -5.0);
    float3 rd = normalize(float3(in.uv, 1.0));
    
    float t = 0.0; 
    for(int i = 0; i < 100; i++) {
        float3 p = ro + rd * t;
        float d = map(p, sdfNodes, sdfNodeCount);
        
        if(d < 0.001) {
            float col = 1.0 - (float(i) / 100.0);
            return float4(col, col * 0.2, col * 0.2, 1.0);
        }
        if(t > 100.0) break;
        t += d;
    }
    
    return float4(0.1, 0.1, 0.1, 1.0);
}