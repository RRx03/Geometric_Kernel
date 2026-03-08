#include <metal_stdlib>
using namespace metal;
#include "../src/Shared.h"

struct RasterizerData {
    float4 position [[position]];
    float4 color;
};

vertex RasterizerData vertex_main(
    uint vertexID [[vertex_id]],
    constant VertexData* vertices [[buffer(0)]],
    constant Uniforms& uniforms   [[buffer(1)]]
) {
    RasterizerData out;
    
    float4 pos = float4(vertices[vertexID].position, 1.0);
    
    
    
    float4 worldPos = uniforms.modelMatrix * pos;
    float4 viewPos  = uniforms.viewMatrix * worldPos;
    out.position    = uniforms.projectionMatrix * viewPos;
    
    out.color = float4(vertices[vertexID].color, 1.0);
    return out;
}

fragment float4 fragment_main(RasterizerData in [[stage_in]]) {
    return in.color;
}
kernel void compute_main(
    device float* resultBuffer [[buffer(0)]],
    uint index [[thread_position_in_grid]]
) {
    resultBuffer[index] = float(index) * 0.5;
}