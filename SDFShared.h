#ifndef SDFShared_h
#define SDFShared_h

#ifndef __METAL_VERSION__
    #include <cstddef>
#endif

enum SDFNodeType {
    SDF_DATA_CARRIER            = -1,
    SDF_TYPE_SPHERE             = 0,
    SDF_TYPE_BOX                = 1,
    SDF_TYPE_CYLINDER           = 2,
    SDF_TYPE_TORUS              = 3,
    SDF_TYPE_CAPSULE            = 4,
    SDF_TYPE_CIRCLE_2D          = 10,
    SDF_TYPE_RECT_2D            = 11,
    SDF_TYPE_BEZIER2D           = 12,
    SDF_TYPE_CUBIC_BEZIER2D     = 13,
    SDF_TYPE_COMPOSITE_SPLINE2D = 14,
    SDF_OP_UNION                = 100,
    SDF_OP_SUBTRACT             = 101,
    SDF_OP_INTERSECT            = 102,
    SDF_OP_SMOOTH_UNION         = 103,
    SDF_OP_TRANSFORM            = 200
};

// 64 bytes = 16 x 4-byte fields
struct SDFNodeGPU {
    int   type;                  //  0
    int   leftChildIndex;        //  4
    int   rightChildIndex;       //  8
    int   _pad0;                 // 12
    float pos_x, pos_y, pos_z;  // 16, 20, 24
    float smoothFactor;          // 28
    float param_x, param_y;     // 32, 36
    float param_z, param_w;     // 40, 44
    float _r0, _r1, _r2, _r3;  // 48, 52, 56, 60
};

// 64 bytes
struct Uniforms {
    float camPosX,   camPosY,   camPosZ;    float _p1;
    float camFwdX,   camFwdY,   camFwdZ;    float _p2;
    float camRightX, camRightY, camRightZ;  float _p3;
    float camUpX,    camUpY,    camUpZ;     float _p4;
};

// 96 bytes
struct RenderParams {
    int   maxSteps;
    float maxDistance;
    float minHitEps;
    float relativeHitEps;
    float stepSafetyFactor;
    float minNormalEps;
    float relativeNormalEps;
    float ambient;
    float diffuseStrength;
    float specularStrength;
    float specularPower;
    float _rp0;
    float baseColorR, baseColorG, baseColorB; float _rp1;
    float bgBottomR,  bgBottomG,  bgBottomB;  float _rp2;
    float bgTopR,     bgTopG,     bgTopB;     float _rp3;
};

#ifndef __METAL_VERSION__
static_assert(sizeof(SDFNodeGPU)  == 64, "SDFNodeGPU must be 64 bytes");
static_assert(sizeof(Uniforms)    == 64, "Uniforms must be 64 bytes");
static_assert(sizeof(RenderParams)== 96, "RenderParams must be 96 bytes");
static_assert(offsetof(SDFNodeGPU, pos_x)       == 16, "pos_x at 16");
static_assert(offsetof(SDFNodeGPU, smoothFactor) == 28, "smoothFactor at 28");
static_assert(offsetof(SDFNodeGPU, param_x)     == 32, "param_x at 32");
#endif

#endif
