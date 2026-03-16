#include <metal_stdlib>
using namespace metal;
#include "../SDFShared.h"

constant int C_DC=-1,C_SPHERE=0,C_BOX=1,C_CYL=2,C_TORUS=3,C_CAPSULE=4;
constant int C_CIRCLE2D=10,C_RECT2D=11,C_BEZ2D=12,C_CBEZ2D=13,C_CSPLINE2D=14;
constant int C_UNION=100,C_SUB=101,C_INTER=102,C_SMOOTH=103,C_XFORM=200;

struct RasterizerData { float4 position [[position]]; float2 uv; };

float3 ndP(SDFNodeGPU n) { return float3(n.pos_x,n.pos_y,n.pos_z); }
float2 ndP2(SDFNodeGPU n) { return float2(n.pos_x,n.pos_y); }
float4 ndQ(SDFNodeGPU n) { return float4(n.param_x,n.param_y,n.param_z,n.param_w); }

// ── Bézier quadratique ──
float2 bq(float2 A, float2 B, float2 C, float t) {
    float u=1.0-t; return u*u*A+2.0*u*t*B+t*t*C;
}

// Distance to quadratic Bézier — 4 coarse + 6 ternary = 10 evals
float dBQ(float2 p, float2 A, float2 B, float2 C) {
    float bt=0, bd=1e10;
    for (int j=0;j<=4;j++) {
        float t=float(j)*0.25;
        float d=length(p-bq(A,B,C,t));
        if(d<bd){bd=d;bt=t;}
    }
    float lo=max(0.0,bt-0.25), hi=min(1.0,bt+0.25);
    for (int i=0;i<6;i++) {
        float m1=lo+(hi-lo)/3.0, m2=hi-(hi-lo)/3.0;
        if(length(p-bq(A,B,C,m1))<length(p-bq(A,B,C,m2))) hi=m2; else lo=m1;
    }
    return length(p-bq(A,B,C,(lo+hi)*0.5));
}

// ── Bézier cubique ──
float2 bc(float2 P0,float2 P1,float2 P2,float2 P3,float t) {
    float u=1.0-t; return u*u*u*P0+3.0*u*u*t*P1+3.0*u*t*t*P2+t*t*t*P3;
}
float dBC(float2 p, float2 P0,float2 P1,float2 P2,float2 P3) {
    float bt=0,bd=1e10;
    for(int j=0;j<=6;j++){float t=float(j)/6.0;float d=length(p-bc(P0,P1,P2,P3,t));if(d<bd){bd=d;bt=t;}}
    float lo=max(0.0,bt-1.0/6.0),hi=min(1.0,bt+1.0/6.0);
    for(int i=0;i<6;i++){float m1=lo+(hi-lo)/3.0,m2=hi-(hi-lo)/3.0;
        if(length(p-bc(P0,P1,P2,P3,m1))<length(p-bc(P0,P1,P2,P3,m2)))hi=m2;else lo=m1;}
    return length(p-bc(P0,P1,P2,P3,(lo+hi)*0.5));
}

// ── CompositeSpline2D ──
// Uses ORIGINAL B-spline control points (not polyline).
// Distance: min over Bézier segments (C1 smooth → clean normals).
// Sign: ray crossing on the B-spline sampled as polygon (fast, robust).
float sdCS(float2 p, constant SDFNodeGPU* nd, int hi, int N, float th) {
    float2 pts[64]; int pi=0;
    int ci=hi+1, mc=(N+2)/3;
    for(int c=0;c<mc&&pi<N;c++){
        SDFNodeGPU dc=nd[ci+c];
        if(pi<N)pts[pi++]=float2(dc.pos_x,dc.pos_y);
        if(pi<N)pts[pi++]=float2(dc.param_x,dc.param_y);
        if(pi<N)pts[pi++]=float2(dc.param_z,dc.param_w);
    }
    N=pi; if(N<2)return 1e10;

    // ── Distance (min over B-spline Bézier segments) ──
    float md=1e10;
    if(N==2){md=min(md,dBQ(p,pts[0],(pts[0]+pts[1])*0.5,pts[1]));}
    else if(N==3){md=min(md,dBQ(p,pts[0],pts[1],pts[2]));}
    else {
        md=min(md,dBQ(p,pts[0],pts[1],(pts[1]+pts[2])*0.5));
        for(int s=1;s<N-3;s++)
            md=min(md,dBQ(p,(pts[s]+pts[s+1])*0.5,pts[s+1],(pts[s+1]+pts[s+2])*0.5));
        md=min(md,dBQ(p,(pts[N-3]+pts[N-2])*0.5,pts[N-2],pts[N-1]));
    }

    if(th>1e-6)return md-th;

    // ── Sign: ray crossing on sampled polygon ──
    // Sample each Bézier segment at 3 points → polygon of ~3*(N-2) edges
    // + 3 closure edges to axis
    // Cast ray in +x direction, count crossings
    float2 poly[128]; int pn=0;

    // Sample B-spline segments into polygon
    if(N==2){
        float2 A=pts[0],B=(pts[0]+pts[1])*0.5,C=pts[1];
        for(int j=0;j<=3;j++){float t=float(j)/3.0;
            float2 q=bq(A,B,C,t);
            if(pn==0||length(q-poly[pn-1])>1e-8)poly[pn++]=q;}
    } else if(N==3){
        for(int j=0;j<=3;j++){float t=float(j)/3.0;
            float2 q=bq(pts[0],pts[1],pts[2],t);
            if(pn==0||length(q-poly[pn-1])>1e-8)poly[pn++]=q;}
    } else {
        // First segment
        {float2 A=pts[0],B=pts[1],C=(pts[1]+pts[2])*0.5;
         for(int j=0;j<=2;j++){float2 q=bq(A,B,C,float(j)/2.0);
            if(pn==0||length(q-poly[pn-1])>1e-8)poly[pn++]=q;}}
        // Internal segments
        for(int s=1;s<N-3;s++){
            float2 A=(pts[s]+pts[s+1])*0.5,B=pts[s+1],C=(pts[s+1]+pts[s+2])*0.5;
            for(int j=1;j<=2;j++){float2 q=bq(A,B,C,float(j)/2.0);
                if(pn==0||length(q-poly[pn-1])>1e-8)poly[pn++]=q;}}
        // Last segment
        {float2 A=(pts[N-3]+pts[N-2])*0.5,B=pts[N-2],C=pts[N-1];
         for(int j=1;j<=3;j++){float2 q=bq(A,B,C,float(j)/3.0);
            if(pn==0||length(q-poly[pn-1])>1e-8)poly[pn++]=q;}}
    }

    // Add closure: last→axis, axis→axis, axis→first
    if(pn>0){
        poly[pn++]=float2(0,poly[pn-1].y);
        poly[pn++]=float2(0,poly[0].y);
        poly[pn++]=poly[0]; // close the loop
    }

    // Count ray crossings (horizontal ray in +x from p)
    int cx=0;
    for(int s=0;s<pn-1;s++){
        float2 a=poly[s],b=poly[s+1];
        if((a.y>p.y)!=(b.y>p.y)){
            float rx=a.x+(p.y-a.y)/(b.y-a.y)*(b.x-a.x);
            if(p.x<rx) cx++;
        }
    }

    return (cx%2==1)?-md:md;
}

// ── map() ──
float map(float3 pos, constant SDFNodeGPU* nd, int nc) {
    float stk[64]; int sp=0;
    for(int i=0;i<nc;i++){
        SDFNodeGPU n=nd[i]; if(n.type==C_DC)continue;
        float3 np=ndP(n); float4 pm=ndQ(n);
        if     (n.type==C_SPHERE)  {stk[sp++]=length(pos-np)-pm.x;}
        else if(n.type==C_BOX)     {float3 d=abs(pos-np)-pm.xyz;stk[sp++]=length(max(d,0.0))+min(max(d.x,max(d.y,d.z)),0.0);}
        else if(n.type==C_CYL)     {float3 lp=pos-np;float dR=length(lp.xz)-pm.x,dA=abs(lp.y)-pm.y;
                                    stk[sp++]=length(max(float2(dR,dA),0.0))+min(max(dR,dA),0.0);}
        else if(n.type==C_TORUS)   {float3 lp=pos-np;float qr=length(lp.xz)-pm.x;stk[sp++]=length(float2(qr,lp.y))-pm.y;}
        else if(n.type==C_CAPSULE) {float3 A=np,B=pm.xyz;float r=pm.w,bb=dot(B-A,B-A);
                                    float t=bb>1e-10?clamp(dot(pos-A,B-A)/bb,0.0,1.0):0.0;stk[sp++]=length(pos-A-(B-A)*t)-r;}
        else if(n.type==C_CIRCLE2D){float2 p2=float2(length(pos.xz),pos.y);stk[sp++]=length(p2-ndP2(n))-pm.x;}
        else if(n.type==C_RECT2D)  {float2 p2=float2(length(pos.xz),pos.y);float2 d=abs(p2-ndP2(n))-pm.xy;
                                    stk[sp++]=length(max(d,0.0))+min(max(d.x,d.y),0.0);}
        else if(n.type==C_BEZ2D)   {float2 p2=float2(length(pos.xz),pos.y);stk[sp++]=dBQ(p2,ndP2(n),pm.xy,pm.zw)-n.pos_z;}
        else if(n.type==C_CBEZ2D)  {SDFNodeGPU ex=nd[i+1];float2 p2=float2(length(pos.xz),pos.y);
                                    stk[sp++]=dBC(p2,ndP2(n),pm.xy,pm.zw,float2(ex.pos_x,ex.pos_y))-n.pos_z;i++;}
        else if(n.type==C_CSPLINE2D){float2 p2=float2(length(pos.xz),pos.y);int N=int(pm.x);
                                     stk[sp++]=sdCS(p2,nd,i,N,pm.y);i+=(N+2)/3;}
        else if(n.type==C_UNION)   {float d2=stk[--sp],d1=stk[--sp];stk[sp++]=min(d1,d2);}
        else if(n.type==C_SUB)     {float d2=stk[--sp],d1=stk[--sp];stk[sp++]=max(d1,-d2);}
        else if(n.type==C_INTER)   {float d2=stk[--sp],d1=stk[--sp];stk[sp++]=max(d1,d2);}
        else if(n.type==C_SMOOTH)  {float d2=stk[--sp],d1=stk[--sp];float k=n.smoothFactor;
                                    if(k<1e-10){stk[sp++]=min(d1,d2);}else{
                                    float h=clamp(0.5+0.5*(d2-d1)/k,0.0,1.0);stk[sp++]=d2*(1.0-h)+d1*h-k*h*(1.0-h);}}
        else if(n.type==C_XFORM)   {float sf=pm.x;float cd=stk[--sp];stk[sp++]=cd*sf;i+=2;}
    }
    return sp>=1?stk[0]:1e10;
}

float3 calcN(float3 p,float rt,constant SDFNodeGPU* nd,int nc,constant RenderParams& rp){
    float eps=max(rp.minNormalEps,rt*rp.relativeNormalEps);float2 e=float2(eps,0);
    return normalize(float3(map(p+e.xyy,nd,nc)-map(p-e.xyy,nd,nc),
                            map(p+e.yxy,nd,nc)-map(p-e.yxy,nd,nc),
                            map(p+e.yyx,nd,nc)-map(p-e.yyx,nd,nc)));}

vertex RasterizerData vertex_main(uint vid [[vertex_id]]){
    RasterizerData out;float2 g[4]={float2(-1,-1),float2(1,-1),float2(-1,1),float2(1,1)};
    out.position=float4(g[vid],0,1);out.uv=g[vid];return out;}

fragment float4 fragment_main(
    RasterizerData in [[stage_in]],
    constant Uniforms& u [[buffer(1)]],
    constant SDFNodeGPU* sdf [[buffer(2)]],
    constant int& cnt [[buffer(3)]],
    constant RenderParams& rp [[buffer(4)]]
){
    float3 ro=float3(u.camPosX,u.camPosY,u.camPosZ);
    float3 fwd=float3(u.camFwdX,u.camFwdY,u.camFwdZ);
    float3 rt=float3(u.camRightX,u.camRightY,u.camRightZ);
    float3 up=float3(u.camUpX,u.camUpY,u.camUpZ);
    float aspect=u.aspectRatio;
    float3 rd=normalize(in.uv.x*aspect*rt+in.uv.y*up+fwd);

    float bgMix=in.uv.y*0.5+0.5;
    float3 bgCol=mix(float3(rp.bgBottomR,rp.bgBottomG,rp.bgBottomB),
                     float3(rp.bgTopR,rp.bgTopG,rp.bgTopB),bgMix);

    float t=0;int steps=0;
    for(int i=0;i<rp.maxSteps;i++){
        float3 p=ro+rd*t;float d=map(p,sdf,cnt);
        float he=max(rp.minHitEps,t*rp.relativeHitEps);
        if(d<he){
            float3 n=calcN(p,t,sdf,cnt,rp);
            float3 ld=normalize(rt+2.0*up+fwd);
            float diff=max(dot(n,ld),0.0);
            float spec=pow(max(dot(reflect(-ld,n),-rd),0.0),rp.specularPower);
            float ao=1.0-float(steps)/float(rp.maxSteps)*0.5;
            float3 col=float3(rp.baseColorR,rp.baseColorG,rp.baseColorB)*(rp.ambient+rp.diffuseStrength*diff)*ao+rp.specularStrength*spec;
            return float4(col,1.0);
        }
        if(t>rp.maxDistance)break;
        t+=d*rp.stepSafetyFactor;steps++;
    }
    return float4(bgCol,1.0);
}
