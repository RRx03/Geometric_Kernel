#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "Renderer.hpp"
#include "../metal-cpp/Metal/Metal.hpp"
#include "../metal-cpp/QuartzCore/QuartzCore.hpp"
#include "MetalBridge.h"
#include <SDL.h>
#include <cstring>
#include <dispatch/dispatch.h>
#include <iostream>
#include <simd/simd.h>
#include <stdexcept>

#define DEV ((MTL::Device *)_device)
#define QUEUE ((MTL::CommandQueue *)_cmdQueue)
#define LAYER ((CA::MetalLayer *)_layer)
#define PSO ((MTL::RenderPipelineState *)_pso)
#define DS ((MTL::DepthStencilState *)_depthState)
#define UBUF ((MTL::Buffer *)_uniformBuf)
#define SBUF ((MTL::Buffer *)_sdfBuf)
#define CBUF ((MTL::Buffer *)_countBuf)
#define RBUF ((MTL::Buffer *)_rpBuf)
#define MSAA ((MTL::Texture *)_msaaTex)
#define DEPTH ((MTL::Texture *)_depthTex)
#define SEMA ((dispatch_semaphore_t)_frameSema)

Renderer::Renderer(SDL_Window *window, const RenderConfig &config)
    : _samples(config.msaaSamples), _cfg(config) {
  camera.distance = config.camDistance;
  camera.azimuth = config.camAzimuth;
  camera.elevation = config.camElevation;
  camera.target = simd_make_float3(config.camTarget[0], config.camTarget[1],
                                   config.camTarget[2]);
  _device = MTL::CreateSystemDefaultDevice();
  if (!_device)
    throw std::runtime_error("No Metal device");
  _cmdQueue = DEV->newCommandQueue();
  _layer = MetalBridge_AttachLayer(window, _device);
  if (!_layer)
    throw std::runtime_error("No Metal layer");
  _frameSema = (void *)dispatch_semaphore_create(1);
  buildShaders();
  buildBuffers();
}
Renderer::~Renderer() {
  if (RBUF)
    RBUF->release();
  if (CBUF)
    CBUF->release();
  if (SBUF)
    SBUF->release();
  if (UBUF)
    UBUF->release();
  if (MSAA)
    MSAA->release();
  if (DEPTH)
    DEPTH->release();
  if (DS)
    DS->release();
  if (PSO)
    PSO->release();
  if (QUEUE)
    QUEUE->release();
}
void Renderer::buildShaders() {
  NS::Error *err = nullptr;
  MTL::Library *lib = DEV->newDefaultLibrary();
  if (!lib) {
    auto p =
        NS::String::string("build/default.metallib", NS::UTF8StringEncoding);
    lib = DEV->newLibrary(p, &err);
  }
  if (!lib) {
    auto p = NS::String::string("default.metallib", NS::UTF8StringEncoding);
    lib = DEV->newLibrary(p, &err);
  }
  if (!lib) {
    std::cerr << "Metal lib error\n";
    throw std::runtime_error("No Metal lib");
  }
  auto vFn = lib->newFunction(
      NS::String::string("vertex_main", NS::UTF8StringEncoding));
  auto fFn = lib->newFunction(
      NS::String::string("fragment_main", NS::UTF8StringEncoding));
  if (!vFn || !fFn)
    throw std::runtime_error("Shader funcs not found");
  auto ds = MTL::DepthStencilDescriptor::alloc()->init();
  ds->setDepthCompareFunction(MTL::CompareFunctionLess);
  ds->setDepthWriteEnabled(true);
  _depthState = DEV->newDepthStencilState(ds);
  ds->release();
  auto pd = MTL::RenderPipelineDescriptor::alloc()->init();
  pd->setVertexFunction(vFn);
  pd->setFragmentFunction(fFn);
  pd->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
  pd->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);
  pd->setRasterSampleCount(_samples);
  _pso = DEV->newRenderPipelineState(pd, &err);
  if (!_pso) {
    std::cerr << "PSO error\n";
    throw std::runtime_error("PSO fail");
  }
  vFn->release();
  fFn->release();
  pd->release();
  lib->release();
}
void Renderer::buildBuffers() {
  _uniformBuf =
      DEV->newBuffer(sizeof(Uniforms), MTL::ResourceStorageModeShared);
  int z = 0;
  _countBuf = DEV->newBuffer(&z, sizeof(int), MTL::ResourceStorageModeShared);
  RenderParams rp = _cfg.toGPUParams();
  _rpBuf =
      DEV->newBuffer(&rp, sizeof(RenderParams), MTL::ResourceStorageModeShared);
}
void Renderer::loadGeometry(const std::vector<SDFNodeGPU> &nodes) {
  _sdfCount = (int)nodes.size();
  if (SBUF) {
    SBUF->release();
    _sdfBuf = nullptr;
  }
  if (_sdfCount > 0)
    _sdfBuf = DEV->newBuffer(nodes.data(), nodes.size() * sizeof(SDFNodeGPU),
                             MTL::ResourceStorageModeShared);
  memcpy(CBUF->contents(), &_sdfCount, sizeof(int));
}
void Renderer::updateRenderParams(const RenderConfig &config) {
  _cfg = config;
  RenderParams rp = config.toGPUParams();
  memcpy(RBUF->contents(), &rp, sizeof(RenderParams));
}
void Renderer::resize(int w, int h) {
  if (w < 1 || h < 1)
    return;
  _width = w;
  _height = h;
  LAYER->setDrawableSize(CGSizeMake(w, h));
  if (MSAA)
    MSAA->release();
  auto md = MTL::TextureDescriptor::alloc()->init();
  md->setTextureType(MTL::TextureType2DMultisample);
  md->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
  md->setWidth(w);
  md->setHeight(h);
  md->setSampleCount(_samples);
  md->setUsage(MTL::TextureUsageRenderTarget);
  md->setStorageMode(MTL::StorageModePrivate);
  _msaaTex = DEV->newTexture(md);
  md->release();
  if (DEPTH)
    DEPTH->release();
  auto dd = MTL::TextureDescriptor::alloc()->init();
  dd->setTextureType(MTL::TextureType2DMultisample);
  dd->setPixelFormat(MTL::PixelFormatDepth32Float);
  dd->setWidth(w);
  dd->setHeight(h);
  dd->setSampleCount(_samples);
  dd->setUsage(MTL::TextureUsageRenderTarget);
  dd->setStorageMode(MTL::StorageModePrivate);
  _depthTex = DEV->newTexture(dd);
  dd->release();
}
void Renderer::renderFrame(float dt) {
  dispatch_semaphore_wait(SEMA, DISPATCH_TIME_FOREVER);
  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
  camera.update(dt);
  Uniforms u = camera.computeUniforms();
  u.aspectRatio = (_height > 0) ? (float)_width / (float)_height : 1.0f;
  memcpy(UBUF->contents(), &u, sizeof(Uniforms));
  CA::MetalDrawable *drawable = LAYER->nextDrawable();
  if (!drawable) {
    dispatch_semaphore_signal(SEMA);
    pool->release();
    return;
  }
  auto cb = QUEUE->commandBuffer();
  dispatch_semaphore_t sem = SEMA;
  cb->addCompletedHandler(^(MTL::CommandBuffer *) {
    dispatch_semaphore_signal(sem);
  });
  auto rpd = MTL::RenderPassDescriptor::alloc()->init();
  rpd->colorAttachments()->object(0)->setTexture(MSAA);
  rpd->colorAttachments()->object(0)->setResolveTexture(drawable->texture());
  rpd->colorAttachments()->object(0)->setLoadAction(MTL::LoadActionClear);
  rpd->colorAttachments()->object(0)->setStoreAction(
      MTL::StoreActionMultisampleResolve);
  rpd->colorAttachments()->object(0)->setClearColor(
      MTL::ClearColor(0.15, 0.15, 0.15, 1));
  rpd->depthAttachment()->setTexture(DEPTH);
  rpd->depthAttachment()->setLoadAction(MTL::LoadActionClear);
  rpd->depthAttachment()->setStoreAction(MTL::StoreActionDontCare);
  rpd->depthAttachment()->setClearDepth(1.0);
  auto enc = cb->renderCommandEncoder(rpd);
  enc->setRenderPipelineState(PSO);
  enc->setDepthStencilState(DS);
  enc->setFragmentBuffer(UBUF, 0, 1);
  if (SBUF)
    enc->setFragmentBuffer(SBUF, 0, 2);
  enc->setFragmentBuffer(CBUF, 0, 3);
  enc->setFragmentBuffer(RBUF, 0, 4);
  enc->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0),
                      NS::UInteger(4));
  enc->endEncoding();
  cb->presentDrawable(drawable);
  cb->commit();
  rpd->release();
  pool->release();
}