#include "Renderer.hpp"
#include "MathUtils.h"
#include "SDFNode.hpp"
#include "SceneParser.hpp"
#include "Shared.h"
#include <iostream>
Renderer::Renderer(SDL_Window *window) {
  this->_device = MTL::CreateSystemDefaultDevice();
  if (!this->_device) {
    throw std::runtime_error("No Metal GPU found");
  }
  this->_commandQueue = this->_device->newCommandQueue();

  SDL_MetalView view = SDL_Metal_CreateView(window);
  void *layer_ptr = SDL_Metal_GetLayer(view);

  this->_layer = reinterpret_cast<CA::MetalLayer *>(layer_ptr);

  this->_layer->setDevice(_device);
  this->_layer->setPixelFormat(MTL::PixelFormatBGRA8Unorm);

  buildShaders();
  buildBuffers();
}

Renderer::~Renderer() {

  if (_depthTexture)
    _depthTexture->release();
  if (_uniformBuffer)
    _uniformBuffer->release();

  if (_vertexBuffer)
    _vertexBuffer->release();
  if (_renderPSO)
    _renderPSO->release();

  if (_depthStencilState)
    _depthStencilState->release();

  if (_commandQueue)
    _commandQueue->release();
  if (_device)
    _device->release();
  if (_msaaTexture)
    _msaaTexture->release();
}

void Renderer::buildShaders() {
  NS::Error *error = nullptr;

  MTL::Library *defaultLibrary = _device->newLibrary(
      NS::String::string("./build/default.metallib", NS::UTF8StringEncoding),
      &error);

  if (!defaultLibrary) {
    std::cerr << "Erreur chargement bibliothèque Metal: "
              << error->localizedDescription()->utf8String() << std::endl;
    return;
  }

  MTL::Function *vertexFn = defaultLibrary->newFunction(
      NS::String::string("vertex_main", NS::UTF8StringEncoding));
  MTL::Function *fragFn = defaultLibrary->newFunction(
      NS::String::string("fragment_main", NS::UTF8StringEncoding));

  MTL::DepthStencilDescriptor *depthDesc =
      MTL::DepthStencilDescriptor::alloc()->init();
  depthDesc->setDepthCompareFunction(MTL::CompareFunctionLess);
  depthDesc->setDepthWriteEnabled(true);
  _depthStencilState = _device->newDepthStencilState(depthDesc);
  depthDesc->release();

  MTL::RenderPipelineDescriptor *pipeDesc =
      MTL::RenderPipelineDescriptor::alloc()->init();
  pipeDesc->setVertexFunction(vertexFn);
  pipeDesc->setFragmentFunction(fragFn);
  pipeDesc->colorAttachments()->object(0)->setPixelFormat(
      MTL::PixelFormatBGRA8Unorm);
  pipeDesc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);
  pipeDesc->setRasterSampleCount(_sampleCount);

  _renderPSO = _device->newRenderPipelineState(pipeDesc, &error);
  if (!_renderPSO)
    std::cerr << "Erreur PSO Graphique: "
              << error->localizedDescription()->utf8String() << std::endl;

  vertexFn->release();
  fragFn->release();

  pipeDesc->release();
  defaultLibrary->release();
}
void Renderer::buildBuffers() {
  auto myPart = SceneParser::parseFile("scene.json");

  if (!myPart) {
    std::cerr << "ERREUR: Impossible de parser scene.json ou type inconnu."
              << std::endl;
    return;
  }

  std::vector<SDFNodeGPU> gpuSDFArray;
  myPart->flatten(gpuSDFArray);
  _sdfNodeCount = (int)gpuSDFArray.size();

  _sdfBuffer = _device->newBuffer(gpuSDFArray.data(),
                                  gpuSDFArray.size() * sizeof(SDFNodeGPU),
                                  MTL::ResourceStorageModeShared);

  _uniformBuffer =
      _device->newBuffer(sizeof(Uniforms), MTL::ResourceStorageModeShared);
}
void Renderer::orbit(float dx, float dy) {
  _camAzimuth -= dx * 0.01f;
  _camElevation += dy * 0.01f;
  _camElevation = std::max(-1.5f, std::min(1.5f, _camElevation));
}

void Renderer::pan(float dx, float dy) {
  float panSpeed = 0.01f;
  _camTarget.x -=
      cos(_camAzimuth) * dx * panSpeed + sin(_camAzimuth) * dy * panSpeed;
  _camTarget.y += dy * panSpeed;
  _camTarget.z -=
      sin(_camAzimuth) * dx * panSpeed - cos(_camAzimuth) * dy * panSpeed;
}

void Renderer::zoom(float dz) {
  _camDistance -= dz * 0.5f;
  _camDistance = std::max(0.5f, _camDistance);
}
void Renderer::updateUniforms() {
  Uniforms u;

  float cx =
      _camTarget.x + _camDistance * cos(_camElevation) * sin(_camAzimuth);
  float cy = _camTarget.y + _camDistance * sin(_camElevation);
  float cz =
      _camTarget.z + _camDistance * cos(_camElevation) * cos(_camAzimuth);

  u.camPos = {cx, cy, cz};

  u.camForward = simd_normalize(_camTarget - u.camPos);
  simd::float3 worldUp = {0.0f, 1.0f, 0.0f};
  u.camRight = simd_normalize(simd_cross(worldUp, u.camForward));
  u.camUp = simd_cross(u.camForward, u.camRight);

  void *ptr = _uniformBuffer->contents();
  memcpy(ptr, &u, sizeof(Uniforms));
}

void Renderer::resize(int width, int height) {
  _width = width;
  _height = height;
  _layer->setDrawableSize(CGSizeMake(width, height));

  if (_msaaTexture)
    _msaaTexture->release();
  if (_depthTexture)
    _depthTexture->release();

  MTL::TextureDescriptor *msaaDesc = MTL::TextureDescriptor::alloc()->init();
  msaaDesc->setTextureType(MTL::TextureType2DMultisample);
  msaaDesc->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
  msaaDesc->setWidth(width);
  msaaDesc->setHeight(height);
  msaaDesc->setSampleCount(_sampleCount);
  msaaDesc->setUsage(MTL::TextureUsageRenderTarget);
  msaaDesc->setStorageMode(MTL::StorageModePrivate);

  _msaaTexture = _device->newTexture(msaaDesc);
  msaaDesc->release();

  MTL::TextureDescriptor *depthDesc = MTL::TextureDescriptor::alloc()->init();
  depthDesc->setTextureType(MTL::TextureType2DMultisample);
  depthDesc->setPixelFormat(MTL::PixelFormatDepth32Float);
  depthDesc->setWidth(width);
  depthDesc->setHeight(height);
  depthDesc->setSampleCount(_sampleCount);
  depthDesc->setUsage(MTL::TextureUsageRenderTarget);
  depthDesc->setStorageMode(MTL::StorageModePrivate);

  _depthTexture = _device->newTexture(depthDesc);
  depthDesc->release();
}

void Renderer::renderFrame() {

  updateUniforms();

  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
  CA::MetalDrawable *drawable = _layer->nextDrawable();

  if (drawable) {

    MTL::CommandBuffer *buffer = _commandQueue->commandBuffer();

    MTL::RenderPassDescriptor *pass =
        MTL::RenderPassDescriptor::renderPassDescriptor();
    MTL::RenderPassColorAttachmentDescriptor *colorAttachment =
        pass->colorAttachments()->object(0);
    colorAttachment->setTexture(_msaaTexture);
    colorAttachment->setResolveTexture(drawable->texture());

    colorAttachment->setLoadAction(MTL::LoadActionClear);
    colorAttachment->setClearColor(MTL::ClearColor::Make(0.1, 0.1, 0.1, 1));

    colorAttachment->setStoreAction(MTL::StoreActionMultisampleResolve);

    MTL::RenderPassDepthAttachmentDescriptor *depthAttachment =
        pass->depthAttachment();
    depthAttachment->setTexture(_depthTexture);
    depthAttachment->setLoadAction(MTL::LoadActionClear);
    depthAttachment->setClearDepth(1.0);
    depthAttachment->setStoreAction(MTL::StoreActionDontCare);

    MTL::RenderCommandEncoder *renderEncoder =
        buffer->renderCommandEncoder(pass);

    renderEncoder->setRenderPipelineState(_renderPSO);
    renderEncoder->setDepthStencilState(_depthStencilState);

    renderEncoder->setFragmentBuffer(_uniformBuffer, 0, 1);

    renderEncoder->setFragmentBuffer(_sdfBuffer, 0, 2);
    renderEncoder->setFragmentBytes(&_sdfNodeCount, sizeof(int), 3);

    renderEncoder->drawPrimitives(MTL::PrimitiveTypeTriangleStrip,
                                  NS::UInteger(0), NS::UInteger(4));

    renderEncoder->endEncoding();

    buffer->presentDrawable(drawable);
    buffer->commit();
  }

  pool->release();
}