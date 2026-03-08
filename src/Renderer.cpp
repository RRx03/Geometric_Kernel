#include "Renderer.hpp"
#include "MathUtils.h"
#include "SDFNode.hpp"
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
  auto sphere = std::make_shared<Geometry::Sphere>(
      simd_make_float3(-0.8f, 0.0f, 0.0f), 1.2f);
  auto box = std::make_shared<Geometry::Box>(
      simd_make_float3(0.8f, 0.0f, 0.0f), simd_make_float3(1.0f, 1.0f, 1.0f));
  auto myPart = std::make_shared<Geometry::Union>(sphere, box);

  std::vector<SDFNodeGPU> gpuSDFArray;
  myPart->flatten(gpuSDFArray);
  _sdfNodeCount = (int)gpuSDFArray.size();

  _sdfBuffer = _device->newBuffer(gpuSDFArray.data(),
                                  gpuSDFArray.size() * sizeof(SDFNodeGPU),
                                  MTL::ResourceStorageModeShared);

  _uniformBuffer =
      _device->newBuffer(sizeof(Uniforms), MTL::ResourceStorageModeShared);
}

void Renderer::updateUniforms() {
  _angle += 0.01f;

  Uniforms u;

  u.modelMatrix = Math::makeYRotation(_angle);

  u.viewMatrix = Math::makeTranslate({0.0f, 0.0f, -3.0f});

  float aspect = (float)_width / (float)_height;
  u.projectionMatrix =
      Math::makePerspective(Math::radians(45.0f), aspect, 0.1f, 100.0f);

  void *ptr = _uniformBuffer->contents();
  memcpy(ptr, &u, sizeof(Uniforms));
}

void Renderer::resize(int width, int height) {
  _width = width;
  _height = height;
  _layer->setDrawableSize(CGSizeMake(width, height));

  // Nettoyage
  if (_msaaTexture)
    _msaaTexture->release();
  if (_depthTexture)
    _depthTexture->release();

  // 1. Texture Couleur MSAA (Celle où on dessine)
  MTL::TextureDescriptor *msaaDesc = MTL::TextureDescriptor::alloc()->init();
  msaaDesc->setTextureType(
      MTL::TextureType2DMultisample); // <--- TYPE MULTISAMPLE
  msaaDesc->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
  msaaDesc->setWidth(width);
  msaaDesc->setHeight(height);
  msaaDesc->setSampleCount(_sampleCount); // <--- 4x
  msaaDesc->setUsage(MTL::TextureUsageRenderTarget);
  msaaDesc->setStorageMode(MTL::StorageModePrivate); // GPU Only

  _msaaTexture = _device->newTexture(msaaDesc);
  msaaDesc->release();

  // 2. Texture Profondeur (Doit aussi être MSAA !)
  MTL::TextureDescriptor *depthDesc = MTL::TextureDescriptor::alloc()->init();
  depthDesc->setTextureType(
      MTL::TextureType2DMultisample); // <--- TYPE MULTISAMPLE
  depthDesc->setPixelFormat(MTL::PixelFormatDepth32Float);
  depthDesc->setWidth(width);
  depthDesc->setHeight(height);
  depthDesc->setSampleCount(_sampleCount); // <--- 4x doit matcher !
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