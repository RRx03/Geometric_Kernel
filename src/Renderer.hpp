#pragma once

#include "metal-cpp/Metal/Metal.hpp"
#include "metal-cpp/QuartzCore/QuartzCore.hpp"
#include <SDL.h>

class Renderer {
public:
  Renderer(SDL_Window *window);
  ~Renderer();
  void renderFrame();
  void resize(int width, int height);

private:
  void buildShaders();
  void buildBuffers();
  void updateUniforms();

  MTL::Device *_device = nullptr;
  MTL::CommandQueue *_commandQueue = nullptr;
  CA::MetalLayer *_layer = nullptr;

  MTL::ComputePipelineState *_computePSO = nullptr;
  MTL::RenderPipelineState *_renderPSO = nullptr;
  MTL::Texture *_depthTexture = nullptr;
  MTL::DepthStencilState *_depthStencilState = nullptr;

  MTL::Buffer *_vertexBuffer = nullptr;
  MTL::Buffer *_computeBuffer = nullptr;
  MTL::Buffer *_uniformBuffer;

  float _angle;        // TEMPLATE
  int _width, _height; // Pour le ratio d'aspect

  MTL::Texture *_msaaTexture = nullptr;
  const int _sampleCount = 4;
};