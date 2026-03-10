#pragma once

#include "SDFShared.h"
#include "metal-cpp/Metal/Metal.hpp"
#include "metal-cpp/QuartzCore/QuartzCore.hpp"
#include <SDL2/SDL.h>
#include <simd/simd.h>
#include <vector>

class Renderer {
public:
  Renderer(SDL_Window *window);
  ~Renderer();
  void renderFrame();
  void resize(int width, int height);

  void orbit(float dx, float dy);
  void pan(float dx, float dy);
  void zoom(float dz);

  void loadGeometry(const std::vector<SDFNodeGPU> &nodes);

private:
  void buildShaders();
  void buildBuffers();
  void updateUniforms();

  // Metal objects
  MTL::Device *_device = nullptr;
  MTL::CommandQueue *_commandQueue = nullptr;
  CA::MetalLayer *_layer = nullptr;

  MTL::RenderPipelineState *_renderPSO = nullptr;
  MTL::Texture *_depthTexture = nullptr;
  MTL::DepthStencilState *_depthStencilState = nullptr;

  MTL::Buffer *_vertexBuffer = nullptr;
  MTL::Buffer *_uniformBuffer = nullptr;

  // SDF data
  MTL::Buffer *_sdfBuffer = nullptr;
  int _sdfNodeCount = 0;

  // Viewport
  int _width = 800;
  int _height = 600;

  // Camera (orbite sphérique)
  float _camDistance = 5.0f;
  float _camAzimuth = 0.0f;
  float _camElevation = 0.0f;
  simd::float3 _camTarget = {0.0f, 0.0f, 0.0f};

  // MSAA
  MTL::Texture *_msaaTexture = nullptr;
  const int _sampleCount = 4;
};