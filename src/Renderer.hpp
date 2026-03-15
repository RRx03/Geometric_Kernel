#pragma once
#include "../SDFShared.h"
#include "Camera.hpp"
#include "RenderConfig.hpp"
#include <vector>

struct SDL_Window;

class Renderer {
public:
  Renderer(SDL_Window *window, const RenderConfig &config);
  ~Renderer();

  void renderFrame(float dt);
  void resize(int width, int height);
  void loadGeometry(const std::vector<SDFNodeGPU> &nodes);
  void updateRenderParams(const RenderConfig &config);

  Camera camera;

  private:
  void buildShaders();
  void buildBuffers();

  // Opaque pointers — actual types from metal-cpp
  void *_device = nullptr;
  void *_cmdQueue = nullptr;
  void *_layer = nullptr;
  void *_pso = nullptr;
  void *_depthState = nullptr;
  void *_uniformBuf = nullptr;
  void *_sdfBuf = nullptr;
  void *_countBuf = nullptr;
  void *_rpBuf = nullptr;
  void *_msaaTex = nullptr;
  void *_depthTex = nullptr;

  int _sdfCount = 0;
  int _width = 1280, _height = 720;
  int _samples = 4;
  RenderConfig _cfg;
};