// ═══════════════════════════════════════════════════════════════
// main.cpp — Geometric Kernel Entry Point
//
// SDL2 event loop, scene loading, keyboard/mouse controls,
// STL export, fullscreen toggle, preset views.
// ═══════════════════════════════════════════════════════════════

#include "../SDFShared.h"
#include "Mesher.hpp"
#include "RenderConfig.hpp"
#include "Renderer.hpp"
#include "SDFEvaluator.hpp"
#include "SDFNode.hpp"
#include "SceneParser.hpp"

#include <SDL.h>
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  // ── Load config ──
  RenderConfig config;
#ifdef HAS_NLOHMANN_JSON
    config = RenderConfig::loadFromFile("render_config.json");
#endif

    // ── Scene file (CLI override or config) ──
    std::string sceneFile = config.sceneFile;
    if (argc > 1)
      sceneFile = argv[1];

    // ── Init SDL ──
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
      std::cerr << "SDL Init error: " << SDL_GetError() << "\n";
      return -1;
    }

    Uint32 winFlags = SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_METAL |
                      SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE;
    if (config.fullscreen)
      winFlags |= SDL_WINDOW_FULLSCREEN_DESKTOP;

    SDL_Window *window = SDL_CreateWindow(
        "Geometric Kernel", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        config.windowWidth, config.windowHeight, winFlags);

    if (!window) {
      std::cerr << "Window error: " << SDL_GetError() << "\n";
      SDL_Quit();
      return -1;
    }

    // ── Renderer ──
    std::unique_ptr<Renderer> renderer;
    try {
      renderer = std::make_unique<Renderer>(window, config);
      int w, h;
      SDL_Metal_GetDrawableSize(window, &w, &h);
      renderer->resize(w, h);
    } catch (const std::exception &e) {
      std::cerr << "Renderer error: " << e.what() << "\n";
      SDL_DestroyWindow(window);
      SDL_Quit();
      return -1;
    }

    // ── Load scene ──
    std::vector<SDFNodeGPU> gpuBuffer;
    SDFEvaluator *evaluatorPtr = nullptr;

    try {
      auto scene = SceneParser::parseFile(sceneFile);
      if (scene.root) {
        scene.root->flatten(gpuBuffer);
        renderer->loadGeometry(gpuBuffer);
        std::cout << "[Main] Scene '" << sceneFile << "': " << gpuBuffer.size()
                  << " GPU nodes\n";
      }
    } catch (const std::exception &e) {
      std::cerr << "[Main] Scene load error: " << e.what() << "\n";
      std::cout << "[Main] Running with empty scene.\n";
    }

    // Create evaluator for STL export and auto-framing
    SDFEvaluator evaluator(gpuBuffer);
    evaluatorPtr = &evaluator;

    // Auto-frame camera to fit the geometry
    if (config.autoFrame && !gpuBuffer.empty()) {
      renderer->camera.autoFrame(evaluator.boundsMin(), evaluator.boundsMax());
    }

    // ── Main loop ──
    bool running = true;
    bool mouseLeftDown = false, mouseRightDown = false;
    bool fullscreen = config.fullscreen;

    auto lastTime = std::chrono::high_resolution_clock::now();

    std::cout << "\n[Controls]\n"
              << "  Left-drag : Orbit\n"
              << "  Right-drag: Pan\n"
              << "  Scroll    : Zoom\n"
              << "  R         : Reset view\n"
              << "  F         : Auto-frame\n"
              << "  F11       : Toggle fullscreen\n"
              << "  1/2/3/4   : Front/Right/Top/3-4 view\n"
              << "  E         : Export STL\n"
              << "  Esc       : Quit\n\n";

    while (running) {
      // Delta time
      auto now = std::chrono::high_resolution_clock::now();
      float dt = std::chrono::duration<float>(now - lastTime).count();
      lastTime = now;

      SDL_Event ev;
      while (SDL_PollEvent(&ev)) {
        switch (ev.type) {

        case SDL_QUIT:
          running = false;
          break;

        case SDL_WINDOWEVENT:
          if (ev.window.event == SDL_WINDOWEVENT_RESIZED ||
              ev.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
            int w, h;
            SDL_Metal_GetDrawableSize(window, &w, &h);
            renderer->resize(w, h);
          }
          break;

        case SDL_KEYDOWN:
          switch (ev.key.keysym.sym) {
          case SDLK_ESCAPE:
            running = false;
            break;
          case SDLK_r:
            renderer->camera.resetView();
            if (config.autoFrame && !gpuBuffer.empty())
              renderer->camera.autoFrame(evaluator.boundsMin(),
                                         evaluator.boundsMax());
            break;
          case SDLK_f:
            if (!gpuBuffer.empty())
              renderer->camera.autoFrame(evaluator.boundsMin(),
                                         evaluator.boundsMax());
            break;
          case SDLK_F11:
            fullscreen = !fullscreen;
            SDL_SetWindowFullscreen(
                window, fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0);
            break;
          case SDLK_1:
            renderer->camera.viewFront();
            break;
          case SDLK_2:
            renderer->camera.viewRight();
            break;
          case SDLK_3:
            renderer->camera.viewTop();
            break;
          case SDLK_4:
            renderer->camera.viewThreeQtr();
            break;
          case SDLK_e:
            if (evaluatorPtr && !gpuBuffer.empty()) {
              // Non-blocking export in background thread
              static bool exporting = false;
              if (!exporting) {
                exporting = true;
                // Capture by value for thread safety
                auto evalCopy = *evaluatorPtr;
                float voxSize = config.minVoxelSize;
                int maxVPD = config.maxVoxelsPerDim;
                float scale = config.exportScale;
                std::thread([evalCopy, voxSize, maxVPD, scale]() {
                  std::cout << "[Export] Starting STL export (background)...\n";
                  Mesher::exportSTL(evalCopy, "export.stl", voxSize, maxVPD,
                                    scale);
                  std::cout << "[Export] Done! File: export.stl\n";
                  exporting = false;
                }).detach();
              } else {
                std::cout << "[Export] Already exporting, please wait.\n";
              }
            }
            break;
          default:
            break;
          }
          break;

        case SDL_MOUSEBUTTONDOWN:
          if (ev.button.button == SDL_BUTTON_LEFT)
            mouseLeftDown = true;
          if (ev.button.button == SDL_BUTTON_RIGHT)
            mouseRightDown = true;
          // Double-click left = auto-frame
          if (ev.button.button == SDL_BUTTON_LEFT && ev.button.clicks == 2) {
            if (!gpuBuffer.empty())
              renderer->camera.autoFrame(evaluator.boundsMin(),
                                         evaluator.boundsMax());
          }
          break;

        case SDL_MOUSEBUTTONUP:
          if (ev.button.button == SDL_BUTTON_LEFT)
            mouseLeftDown = false;
          if (ev.button.button == SDL_BUTTON_RIGHT)
            mouseRightDown = false;
          break;

        case SDL_MOUSEMOTION:
          if (mouseLeftDown) {
            renderer->camera.orbit((float)ev.motion.xrel,
                                   (float)ev.motion.yrel);
          }
          if (mouseRightDown) {
            renderer->camera.pan((float)ev.motion.xrel, (float)ev.motion.yrel);
          }
          break;

        case SDL_MOUSEWHEEL:
          renderer->camera.zoom((float)ev.wheel.y);
          break;
        }
      }

        // Render
        renderer->renderFrame(dt);
    }

    // ── Cleanup ──
    renderer.reset();
    SDL_DestroyWindow(window);
    SDL_Quit();

    std::cout << "[Main] Shutdown complete.\n";
    return 0;
}