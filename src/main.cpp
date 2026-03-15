#include "Mesher.hpp"
#include "SDFEvaluator.hpp"
#include "SceneParser.hpp"
#include "metal-cpp/Foundation/Foundation.hpp"
#include "metal-cpp/Metal/Metal.hpp"
#include "metal-cpp/QuartzCore/QuartzCore.hpp"
#include <Renderer.hpp>
#include <SDL2/SDL.h>
#include <SDL2/SDL_metal.h>
#include <atomic>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <simd/simd.h>
#include <thread>

// ── Auto-detect bounding box from scene JSON ──
// Searches for a Box node and uses its bounds, or falls back to default
static void detectBoundingBox(const std::string &sceneFile, simd::float3 &bbMin,
                              simd::float3 &bbMax) {
  try {
    std::ifstream f(sceneFile);
    if (!f.is_open())
      return;
    auto root = nlohmann::json::parse(f);

    // Recursive search for Box nodes
    std::function<bool(const nlohmann::json &)> findBox =
        [&](const nlohmann::json &j) -> bool {
      if (!j.is_object())
        return false;
      if (j.contains("type") && j["type"] == "Box" && j.contains("bounds")) {
        float bx = j["bounds"][0].get<float>();
        float by = j["bounds"][1].get<float>();
        float bz = j["bounds"][2].get<float>();
        float px = 0, py = 0, pz = 0;
        if (j.contains("position")) {
          px = j["position"][0].get<float>();
          py = j["position"][1].get<float>();
          pz = j["position"][2].get<float>();
        }
        // Add margin
        float margin = 0.01f;
        bbMin = simd_make_float3(px - bx - margin, py - by - margin,
                                 pz - bz - margin);
        bbMax = simd_make_float3(px + bx + margin, py + by + margin,
                                 pz + bz + margin);
        return true;
      }
      for (auto &[key, val] : j.items()) {
        if (findBox(val))
          return true;
      }
      return false;
    };

    findBox(root);
  } catch (...) {
    // Keep defaults
  }
}

int main() {
  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    std::cout << "Erreur Init SDL: " << SDL_GetError() << std::endl;
    return -1;
  }

  SDL_Window *window =
      SDL_CreateWindow("Generative Kernel", SDL_WINDOWPOS_CENTERED,
                       SDL_WINDOWPOS_CENTERED, 800, 600,
                       SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_METAL |
                           SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

  if (!window) {
    std::cout << "Erreur Creation Fenetre: " << SDL_GetError() << std::endl;
    return -1;
  }

  std::unique_ptr<Renderer> renderer;
  try {
    renderer = std::make_unique<Renderer>(window);
    int width, height;
    SDL_GetWindowSize(window, &width, &height);
    renderer->resize(width, height);
  } catch (const std::runtime_error &e) {
    std::cerr << "Erreur Renderer: " << e.what() << std::endl;
    SDL_DestroyWindow(window);
    SDL_Quit();
    return -1;
  }

  // ── Charger la scène ──
  auto myPartTree = SceneParser::parseFile("scene.json");
  std::vector<SDFNodeGPU> flattenedTree;
  if (myPartTree) {
    myPartTree->flatten(flattenedTree);
  }

  std::cout << "Scene chargee: " << flattenedTree.size() << " noeuds GPU"
            << std::endl;

  renderer->loadGeometry(flattenedTree);
  SDFEvaluator solverEvaluator(flattenedTree);

  std::atomic<bool> exportInProgress{false};

  // ── Auto bounding box from scene ──
  simd::float3 bbMin = simd_make_float3(-3.0f, -3.0f, -3.0f);
  simd::float3 bbMax = simd_make_float3(3.0f, 3.0f, 3.0f);
  detectBoundingBox("scene.json", bbMin, bbMax);

  float bbSize = simd_length(bbMax - bbMin);
  std::cout << "Bounding box: [" << bbMin.x << "," << bbMin.y << "," << bbMin.z
            << "] → [" << bbMax.x << "," << bbMax.y << "," << bbMax.z << "]"
            << std::endl;
  std::cout << "Taille diag: " << bbSize * 1000.0f << " mm" << std::endl;

  constexpr float STL_EXPORT_SCALE = 1000.0f;

  bool running = true;
  SDL_Event event;

  std::cout << "\n=== CONTROLES ===" << std::endl;
  std::cout << "  Souris gauche : orbiter" << std::endl;
  std::cout << "  Souris droite : pan" << std::endl;
  std::cout << "  Molette       : zoom" << std::endl;
  std::cout << "  E             : export STL (standard)" << std::endl;
  std::cout << "  H             : export STL (haute qualite)" << std::endl;
  std::cout << "  Unites internes: metres (SI)" << std::endl;
  std::cout << "  Unites STL    : millimetres" << std::endl;
  std::cout << "================\n" << std::endl;

  while (running) {
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT)
        running = false;

      if (event.type == SDL_WINDOWEVENT &&
          event.window.event == SDL_WINDOWEVENT_RESIZED)
        renderer->resize(event.window.data1, event.window.data2);

      if (event.type == SDL_MOUSEMOTION) {
        if (event.motion.state & SDL_BUTTON_LMASK)
          renderer->orbit(event.motion.xrel, event.motion.yrel);
        else if (event.motion.state & SDL_BUTTON_RMASK)
          renderer->pan(event.motion.xrel, event.motion.yrel);
      }

      if (event.type == SDL_MOUSEWHEEL)
        renderer->zoom(event.wheel.y);

      if (event.type == SDL_KEYDOWN) {
        float res = 0.0f;
        std::string filename;

        // E = standard (~100 voxels along longest axis)
        // H = haute qualité (~300 voxels along longest axis)
        if (event.key.keysym.sym == SDLK_e) {
          res = bbSize / 100.0f;
          filename = "export.stl";
        } else if (event.key.keysym.sym == SDLK_h) {
          res = bbSize / 300.0f;
          filename = "export_hq.stl";
        }

        if (res > 0.0f && !exportInProgress.load()) {
          exportInProgress.store(true);
          std::thread exportThread([&solverEvaluator, &exportInProgress, bbMin,
                                    bbMax, res, filename]() {
            Mesher::generateSTL(solverEvaluator, bbMin, bbMax, res, filename,
                                STL_EXPORT_SCALE);
            exportInProgress.store(false);
          });
          exportThread.detach();
        } else if (res > 0.0f) {
          std::cout << ">>> Export deja en cours..." << std::endl;
        }
      }
    }

    renderer->renderFrame();
    SDL_Delay(10);
  }

  SDL_DestroyWindow(window);
  SDL_Quit();
  std::cout << "Fermeture." << std::endl;
  return 0;
}