#include <SDL2/SDL.h>
#include <SDL2/SDL_metal.h>
#include <iostream>

#include "metal-cpp/Foundation/Foundation.hpp"
#include "metal-cpp/Metal/Metal.hpp"
#include "metal-cpp/QuartzCore/QuartzCore.hpp"
#include <Renderer.hpp>
#include <simd/simd.h>

int main() {
  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    std::cout << "Erreur Init SDL: " << SDL_GetError() << std::endl;
    return -1;
  }

  SDL_Window *window = SDL_CreateWindow(
      "TestMetal", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 800, 600,
      SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_METAL | SDL_WINDOW_SHOWN);

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
    std::cerr << "Erreur lors de l'initialisation du Renderer: " << e.what()
              << std::endl;
    SDL_DestroyWindow(window);
    SDL_Quit();
    return -1;
  }

  bool running = true;
  SDL_Event event;
  while (running) {
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT) {
        running = false;
      }
      if (event.type == SDL_WINDOWEVENT &&
          event.window.event == SDL_WINDOWEVENT_RESIZED) {
        renderer->resize(event.window.data1, event.window.data2);
      }
      if (event.type == SDL_MOUSEMOTION) {
        if (event.motion.state & SDL_BUTTON_LMASK) {
          renderer->orbit(event.motion.xrel, event.motion.yrel);
        } else if (event.motion.state & SDL_BUTTON_RMASK) {
          renderer->pan(event.motion.xrel, event.motion.yrel);
        }
      }
      if (event.type == SDL_MOUSEWHEEL) {
        renderer->zoom(event.wheel.y);
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