# ═══════════════════════════════════════════════════════════════
# Geometric Kernel — Makefile
#
# Targets:
#   make test   — Golden values (no Metal/SDL)
#   make run    — Build & launch full app
#   make stl    — Build & export STL only (no window)
#   make clean  — Cleanup
#   make init   — Install dependencies
# ═══════════════════════════════════════════════════════════════

CXX       = clang++
OBJCXX    = clang++
CXXFLAGS  = -std=c++17 -O2 -Wall -Wextra -I. -DHAS_NLOHMANN_JSON
MMFLAGS   = -std=c++17 -O2 -Wall -Wextra -I. -DHAS_NLOHMANN_JSON -ObjC++

# Metal shader
METALC    = xcrun -sdk macosx metal
METALLINK = xcrun -sdk macosx metallib

# SDL2
SDL_CFLAGS  = $(shell sdl2-config --cflags 2>/dev/null || echo "-I/opt/homebrew/include/SDL2")
SDL_LDFLAGS = $(shell sdl2-config --libs 2>/dev/null || echo "-L/opt/homebrew/lib -lSDL2")

# JSON (header-only) — try pkg-config, then brew --prefix, then common paths
JSON_CFLAGS = $(shell pkg-config --cflags nlohmann_json 2>/dev/null \
              || echo "-I$(shell brew --prefix nlohmann-json 2>/dev/null)/include" \
              || echo "-I/opt/homebrew/include")

# Frameworks
FRAMEWORKS = -framework Metal -framework QuartzCore -framework Foundation \
             -framework Cocoa -framework CoreGraphics

BUILD = build

.PHONY: all test run clean init

all: test

# ═══════════════════════════════════════════════════════════════
# Tests (no Metal, no SDL, no JSON required)
# ═══════════════════════════════════════════════════════════════
test: $(BUILD)/test_primitives
	@echo ""
	@echo "═══ Running Golden Values Tests ═══"
	@./$(BUILD)/test_primitives

$(BUILD)/test_primitives: tests/test_primitives.cpp SDFShared.h \
		src/SDFMath.hpp src/SDFNode.hpp src/SDFEvaluator.hpp | $(BUILD)
	$(CXX) -std=c++17 -O2 -Wall -Wextra -I. -Wno-unused-parameter \
		tests/test_primitives.cpp -o $@

# ═══════════════════════════════════════════════════════════════
# Full Application
# ═══════════════════════════════════════════════════════════════
run: $(BUILD)/kernel $(BUILD)/default.metallib
	@echo ""
	@echo "═══ Launching Geometric Kernel ═══"
	@cd $(BUILD) && ./kernel ../scenes/test_nozzle.json

KERNEL_OBJS = $(BUILD)/main.o $(BUILD)/Renderer.o $(BUILD)/MetalBridge.o $(BUILD)/Mesher.o

$(BUILD)/kernel: $(KERNEL_OBJS) | $(BUILD)
	$(OBJCXX) $(KERNEL_OBJS) $(SDL_LDFLAGS) $(FRAMEWORKS) -o $@

$(BUILD)/main.o: src/main.cpp src/Renderer.hpp src/SDFNode.hpp \
		src/SDFEvaluator.hpp src/SceneParser.hpp src/RenderConfig.hpp \
		src/Camera.hpp src/Mesher.hpp SDFShared.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(SDL_CFLAGS) $(JSON_CFLAGS) -c src/main.cpp -o $@

# Pure C++ — includes metal-cpp with _PRIVATE_IMPLEMENTATION
$(BUILD)/Renderer.o: src/Renderer.cpp src/Renderer.hpp src/Camera.hpp \
		src/RenderConfig.hpp src/MetalBridge.h SDFShared.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(SDL_CFLAGS) $(JSON_CFLAGS) -c src/Renderer.cpp -o $@

# Obj-C++ bridge — NO metal-cpp, only Cocoa + CAMetalLayer + SDL
$(BUILD)/MetalBridge.o: src/MetalBridge.mm src/MetalBridge.h | $(BUILD)
	$(OBJCXX) $(MMFLAGS) $(SDL_CFLAGS) -c src/MetalBridge.mm -o $@

$(BUILD)/Mesher.o: src/Mesher.cpp src/Mesher.hpp src/SDFEvaluator.hpp \
		src/MCTables.h SDFShared.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c src/Mesher.cpp -o $@

# ═══════════════════════════════════════════════════════════════
# Metal Shader — compile to .metallib in one step to avoid version mismatch
# ═══════════════════════════════════════════════════════════════
$(BUILD)/default.metallib: shaders/kernel.metal SDFShared.h | $(BUILD)
	xcrun -sdk macosx metal -std=macos-metal2.0 -O2 -I. \
		-frecord-sources \
		shaders/kernel.metal -o $(BUILD)/default.metallib
	@echo "[Metal] Shader compiled successfully"

# ═══════════════════════════════════════════════════════════════
$(BUILD):
	@mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)

init:
	@echo "Installing dependencies..."
	brew install sdl2 nlohmann-json
	@echo "Done. Run 'make test' then 'make run'."