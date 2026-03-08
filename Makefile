
APP_NAME = app
BUILD_DIR = ./build
SRC_FOLDER = ./src
OBJS_FOLDER = $(BUILD_DIR)/objs
BIN_PATH = $(BUILD_DIR)/$(APP_NAME)
METALLIB = $(BUILD_DIR)/default.metallib

CXX = clang++
CXXFLAGS = -std=c++17 -I. -I$(SRC_FOLDER) -I/opt/homebrew/include -MMD -MP 
SDL2_CFLAGS := $(shell sdl2-config --cflags)
SDL2_LDFLAGS := $(shell sdl2-config --libs)
FRAMEWORKS = -framework Metal -framework Foundation -framework QuartzCore -framework CoreGraphics

SRCS := $(wildcard $(SRC_FOLDER)/*.cpp)
OBJS := $(patsubst $(SRC_FOLDER)/%.cpp, $(OBJS_FOLDER)/%.o, $(SRCS))

DEPS := $(OBJS:.o=.d)

.PHONY: all clean run copy_shaders


all: $(BIN_PATH) $(METALLIB)

$(BIN_PATH): $(OBJS)
	@echo "Linkage de l'exécutable..."
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OBJS) $(FRAMEWORKS) $(SDL2_LDFLAGS) -o $@
	@echo "Succès ! Exécutable créé ici : $@"

$(OBJS_FOLDER)/%.o: $(SRC_FOLDER)/%.cpp
	@echo "Compilation de $< ..."
	@mkdir -p $(OBJS_FOLDER)
	$(CXX) $(CXXFLAGS) $(SDL2_CFLAGS) -c $< -o $@

$(METALLIB): shaders/shader.metal
	@echo "Compilation des Shaders..."
	@mkdir -p $(BUILD_DIR)
	xcrun -sdk macosx metal -c shaders/shader.metal -o $(BUILD_DIR)/shader.air
	xcrun -sdk macosx metallib $(BUILD_DIR)/shader.air -o $(METALLIB)
	@rm $(BUILD_DIR)/shader.air 
	@echo "Shaders compilés dans $(METALLIB)"

-include $(DEPS)
init: 
	brew install nlohmann-json
clean:
	rm -rf $(BUILD_DIR)
	@echo "Dossier build nettoyé."

run: all
	$(BIN_PATH)

bear:
	make clean
	bear -- make
	
setup : 
	xcode-select --install	