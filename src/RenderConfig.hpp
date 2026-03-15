#pragma once
// ═══════════════════════════════════════════════════════════════
// RenderConfig.hpp — Configuration du kernel depuis JSON
//
// Tous les paramètres sont optionnels. Si le fichier est absent
// ou incomplet, les valeurs par défaut sont utilisées.
// ═══════════════════════════════════════════════════════════════

#include "../SDFShared.h"
#include <fstream>
#include <iostream>
#include <string>

// Tentative d'include de nlohmann/json — optionnel pour les tests
#ifdef HAS_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#endif

struct RenderConfig {
    // ── Display ──
    int   windowWidth      = 1280;
    int   windowHeight     = 720;
    bool  fullscreen       = false;
    int   msaaSamples      = 4;
    bool  vsync            = true;

    // ── Ray marcher ──
    int   maxSteps         = 256;
    float maxDistance       = 100.0f;
    float minHitEps        = 5e-5f;
    float relativeHitEps   = 3e-4f;
    float stepSafetyFactor = 0.8f;

    // ── Normals ──
    float minNormalEps     = 1e-4f;
    float relativeNormalEps= 3e-4f;

    // ── Lighting ──
    float ambient          = 0.15f;
    float diffuseStrength  = 0.7f;
    float specularStrength = 0.15f;
    float specularPower    = 32.0f;
    float baseColor[3]     = {0.75f, 0.78f, 0.82f};
    float bgBottom[3]      = {0.12f, 0.12f, 0.14f};
    float bgTop[3]         = {0.22f, 0.22f, 0.25f};

    // ── Camera ──
    float camDistance       = 0.5f;
    float camAzimuth        = 0.5f;
    float camElevation      = 0.3f;
    float camTarget[3]      = {0.0f, 0.0f, 0.0f};
    bool  autoFrame         = true;

    // ── Export ──
    std::string exportFormat    = "stl_binary";
    float exportScale           = 1000.0f;  // m → mm
    int   maxVoxelsPerDim       = 512;
    float minVoxelSize          = 0.0f;     // 0 = auto
    bool  autoResolution        = true;

    // ── Winding ──
    int   windingKMin           = 2;
    int   windingKMax           = 16;
    float windingCurvThreshold  = 0.01f;

    // ── Scene ──
    std::string sceneFile       = "scene.json";
    std::string displayMode     = "auto";

    // ── Build RenderParams struct for GPU ──
    RenderParams toGPUParams() const {
        RenderParams rp{};
        rp.maxSteps         = maxSteps;
        rp.maxDistance       = maxDistance;
        rp.minHitEps        = minHitEps;
        rp.relativeHitEps   = relativeHitEps;
        rp.stepSafetyFactor = stepSafetyFactor;
        rp.minNormalEps     = minNormalEps;
        rp.relativeNormalEps= relativeNormalEps;
        rp.ambient          = ambient;
        rp.diffuseStrength  = diffuseStrength;
        rp.specularStrength = specularStrength;
        rp.specularPower    = specularPower;
        rp.baseColorR = baseColor[0]; rp.baseColorG = baseColor[1]; rp.baseColorB = baseColor[2];
        rp.bgBottomR  = bgBottom[0];  rp.bgBottomG  = bgBottom[1];  rp.bgBottomB  = bgBottom[2];
        rp.bgTopR     = bgTop[0];     rp.bgTopG     = bgTop[1];     rp.bgTopB     = bgTop[2];
        return rp;
    }

#ifdef HAS_NLOHMANN_JSON
    // Chargement depuis fichier JSON (tous les champs optionnels)
    static RenderConfig loadFromFile(const std::string& filepath) {
        RenderConfig cfg;
        std::ifstream f(filepath);
        if (!f.is_open()) {
            std::cout << "[RenderConfig] Fichier '" << filepath
                      << "' absent, utilisation des valeurs par defaut.\n";
            return cfg;
        }

        nlohmann::json data;
        try {
            data = nlohmann::json::parse(f);
        } catch (const nlohmann::json::parse_error& e) {
            std::cerr << "[RenderConfig] JSON invalide: " << e.what()
                      << " — valeurs par defaut.\n";
            return cfg;
        }

        // Helper macro
        #define LOAD_INT(section, field, var) \
            if (data.contains(section) && data[section].contains(field)) \
                var = data[section][field].get<int>();
        #define LOAD_FLOAT(section, field, var) \
            if (data.contains(section) && data[section].contains(field)) \
                var = data[section][field].get<float>();
        #define LOAD_BOOL(section, field, var) \
            if (data.contains(section) && data[section].contains(field)) \
                var = data[section][field].get<bool>();
        #define LOAD_STR(section, field, var) \
            if (data.contains(section) && data[section].contains(field)) \
                var = data[section][field].get<std::string>();
        #define LOAD_F3(section, field, var) \
            if (data.contains(section) && data[section].contains(field)) { \
                auto& a = data[section][field]; \
                var[0] = a[0].get<float>(); var[1] = a[1].get<float>(); var[2] = a[2].get<float>(); \
            }

        LOAD_INT("display", "width", cfg.windowWidth);
        LOAD_INT("display", "height", cfg.windowHeight);
        LOAD_BOOL("display", "fullscreen", cfg.fullscreen);
        LOAD_INT("display", "msaa_samples", cfg.msaaSamples);
        LOAD_BOOL("display", "vsync", cfg.vsync);

        LOAD_INT("ray_marcher", "max_steps", cfg.maxSteps);
        LOAD_FLOAT("ray_marcher", "max_distance", cfg.maxDistance);
        LOAD_FLOAT("ray_marcher", "min_hit_eps", cfg.minHitEps);
        LOAD_FLOAT("ray_marcher", "relative_hit_eps", cfg.relativeHitEps);
        LOAD_FLOAT("ray_marcher", "step_safety_factor", cfg.stepSafetyFactor);

        LOAD_FLOAT("normals", "min_eps", cfg.minNormalEps);
        LOAD_FLOAT("normals", "relative_eps", cfg.relativeNormalEps);

        LOAD_FLOAT("lighting", "ambient", cfg.ambient);
        LOAD_FLOAT("lighting", "diffuse_strength", cfg.diffuseStrength);
        LOAD_FLOAT("lighting", "specular_strength", cfg.specularStrength);
        LOAD_FLOAT("lighting", "specular_power", cfg.specularPower);
        LOAD_F3("lighting", "base_color", cfg.baseColor);
        LOAD_F3("lighting", "background_bottom", cfg.bgBottom);
        LOAD_F3("lighting", "background_top", cfg.bgTop);

        LOAD_FLOAT("camera", "initial_distance", cfg.camDistance);
        LOAD_FLOAT("camera", "initial_azimuth", cfg.camAzimuth);
        LOAD_FLOAT("camera", "initial_elevation", cfg.camElevation);
        LOAD_F3("camera", "initial_target", cfg.camTarget);
        LOAD_BOOL("camera", "auto_frame", cfg.autoFrame);

        LOAD_STR("export", "format", cfg.exportFormat);
        LOAD_FLOAT("export", "scale_factor", cfg.exportScale);
        LOAD_INT("export", "max_voxels_per_dim", cfg.maxVoxelsPerDim);
        LOAD_FLOAT("export", "min_voxel_size", cfg.minVoxelSize);
        LOAD_BOOL("export", "auto_resolution", cfg.autoResolution);

        LOAD_INT("winding", "k_min", cfg.windingKMin);
        LOAD_INT("winding", "k_max", cfg.windingKMax);
        LOAD_FLOAT("winding", "curvature_threshold", cfg.windingCurvThreshold);

        LOAD_STR("scene", "file", cfg.sceneFile);
        LOAD_STR("scene", "display_mode", cfg.displayMode);

        #undef LOAD_INT
        #undef LOAD_FLOAT
        #undef LOAD_BOOL
        #undef LOAD_STR
        #undef LOAD_F3

        std::cout << "[RenderConfig] Charge depuis '" << filepath << "'\n";
        return cfg;
    }
#endif // HAS_NLOHMANN_JSON
};