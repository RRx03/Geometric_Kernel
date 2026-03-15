#pragma once
// ═══════════════════════════════════════════════════════════════
// SceneParser.hpp — JSON → Arbre SDFNode
//
// Parse le geometry.json produit par Athena et construit
// l'arbre AST de primitives SDF.
// ═══════════════════════════════════════════════════════════════

#include "SDFNode.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

using json = nlohmann::json;

class SceneParser {
public:
    struct SceneInfo {
        std::shared_ptr<Geometry::SDFNode> root;
        std::string displayMode = "auto"; // "3d", "2d_axisymmetric", "2d", "auto"
    };

    static SceneInfo parseFile(const std::string& filepath) {
        std::ifstream f(filepath);
        if (!f.is_open())
            throw std::runtime_error("SceneParser: impossible d'ouvrir '" + filepath + "'");

        json data;
        try {
            data = json::parse(f);
        } catch (const json::parse_error& e) {
            throw std::runtime_error("SceneParser: JSON invalide dans '"
                + filepath + "': " + e.what());
        }

        SceneInfo info;

        // Mode d'affichage
        if (data.contains("display_mode")) {
            info.displayMode = data["display_mode"].get<std::string>();
        }

        // Racine géométrique
        if (data.contains("geometry")) {
            info.root = parseNode(data["geometry"]);
        } else {
            info.root = parseNode(data);
        }

        return info;
    }

private:
    // Vérifie qu'un champ existe, sinon exception claire
    static void req(const json& j, const std::string& field, const std::string& type) {
        if (!j.contains(field))
            throw std::runtime_error("SceneParser: '" + type + "' requiert '"
                + field + "'. JSON: " + j.dump(2).substr(0, 200));
    }

    static std::shared_ptr<Geometry::SDFNode> parseNode(const json& j) {
        if (!j.contains("type"))
            throw std::runtime_error("SceneParser: noeud sans 'type'. JSON: "
                + j.dump(2).substr(0, 200));

        std::string type = j["type"];

        // ── Opérations CSG ──

        if (type == "Union") {
            req(j, "left", type); req(j, "right", type);
            return std::make_shared<Geometry::Union>(
                parseNode(j["left"]), parseNode(j["right"]));
        }
        if (type == "Subtract") {
            req(j, "base", type); req(j, "subtract", type);
            return std::make_shared<Geometry::Subtract>(
                parseNode(j["base"]), parseNode(j["subtract"]));
        }
        if (type == "Intersect") {
            req(j, "left", type); req(j, "right", type);
            return std::make_shared<Geometry::Intersect>(
                parseNode(j["left"]), parseNode(j["right"]));
        }
        if (type == "SmoothUnion") {
            req(j, "left", type); req(j, "right", type); req(j, "smoothFactor", type);
            return std::make_shared<Geometry::SmoothUnion>(
                parseNode(j["left"]), parseNode(j["right"]),
                j["smoothFactor"].get<float>());
        }

        // ── Transformation ──

        if (type == "Transform") {
            req(j, "child", type);
            simd::float3 trans = {0, 0, 0};
            simd::float3 axis  = {0, 1, 0};
            float angle = 0.0f;
            float scale = 1.0f;

            if (j.contains("translate")) {
                auto& t = j["translate"];
                trans = simd_make_float3(t[0].get<float>(), t[1].get<float>(), t[2].get<float>());
            }
            if (j.contains("rotate")) {
                auto& r = j["rotate"];
                if (r.contains("axis")) {
                    auto& a = r["axis"];
                    axis = simd_make_float3(a[0].get<float>(), a[1].get<float>(), a[2].get<float>());
                    axis = simd_normalize(axis);
                }
                if (r.contains("angle_deg"))
                    angle = r["angle_deg"].get<float>() * 3.14159265f / 180.0f;
                if (r.contains("angle_rad"))
                    angle = r["angle_rad"].get<float>();
            }
            if (j.contains("scale"))
                scale = j["scale"].get<float>();

            return std::make_shared<Geometry::Transform>(
                parseNode(j["child"]), trans, axis, angle, scale);
        }

        // ── Primitives 3D ──

        if (type == "Sphere") {
            req(j, "position", type); req(j, "radius", type);
            auto& p = j["position"];
            return std::make_shared<Geometry::Sphere>(
                simd_make_float3(p[0].get<float>(), p[1].get<float>(), p[2].get<float>()),
                j["radius"].get<float>());
        }
        if (type == "Box") {
            req(j, "position", type); req(j, "bounds", type);
            auto& p = j["position"];
            auto& b = j["bounds"];
            return std::make_shared<Geometry::Box>(
                simd_make_float3(p[0].get<float>(), p[1].get<float>(), p[2].get<float>()),
                simd_make_float3(b[0].get<float>(), b[1].get<float>(), b[2].get<float>()));
        }
        if (type == "Cylinder") {
            req(j, "position", type); req(j, "radius", type); req(j, "height", type);
            auto& p = j["position"];
            return std::make_shared<Geometry::Cylinder>(
                simd_make_float3(p[0].get<float>(), p[1].get<float>(), p[2].get<float>()),
                j["radius"].get<float>(),
                j["height"].get<float>() * 0.5f); // height → halfHeight
        }
        if (type == "Torus") {
            req(j, "position", type); req(j, "majorRadius", type); req(j, "minorRadius", type);
            auto& p = j["position"];
            return std::make_shared<Geometry::Torus>(
                simd_make_float3(p[0].get<float>(), p[1].get<float>(), p[2].get<float>()),
                j["majorRadius"].get<float>(), j["minorRadius"].get<float>());
        }
        if (type == "Capsule") {
            req(j, "pointA", type); req(j, "pointB", type); req(j, "radius", type);
            auto& a = j["pointA"];
            auto& b = j["pointB"];
            return std::make_shared<Geometry::Capsule>(
                simd_make_float3(a[0].get<float>(), a[1].get<float>(), a[2].get<float>()),
                simd_make_float3(b[0].get<float>(), b[1].get<float>(), b[2].get<float>()),
                j["radius"].get<float>());
        }

        // ── Primitives 2D ──

        if (type == "CompositeSpline2D") {
            req(j, "points", type);
            float thickness = j.value("thickness", 0.0f);
            std::vector<simd::float2> pts;
            for (auto& pt : j["points"]) {
                pts.push_back(simd_make_float2(pt[0].get<float>(), pt[1].get<float>()));
            }
            return std::make_shared<Geometry::CompositeSpline2D>(std::move(pts), thickness);
        }
        if (type == "Bezier2D") {
            req(j, "p0", type); req(j, "p1", type); req(j, "p2", type);
            float thickness = j.value("thickness", 0.01f);
            return std::make_shared<Geometry::Bezier2D>(
                simd_make_float2(j["p0"][0].get<float>(), j["p0"][1].get<float>()),
                simd_make_float2(j["p1"][0].get<float>(), j["p1"][1].get<float>()),
                simd_make_float2(j["p2"][0].get<float>(), j["p2"][1].get<float>()),
                thickness);
        }
        if (type == "CubicBezier2D") {
            req(j, "p0", type); req(j, "p1", type); req(j, "p2", type); req(j, "p3", type);
            float thickness = j.value("thickness", 0.01f);
            return std::make_shared<Geometry::CubicBezier2D>(
                simd_make_float2(j["p0"][0].get<float>(), j["p0"][1].get<float>()),
                simd_make_float2(j["p1"][0].get<float>(), j["p1"][1].get<float>()),
                simd_make_float2(j["p2"][0].get<float>(), j["p2"][1].get<float>()),
                simd_make_float2(j["p3"][0].get<float>(), j["p3"][1].get<float>()),
                thickness);
        }
        if (type == "Circle2D") {
            req(j, "center", type); req(j, "radius", type);
            return std::make_shared<Geometry::Circle2D>(
                simd_make_float2(j["center"][0].get<float>(), j["center"][1].get<float>()),
                j["radius"].get<float>());
        }
        if (type == "Rect2D") {
            req(j, "center", type); req(j, "halfExtents", type);
            return std::make_shared<Geometry::Rect2D>(
                simd_make_float2(j["center"][0].get<float>(), j["center"][1].get<float>()),
                simd_make_float2(j["halfExtents"][0].get<float>(), j["halfExtents"][1].get<float>()));
        }

        throw std::runtime_error("SceneParser: type inconnu '" + type + "'");
    }
};
