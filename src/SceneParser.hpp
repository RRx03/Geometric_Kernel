#pragma once
#include "SDFNode.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

using json = nlohmann::json;

class SceneParser {
public:
  static std::shared_ptr<Geometry::SDFNode>
  parseFile(const std::string &filepath) {
    std::ifstream f(filepath);
    if (!f.is_open())
      throw std::runtime_error("SceneParser: impossible d'ouvrir '" + filepath +
                               "'");
    json data;
    try {
      data = json::parse(f);
    } catch (const json::parse_error &e) {
      throw std::runtime_error("SceneParser: JSON invalide dans '" + filepath +
                               "': " + e.what());
    }
    return parseNode(data);
  }

private:
  static void req(const json &j, const std::string &f, const std::string &t) {
    if (!j.contains(f))
      throw std::runtime_error("SceneParser: '" + t + "' requiert '" + f + "'");
  }

  static std::shared_ptr<Geometry::SDFNode> parseNode(const json &j) {
    if (!j.contains("type"))
      throw std::runtime_error("SceneParser: noeud sans 'type'. JSON: " +
                               j.dump(2).substr(0, 200));
    std::string type = j["type"];

    // CSG
    if (type == "Union") {
      req(j, "left", type);
      req(j, "right", type);
      return std::make_shared<Geometry::Union>(parseNode(j["left"]),
                                               parseNode(j["right"]));
    }
    if (type == "Subtract") {
      req(j, "base", type);
      req(j, "subtract", type);
      return std::make_shared<Geometry::Subtract>(parseNode(j["base"]),
                                                  parseNode(j["subtract"]));
    }
    if (type == "Intersect") {
      req(j, "left", type);
      req(j, "right", type);
      return std::make_shared<Geometry::Intersect>(parseNode(j["left"]),
                                                   parseNode(j["right"]));
    }
    if (type == "SmoothUnion") {
      req(j, "left", type);
      req(j, "right", type);
      req(j, "smoothFactor", type);
      return std::make_shared<Geometry::SmoothUnion>(
          parseNode(j["left"]), parseNode(j["right"]),
          j["smoothFactor"].get<float>());
    }

    // 3D
    if (type == "Sphere") {
      req(j, "position", type);
      req(j, "radius", type);
      return std::make_shared<Geometry::Sphere>(
          simd::float3{j["position"][0], j["position"][1], j["position"][2]},
          j["radius"].get<float>());
    }
    if (type == "Box") {
      req(j, "position", type);
      req(j, "bounds", type);
      return std::make_shared<Geometry::Box>(
          simd::float3{j["position"][0], j["position"][1], j["position"][2]},
          simd::float3{j["bounds"][0], j["bounds"][1], j["bounds"][2]});
    }

    // 2D
    if (type == "Circle2D") {
      req(j, "position", type);
      req(j, "radius", type);
      return std::make_shared<Geometry::Circle2D>(
          simd::float3{j["position"][0], j["position"][1], 0},
          j["radius"].get<float>());
    }
    if (type == "Rect2D") {
      req(j, "position", type);
      req(j, "bounds", type);
      return std::make_shared<Geometry::Rect2D>(
          simd::float3{j["position"][0], j["position"][1], 0},
          simd::float3{j["bounds"][0], j["bounds"][1], 0});
    }
    if (type == "Bezier2D") {
      req(j, "p0", type);
      req(j, "p1", type);
      req(j, "p2", type);
      req(j, "thickness", type);
      return std::make_shared<Geometry::Bezier2D>(
          simd::float2{j["p0"][0], j["p0"][1]},
          simd::float2{j["p1"][0], j["p1"][1]},
          simd::float2{j["p2"][0], j["p2"][1]}, j["thickness"].get<float>());
    }
    if (type == "CubicBezier2D") {
      req(j, "p0", type);
      req(j, "p1", type);
      req(j, "p2", type);
      req(j, "p3", type);
      req(j, "thickness", type);
      return std::make_shared<Geometry::CubicBezier2D>(
          simd::float2{j["p0"][0], j["p0"][1]},
          simd::float2{j["p1"][0], j["p1"][1]},
          simd::float2{j["p2"][0], j["p2"][1]},
          simd::float2{j["p3"][0], j["p3"][1]}, j["thickness"].get<float>());
    }

    if (type == "CompositeSpline2D") {
      req(j, "points", type);
      req(j, "thickness", type);
      const auto &pts = j["points"];
      if (!pts.is_array() || pts.size() < 2)
        throw std::runtime_error("CompositeSpline2D: au moins 2 points requis");
      std::vector<simd::float2> points;
      points.reserve(pts.size());
      for (const auto &p : pts) {
        if (!p.is_array() || p.size() < 2)
          throw std::runtime_error(
              "CompositeSpline2D: chaque point doit etre [r, y]");
        points.push_back(simd::float2{p[0].get<float>(), p[1].get<float>()});
      }
      return std::make_shared<Geometry::CompositeSpline2D>(
          std::move(points), j["thickness"].get<float>());
    }

    throw std::runtime_error("SceneParser: type inconnu: '" + type + "'");
  }
};