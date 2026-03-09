#pragma once
#include "SDFNode.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class SceneParser {
public:
  static std::shared_ptr<Geometry::SDFNode>
  parseFile(const std::string &filepath) {
    std::ifstream f(filepath);
    json data = json::parse(f);
    return parseNode(data);
  }

private:
  static std::shared_ptr<Geometry::SDFNode> parseNode(const json &j) {
    std::string type = j["type"];

    if (type == "Union") {
      return std::make_shared<Geometry::Union>(parseNode(j["left"]),
                                               parseNode(j["right"]));
    } else if (type == "Subtract") {
      return std::make_shared<Geometry::Subtract>(parseNode(j["base"]),
                                                  parseNode(j["subtract"]));
    } else if (type == "Intersect") {
      return std::make_shared<Geometry::Intersect>(parseNode(j["left"]),
                                                   parseNode(j["right"]));
    } else if (type == "Circle2D") {
      simd::float3 pos = {j["position"][0], j["position"][1], 0.0f};
      float radius = j["radius"];
      return std::make_shared<Geometry::Circle2D>(pos, radius);
    } else if (type == "Rect2D") {
      simd::float3 pos = {j["position"][0], j["position"][1], 0.0f};
      simd::float3 bounds = {j["bounds"][0], j["bounds"][1], 0.0f};
      return std::make_shared<Geometry::Rect2D>(pos, bounds);
    } else if (type == "Bezier2D") {
      simd::float2 p0 = {j["p0"][0], j["p0"][1]};
      simd::float2 p1 = {j["p1"][0], j["p1"][1]};
      simd::float2 p2 = {j["p2"][0], j["p2"][1]};
      float thickness = j["thickness"];
      return std::make_shared<Geometry::Bezier2D>(p0, p1, p2, thickness);
    }

    return nullptr;
  }
};