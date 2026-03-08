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
    } else if (type == "Circle2D") {
      simd::float3 pos = {j["position"][0], j["position"][1], 0.0f};
      float radius = j["radius"];
      // Il te faudra créer la classe Geometry::Circle2D dans SDFNode.hpp
      return std::make_shared<Geometry::Circle2D>(pos, radius);
    } else if (type == "Rect2D") {
      simd::float3 pos = {j["position"][0], j["position"][1], 0.0f};
      simd::float3 bounds = {j["bounds"][0], j["bounds"][1], 0.0f};
      return std::make_shared<Geometry::Rect2D>(pos, bounds);
    }

    return nullptr;
  }
};