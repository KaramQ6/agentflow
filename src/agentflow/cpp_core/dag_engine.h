#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace agentflow {
namespace cpp_core {

std::vector<std::vector<std::string>> topological_sort_levels(
    const std::vector<std::string>& nodes,
    const std::vector<std::pair<std::string, std::string>>& edges);

void execute_level_parallel(
    const std::vector<std::string>& level_nodes,
    const std::function<void(const std::string&)>& callback,
    size_t max_threads);

}  // namespace cpp_core
}  // namespace agentflow
