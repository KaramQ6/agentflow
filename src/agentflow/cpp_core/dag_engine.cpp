#include "dag_engine.h"

#include <algorithm>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace agentflow {
namespace cpp_core {

std::vector<std::vector<std::string>> topological_sort_levels(
    const std::vector<std::string>& nodes,
    const std::vector<std::pair<std::string, std::string>>& edges) {

    std::unordered_map<std::string, int> in_degree;
    std::unordered_map<std::string, std::vector<std::string>> dependents;
    for (const auto& n : nodes) {
        in_degree[n] = 0;
        dependents[n] = {};
    }

    for (const auto& [from, to] : edges) {
        if (in_degree.find(from) == in_degree.end() || in_degree.find(to) == in_degree.end()) {
            throw std::invalid_argument("Edge references unknown node: " + from + " -> " + to);
        }
        in_degree[to]++;
        dependents[from].push_back(to);
    }

    std::unordered_map<std::string, int> index_of;
    for (size_t i = 0; i < nodes.size(); ++i) {
        index_of[nodes[i]] = static_cast<int>(i);
    }

    std::queue<std::string> q;
    for (const auto& n : nodes) {
        if (in_degree[n] == 0) {
            q.push(n);
        }
    }

    std::vector<std::vector<std::string>> levels;
    size_t processed = 0;

    while (!q.empty()) {
        size_t level_size = q.size();
        std::vector<std::string> level;
        for (size_t i = 0; i < level_size; ++i) {
            std::string node = q.front();
            q.pop();
            level.push_back(node);
            processed++;
            for (const auto& child : dependents[node]) {
                in_degree[child]--;
                if (in_degree[child] == 0) {
                    q.push(child);
                }
            }
        }
        levels.push_back(level);
    }

    if (processed != nodes.size()) {
        throw std::runtime_error("Cycle detected in dependency graph");
    }

    return levels;
}

void execute_level_parallel(
    const std::vector<std::string>& level_nodes,
    const std::function<void(const std::string&)>& callback,
    size_t max_threads) {

    if (level_nodes.empty()) return;

    size_t num_threads = std::min(level_nodes.size(), max_threads);
    if (num_threads == 0) num_threads = 1;

    std::vector<std::thread> threads;
    std::mutex index_mutex;
    size_t next_index = 0;

    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            while (true) {
                size_t idx;
                {
                    std::lock_guard<std::mutex> lock(index_mutex);
                    if (next_index >= level_nodes.size()) break;
                    idx = next_index++;
                }
                callback(level_nodes[idx]);
            }
        });
    }

    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }
}

}  // namespace cpp_core
}  // namespace agentflow
