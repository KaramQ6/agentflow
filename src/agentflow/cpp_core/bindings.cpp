#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dag_engine.h"

namespace py = pybind11;

PYBIND11_MODULE(_agentflow_cpp, m) {
    m.doc() = "C++ accelerated DAG engine for AgentFlow pipelines";

    m.def(
        "topological_sort_levels",
        [](const std::vector<std::string>& nodes,
           const std::vector<std::pair<std::string, std::string>>& edges)
            -> std::vector<std::vector<std::string>> {
            py::gil_scoped_release release;
            return agentflow::cpp_core::topological_sort_levels(nodes, edges);
        },
        py::arg("nodes"),
        py::arg("edges"),
        "Perform Kahn's topological sort and return execution levels.");

    m.def(
        "execute_level_parallel",
        [](const std::vector<std::string>& level_nodes,
           const std::function<void(const std::string&)>& callback,
           size_t max_threads) {
            py::gil_scoped_release release;
            agentflow::cpp_core::execute_level_parallel(level_nodes, callback, max_threads);
        },
        py::arg("level_nodes"),
        py::arg("callback"),
        py::arg("max_threads") = 0,
        "Execute callbacks for a set of nodes in parallel using C++ std::thread pool.");
}
