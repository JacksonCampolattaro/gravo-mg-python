#include <gravomg/multigrid.h>
#include <gravomg/sampling.h>
#include <gravomg/utility.h>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Sparse>

namespace py = pybind11;

PYBIND11_MODULE(gravomg_bindings, m) {
    m.doc() = "Gravo MG bindings";

    // Types
    py::enum_<GravoMG::Weighting>(m, "Weighting")
            .value("BARYCENTRIC", GravoMG::Weighting::BARYCENTRIC)
            .value("UNIFORM", GravoMG::Weighting::UNIFORM)
            .value("INVDIST", GravoMG::Weighting::INVDIST);

    // Utility
    m.def("to_edge_distance_matrix", &GravoMG::toEdgeDistanceMatrix);
    m.def("extract_edges", &GravoMG::extractEdges);

    // Sampling
    m.def("fast_disc_sample", &GravoMG::fastDiscSample);

    // Prolongation
    m.def("assign_parents", &GravoMG::assignParents);
    m.def("average_edge_length", &GravoMG::averageEdgeLength);
    m.def("extract_coarse_edges", &GravoMG::extractCoarseEdges);
    m.def("coarse_from_mean_of_fine_children", &GravoMG::coarseFromMeanOfFineChildren);
    m.def("construct_voronoi_triangles", &GravoMG::constructVoronoiTriangles);
    m.def("construct_prolongation", &GravoMG::constructProlongation);
    m.def("projected_points", &GravoMG::projectedPoints);
}