#include <gravomg/multigrid.h>
#include <gravomg/sampling.h>

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
    m.def("to_homogenous", &GravoMG::toHomogenous);
    m.def("extract_edges", &GravoMG::extractEdges);

    // Sampling
    m.def("maximum_delta_independent_set", &GravoMG::maximumDeltaIndependentSet);
    m.def("maximum_delta_independent_set_with_distances", &GravoMG::maximumDeltaIndependentSetWithDistances);
    m.def("fast_disc_sample", &GravoMG::fastDiscSample);

    // Prolongation
    m.def("assign_parents", &GravoMG::assignParents);
    m.def("extract_coarse_edges", &GravoMG::extractCoarseEdges);
    m.def("coarse_from_mean_of_fine_children", &GravoMG::coarseFromMeanOfFineChildren);
    m.def("average_edge_length", &GravoMG::averageEdgeLength);
    m.def("construct_voronoi_triangles", &GravoMG::constructVoronoiTriangles);
    m.def("construct_prolongation", &GravoMG::constructProlongation);

//    py::class_<MultigridSolver>(m, "MultigridSolver")
//        .def(py::init<Eigen::MatrixXd, Eigen::MatrixXi, Eigen::SparseMatrix<double>, double, int, int, double, int, int, int, int, bool, Sampling, Weighting, bool>())
//        .def("solve", &MultigridSolver::solve, py::arg("lhs"), py::arg("rhs"))
//        .def("residual", &MultigridSolver::residual, py::arg("lhs"), py::arg("rhs"), py::arg("solution"), py::arg("type") = 2)
//        .def("prolongation_matrices", &MultigridSolver::prolongation_matrices)
//        .def("set_prolongation_matrices", &MultigridSolver::set_prolongation_matrices, py::arg("U"))
//        .def("sampling_indices", &MultigridSolver::sampling_indices)
//        .def("all_triangles", &MultigridSolver::all_triangles)
//        .def("nearest_source", &MultigridSolver::nearest_source);
}