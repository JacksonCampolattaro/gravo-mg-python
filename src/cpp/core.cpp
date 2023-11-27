#include "gravomg/multigrid.h"
#include "gravomg/sampling.h"
#include "gravomg/utility.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Eigen/Sparse"

namespace py = pybind11;

PYBIND11_MODULE(gravomg_bindings, m) {
    m.doc() = "Gravo MG bindings";

    py::enum_<Sampling>(m, "Sampling")
            .value("FASTDISK", FASTDISK)
            .value("RANDOM", RANDOM)
            .value("MIS", MIS);

    py::enum_<Weighting>(m, "Weighting")
            .value("BARYCENTRIC", BARYCENTRIC)
            .value("UNIFORM", UNIFORM)
            .value("INVDIST", INVDIST);

    m.def("fast_disc_sample", &GravoMG::fastDiskSample);

    m.def("construct_prolongations", &GravoMG::constructProlongations);

    m.def("maximum_delta_independent_set", &GravoMG::maximumDeltaIndependentSetWithDistances);

    m.def("maximum_delta_independent_set_with_distances", &GravoMG::maximumDeltaIndependentSetWithDistances);

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