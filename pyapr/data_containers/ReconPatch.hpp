//
// Created by joel on 01.03.21.
//

#ifndef PYLIBAPR_RECONPATCH_HPP
#define PYLIBAPR_RECONPATCH_HPP

#include "numerics/APRReconstruction.hpp"


void AddReconPatch(pybind11::module &m) {

    py::class_<ReconPatch>(m, "ReconPatch")
            .def(py::init())
            .def("__repr__", [](ReconPatch& p) {
                return "ReconPatch: z: [" + std::to_string(p.z_begin) + ", " + std::to_string(p.z_end) + ")" +
                        ", x: [" + std::to_string(p.x_begin) + ", " + std::to_string(p.x_end) + ")" +
                        ", y: [" + std::to_string(p.y_begin) + ", " + std::to_string(p.y_end) + ")" +
                        ", level_delta = " + std::to_string(p.level_delta);
            })
            .def_readwrite("z_begin", &ReconPatch::z_begin)
            .def_readwrite("z_end", &ReconPatch::z_end)
            .def_readwrite("x_begin", &ReconPatch::x_begin)
            .def_readwrite("x_end", &ReconPatch::x_end)
            .def_readwrite("y_begin", &ReconPatch::y_begin)
            .def_readwrite("y_end", &ReconPatch::y_end)
            .def_readwrite("level_delta", &ReconPatch::level_delta)
            .def("check_limits", &ReconPatch::check_limits, "check patch limits against APR dimensions")
            .def("size", &ReconPatch::size, "return the number of pixels in the patch region");
}


#endif //PYLIBAPR_RECONPATCH_HPP
