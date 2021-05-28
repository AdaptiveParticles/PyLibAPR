//
// Created by Joel Jonsson on 07.05.20.
//

#ifndef PYLIBAPR_PYLINEARITERATOR_HPP
#define PYLIBAPR_PYLINEARITERATOR_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

#include "data_structures/APR/APR.hpp"

namespace py = pybind11;


// -------- wrapper -------------------------------------------------
void AddPyLinearIterator(pybind11::module &m, const std::string &modulename) {
    py::class_<LinearIterator>(m, modulename.c_str())
            .def(py::init())
            .def("total_number_particles", &LinearIterator::total_number_particles, "return number of particles",
                 py::arg("level")=0)
            .def("level_min", &LinearIterator::level_min, "return the minimum resolution level")
            .def("level_max", &LinearIterator::level_max, "return the maximum resolution level")
            .def("x_num", &LinearIterator::x_num,  "Gives the maximum bounds in the x direction for the given level")
            .def("y_num", &LinearIterator::y_num,  "Gives the maximum bounds in the y direction for the given level")
            .def("z_num", &LinearIterator::z_num,  "Gives the maximum bounds in the z direction for the given level")
            .def("y", &LinearIterator::get_y, "returns the y-coordinate of a given particle index")
            .def("begin", &LinearIterator::begin, "returns the index of the first particle at a given combination of level, z, and x",
                        py::arg("level"), py::arg("z"), py::arg("z"))
            .def("end", &LinearIterator::end, "returns the end index of the current row (l, z, x combination)");
}

#endif //PYLIBAPR_PYLINEARITERATOR_HPP
