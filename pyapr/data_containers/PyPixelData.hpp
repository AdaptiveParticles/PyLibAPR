//
// Created by Joel Jonsson on 29.06.18.
//

#ifndef PYLIBAPR_PYPIXELDATA_HPP
#define PYLIBAPR_PYPIXELDATA_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "data_structures/Mesh/PixelData.hpp"

namespace py = pybind11;


// Currently only using this to return PixelData objects to python as arrays without copy. It could be made more
// complete with constructors and methods, but I don't think it is necessary.

template<typename DataType>
void AddPyPixelData(pybind11::module &m, const std::string &aTypeString) {
    using PixelDataType = PixelData<DataType>;
    std::string typeStr = "PixelData" + aTypeString;
    py::class_<PixelDataType>(m, typeStr.c_str(), py::buffer_protocol())
            .def(py::init())
            .def_buffer([](PixelDataType &a) -> py::buffer_info{
                return py::buffer_info(
                        a.mesh.get(),
                        sizeof(DataType),
                        py::format_descriptor<DataType>::format(),
                        3,
                        {a.z_num, a.x_num, a.y_num},
                        {sizeof(DataType) * a.x_num * a.y_num, sizeof(DataType) * a.y_num, sizeof(DataType)}
                );
            });
}

#endif //PYLIBAPR_PYPIXELDATA_HPP
