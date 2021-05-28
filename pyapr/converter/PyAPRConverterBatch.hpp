//
// Created by Joel Jonsson on 05.11.20.
//
#ifndef PYLIBAPR_PYAPRCONVERTERBATCH_HPP
#define PYLIBAPR_PYAPRCONVERTERBATCH_HPP

#include "algorithm/APRConverterBatch.hpp"
#include "data_containers/PyAPR.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template<typename T>
class PyAPRConverterBatch : public APRConverterBatch<T> {

public:

    PyAPRConverterBatch() : APRConverterBatch<T>() {}

    /**
     * Set the parameters to be used during conversion.
     * @param par
     */
    void set_parameters(APRParameters &par) { this->par = par; }

    /**
     * return current parameter set
     * @return
     */
    APRParameters get_parameters() { return this->par; }
};

template<typename DataType>
void AddPyAPRConverterBatch(pybind11::module &m, const std::string &aTypeString) {
    using converter = PyAPRConverterBatch<DataType>;
    std::string typeStr = aTypeString + "ConverterBatch";
    py::class_<converter>(m, typeStr.c_str())
            .def(py::init())
            .def_readwrite("verbose", &converter::verbose, "print timings and additional information")
            .def_readwrite("z_block_size", &converter::z_block_size, "number of z slices to process simultaneously")
            .def_readwrite("z_ghost_size", &converter::ghost_z, "number of \'ghost slices\' on each side of each block")
            .def("get_apr", &converter::get_apr, "compute APR from an image (input as a numpy array)")
            .def("set_parameters", &converter::set_parameters, "set parameters")
            .def("get_parameters", &converter::get_parameters, "get parameters");
}



#endif //PYLIBAPR_PYAPRCONVERTERBATCH_HPP
