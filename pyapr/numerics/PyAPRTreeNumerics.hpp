//
// Created by joeljonsson on 04.03.21.
//

#ifndef PYLIBAPR_PYAPRTREENUMERICS_HPP
#define PYLIBAPR_PYAPRTREENUMERICS_HPP

#include "data_containers/PyAPR.hpp"
#include "numerics/APRTreeNumerics.hpp"

namespace PyAPRTreeNumerics {
    template<typename S,typename T>
    void fill_tree_mean(APR &apr, const PyParticleData<S>& particle_data, PyParticleData<T>& tree_data) {
        APRTreeNumerics::fill_tree_mean(apr, particle_data, tree_data);
    }

    template<typename S,typename T>
    void fill_tree_min(APR &apr, const PyParticleData<S>& particle_data, PyParticleData<T>& tree_data) {
        APRTreeNumerics::fill_tree_min(apr, particle_data, tree_data);
    }

    template<typename S,typename T>
    void fill_tree_max(APR &apr, const PyParticleData<S>& particle_data, PyParticleData<T>& tree_data) {
        APRTreeNumerics::fill_tree_max(apr, particle_data, tree_data);
    }
}


void AddPyAPRTreeNumerics(py::module &p, const std::string &modulename) {

    auto m = p.def_submodule(modulename.c_str());

    m.def("fill_tree_mean", &PyAPRTreeNumerics::fill_tree_mean<uint16_t, uint16_t>,
          "Compute interior tree particle values by average downsampling",
          py::arg("apr"), py::arg("particle_data"), py::arg("tree_data"));
    m.def("fill_tree_mean", &PyAPRTreeNumerics::fill_tree_mean<uint16_t, float>,
          "Compute interior tree particle values by average downsampling",
          py::arg("apr"), py::arg("particle_data"), py::arg("tree_data"));
    m.def("fill_tree_mean", &PyAPRTreeNumerics::fill_tree_mean<float, float>,
          "Compute interior tree particle values by average downsampling",
          py::arg("apr"), py::arg("particle_data"), py::arg("tree_data"));


    m.def("fill_tree_max", &PyAPRTreeNumerics::fill_tree_max<uint16_t, uint16_t>,
          "Compute interior tree particle values by max downsampling",
          py::arg("apr"), py::arg("particle_data"), py::arg("tree_data"));
    m.def("fill_tree_max", &PyAPRTreeNumerics::fill_tree_max<uint16_t, float>,
          "Compute interior tree particle values by max downsampling",
          py::arg("apr"), py::arg("particle_data"), py::arg("tree_data"));
    m.def("fill_tree_max", &PyAPRTreeNumerics::fill_tree_max<float, float>,
          "Compute interior tree particle values by max downsampling",
          py::arg("apr"), py::arg("particle_data"), py::arg("tree_data"));


    m.def("fill_tree_min", &PyAPRTreeNumerics::fill_tree_min<uint16_t, uint16_t>,
          "Compute interior tree particle values by min downsampling",
          py::arg("apr"), py::arg("particle_data"), py::arg("tree_data"));
    m.def("fill_tree_min", &PyAPRTreeNumerics::fill_tree_min<uint16_t, float>,
          "Compute interior tree particle values by min downsampling",
          py::arg("apr"), py::arg("particle_data"), py::arg("tree_data"));
    m.def("fill_tree_min", &PyAPRTreeNumerics::fill_tree_min<float, float>,
          "Compute interior tree particle values by min downsampling",
          py::arg("apr"), py::arg("particle_data"), py::arg("tree_data"));
}

#endif //PYLIBAPR_PYAPRTREENUMERICS_HPP
