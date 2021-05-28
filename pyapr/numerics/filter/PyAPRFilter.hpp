
#ifndef PYLIBAPR_PYAPRFILTER_HPP
#define PYLIBAPR_PYAPRFILTER_HPP

#include "data_structures/APR/APR.hpp"
#include "data_containers/PyAPR.hpp"
#include "numerics/APRReconstruction.hpp"
#include "numerics/APRFilter.hpp"
#include "numerics/APRNumerics.hpp"
#include "numerics/APRStencil.hpp"

#ifdef PYAPR_USE_CUDA
#include "numerics/APRNumericsGPU.hpp"
#include "numerics/PixelNumericsGPU.hpp"
#endif

namespace py = pybind11;

template<typename T>
void get_ds_stencil_vec(py::buffer_info& stencil_buf, VectorData<T>& stencil_vec, int num_levels, bool normalize = false) {

    auto* stencil_ptr = static_cast<T*>(stencil_buf.ptr);

    PixelData<T> stencil;
    stencil.init_from_mesh(stencil_buf.shape[2], stencil_buf.shape[1], stencil_buf.shape[0], stencil_ptr); // may lead to memory issues

    APRStencil::get_downsampled_stencils(stencil, stencil_vec, num_levels, normalize);
}

template<typename T>
void get_ds_stencil_vec(py::buffer_info& stencil_buf, std::vector<PixelData<T>>& stencil_vec, int num_levels, bool normalize = false) {

    auto* stencil_ptr = static_cast<T*>(stencil_buf.ptr);

    PixelData<T> stencil;
    stencil.init_from_mesh(stencil_buf.shape[2], stencil_buf.shape[1], stencil_buf.shape[0], stencil_ptr); // may lead to memory issues

    APRStencil::get_downsampled_stencils(stencil, stencil_vec, num_levels, normalize);
}


template<typename inputType, typename stencilType>
void convolve(APR& apr, const PyParticleData<inputType>& input_parts, PyParticleData<stencilType>& output_parts,
              py::array_t<stencilType>& stencil, bool use_stencil_downsample, bool normalize_stencil, bool use_reflective_boundary) {

    auto stencil_buf = stencil.request();
    std::vector<PixelData<stencilType>> stencil_vec;
    int nlevels = use_stencil_downsample ? apr.level_max() - apr.level_min() : 1;
    get_ds_stencil_vec(stencil_buf, stencil_vec, nlevels, normalize_stencil);

    APRFilter::convolve(apr, stencil_vec, input_parts, output_parts, use_reflective_boundary);
}


template<typename inputType, typename stencilType>
void convolve_pencil(APR& apr, PyParticleData<inputType>& input_parts, PyParticleData<stencilType>& output_parts,
                     py::array_t<stencilType>& stencil, bool use_stencil_downsample, bool normalize_stencil, bool use_reflective_boundary) {

    auto stencil_buf = stencil.request();
    std::vector<PixelData<stencilType>> stencil_vec;
    int nlevels = use_stencil_downsample ? apr.level_max() - apr.level_min() : 1;
    get_ds_stencil_vec(stencil_buf, stencil_vec, nlevels, normalize_stencil);

    APRFilter::convolve_pencil(apr, stencil_vec, input_parts, output_parts, use_reflective_boundary);
}


#ifdef PYAPR_USE_CUDA

template<typename inputType, typename stencilType>
void convolve_cuda(APR& apr, PyParticleData<inputType>& input_parts, PyParticleData<stencilType>& output_parts,
                   py::array_t<stencilType>& stencil, bool use_stencil_downsample, bool normalize_stencil, bool use_reflective_boundary) {

    auto stencil_buf = stencil.request();
    int stencil_size;

    if( stencil_buf.ndim == 3 ) {
        stencil_size = stencil_buf.shape[0];
        if( ((stencil_size != 3) && (stencil_size != 5)) || (stencil_buf.shape[1] != stencil_size) || (stencil_buf.shape[2] != stencil_size) ) {
            throw std::invalid_argument("stencil must have shape (3, 3, 3) or (5, 5, 5)");
        }
    } else {
        throw std::invalid_argument("stencil must have 3 dimensions");
    }

    // copy stencil to VectorData TODO: add a copy-free option
    int num_stencil_elements = stencil_size * stencil_size * stencil_size;
    VectorData<stencilType> stencil_vec;
    stencil_vec.resize(num_stencil_elements);
    auto* stencil_ptr = static_cast<stencilType*>(stencil_buf.ptr);
    std::copy(stencil_ptr, stencil_ptr+num_stencil_elements, stencil_vec.begin());

    VectorData<stencilType> tree_data;
    auto access = apr.gpuAPRHelper();
    auto tree_access = apr.gpuTreeHelper();

    if(stencil_size == 3) {
        isotropic_convolve_333(access, tree_access, input_parts.data, output_parts.data, stencil_vec,
                               tree_data, use_reflective_boundary, use_stencil_downsample, normalize_stencil);
    } else {
        isotropic_convolve_555(access, tree_access, input_parts.data, output_parts.data, stencil_vec,
                               tree_data, use_reflective_boundary, use_stencil_downsample, normalize_stencil);
    }
}

#endif


void AddPyAPRFilter(py::module &m, const std::string &modulename) {

    auto m2 = m.def_submodule(modulename.c_str());
    m2.def("convolve", &convolve<float, float>, "Convolve an APR with a stencil",
            py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("stencil"),
            py::arg("use_stencil_downsample")=true, py::arg("normalize_stencil")=false, py::arg("use_reflective_boundary")=false);
    m2.def("convolve", &convolve<uint16_t, float>, "Convolve an APR with a stencil",
            py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("stencil"),
            py::arg("use_stencil_downsample")=true, py::arg("normalize_stencil")=false, py::arg("use_reflective_boundary")=false);

    m2.def("convolve_pencil", &convolve_pencil<float, float>, "Convolve an APR with a stencil",
            py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("stencil"),
            py::arg("use_stencil_downsample")=true, py::arg("normalize_stencil")=false, py::arg("use_reflective_boundary")=false);
    m2.def("convolve_pencil", &convolve_pencil<uint16_t, float>, "Convolve an APR with a stencil",
            py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("stencil"),
            py::arg("use_stencil_downsample")=true, py::arg("normalize_stencil")=false, py::arg("use_reflective_boundary")=false);

#ifdef PYAPR_USE_CUDA
    m2.def("convolve_cuda", &convolve_cuda<float, float>, "Filter an APR with a stencil",
            py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("stencil"),
            py::arg("use_stencil_downsample")=true, py::arg("normalize_stencil")=false, py::arg("use_reflective_boundary")=false);
    m2.def("convolve_cuda", &convolve_cuda<uint16_t, float>, "Filter an APR with a stencil",
            py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("stencil"),
            py::arg("use_stencil_downsample")=true, py::arg("normalize_stencil")=false, py::arg("use_reflective_boundary")=false);
#endif
}

#endif //PYLIBAPR_PYAPRFILTER_HPP