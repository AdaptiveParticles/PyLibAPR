
#ifndef PYLIBAPR_PYAPRNUMERICS_HPP
#define PYLIBAPR_PYAPRNUMERICS_HPP

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


template<typename inputType, typename stencilType>
void gradient(PyAPR& apr, const PyParticleData<inputType>& input_parts, PyParticleData<stencilType>& output_parts,
              int dimension, float delta, bool sobel=false) {
    if(sobel) {
        APRNumerics::gradient_sobel(apr.apr, input_parts.parts, output_parts.parts, dimension, delta);
    } else {
        APRNumerics::gradient_cfd(apr.apr, input_parts.parts, output_parts.parts, dimension, delta);
    }
}


template<typename inputType, typename stencilType>
void gradient_magnitude(PyAPR& apr, const PyParticleData<inputType>& input_parts, PyParticleData<stencilType>& output_parts,
                        const std::vector<float>& deltas = {1.0f, 1.0f, 1.0f}, bool sobel=false) {

    if(sobel) {
        APRNumerics::gradient_magnitude_sobel(apr.apr, input_parts.parts, output_parts.parts, deltas);
    } else {
        APRNumerics::gradient_magnitude_cfd(apr.apr, input_parts.parts, output_parts.parts, deltas);
    }
}


template<typename inputType, typename outputType>
void local_std(PyAPR& apr, const PyParticleData<inputType>& input_parts, PyParticleData<outputType>& output_parts,
               const std::vector<int>& size = {3, 3, 3}) {
    APRNumerics::local_std(apr.apr, input_parts.parts, output_parts.parts, size);
}



template<typename inputType, typename stencilType>
void richardson_lucy_cpu(PyAPR& apr, PyParticleData<inputType>& input_parts, PyParticleData<stencilType>& output_parts,
                            py::array_t<stencilType>& stencil, int niter, bool use_stencil_downsample, bool normalize_stencil, bool resume) {

    auto stencil_buf = stencil.request();
    auto* stencil_ptr = static_cast<stencilType*>(stencil_buf.ptr);
    PixelData<stencilType> psf;
    psf.init_from_mesh(stencil_buf.shape[2], stencil_buf.shape[1], stencil_buf.shape[0], stencil_ptr);

    APRNumerics::richardson_lucy(apr.apr, input_parts.parts, output_parts.parts, psf, niter,
                                 use_stencil_downsample, normalize_stencil, resume);
}


template<typename inputType, typename stencilType>
void richardson_lucy_tv_cpu(PyAPR& apr, PyParticleData<inputType>& input_parts, PyParticleData<stencilType>& output_parts,
                            py::array_t<stencilType>& stencil, int niter, float reg_factor, bool use_stencil_downsample,
                            bool normalize_stencil, bool resume) {

    auto stencil_buf = stencil.request();
    auto* stencil_ptr = static_cast<float*>(stencil_buf.ptr);
    PixelData<float> psf;
    psf.init_from_mesh(stencil_buf.shape[2], stencil_buf.shape[1], stencil_buf.shape[0], stencil_ptr);

    APRNumerics::richardson_lucy_tv(apr.apr, input_parts.parts, output_parts.parts, psf, niter, reg_factor,
                                    use_stencil_downsample, normalize_stencil, resume);
}


template<typename InputType, typename OutputType>
void adaptive_min(PyAPR& apr, const PyParticleData<InputType>& input_parts, PyParticleData<OutputType>& output_parts,
                  int num_tree_smooth=3, int level_delta=1, int num_part_smooth=2) {

    APRNumerics::adaptive_min(apr.apr, input_parts.parts, output_parts.parts, num_tree_smooth, level_delta, num_part_smooth);
}


template<typename InputType, typename OutputType>
void adaptive_max(PyAPR& apr, const PyParticleData<InputType>& input_parts, PyParticleData<OutputType>& output_parts,
                  int num_tree_smooth=3, int level_delta=1, int num_part_smooth=2) {

    APRNumerics::adaptive_max(apr.apr, input_parts.parts, output_parts.parts, num_tree_smooth, level_delta, num_part_smooth);
}


/**
 * Apply a filter to each particle and its face-side neighbours in a given dimension.
 */
template<typename S,typename U>
void face_neighbour_filter(PyAPR &apr, const PyParticleData<S>& input_parts, PyParticleData<U>& output_parts,
                           const std::vector<float>& filter, int dimension) {
    APRNumerics::face_neighbour_filter(apr.apr, input_parts.parts, output_parts.parts, filter, dimension);
}

/**
 * Successively apply a filter to each particle and its face-side neighbours in each dimension (y -> x -> z)
 */
template<typename S,typename U>
void seperable_face_neighbour_filter(PyAPR &apr, const PyParticleData<S>& input_parts, PyParticleData<U>& output_parts,
                                     const std::vector<float>& filter, int repeats = 1) {
    APRNumerics::seperable_face_neighbour_filter(apr.apr, input_parts.parts, output_parts.parts, filter, repeats);
}


#ifdef PYAPR_USE_CUDA

template<typename inputType, typename stencilType>
void richardson_lucy_cuda(PyAPR& apr, PyParticleData<inputType>& input_parts, PyParticleData<stencilType>& output_parts,
                          py::array_t<stencilType>& stencil, int niter, bool use_stencil_downsample, bool normalize_stencil,
                          bool resume) {

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

    // copy stencil to VectorData
    auto* stencil_ptr = static_cast<stencilType*>(stencil_buf.ptr);
    PixelData<stencilType> stencil_pd;
    stencil_pd.init_from_mesh(stencil_size, stencil_size, stencil_size, stencil_ptr);

    auto access = apr.apr.gpuAPRHelper();
    auto tree_access = apr.apr.gpuTreeHelper();

    richardson_lucy(access, tree_access, input_parts.parts.data, output_parts.parts.data, stencil_pd, niter,
                    use_stencil_downsample, normalize_stencil, resume);
}


template<typename inputType, typename stencilType>
void richardson_lucy_pixel_cuda(py::array_t<inputType> input, py::array_t<stencilType> output, py::array_t<stencilType>& stencil, int niter) {

    auto stencil_buf = stencil.request();
    auto input_buf = input.request();
    auto output_buf = output.request();

    for( int i = 0; i < 3; i++) {
        if(input_buf.shape[i] == 0 || input_buf.shape[i] != output_buf.shape[i]) {
            throw std::invalid_argument("input and output must have the same shape and 3 non-zero dimensions");
        }
    }

    int stencil_size;

    if( stencil_buf.ndim == 3 ) {
        stencil_size = stencil_buf.shape[0];
        if( ((stencil_size != 3) && (stencil_size != 5)) || (stencil_buf.shape[1] != stencil_size) || (stencil_buf.shape[2] != stencil_size) ) {
            throw std::invalid_argument("stencil must have shape (3, 3, 3) or (5, 5, 5)");
        }
    } else {
        throw std::invalid_argument("stencil must have 3 dimensions");
    }

    PixelData<inputType> input_pd;
    PixelData<stencilType> output_pd;
    PixelData<stencilType> psf_pd;

    auto* input_ptr = static_cast<inputType*>(input_buf.ptr);
    auto* output_ptr = static_cast<stencilType*>(output_buf.ptr);
    auto* psf_ptr = static_cast<stencilType*>(stencil_buf.ptr);

    input_pd.init_from_mesh(input_buf.shape[2], input_buf.shape[1], input_buf.shape[0], input_ptr);
    output_pd.init_from_mesh(output_buf.shape[2], output_buf.shape[1], output_buf.shape[0], output_ptr);
    psf_pd.init_from_mesh(stencil_size, stencil_size, stencil_size, psf_ptr);

    richardson_lucy_pixel(input_pd, output_pd, psf_pd, niter);
}

#endif


void AddPyAPRNumerics(py::module &p, const std::string &modulename) {

    auto m = p.def_submodule(modulename.c_str());

    m.def("gradient", &gradient<uint16_t, float>, "adaptive gradient filter",
          py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("dimension"),
          py::arg("delta")=1.0f, py::arg("sobel")=false);
    m.def("gradient", &gradient<float, float>, "adaptive gradient filter",
          py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("dimension"),
          py::arg("delta")=1.0f, py::arg("sobel")=false);

    m.def("gradient_magnitude", &gradient_magnitude<uint16_t, float>, "adaptive gradient magnitude",
          py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"),
          py::arg("deltas"), py::arg("sobel")=false);
    m.def("gradient_magnitude", &gradient_magnitude<float, float>, "adaptive gradient magnitude",
          py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"),
          py::arg("deltas"), py::arg("sobel")=false);

    m.def("local_std", &local_std<uint16_t, float>, "compute local standard deviation around each particle",
          py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("size"));
    m.def("local_std", &local_std<float, float>, "compute local standard deviation around each particle",
          py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("size"));

    m.def("richardson_lucy", &richardson_lucy_cpu<float, float>, "APR LR deconvolution",
          py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("stencil"), py::arg("niter"),
          py::arg("use_stencil_downsample")=true, py::arg("normalize_stencil")=false, py::arg("resume")=false);
    m.def("richardson_lucy", &richardson_lucy_cpu<uint16_t, float>, "APR LR deconvolution",
          py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("stencil"), py::arg("niter"),
          py::arg("use_stencil_downsample")=true, py::arg("normalize_stencil")=false, py::arg("resume")=false);

    m.def("richardson_lucy_tv", &richardson_lucy_tv_cpu<float, float>, "APR LR deconvolution with total variation regularization",
          py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("stencil"), py::arg("niter"),
          py::arg("reg_factor"), py::arg("use_stencil_downsample")=true, py::arg("normalize_stencil")=false, py::arg("resume")=false);
    m.def("richardson_lucy_tv", &richardson_lucy_tv_cpu<uint16_t, float>, "APR LR deconvolution with total variation regularization",
          py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("stencil"), py::arg("niter"),
          py::arg("reg_factor"), py::arg("use_stencil_downsample")=true, py::arg("normalize_stencil")=false, py::arg("resume")=false);

    m.def("adaptive_min", &adaptive_min<uint16_t, float>, "computes a smoothed local minimum at each particle location",
          py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("num_tree_smooth")=3, py::arg("level_delta")=1,
          py::arg("num_part_smooth")=2);
    m.def("adaptive_min", &adaptive_min<float, float>, "computes a smoothed local minimum at each particle location",
          py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("num_tree_smooth")=3, py::arg("level_delta")=1,
          py::arg("num_part_smooth")=2);

    m.def("adaptive_max", &adaptive_max<uint16_t, float>, "computes a smoothed local maximum at each particle location",
          py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("num_tree_smooth")=3, py::arg("level_delta")=1,
          py::arg("num_part_smooth")=2);
    m.def("adaptive_max", &adaptive_max<float, float>, "computes a smoothed local maximum at each particle location",
          py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("num_tree_smooth")=3, py::arg("level_delta")=1,
          py::arg("num_part_smooth")=2);

    m.def("face_neighbour_filter", &face_neighbour_filter<uint16_t, float>, "apply a filter to each particle and its face-side neighbours in a given dimension",
          py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("filter"), py::arg("dimension"));
    m.def("face_neighbour_filter", &face_neighbour_filter<float, float>, "apply a filter to each particle and its face-side neighbours in a given dimension",
          py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("filter"), py::arg("dimension"));

    m.def("seperable_face_neighbour_filter", &seperable_face_neighbour_filter<uint16_t, float>, "apply face_neighbour_filter separably in each dimension",
          py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("filter"), py::arg("repeats")=1);
    m.def("seperable_face_neighbour_filter", &seperable_face_neighbour_filter<float, float>, "apply face_neighbour_filter separably in each dimension",
          py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("filter"), py::arg("repeats")=1);

#ifdef PYAPR_USE_CUDA
    m.def("richardson_lucy_cuda", &richardson_lucy_cuda<float, float>, "run APR LR deconvolution on GPU",
          py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("stencil"), py::arg("niter"),
          py::arg("use_stencil_downsample")=true, py::arg("normalize_stencil")=false, py::arg("resume")=false);
    m.def("richardson_lucy_cuda", &richardson_lucy_cuda<uint16_t, float>, "run APR LR deconvolution on GPU",
          py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("stencil"), py::arg("niter"),
          py::arg("use_stencil_downsample")=true, py::arg("normalize_stencil")=false, py::arg("resume")=false);

    m.def("richardson_lucy_pixel_cuda", &richardson_lucy_pixel_cuda<float, float>, "run pixel LR deconvolution on GPU");
    m.def("richardson_lucy_pixel_cuda", &richardson_lucy_pixel_cuda<uint16_t, float>, "run pixel LR deconvolution on GPU");
#endif
}

#endif //PYLIBAPR_PYAPRNUMERICS_HPP

