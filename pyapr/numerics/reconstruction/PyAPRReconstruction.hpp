
#ifndef PYLIBAPR_PYAPRRECONSTRUCTION_HPP
#define PYLIBAPR_PYAPRRECONSTRUCTION_HPP

#include "data_containers/PyPixelData.hpp"
#include "data_containers/PyAPR.hpp"
#include "numerics/APRReconstruction.hpp"
#include "numerics/APRTreeNumerics.hpp"

namespace py = pybind11;


namespace PyAPRReconstruction {

    /**
     * Constant reconstruction (full volume)
     */
    template<typename S, typename T>
    void reconstruct_constant_inplace(APR& apr, PyParticleData<S>& parts, py::array_t<T, py::array::c_style>& arr) {

        auto buf = arr.request(true);
        auto* ptr = static_cast<T*>(buf.ptr);
        PixelData<T> recon;
        recon.init_from_mesh(apr.org_dims(0), apr.org_dims(1), apr.org_dims(2), ptr);

        const size_t size_alloc = std::accumulate(buf.shape.begin(), buf.shape.end(), 1, std::multiplies<size_t>());
        if(recon.mesh.size() != size_alloc) {
            throw std::invalid_argument("input array size does not agree with APR dimensions");
        }

        APRReconstruction::reconstruct_constant(apr, recon, parts);
    }

    /**
     * Level reconstruction (full volume)
     */
    template<typename S>
    void reconstruct_level_inplace(APR& apr, py::array_t<S, py::array::c_style>& arr) {

        auto buf = arr.request(true);
        auto* ptr = static_cast<S*>(buf.ptr);
        PixelData<S> recon;
        recon.init_from_mesh(apr.org_dims(0), apr.org_dims(1), apr.org_dims(2), ptr);

        const size_t size_alloc = std::accumulate(buf.shape.begin(), buf.shape.end(), 1, std::multiplies<size_t>());
        if(recon.mesh.size() != size_alloc) {
            throw std::invalid_argument("input array size does not agree with APR dimensions");
        }
        APRReconstruction::reconstruct_level(apr, recon);
    }

    /**
     * Smooth reconstruction (full volume)
     */
    template<typename S, typename T>
    void reconstruct_smooth_inplace(APR& apr, PyParticleData<S>& parts, py::array_t<T, py::array::c_style>& arr) {

        auto buf = arr.request(true);
        auto* ptr = static_cast<T*>(buf.ptr);
        PixelData<T> recon;
        recon.init_from_mesh(apr.org_dims(0), apr.org_dims(1), apr.org_dims(2), ptr);

        const size_t size_alloc = std::accumulate(buf.shape.begin(), buf.shape.end(), 1, std::multiplies<size_t>());
        if(recon.mesh.size() != size_alloc) {
            throw std::invalid_argument("input array size does not agree with APR dimensions");
        }

        APRReconstruction::reconstruct_smooth(apr, recon, parts);
    }

    /**
     * Constant reconstruction (patch)
     */
    template<typename S, typename T>
    void
    reconstruct_constant_patch_inplace(APR& apr, PyParticleData<S>& parts, PyParticleData<T>& tree_parts,
                                       ReconPatch& patch, py::array_t<S, py::array::c_style>& arr) {
        auto buf = arr.request(true);
        auto* ptr = static_cast<S*>(buf.ptr);
        PixelData<S> recon;

        if(buf.ndim == 3) {
            recon.init_from_mesh(buf.shape[2], buf.shape[1], buf.shape[0], ptr);
        } else if(buf.ndim == 2) {
            recon.init_from_mesh(buf.shape[1], buf.shape[0], 1, ptr);
        } else if(buf.ndim == 1) {
            recon.init_from_mesh(buf.shape[0], 1, 1, ptr);
        } else {
            throw std::invalid_argument("input array must be 1-3 dimensional");
        }

        //patch.check_limits(aPyAPR);   // assuming check_limits has been called from python to initialize array
        const size_t psize = (patch.z_end-patch.z_begin) *
                             (patch.x_end-patch.x_begin) *
                             (patch.y_end-patch.y_begin);
        if(recon.mesh.size() != psize) {
            throw std::invalid_argument("input array must agree with patch size!");
        }

        APRReconstruction::reconstruct_constant(apr, recon, parts, tree_parts, patch);
    }


    /**
     * Level reconstruction (patch)
     */
    template<typename S>
    void
    reconstruct_level_patch_inplace(APR& apr, ReconPatch& patch, py::array_t<S, py::array::c_style>& arr) {
        auto buf = arr.request(true);
        auto* ptr = static_cast<S*>(buf.ptr);
        PixelData<S> recon;

        if(buf.ndim == 3) {
            recon.init_from_mesh(buf.shape[2], buf.shape[1], buf.shape[0], ptr);
        } else if(buf.ndim == 2) {
            recon.init_from_mesh(buf.shape[1], buf.shape[0], 1, ptr);
        } else if(buf.ndim == 1) {
            recon.init_from_mesh(buf.shape[0], 1, 1, ptr);
        } else {
            throw std::invalid_argument("input array must be 1-3 dimensional");
        }

        //patch.check_limits(aPyAPR);   // assuming check_limits has been called from python to initialize array
        const size_t psize = (patch.z_end-patch.z_begin) *
                             (patch.x_end-patch.x_begin) *
                             (patch.y_end-patch.y_begin);
        if(recon.mesh.size() != psize) {
            throw std::invalid_argument("input array must agree with patch size!");
        }

        APRReconstruction::reconstruct_level(apr, recon, patch);
    }


    /**
     * smooth reconstruction (patch)
     */
    template<typename S, typename T>
    void
    reconstruct_smooth_patch_inplace(APR& apr, PyParticleData<S>& parts, PyParticleData<T>& tree_parts,
                                     ReconPatch& patch, py::array_t<S, py::array::c_style>& arr) {
        auto buf = arr.request(true);
        auto* ptr = static_cast<S*>(buf.ptr);
        PixelData<S> recon;

        if(buf.ndim == 3) {
            recon.init_from_mesh(buf.shape[2], buf.shape[1], buf.shape[0], ptr);
        } else if(buf.ndim == 2) {
            recon.init_from_mesh(buf.shape[1], buf.shape[0], 1, ptr);
        } else if(buf.ndim == 1) {
            recon.init_from_mesh(buf.shape[0], 1, 1, ptr);
        } else {
            throw std::invalid_argument("input array must be 1-3 dimensional");
        }

        //patch.check_limits(aPyAPR);   // assuming check_limits has been called from python to initialize array
        const size_t psize = (patch.z_end-patch.z_begin) *
                             (patch.x_end-patch.x_begin) *
                             (patch.y_end-patch.y_begin);
        if(recon.mesh.size() != psize) {
            throw std::invalid_argument("input array must agree with patch size!");
        }

        APRReconstruction::reconstruct_smooth(apr, recon, parts, tree_parts, patch);
    }

}

void AddPyAPRReconstruction(py::module &m, const std::string &modulename) {

    auto m2 = m.def_submodule(modulename.c_str());

    /// constant reconstruction (full volume) into preallocated numpy array
    m2.def("reconstruct_constant_inplace", &PyAPRReconstruction::reconstruct_constant_inplace<uint16_t, uint16_t>, "Piecewise constant reconstruction",
           py::arg("APR"), py::arg("parts"), py::arg("arr"));
    m2.def("reconstruct_constant_inplace", &PyAPRReconstruction::reconstruct_constant_inplace<uint16_t, float>, "Piecewise constant reconstruction",
           py::arg("APR"), py::arg("parts"), py::arg("arr"));
    m2.def("reconstruct_constant_inplace", &PyAPRReconstruction::reconstruct_constant_inplace<float, float>, "Piecewise constant reconstruction",
           py::arg("APR"), py::arg("parts"), py::arg("arr"));

    /// level reconstruction (full volume) into preallocated numpy array
    m2.def("reconstruct_level_inplace", &PyAPRReconstruction::reconstruct_level_inplace<uint8_t>, "Particle level reconstruction",
           py::arg("APR"), py::arg("arr"));
    m2.def("reconstruct_level_inplace", &PyAPRReconstruction::reconstruct_level_inplace<uint16_t>, "Particle level reconstruction",
           py::arg("APR"), py::arg("arr"));
    m2.def("reconstruct_level_inplace", &PyAPRReconstruction::reconstruct_level_inplace<float>, "Particle level reconstruction",
           py::arg("APR"), py::arg("arr"));

    /// smooth reconstruction (full volume) into preallocated numpy array
    m2.def("reconstruct_smooth_inplace", &PyAPRReconstruction::reconstruct_smooth_inplace<uint16_t, uint16_t>, "Smooth reconstruction",
           py::arg("APR"), py::arg("parts"), py::arg("arr"));
    m2.def("reconstruct_smooth_inplace", &PyAPRReconstruction::reconstruct_smooth_inplace<uint16_t, float>, "Smooth reconstruction",
           py::arg("APR"), py::arg("parts"), py::arg("arr"));
    m2.def("reconstruct_smooth_inplace", &PyAPRReconstruction::reconstruct_smooth_inplace<float, float>, "Smooth reconstruction",
           py::arg("APR"), py::arg("parts"), py::arg("arr"));


    /// constant reconstruction (patch) into preallocated numpy array
    m2.def("reconstruct_constant_patch_inplace", &PyAPRReconstruction::reconstruct_constant_patch_inplace<uint16_t, uint16_t>,
           "Piecewise constant patch reconstruction",
           py::arg("APR"), py::arg("parts"), py::arg("tree_parts"), py::arg("patch"), py::arg("arr"));
    m2.def("reconstruct_constant_patch_inplace", &PyAPRReconstruction::reconstruct_constant_patch_inplace<uint16_t, float>,
           "Piecewise constant patch reconstruction",
           py::arg("APR"), py::arg("parts"), py::arg("tree_parts"), py::arg("patch"), py::arg("arr"));
    m2.def("reconstruct_constant_patch_inplace", &PyAPRReconstruction::reconstruct_constant_patch_inplace<float, float>,
           "Piecewise constant patch reconstruction",
           py::arg("APR"), py::arg("parts"), py::arg("tree_parts"), py::arg("patch"), py::arg("arr"));

    /// level reconstruction (patch) into preallocated numpy array
    m2.def("reconstruct_level_patch_inplace", &PyAPRReconstruction::reconstruct_level_patch_inplace<uint8_t>,
           "Particle level patch reconstruction", py::arg("APR"), py::arg("patch"), py::arg("arr"));
    m2.def("reconstruct_level_patch_inplace", &PyAPRReconstruction::reconstruct_level_patch_inplace<uint16_t>,
           "Particle level patch reconstruction", py::arg("APR"), py::arg("patch"), py::arg("arr"));
    m2.def("reconstruct_level_patch_inplace", &PyAPRReconstruction::reconstruct_level_patch_inplace<float>,
           "Particle level patch reconstruction", py::arg("APR"), py::arg("patch"), py::arg("arr"));

    /// smooth reconstruction (patch) into preallocated numpy array
    m2.def("reconstruct_smooth_patch_inplace", &PyAPRReconstruction::reconstruct_smooth_patch_inplace<uint16_t, uint16_t>,
           "Smooth patch reconstruction",
           py::arg("APR"), py::arg("parts"), py::arg("tree_parts"), py::arg("patch"), py::arg("arr"));
    m2.def("reconstruct_smooth_patch_inplace", &PyAPRReconstruction::reconstruct_smooth_patch_inplace<uint16_t, float>,
           "Smooth patch reconstruction",
           py::arg("APR"), py::arg("parts"), py::arg("tree_parts"), py::arg("patch"), py::arg("arr"));
    m2.def("reconstruct_smooth_patch_inplace", &PyAPRReconstruction::reconstruct_smooth_patch_inplace<float, float>,
           "Smooth patch reconstruction",
           py::arg("APR"), py::arg("parts"), py::arg("tree_parts"), py::arg("patch"), py::arg("arr"));
}


#endif //PYLIBAPR_PYAPRRECONSTRUCTION_HPP
