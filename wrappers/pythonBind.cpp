//
// Created by Krzysztof Gonciarz on 5/7/18.
// Modified by Joel Jonsson on 2/5/19.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <ConfigAPR.h>

#include "data_containers/PyPixelData.hpp"
#include "data_containers/PyAPR.hpp"
#include "data_containers/PyAPRParameters.hpp"
#include "data_containers/PyParticleData.hpp"
#include "data_containers/ReconPatch.hpp"
#include "data_containers/iterators/PyLinearIterator.hpp"
#include "numerics/PyAPRNumerics.hpp"
#include "numerics/PyAPRTreeNumerics.hpp"
#include "numerics/reconstruction/PyAPRReconstruction.hpp"
#include "numerics/filter/PyAPRFilter.hpp"
#include "numerics/segmentation/PyAPRSegmentation.hpp"
#include "numerics/transform/PyAPRTransform.hpp"
#include "converter/PyAPRConverter.hpp"
#include "converter/PyAPRConverterBatch.hpp"
#include "io/PyAPRFile.hpp"
#include "viewer/ViewerHelpers.hpp"
#include "viewer/PyAPRRaycaster.hpp"

namespace py = pybind11;

// -------- Check if properly configured in CMAKE -----------------------------
#ifndef APR_PYTHON_MODULE_NAME
#error "Name of APR module (python binding) is not defined!"
#endif

// -------- Definition of python module ---------------------------------------
PYBIND11_MODULE(APR_PYTHON_MODULE_NAME, m) {
    m.doc() = "python binding for LibAPR library";
    m.attr("__version__") = py::str(ConfigAPR::APR_VERSION);

    py::module data_containers = m.def_submodule("data_containers");

    //wrap the PyAPR class
    AddPyAPR(data_containers, "APR");

    //wrap the PyPixelData class for different data types
    AddPyPixelData<uint8_t>(data_containers, "Byte");
    AddPyPixelData<uint16_t>(data_containers, "Short");
    AddPyPixelData<float>(data_containers, "Float");

    // wrap the APRParameters class
    AddPyAPRParameters(data_containers);

    // wrap the PyParticleData class for different data types
    AddPyParticleData<uint8_t>(data_containers, "Byte");
    AddPyParticleData<float>(data_containers, "Float");
    AddPyParticleData<uint16_t>(data_containers, "Short");

    AddReconPatch(data_containers);

    // wrap PyLinearIterator
    AddPyLinearIterator(data_containers, "iterators");

    // wrap numerics module and submodules
    py::module numerics = m.def_submodule("numerics");
    AddPyAPRNumerics(numerics, "aprnumerics");
    AddPyAPRTreeNumerics(numerics, "treenumerics");
    AddPyAPRReconstruction(numerics, "reconstruction");
    AddPyAPRFilter(numerics, "filter");
    AddPyAPRSegmentation(numerics, "segmentation");
    AddPyAPRTransform(numerics, "transform");

    // wrap APRConverter for different data types
    py::module converter = m.def_submodule("converter");
    AddPyAPRConverter<float>(converter, "Float");
    AddPyAPRConverter<uint16_t>(converter, "Short");
    AddPyAPRConverterBatch<float>(converter, "Float");
    AddPyAPRConverterBatch<uint16_t>(converter, "Short");
    //AddPyAPRConverter<uint8_t>(converter, "Byte");  // need to fix or disable APRConverter GPU steps to include this

    // wrap APRFile
    py::module io = m.def_submodule("io");
    AddPyAPRFile(io, "APRFile");

    // wrap visualization functions
    py::module viewer = m.def_submodule("viewer");
    AddViewerHelpers(viewer,"viewerHelp");
    AddPyAPRRaycaster(viewer,"raycaster");
}