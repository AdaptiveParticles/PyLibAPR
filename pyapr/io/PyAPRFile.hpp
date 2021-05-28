//
// Created by Joel Jonsson on 22.05.19.
//

#ifndef PYLIBAPR_PYAPRFILE_HPP
#define PYLIBAPR_PYAPRFILE_HPP

#include "io/APRFile.hpp"
#include "data_containers/PyAPR.hpp"
#include "data_containers/PyParticleData.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

class PyAPRFile : public APRFile {
public:
    PyAPRFile() : APRFile() {}

    template<typename DataType>
    bool read_particles_py(APR& apr, std::string particles_name, PyParticleData<DataType>& particles,
                           bool apr_or_tree = true, uint64_t t = 0, std::string channel_name = "t") {
        return this->read_particles(apr, particles_name, particles, apr_or_tree, t, channel_name);
    }
};

// -------- wrapper -------------------------------------------------
void AddPyAPRFile(pybind11::module &m, const std::string &modulename) {

    py::class_<APRFile>(m, "APRFile_CPP");

    py::class_<PyAPRFile, APRFile>(m, modulename.c_str())
            .def(py::init())
            .def("open", &PyAPRFile::open, py::arg("file_name"), py::arg("read_write")="WRITE", "open a file for reading and/or writing")
            .def("close", &PyAPRFile::close, "close the file")
            .def("set_read_write_tree", &PyAPRFile::set_read_write_tree, "set whether the interior APRTree access should also be written and read.")
            .def("set_write_linear_flag", &PyAPRFile::set_write_linear_flag, "write linear access structure?")

            .def("get_particles_names", &PyAPRFile::get_particles_names, "return list of field names for stored particle values",
                 py::arg("apr_or_tree")=true, py::arg("t")=0, py::arg("channel_name")="t")
            .def("get_channel_names", &PyAPRFile::get_channel_names, "return list of channel names")

            .def("write_apr", &PyAPRFile::write_apr, py::arg("apr"), py::arg("t")=0, py::arg("channel_name")="t", "write apr to file")
            .def("write_apr_append", &PyAPRFile::write_apr_append, "write the APR to file and append it as the next time point")

            .def("read_apr", &PyAPRFile::read_apr, py::arg("apr"), py::arg("t")=0, py::arg("channel_name")="t", "read an APR from file")

            .def("write_particles", &PyAPRFile::write_particles<uint8_t>, "write particles to file",
                 py::arg("particles_name"), py::arg("particles"), py::arg("apr_or_tree")=true, py::arg("t")=0, py::arg("channel_name")="t")
            .def("write_particles", &PyAPRFile::write_particles<uint16_t>, "write particles to file",
                 py::arg("particles_name"), py::arg("particles"), py::arg("apr_or_tree")=true, py::arg("t")=0, py::arg("channel_name")="t")
            .def("write_particles", &PyAPRFile::write_particles<float>, "write particles to file",
                 py::arg("particles_name"), py::arg("particles"), py::arg("apr_or_tree")=true, py::arg("t")=0, py::arg("channel_name")="t")

            .def("read_particles", &PyAPRFile::read_particles_py<uint8_t>, "read particles from file",
                 py::arg("apr"), py::arg("particles_name"), py::arg("particles"), py::arg("apr_or_tree")=true, py::arg("t")=0, py::arg("channel_name")="t")
            .def("read_particles", &PyAPRFile::read_particles_py<uint16_t>, "read particles from file",
                 py::arg("apr"), py::arg("particles_name"), py::arg("particles"), py::arg("apr_or_tree")=true, py::arg("t")=0, py::arg("channel_name")="t")
            .def("read_particles", &PyAPRFile::read_particles_py<float>, "read particles from file",
                 py::arg("apr"), py::arg("particles_name"), py::arg("particles"), py::arg("apr_or_tree")=true, py::arg("t")=0, py::arg("channel_name")="t")

            .def("current_file_size_GB", &PyAPRFile::current_file_size_GB, "get current file size in GB")
            .def("current_file_size_MB", &PyAPRFile::current_file_size_MB, "get current file size in MB");
}

#endif //PYLIBAPR_PYAPRFILE_HPP
