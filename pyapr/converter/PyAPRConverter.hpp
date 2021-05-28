//
// Created by Joel Jonsson on 21.05.19.
//
#ifndef PYLIBAPR_PYAPRCONVERTER_HPP
#define PYLIBAPR_PYAPRCONVERTER_HPP

#include "algorithm/APRConverter.hpp"
#include "data_containers/PyAPR.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template<typename T>
class PyAPRConverter : public APRConverter<T> {

public:

    PyAPRConverter() : APRConverter<T>() {}

    /**
     * set the parameters to be used during conversion.
     * @param par
     */
    void set_parameters(APRParameters &par) {
        this->par = par;
    }

    APRParameters get_parameters() {
        return this->par;
    }

    PixelData<T>& get_gradient() {
        return this->grad_temp;
    }

    PixelData<float>& get_intensity_scale() {
        return this->local_scale_temp;
    }

    PixelData<float>& get_smooth_image() {
        return this->local_scale_temp2;
    }

    /**
     * Compute an APR from a given image, given as a numpy array.
     * @param aPyAPR
     * @param img
     */
    void get_apr_py(APR& apr, py::array_t<T>& img) {

        auto buf = img.request(false);

        int y_num, x_num, z_num;

        auto shape = buf.shape;

        if(buf.ndim == 3) {
            y_num = shape[2];
            x_num = shape[1];
            z_num = shape[0];
        } else if (buf.ndim == 2) {
            y_num = shape[1];
            x_num = shape[0];
            z_num = 1;
        } else if (buf.ndim == 1) {
            y_num = shape[0];
            x_num = 1;
            z_num = 1;
        } else {
            throw std::invalid_argument("input array must be of dimension at most 3");
        }

        auto ptr = static_cast<T*>(buf.ptr);
        PixelData<T> pd;
        pd.init_from_mesh(y_num, x_num, z_num, ptr);

        this->get_apr(apr, pd);
    }

    void get_apr_step1(APR& apr, py::array_t<T>& img){

        auto buf = img.request(false);

        unsigned int y_num, x_num, z_num;

        auto shape = buf.shape;

        if(buf.ndim == 3) {

            y_num = shape[2];
            x_num = shape[1];
            z_num = shape[0];
        } else if (buf.ndim == 2) {

            y_num = shape[1];
            x_num = shape[0];
            z_num = 1;
        } else if (buf.ndim == 1) {

            y_num = shape[0];
            x_num = 1;
            z_num = 1;
        } else {
            throw std::invalid_argument("input array must be of dimension at most 3");
        }

        auto ptr = static_cast<T*>(buf.ptr);

        PixelData<T> pd;
        pd.init_from_mesh(y_num, x_num, z_num, ptr);

        APRTimer timer(this->verbose);

        timer.start_timer("Compute filtering steps of APR");

        this->get_lrf(apr, pd);

        timer.stop_timer();

    }

    void get_apr_step2(APR &apr, APRParameters& par){

        this->par = par;
        this->get_ds(apr);
    }

    static inline uint32_t asmlog_2(const uint32_t x) {
        if (x == 0) return 0;
        return (31 - __builtin_clz (x));
    }

    void get_level_slice(int z_slice, py::array_t<float>& input, APRParameters& par, APR& apr) {

        auto buf = input.request();
        auto *ptr = static_cast<float*>(buf.ptr);
        PixelData<float> level_output;
        level_output.init_from_mesh(buf.shape[1], buf.shape[0], 1, ptr); // may lead to memory issues

        const float scale_factor = pow(2, apr.level_max())/par.rel_error;

        const uint32_t l_max = apr.level_max();

        for(int x = 0; x < level_output.x_num; x++){
            for(int y = 0;y < level_output.y_num; y++){

               T _grad = this->grad_temp.at(y,x,z_slice);
               float _lis = this->local_scale_temp.at(y,x,z_slice);
               float _intensity = this->local_scale_temp2.at(y,x,z_slice);

               if(_grad < par.grad_th){
                   _grad = 0;
               }

               if(_lis < par.sigma_th){
                   _lis = par.sigma_th;
               }

               if(_intensity < par.Ip_th + this->bspline_offset){
                   _grad = 0;
               }

               auto level = std::min(asmlog_2(scale_factor*_grad/_lis), l_max);
               if(level < (l_max-1)){
                    level = l_max - 2;
               }

               level_output.at(y,x,0) = level;
            }
        }
    }
};

template<typename DataType>
void AddPyAPRConverter(pybind11::module &m, const std::string &aTypeString) {
    using converter = PyAPRConverter<DataType>;
    std::string typeStr = aTypeString + "Converter";
    py::class_<converter>(m, typeStr.c_str())
            .def(py::init())
            .def_readwrite("verbose", &converter::verbose)
            .def("get_apr", &converter::get_apr_py, "compute APR from an image (input as a numpy array)")
            .def("set_parameters", &converter::set_parameters, "set parameters")
            .def("get_parameters", &converter::get_parameters, "get parameters")
            .def("get_gradient", &converter::get_gradient, "return B-spline gradient",
                 py::return_value_policy::reference)
            .def("get_intensity_scale", &converter::get_intensity_scale, "return local intensity scale",
                 py::return_value_policy::reference)
            .def("get_smooth_image", &converter::get_smooth_image, "return B-spline smoothed image",
                 py::return_value_policy::reference)
            .def("get_level_slice", &converter::get_level_slice,
                 "gets the current level slice for the applied parameters")
            .def("get_apr_step1", &converter::get_apr_step1,
                 "Interactive APR generation")
            .def("get_apr_step2", &converter::get_apr_step2,
                 "Interactive APR generation");
}



#endif //PYLIBAPR_PYAPRCONVERTER_HPP
