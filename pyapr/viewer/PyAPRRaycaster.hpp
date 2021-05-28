
#ifndef PYLIBAPR_PYAPRRAYCASTER_HPP
#define PYLIBAPR_PYAPRRAYCASTER_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "numerics/APRRaycaster.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace py = pybind11;

/**
 * @tparam T data type of particle values
 */

class PyAPRRaycaster {

    APRRaycaster apr_raycaster;

    float current_angle = 0;

    ReconPatch rp;

    float z = 0;


public:

    PyAPRRaycaster() {
        rp.level_delta = 0;
    }

    void set_verbose(bool verboseMode) { apr_raycaster.verbose = verboseMode; }

    void set_angle(float angle){
        current_angle = angle;
    }

    void set_phi(float angle){
        //apr_raycaster.phi = -3.14/2 + fmod(angle-3.14f/2.0f,(3.14f));
        apr_raycaster.phi = angle;
    }

    void increment_angle(float angle){
        current_angle += angle;
    }

    void increment_phi(float angle){
        // this is required due to the fixed domain of phi (from -pi/2 to pi/2) I used a cos instead of sin.

        //float curr_x =
        //float curr_y =
        //float curr_x =

        float new_angle = apr_raycaster.phi_s + angle;
        if(new_angle > M_PI/2){
            double diff = new_angle - M_PI/2;
            diff = std::min(diff, M_PI);
            apr_raycaster.phi_s = -M_PI/2 + diff;
            apr_raycaster.phi += angle;
            //current_angle += 3.14;
        } else if (new_angle < -M_PI/2){

            double diff = -new_angle - M_PI/2;
            diff = std::min(diff, M_PI);

            apr_raycaster.phi = M_PI/2 - diff;

            apr_raycaster.phi_s =  M_PI/2 - diff;
            apr_raycaster.phi += angle;

            //current_angle += 3.14;
        } else {
            apr_raycaster.phi_s +=  angle;
            apr_raycaster.phi += angle;
        }

       // apr_raycaster.phi = new_angle;
    }

    void set_level_delta(int level_delta){
        this->rp.level_delta = level_delta;
    }

    void set_z_anisotropy(float aniso){
        apr_raycaster.scale_z = aniso;
    }

    void set_radius(float radius){
        apr_raycaster.radius_factor = radius;
    }

    float get_radius(){
        return apr_raycaster.radius_factor;
    }


    void get_view(APR& apr, PyParticleData<uint16_t>& particles, PyParticleData<float>& particles_tree, py::array_t<uint16_t>& input) {

        py::gil_scoped_acquire acquire;

        auto buf = input.request();

        auto *ptr = static_cast<uint16_t*>(buf.ptr);

        PixelData<uint16_t> input_img;

//        std::cout << apr_raycaster.phi << std::endl;
//        std::cout << current_angle << std::endl;

        input_img.init_from_mesh(buf.shape[0], buf.shape[1], 1, ptr); // may lead to memory issues

        apr_raycaster.theta_0 = current_angle; //start
        apr_raycaster.theta_final = 0; //stop radians
        apr_raycaster.theta_delta = (apr_raycaster.theta_final - apr_raycaster.theta_0); //steps

        apr_raycaster.scale_down = pow(2,rp.level_delta);

        apr_raycaster.perform_raycast_patch(apr,particles,particles_tree,input_img,rp,[] (const uint16_t& a,const uint16_t& b) {return std::max(a,b);});

        py::gil_scoped_release release;

        //apr_raycaster.perform_raycast(aPyAPR.apr,particles.parts,input_img,[] (const uint16_t& a,const uint16_t& b) {return std::max(a,b);});
    }


};

void AddPyAPRRaycaster(pybind11::module &m,  const std::string &modulename) {

     py::class_<PyAPRRaycaster>(m, modulename.c_str())
        .def(py::init())
        .def("set_verbose", &PyAPRRaycaster::set_verbose, "set verbose mode for projection timer")
        .def("set_angle",&PyAPRRaycaster::set_angle, "demo")
        .def("set_level_delta",&PyAPRRaycaster::set_level_delta, "demo")
        .def("set_z_anisotropy",&PyAPRRaycaster::set_z_anisotropy, "demo")
        .def("set_radius",&PyAPRRaycaster::set_radius, "demo")
        .def("set_phi",&PyAPRRaycaster::set_phi, "demo")
        .def("get_radius",&PyAPRRaycaster::get_radius, "demo")
        .def("increment_phi",&PyAPRRaycaster::increment_phi, "demo")
        .def("increment_angle",&PyAPRRaycaster::increment_angle, "demo")
        .def("get_view", &PyAPRRaycaster::get_view, "demo");
}

#endif //PYLIBAPR_PYAPRRAYCASTER_HPP
