
#ifndef PYLIBAPR_PYAPRTRANSFORM_HPP
#define PYLIBAPR_PYAPRTRANSFORM_HPP

#include "data_structures/APR/APR.hpp"

#include "data_containers/PyAPR.hpp"
#include "data_containers/PyParticleData.hpp"
#include <math.h>
#include <stdlib.h>
#include <algorithm>

namespace py = pybind11;


template<typename T>
void dilation_py(APR& apr, PyParticleData<T>& parts, bool binary=false, int radius=1) {

    if(binary) {
        dilation_binary(apr, parts, radius);
    } else {
        dilation(apr, parts, radius);
    }
}


template<typename T>
void dilation(APR& apr, ParticleData<T>& parts, const int radius) {
    auto apr_it = apr.random_iterator();
    auto neigh_it = apr.random_iterator();

    ParticleData<T> output;
    output.init(apr);

    for(int i = 0; i < radius; i++) {
        for(int level = apr.level_max(); level > apr.level_min(); --level) {

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(apr_it, neigh_it)
#endif
            for(int z = 0; z < apr_it.z_num(level); ++z) {
                for(int x = 0; x < apr_it.x_num(level); ++x) {
                    for(apr_it.begin(level, z, x); apr_it < apr_it.end(); ++apr_it) {

                        uint64_t ct_id = apr_it;
                        T val = parts[ct_id];

                        // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                        // Edges are bidirectional, so we only need the positive directions
                        for (int direction = 0; direction < 6; direction++) {
                            apr_it.find_neighbours_in_direction(direction);

                            // For each face, there can be 0-4 neighbours
                            for (int index = 0; index < apr_it.number_neighbours_in_direction(direction); ++index) {
                                if (neigh_it.set_neighbour_iterator(apr_it, direction, index)) {
                                    //will return true if there is a neighbour defined

                                    val = std::max(val, parts[neigh_it]);
                                }
                            }
                        }
                        output[ct_id] = val;
                    }
                }
            }
        }
        output.swap(parts);
    }
}


template<typename T>
void dilation_binary(APR& apr, ParticleData<T>& parts, const int radius) {
    auto apr_it = apr.random_iterator();
    auto neigh_it = apr.random_iterator();

    ParticleData<T> output;
    output.init(apr);

    for(int i = 0; i < radius; i++) {
        for(int level = apr.level_max(); level > apr.level_min(); --level) {

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(apr_it, neigh_it)
#endif
            for(int z = 0; z < apr_it.z_num(level); ++z) {
                for(int x = 0; x < apr_it.x_num(level); ++x) {
                    for(apr_it.begin(level, z, x); apr_it < apr_it.end(); ++apr_it) {

                        uint64_t ct_id = apr_it;
                        if(find_positive_neighbour(apr_it, neigh_it, parts)) {
                            output[ct_id] = 1;
                        } else {
                            output[ct_id] = parts[ct_id];
                        }
                    }
                }
            }
        }
        output.swap(parts);
    }
}


template<typename T>
void erosion_py(APR& apr, PyParticleData<T>& parts, bool binary=false, int radius=1) {

    if(binary) {
        erosion_binary(apr, parts, radius);
    } else {
        erosion(apr, parts, radius);
    }
}


template<typename T>
void erosion(APR& apr, ParticleData<T>& parts, const int radius) {
    auto apr_it = apr.random_iterator();
    auto neigh_it = apr.random_iterator();

    ParticleData<T> output;
    output.init(apr);

    for(int i = 0; i < radius; i++) {
        for(int level = apr.level_max(); level > apr.level_min(); --level) {

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(apr_it, neigh_it)
#endif
            for(int z = 0; z < apr_it.z_num(level); ++z) {
                for(int x = 0; x < apr_it.x_num(level); ++x) {
                    for(apr_it.begin(level, z, x); apr_it < apr_it.end(); ++apr_it) {

                        uint64_t ct_id = apr_it;
                        T val = parts[ct_id];

                        // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                        // Edges are bidirectional, so we only need the positive directions
                        for (int direction = 0; direction < 6; direction++) {
                            apr_it.find_neighbours_in_direction(direction);

                            // For each face, there can be 0-4 neighbours
                            for (int index = 0; index < apr_it.number_neighbours_in_direction(direction); ++index) {
                                if (neigh_it.set_neighbour_iterator(apr_it, direction, index)) {
                                    //will return true if there is a neighbour defined

                                    val = std::min(val, parts[neigh_it]);
                                }
                            }
                        }
                        output[ct_id] = val;
                    }
                }
            }
        }
        output.swap(parts);
    }
}


template<typename T>
void erosion_binary(APR& apr, ParticleData<T>& parts, const int radius) {
    auto apr_it = apr.random_iterator();
    auto neigh_it = apr.random_iterator();

    ParticleData<T> output;
    output.init(apr);

    for(int i = 0; i < radius; i++) {
        for(int level = apr.level_max(); level > apr.level_min(); --level) {

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(apr_it, neigh_it)
#endif
            for(int z = 0; z < apr_it.z_num(level); ++z) {
                for(int x = 0; x < apr_it.x_num(level); ++x) {
                    for(apr_it.begin(level, z, x); apr_it < apr_it.end(); ++apr_it) {

                        uint64_t ct_id = apr_it;
                        if(find_zero_neighbour(apr_it, neigh_it, parts)) {
                            output[ct_id] = 0;
                        } else {
                            output[ct_id] = parts[ct_id];
                        }
                    }
                }
            }
        }
        output.swap(parts);
    }
}


template<typename APRIteratorType, typename NeighbourIteratorType, typename T>
inline bool find_zero_neighbour(APRIteratorType& apr_it, NeighbourIteratorType& neigh_it, ParticleData<T>& parts) {
    // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
    // Edges are bidirectional, so we only need the positive directions
    for (int direction = 0; direction < 6; direction++) {
        apr_it.find_neighbours_in_direction(direction);

        // For each face, there can be 0-4 neighbours
        for (int index = 0; index < apr_it.number_neighbours_in_direction(direction); ++index) {
            if (neigh_it.set_neighbour_iterator(apr_it, direction, index)) {
                //will return true if there is a neighbour defined
                if(!parts[neigh_it]) {
                    return true;
                }
            }
        }
    }
    return false;
}


template<typename APRIteratorType, typename NeighbourIteratorType, typename T>
inline bool find_positive_neighbour(APRIteratorType& apr_it, NeighbourIteratorType& neigh_it, ParticleData<T>& parts) {
    // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
    // Edges are bidirectional, so we only need the positive directions
    for (int direction = 0; direction < 6; direction++) {
        apr_it.find_neighbours_in_direction(direction);

        // For each face, there can be 0-4 neighbours
        for (int index = 0; index < apr_it.number_neighbours_in_direction(direction); ++index) {
            if (neigh_it.set_neighbour_iterator(apr_it, direction, index)) {
                //will return true if there is a neighbour defined
                if(parts[neigh_it]) {
                    return true;
                }
            }
        }
    }
    return false;
}



template<typename T>
void find_perimeter(APR& apr, PyParticleData<T>& parts, PyParticleData<T>& perimeter) {
    auto apr_it = apr.random_iterator();
    auto neigh_it = apr.random_iterator();

    perimeter.init(apr);  //values are initialized to 0

    for(int level = apr.level_max(); level > apr.level_min(); --level) {

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(apr_it, neigh_it)
#endif
        for(int z = 0; z < apr_it.z_num(level); ++z) {
            for(int x = 0; x < apr_it.x_num(level); ++x) {
                for(apr_it.begin(level, z, x); apr_it < apr_it.end(); ++apr_it) {

                    uint64_t ct_id = apr_it;
                    if(parts[ct_id] > 0){
                        if(find_zero_neighbour(apr_it, neigh_it, parts)) {
                            perimeter[ct_id] = 1;
                        }
                    }
                }
            }
        }
    }
}


/**
 * Remove objects smaller than a given volume (in pixels). Assumes that each object is labeled with a distinct number,
 * for example the output of connected_component.
 * @param apr
 * @param object_labels
 * @param min_volume
 */
void remove_small_objects(APR& apr, PyParticleData<uint16_t>& object_labels, const int min_volume) {
    auto max_label = object_labels.max();

    std::vector<uint16_t> bin_counts(max_label+1, 0);
    auto it = apr.iterator();
    const int ndim = it.number_dimensions();

    for(int level = it.level_min(); level <= it.level_max(); ++level) {
        const int particle_volume = std::pow(2, ndim*(it.level_max() - level));

        for(uint64_t idx = it.particles_level_begin(level); idx < it.particles_level_end(level); ++idx) {
            bin_counts[object_labels[idx]] += particle_volume;
        }
    }

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(size_t idx = 0; idx < it.total_number_particles(); ++idx) {
        if(bin_counts[object_labels[idx]] < min_volume) {
            object_labels[idx] = 0;
        }
    }
}


void find_objects(APR& apr, PyParticleData<uint16_t>& labels, py::array_t<int>& min_coords, py::array_t<int>& max_coords) {
    // assumes min_coords initialized to a >= the maximum (original) dimension and max_coords to 0
    auto minc = min_coords.mutable_unchecked<2>(); // Will throw if ndim != 2 or flags.writeable is false
    auto maxc = max_coords.mutable_unchecked<2>();

    auto it = apr.iterator();
    const int z_num = it.z_num(it.level_max());
    const int x_num = it.x_num(it.level_max());
    const int y_num = it.y_num(it.level_max());

    for(int level = it.level_min(); level <= it.level_max(); ++level) {
        const int lsize = it.level_size(level);
        for(int z = 0; z < it.z_num(level); ++z) {
            const int zl = z * lsize;
            const int zh = std::min(zl + lsize, z_num);
            for(int x = 0; x < it.x_num(level); ++x) {
                const int xl = x * lsize;
                const int xh = std::min(xl + lsize, x_num);
                for(it.begin(level, z, x); it < it.end(); ++it) {
                    if(labels[it] > 0) {
                        const int lab = labels[it];
                        const int yl = it.y() * lsize;
                        const int yh = std::min(yl + lsize, y_num);

                        minc(lab, 0) = std::min(minc(lab, 0), zl);
                        minc(lab, 1) = std::min(minc(lab, 1), xl);
                        minc(lab, 2) = std::min(minc(lab, 2), yl);

                        maxc(lab, 0) = std::max(maxc(lab, 0), zh);
                        maxc(lab, 1) = std::max(maxc(lab, 1), xh);
                        maxc(lab, 2) = std::max(maxc(lab, 2), yh);
                    }
                }
            }
        }
    }
}


template<typename T>
void maximum_projection_y(APR& apr, PyParticleData<T>& parts, py::array_t<float>& proj) {
    // assumes proj is of shape (z_num, x_num) initialized to 0
    auto mip = proj.mutable_unchecked<2>();
    auto it = apr.iterator();

    const int z_num = it.z_num(it.level_max());
    const int x_num = it.x_num(it.level_max());

    for(int level = it.level_max(); level >= it.level_min(); --level) {
        const int level_size = it.level_size(level);
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it)
#endif
        for(int z = 0; z < it.z_num(level); ++z) {
            const int z_m = z * level_size;
            for(int x = 0; x < it.x_num(level); ++x) {
                const int x_m = x * level_size;

                float row_max = std::numeric_limits<float>::min();
                for(it.begin(level, z, x); it < it.end(); ++it) {
                    row_max = row_max < parts[it] ? parts[it] : row_max;
                }

                for (int i = z_m; i < std::min(z_m + level_size, z_num); ++i) {
                    for (int j = x_m; j < std::min(x_m + level_size, x_num); ++j) {
                        mip(i, j) = (mip(i, j) < row_max) ? row_max : mip(i, j);
                    }
                }
            }
        }
    }
}


template<typename T>
void maximum_projection_x(APR& apr, PyParticleData<T>& parts, py::array_t<float>& proj) {
    // assumes proj is of shape (z_num, y_num) initialized to 0
    auto mip = proj.mutable_unchecked<2>();
    auto it = apr.iterator();

    const int z_num = it.z_num(it.level_max());
    const int y_num = it.y_num(it.level_max());

    for(int level = it.level_max(); level >= it.level_min(); --level) {
        const int level_size = it.level_size(level);
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it)
#endif
        for(int z = 0; z < it.z_num(level); ++z) {
            const int z_m = z * level_size;
            for(int x = 0; x < it.x_num(level); ++x) {
                for(it.begin(level, z, x); it < it.end(); ++it) {

                    const int y_m = it.y() * level_size;
                    for (int i = z_m; i < std::min(z_m + level_size, z_num); ++i) {
                        for (int j = y_m; j < std::min(y_m + level_size, y_num); ++j) {
                            mip(i, j) = (mip(i, j) < parts[it]) ? parts[it] : mip(i, j);
                        }
                    }
                }
            }
        }
    }
}


template<typename T>
void maximum_projection_z(APR& apr, PyParticleData<T>& parts, py::array_t<float>& proj) {
    // assumes proj is of shape (z_num, y_num) initialized to 0
    auto mip = proj.mutable_unchecked<2>();
    auto it = apr.iterator();

    const int x_num = it.x_num(it.level_max());
    const int y_num = it.y_num(it.level_max());

    for(int level = it.level_max(); level >= it.level_min(); --level) {
        const int level_size = it.level_size(level);
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it)
#endif
        for(int x = 0; x < it.x_num(level); ++x) {  // swapped x and z loops to avoid race conditions
            const int x_m = x * level_size;
            for(int z = 0; z < it.z_num(level); ++z) {
                for(it.begin(level, z, x); it < it.end(); ++it) {

                    const int y_m = it.y() * level_size;
                    for (int i = x_m; i < std::min(x_m + level_size, x_num); ++i) {
                        for (int j = y_m; j < std::min(y_m + level_size, y_num); ++j) {
                            mip(i, j) = (mip(i, j) < parts[it]) ? parts[it] : mip(i, j);
                        }
                    }
                }
            }
        }
    }
}


/// maximum projections in a subregion of the image, specified by a ReconPatch struct

template<typename T>
void maximum_projection_y_patch(APR& apr, PyParticleData<T>& parts, py::array_t<float>& proj, ReconPatch& patch) {
    // assumes proj is of shape (z_num, x_num) initialized to 0
    auto mip = proj.mutable_unchecked<2>();
    auto it = apr.iterator();

    int tmp = patch.level_delta;
    patch.level_delta = 0;
    patch.check_limits(apr);
    patch.level_delta = tmp;

    for(int level = it.level_max(); level > it.level_min(); --level) {
        const int level_size = it.level_size(level);

        const int z_begin_l = patch.z_begin / level_size;
        const int x_begin_l = patch.x_begin / level_size;
        const int y_begin_l = patch.y_begin / level_size;
        const int z_end_l = (patch.z_end + level_size - 1) / level_size;
        const int x_end_l = (patch.x_end + level_size - 1) / level_size;
        const int y_end_l = (patch.y_end + level_size - 1) / level_size;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it)
#endif
        for(int z = z_begin_l; z < z_end_l; ++z) {
            const int zp_begin = std::max(z * level_size, patch.z_begin) - patch.z_begin;
            const int zp_end = std::min((z+1) * level_size, patch.z_end) - patch.z_begin;

            for(int x = x_begin_l; x < x_end_l; ++x) {

                const int xp_begin = std::max(x * level_size, patch.x_begin) - patch.x_begin;
                const int xp_end = std::min((x+1) * level_size, patch.x_end) - patch.x_begin;

                // find start of y region
                it.begin(level, z, x);
                while(it.y() < y_begin_l && it < it.end()) { ++it; }

                // find max in y region
                float row_max = std::numeric_limits<float>::min();
                for(; (it < it.end()) && (it.y() < y_end_l); ++it) {
                    row_max = (row_max < parts[it]) ? parts[it] : row_max;
                }

                // compare to projection
                for (int i = zp_begin; i < zp_end; ++i) {
                    for (int j = xp_begin; j < xp_end; ++j) {
                        mip(i, j) = (mip(i, j) < row_max) ? row_max : mip(i, j);
                    }
                }
            }
        }
    }
}


template<typename T>
void maximum_projection_x_patch(APR& apr, PyParticleData<T>& parts, py::array_t<float>& proj, ReconPatch& patch) {
    // assumes proj is of shape (z_num, y_num) initialized to 0
    auto mip = proj.mutable_unchecked<2>();
    auto it = apr.iterator();

    int tmp = patch.level_delta;
    patch.level_delta = 0;
    patch.check_limits(apr);
    patch.level_delta = tmp;

    for(int level = it.level_max(); level > it.level_min(); --level) {
        const int level_size = it.level_size(level);

        const int z_begin_l = patch.z_begin / level_size;
        const int x_begin_l = patch.x_begin / level_size;
        const int y_begin_l = patch.y_begin / level_size;
        const int z_end_l = (patch.z_end + level_size - 1) / level_size;
        const int x_end_l = (patch.x_end + level_size - 1) / level_size;
        const int y_end_l = (patch.y_end + level_size - 1) / level_size;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it)
#endif
        for(int z = z_begin_l; z < z_end_l; ++z) {
            const int zp_begin = std::max(z * level_size, patch.z_begin) - patch.z_begin;
            const int zp_end = std::min((z+1) * level_size, patch.z_end) - patch.z_begin;

            for(int x = x_begin_l; x < x_end_l; ++x) {

                // find start of y region
                it.begin(level, z, x);
                while(it.y() < y_begin_l && it < it.end()) { ++it; }

                // iterate over y region and project
                for(; (it < it.end()) && (it.y() < y_end_l); ++it) {
                    const int yp_begin = std::max(it.y() * level_size, patch.y_begin) - patch.y_begin;
                    const int yp_end = std::min((it.y()+1) * level_size, patch.y_end) - patch.y_begin;

                    for (int i = zp_begin; i < zp_end; ++i) {
                        for (int j = yp_begin; j < yp_end; ++j) {
                            mip(i, j) = (mip(i, j) < parts[it]) ? parts[it] : mip(i, j);
                        }
                    }
                }
            }
        }
    }
}


template<typename T>
void maximum_projection_z_patch(APR& apr, PyParticleData<T>& parts, py::array_t<float>& proj, ReconPatch& patch) {
    // assumes proj is of shape (z_num, y_num) initialized to 0
    auto mip = proj.mutable_unchecked<2>();
    auto it = apr.iterator();

    int tmp = patch.level_delta;
    patch.level_delta = 0;
    patch.check_limits(apr);
    patch.level_delta = tmp;

    for(int level = it.level_max(); level > it.level_min(); --level) {
        const int level_size = it.level_size(level);

        const int z_begin_l = patch.z_begin / level_size;
        const int x_begin_l = patch.x_begin / level_size;
        const int y_begin_l = patch.y_begin / level_size;
        const int z_end_l = (patch.z_end + level_size - 1) / level_size;
        const int x_end_l = (patch.x_end + level_size - 1) / level_size;
        const int y_end_l = (patch.y_end + level_size - 1) / level_size;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it)
#endif
        for(int x = x_begin_l; x < x_end_l; ++x) {  // swapped x and z loops to avoid race conditions
            const int xp_begin = std::max(x * level_size, patch.x_begin) - patch.x_begin;
            const int xp_end = std::min((x+1) * level_size, patch.x_end) - patch.x_begin;

            for(int z = z_begin_l; z < z_end_l; ++z) {

                // find start of y region
                it.begin(level, z, x);
                while(it.y() < y_begin_l && it < it.end()) { ++it; }

                // iterate over y region and project
                for(; (it < it.end()) && (it.y() < y_end_l); ++it) {
                    const int yp_begin = std::max(it.y() * level_size, patch.y_begin) - patch.y_begin;
                    const int yp_end = std::min((it.y()+1) * level_size, patch.y_end) - patch.y_begin;

                    for (int i = xp_begin; i < xp_end; ++i) {
                        for (int j = yp_begin; j < yp_end; ++j) {
                            mip(i, j) = (mip(i, j) < parts[it]) ? parts[it] : mip(i, j);
                        }
                    }
                }
            }
        }
    }
}





void AddPyAPRTransform(py::module &m, const std::string &modulename) {

    auto m2 = m.def_submodule(modulename.c_str());
    m2.def("dilation", &dilation_py<uint16_t>,
           "computes a morphological dilation of the input. each particle takes the value of the maximum of its face-side neighbbours",
           py::arg("apr"), py::arg("parts"), py::arg("binary")=false, py::arg("radius")=1);
    m2.def("dilation", &dilation_py<float>,
           "computes a morphological dilation of the input. each particle takes the value of the maximum of its face-side neighbbours",
           py::arg("apr"), py::arg("parts"), py::arg("binary")=false, py::arg("radius")=1);
    m2.def("erosion", &erosion_py<uint16_t>,
           "computes a morphological erosion of the input. each particle takes the value of the minimum of its face-side neighbbours",
           py::arg("apr"), py::arg("parts"), py::arg("binary")=false, py::arg("radius")=1);
    m2.def("erosion", &erosion_py<float>,
           "computes a morphological erosion of the input. each particle takes the value of the minimum of its face-side neighbbours",
           py::arg("apr"), py::arg("parts"), py::arg("binary")=false, py::arg("radius")=1);

    m2.def("find_perimeter", &find_perimeter<uint16_t>, "find all positive particles with at least one zero neighbour",
           py::arg("apr"), py::arg("parts"), py::arg("perimeter"));
    m2.def("find_perimeter", &find_perimeter<float>, "find all positive particles with at least one zero neighbour",
           py::arg("apr"), py::arg("parts"), py::arg("perimeter"));

    m2.def("remove_small_objects", &remove_small_objects, py::arg("apr"), py::arg("object_labels"), py::arg("min_volume"));
    m2.def("find_objects_cpp", &find_objects, py::arg("apr"), py::arg("labels"), py::arg("min_coords").noconvert(), py::arg("max_coords").noconvert());

    /// y projection
    m2.def("max_projection_y", &maximum_projection_y<float>, "maximum projection along y axis",
            py::arg("apr"), py::arg("parts"), py::arg("proj").noconvert());
    m2.def("max_projection_y", &maximum_projection_y<uint16_t>, "maximum projection along y axis",
           py::arg("apr"), py::arg("parts"), py::arg("proj").noconvert());

    m2.def("max_projection_y", &maximum_projection_y_patch<float>, "maximum projection along y axis",
           py::arg("apr"), py::arg("parts"), py::arg("proj").noconvert(), py::arg("patch"));
    m2.def("max_projection_y", &maximum_projection_y_patch<uint16_t>, "maximum projection along y axis",
           py::arg("apr"), py::arg("parts"), py::arg("proj").noconvert(), py::arg("patch"));

    /// x projection
    m2.def("max_projection_x", &maximum_projection_x<float>, "maximum projection along x axis",
           py::arg("apr"), py::arg("parts"), py::arg("proj").noconvert());
    m2.def("max_projection_x", &maximum_projection_x<uint16_t>, "maximum projection along x axis",
           py::arg("apr"), py::arg("parts"), py::arg("proj").noconvert());

    m2.def("max_projection_x", &maximum_projection_x_patch<float>, "maximum projection along x axis",
           py::arg("apr"), py::arg("parts"), py::arg("proj").noconvert(), py::arg("patch"));
    m2.def("max_projection_x", &maximum_projection_x_patch<uint16_t>, "maximum projection along x axis",
           py::arg("apr"), py::arg("parts"), py::arg("proj").noconvert(), py::arg("patch"));

    /// z projection
    m2.def("max_projection_z", &maximum_projection_z<float>, "maximum projection along z axis",
           py::arg("apr"), py::arg("parts"), py::arg("proj").noconvert());
    m2.def("max_projection_z", &maximum_projection_z<uint16_t>, "maximum projection along z axis",
           py::arg("apr"), py::arg("parts"), py::arg("proj").noconvert());

    m2.def("max_projection_z", &maximum_projection_z_patch<float>, "maximum projection along z axis",
           py::arg("apr"), py::arg("parts"), py::arg("proj").noconvert(), py::arg("patch"));
    m2.def("max_projection_z", &maximum_projection_z_patch<uint16_t>, "maximum projection along z axis",
           py::arg("apr"), py::arg("parts"), py::arg("proj").noconvert(), py::arg("patch"));
}

#endif //PYLIBAPR_PYAPRTRANSFORM_HPP
