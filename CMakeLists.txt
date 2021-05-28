cmake_minimum_required(VERSION 3.9)
project(PyLibAPR DESCRIPTION "Python wrappers for LibAPR")

set(CMAKE_CXX_STANDARD 14)


# APR build options:
option(PYAPR_USE_OPENMP "Use OpenMP for PyAPR functionality" ON)
option(PYAPR_USE_CUDA "Use CUDA for PyAPR functionality" OFF)
option(PYAPR_PREFER_EXTERNAL_LIBAPR "Use installed LibAPR rather than building from submodules" OFF)

##------------- Required libraries -------------##
find_package(HDF5 REQUIRED)
find_package(ZLIB REQUIRED)
find_package(TIFF REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_CFLAGS} -DHAVE_LIBTIFF")
set(CMAKE_CXX_FLAGS "${CMAKE_CXXFLAGS} -DHAVE_LIBTIFF")

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${HDF5_INCLUDE_DIRS} ${ZLIB_INCLUDE_DIRS} ${TIFF_INCLUDE_DIR})

###############################################################################
# Find or build LibAPR
###############################################################################
if(PYAPR_PREFER_EXTERNAL_LIBAPR)
    find_package(APR)
    if(APR_FOUND)
        message(STATUS "Found LibAPR - version ${APR_VERSION}")
        set(APR_LIBRARIES APR::staticLib)
    endif()
endif()

if(NOT PYAPR_PREFER_EXTERNAL_LIBAPR OR NOT APR_FOUND)
    message(STATUS "PyAPR: Building LibAPR from submodules")

    set(APR_TESTS OFF CACHE BOOL "" FORCE)
    set(APR_BUILD_STATIC_LIB ON CACHE BOOL "" FORCE)
    set(APR_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
    set(APR_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(APR_BENCHMARK OFF CACHE BOOL "" FORCE)
    set(APR_DENOISE OFF CACHE BOOL "" FORCE)
    set(APR_USE_LIBTIFF ON CACHE BOOL "" FORCE)
    set(APR_USE_OPENMP ${PYAPR_USE_OPENMP} CACHE BOOL "" FORCE)
    set(APR_USE_CUDA ${PYAPR_USE_CUDA} CACHE BOOL "" FORCE)

    if(CMAKE_COMPILER_IS_GNUCC)
        set(APR_PREFER_EXTERNAL_BLOSC OFF CACHE BOOL "" FORCE)
    else()
        set(APR_PREFER_EXTERNAL_BLOSC ON CACHE BOOL "" FORCE)
    endif()

    add_subdirectory(external/LibAPR)
    include_directories(external/LibAPR/src)
    set(APR_LIBRARIES staticLib)
endif()



###############################################################################
# Handle OpenMP
###############################################################################
find_package(OpenMP)
if(NOT OPENMP_FOUND OR NOT PYAPR_USE_OPENMP)
    message(WARNING "OpenMP support not found or disabled with current compiler. While PyAPR can compile like this, performance might not be optimal. Please see README.md for instructions.")
    set(OPENMP_LINK "")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPYAPR_HAVE_OPENMP -DHAVE_OPENMP ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPYAPR_HAVE_OPENMP -DHAVE_OPENMP ${OpenMP_CXX_FLAGS}")
    set(OPENMP_LINK "OpenMP::OpenMP_CXX")
endif()


###############################################################################
# Handle CUDA
###############################################################################
if(PYAPR_USE_CUDA)
    message(STATUS "APR: Building CUDA for PYAPR")
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread -lineinfo -Xptxas -O3,-v --use_fast_math -DPYAPR_USE_CUDA -DAPR_USE_CUDA")

    enable_language(CUDA)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPYAPR_USE_CUDA -DAPR_USE_CUDA")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPYAPR_USE_CUDA -DAPR_USE_CUDA")
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

endif()

###############################################################################
# Configure compiler options
###############################################################################
# If you ever want to compile with Intel's icc (or any other compiler) provide
# compiler names/paths in cmake command like this:
# CC="icc" CXX="icc" CXXFLAGS="-O3" cmake -DAPR_TESTS=1

if(WIN32)

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DWIN_COMPILE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWIN_COMPILE")

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(STATUS "Compiling on windows with CLANG!")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -fcxx-exceptions")
        set(CMAKE_CXX_FLAGS_DEBUG "/MD /Z7")
        set(CMAKE_CXX_FLAGS_RELEASE "/MD /EHsc /std:c++17 /arch:AVX2 -Xclang -O3 /nologo /fp:fast") #-flto=thin -march=native /O2 /Ob2
    endif()

    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        message(STATUS "Compiling on windows with MSVC!")
        set(CMAKE_CXX_FLAGS_RELEASE "/MD /EHsc /std:c++17 /arch:AVX2 /O2 /Ob2 /nologo /fp:fast")
        set(CMAKE_CXX_FLAGS_DEBUG "/MD /Z7")
    endif()

else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -pedantic ")
    if(CMAKE_COMPILER_IS_GNUCC)
        set(CMAKE_CXX_FLAGS_RELEASE "-O4 -ffast-math")
        set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Bdynamic")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -ldl -lz")
        
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math")
        set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lz")
    endif()
endif()

###############################################################################
# Python wrappers
###############################################################################
message(STATUS "PyAPR: Building PYTHON wrappers")

set(PYBIND11_PYTHON_VERSION 3)

if(${CMAKE_VERSION} VERSION_LESS "3.18.0") #detection syntax changed
    find_package( PythonInterp ${PYTHON_VERSION} REQUIRED )
    find_package( PythonLibs ${PYTHON_VERSION} REQUIRED )
else()
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
endif()

add_subdirectory(external/pybind11)

add_library(maxflow STATIC
            external/maxflow-v3.04.src/graph.cpp
            external/maxflow-v3.04.src/maxflow.cpp)
set_target_properties(maxflow PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

include_directories(pyapr)
include_directories(external)

set(APR_PYTHON_MODULE_NAME _pyaprwrapper)
add_definitions(-DAPR_PYTHON_MODULE_NAME=${APR_PYTHON_MODULE_NAME})
pybind11_add_module(${APR_PYTHON_MODULE_NAME} MODULE wrappers/pythonBind.cpp)
target_include_directories(${APR_PYTHON_MODULE_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/wrappers ${CMAKE_CURRENT_SOURCE_DIR}/pyapr ${CMAKE_CURRENT_SOURCE_DIR}/external)
target_link_libraries(${APR_PYTHON_MODULE_NAME} PRIVATE ${APR_LIBRARIES} pybind11::module ${HDF5_LIBRARIES} ${TIFF_LIBRARIES} ${BLOSC_LIBRARIES} ${ZLIB_LIBRARIES} maxflow ${OPENMP_LINK})
set_target_properties(${APR_PYTHON_MODULE_NAME} PROPERTIES OUTPUT_NAME ${APR_PYTHON_MODULE_NAME})
set_target_properties(${APR_PYTHON_MODULE_NAME} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")
