#!/bin/sh
set -x
git clone https://github.com/microsoft/vcpkg
cd vcpkg
./bootstrap-vcpkg.bat
./vcpkg.exe install blosc:x64-windows tiff:x64-windows hdf5:x64-windows szip:x64-windows


