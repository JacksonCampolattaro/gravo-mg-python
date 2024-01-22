cmake_minimum_required(VERSION 3.16)
project(gravomg_bindings)
set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

FetchContent_Declare(
        gravomg
        GIT_REPOSITORY https://github.com/JacksonCampolattaro/gravo-mg.git
        GIT_TAG main
)
FetchContent_MakeAvailable(gravomg)


# Python Bindings
pybind11_add_module(gravomg_bindings
    src/cpp/core.cpp
)

include_directories(gravomg_bindings ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp)
target_link_libraries(gravomg_bindings PRIVATE gravomg pybind11::pybind11)