cmake_minimum_required(VERSION 2.8.3)
project(bc_exploration)

set(CMAKE_BUILD_TYPE RELEASE)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y -O3 -Wall -fpic")

# find current version of python 2.7
execute_process(
        COMMAND "python2.7" -c "from distutils import sysconfig; print(sysconfig.get_python_inc())"
        OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

include_directories(
        ${PYTHON_INCLUDE_DIRS}
        "${PROJECT_SOURCE_DIR}/inc/"
        "${PROJECT_SOURCE_DIR}/../../deps/pybind11/include"
)

set(SOURCE_FILES
        src/exploration/astar.cpp
        src/exploration/collision.cpp
        src/exploration/util.cpp
        )

add_library(exploration_cpp SHARED ${PROJECT_SOURCE_DIR} ${SOURCE_FILES} src/exploration/python.cpp)
target_link_libraries(exploration_cpp ${PYTHON_LIBRARIES})
set_target_properties(exploration_cpp PROPERTIES SUFFIX ".so" PREFIX "_")
