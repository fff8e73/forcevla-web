cmake_minimum_required(VERSION 3.16)
project(forcevla_offline_qp LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(pinocchio REQUIRED)
find_package(yaml-cpp REQUIRED)
set(DAQP_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/../vendor/daqp)
add_library(compatible_daqp STATIC
  ${DAQP_SOURCE}/src/api.c ${DAQP_SOURCE}/src/auxiliary.c
  ${DAQP_SOURCE}/src/bnb.c ${DAQP_SOURCE}/src/daqp.c
  ${DAQP_SOURCE}/src/daqp_prox.c ${DAQP_SOURCE}/src/factorization.c
  ${DAQP_SOURCE}/src/hierarchical.c ${DAQP_SOURCE}/src/utils.c)
target_include_directories(compatible_daqp PUBLIC ${DAQP_SOURCE}/include)
target_compile_definitions(compatible_daqp PRIVATE PROFILING)
add_executable(offline_qp offline_qp.cpp)
target_include_directories(offline_qp PRIVATE /opt/kernelmind/apex/install/marvin_qp_controller/include)
target_link_libraries(offline_qp PRIVATE /opt/kernelmind/apex/install/marvin_qp_controller/lib/marvin_qp_controller/libqp_controller_core.a compatible_daqp pinocchio::pinocchio yaml-cpp m)
