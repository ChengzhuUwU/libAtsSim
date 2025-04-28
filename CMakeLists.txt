cmake_minimum_required(VERSION 3.27)

project(Target VERSION 0.2)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}  -O3 -DNDEBUG")

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    if (NOT CMAKE_CXX_FLAGS MATCHES "-march" AND NOT CMAKE_CXX_FLAGS MATCHES "-arch" AND NOT CMAKE_OSX_ARCHITECTURES)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
    endif()
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES Clang) # We are using clang-cl
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif()
endif()

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${PROJECT_BINARY_DIR}/bin)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${PROJECT_BINARY_DIR}/bin)



set(THIRDPARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ext)
set(METAL_CPP_DIR ${THIRDPARTY_DIR}/metal\-cpp)

# set(NANOGUI_DIR ${THIRDPARTY_DIR}/nanogui) # 
# add_subdirectory(${NANOGUI_DIR})



set(SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(EXAMPLE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/examples)
set(SELF_RESOURCES_PATH ${CMAKE_CURRENT_SOURCE_DIR}/resources)


configure_file(TargetConfig.h.in TargetConfig.h)

add_subdirectory(${SRC_PATH}) 

if(APPLE)
list(APPEND MAIN_LIBS Scheduler)
link_libraries(${MAIN_LIBS}) 
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
endif()


add_executable(Example1 ${EXAMPLE_PATH}/example1_heft2002.cpp)
add_executable(Example2 ${EXAMPLE_PATH}/example2_async_iter.cpp)
add_executable(Example3 ${EXAMPLE_PATH}/example3_hybrid_cpu_gpu.cpp)
# add_executable(TestHEFT2002 ${SRC_PATH}/heft_2002.cpp)
if(APPLE)
# add_executable(Simulation ${SRC_PATH}/simulation.cpp)
endif()




if(APPLE)
add_custom_target(MetalShader 
                  COMMAND  python3 ${CMAKE_CURRENT_SOURCE_DIR}/compile_shader_and_make_proj.py
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                 )
add_dependencies(Example3 MetalShader)
endif()

# For xcode: cmake -G Xcode ..
