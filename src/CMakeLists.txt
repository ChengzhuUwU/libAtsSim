
set(CODES_DIR 
              ${CMAKE_CURRENT_SOURCE_DIR} 
              ${CMAKE_CURRENT_SOURCE_DIR}/SharedDefine
              ${CMAKE_CURRENT_SOURCE_DIR}/Utils
              ${CMAKE_CURRENT_SOURCE_DIR}/heterogeneous
)

set(SOURCES "")
set(HEADERS "")
foreach(DIR ${CODES_DIR})
    file(GLOB SOURCE ${DIR}/*.cpp ${DIR}/*.cc ${DIR}/*.metal)
    file(GLOB HEADER ${DIR}/*.h ${DIR}/*.hpp)
    list(APPEND SOURCES ${SOURCE})
    list(APPEND HEADERS ${HEADER})

    get_filename_component(DIR_NAME ${DIR} NAME)
    source_group("${DIR_NAME}" FILES ${SOURCE} ${HEADER})
endforeach()

# equal to
# file(GLOB SOURCES "*.cpp" "Utils/*.cpp" "SolverData/*.cpp" "Solver/*.cpp" "Solver/*.metal" "Render/*.cpp" "SharedDefine/*.cpp")
# file(GLOB HEADERS "*.h" "Utils/*.h" "SolverData/*.h" "Solver/*.h" "Render/*.h" "SharedDefine/*.h")


# if(APPLE)
add_library(Scheduler SHARED ${SOURCES} ${HEADERS})
# else()
# add_executable(HeteroScheduler ${SOURCES} ${HEADERS})
# endif()



if(WIN32)
set(CMAKE_PREFIX_PATH   "C:/Applications/vcpkg/installed/x64-windows") # Replace with your vcpkg path
set(TBB_DIR ${THIRDPARTY_DIR}/oneapi-tbb-2021.12.0-win/oneapi-tbb-2021.12.0/lib/cmake/tbb)
# set(NLOPT_DIR ${THIRDPARTY_DIR}/NLopt.v2.6.1.x86_64-w64-mingw32/lib/cmake/nlopt)
endif()


# find_package(OpenMP)
# find_package(embree 4 REQUIRED)
# find_package(CGAL REQUIRED)
# set(Boost_USE_STATIC_RUNTIME OFF) 
# set(Boost_USE_STATIC_LIBS OFF) 
find_package(TBB REQUIRED)
# find_package(Eigen3 REQUIRED)
# find_package(tinyobjloader CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
# find_package(spdlog REQUIRED)
# find_package(Boost REQUIRED) 


if(WIN32)
else()
# find_package(NLopt REQUIRED)
# list(APPEND EXTRA_LIBS ${NLOPT_LIBRARIES})
endif()

# set(Boost_USE_MULTITHREADED ON)  

list(APPEND EXTRA_LIBS TBB::tbb)
list(APPEND EXTRA_LIBS glm::glm)


# add_compile_definitions(__TBB_DYNAMIC_LOAD_ENABLED=0) # tbb
# add_compile_definitions(-DTBB_ENABLE_IPO=OFF) # tbb

if(APPLE)
    # 第三方库：metal
    list(APPEND EXTRA_LIBS "-framework Foundation")
    list(APPEND EXTRA_LIBS "-framework Metal")
    list(APPEND EXTRA_LIBS "-framework QuartzCore")
    list(APPEND EXTRA_LIBS "-framework AppKit")
endif()


# list(APPEND EXTRA_LIBS nanogui)

set(METAL_CPP_DIR ${THIRDPARTY_DIR}/metal\-cpp)
message(STATUS "METAL_CPP_DIR: ${METAL_CPP_DIR}")
target_link_libraries(Scheduler PUBLIC ${EXTRA_LIBS})

target_include_directories(Scheduler PUBLIC ${CODES_DIR})
target_include_directories(Scheduler PUBLIC ${PROJECT_BINARY_DIR}) # projectbinatydir为build目录

if(APPLE)
target_include_directories(Scheduler PUBLIC ${METAL_CPP_DIR})
endif()



# if(APPLE)
# add_custom_target(MetalShader 
#                   COMMAND  python3 ${CMAKE_CURRENT_SOURCE_DIR}/compile_shader_and_make_proj.py
#                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#                  )
# add_dependencies(HeteroScheduler MetalShader)
# endif()

# For xcode: cmake -G Xcode ..
