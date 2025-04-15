cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(my-kernel LANGUAGES CXX CUDA)

cmake_policy(SET CMP0169 OLD)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

set(BUILD_FA3, OFF)

find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

message(STATUS "Detected CUDA_VERSION=${CUDA_VERSION}")
if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.8")
    message("CUDA_VERSION ${CUDA_VERSION} >= 12.8")
elseif ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.4")
    message("CUDA_VERSION ${CUDA_VERSION} >= 12.4")
elseif ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.1")
    message("CUDA_VERSION ${CUDA_VERSION} >= 12.1")
elseif ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "11.8")
    message("CUDA_VERSION ${CUDA_VERSION} >= 11.8")
endif()

find_package(Torch REQUIRED)
# clean Torch Flag
clear_cuda_arches(CMAKE_FLAG)

include(FetchContent)

# ccache option
option(ENABLE_CCACHE "Whether to use ccache" ON)
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND AND ENABLE_CCACHE AND DEFINED ENV{CCACHE_DIR})
    message(STATUS "Building with CCACHE enabled")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "ccache")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "ccache")
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/csrc
    ${repo-cutlass_SOURCE_DIR}/include
    ${repo-cutlass_SOURCE_DIR}/tools/util/include
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(SGL_KERNEL_CUDA_FLAGS
    "-DNDEBUG"
    "-DOPERATOR_NAMESPACE=my-kernel"
    "-O3"
    "-Xcompiler"
    "-fPIC"
    # "-gencode=arch=compute_75,code=sm_75"
    # "-gencode=arch=compute_80,code=sm_80"
    "-gencode=arch=compute_89,code=sm_89"
    "-gencode=arch=compute_90,code=sm_90"
    "-std=c++17"
    "-DFLASHINFER_ENABLE_F16"
    "-DCUTE_USE_PACKED_TUPLE=1"
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1"
    "-DCUTLASS_VERSIONS_GENERATED"
    "-DCUTLASS_TEST_LEVEL=0"
    "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1"
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0"
    "--expt-relaxed-constexpr"
    "-Xcompiler=-Wconversion"
    "-Xcompiler=-fno-strict-aliasing"
)

option(SGL_KERNEL_ENABLE_SM100A "Enable SM100A" OFF)
option(SGL_KERNEL_ENABLE_SM90A  "Enable SM90A"  OFF)
option(SGL_KERNEL_ENABLE_BF16   "Enable BF16"   ON)
option(SGL_KERNEL_ENABLE_FP8    "Enable FP8"    ON)
option(SGL_KERNEL_ENABLE_FP4    "Enable FP4"    OFF)

if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.8" OR SGL_KERNEL_ENABLE_SM100A)
    list(APPEND SGL_KERNEL_CUDA_FLAGS
        "-gencode=arch=compute_100,code=sm_100"
        "-gencode=arch=compute_100a,code=sm_100a"
    )
else()
    list(APPEND SGL_KERNEL_CUDA_FLAGS
        "-use_fast_math"
    )
endif()

if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.4" OR SGL_KERNEL_ENABLE_SM90A)
    set(BUILD_FA3 ON)
    list(APPEND SGL_KERNEL_CUDA_FLAGS
        "-gencode=arch=compute_90a,code=sm_90a"
    )
endif()

if (SGL_KERNEL_ENABLE_BF16)
    list(APPEND SGL_KERNEL_CUDA_FLAGS
        "-DFLASHINFER_ENABLE_BF16"
    )
endif()

if (SGL_KERNEL_ENABLE_FP8)
    list(APPEND SGL_KERNEL_CUDA_FLAGS
        "-DFLASHINFER_ENABLE_FP8"
        "-DFLASHINFER_ENABLE_FP8_E4M3"
        "-DFLASHINFER_ENABLE_FP8_E5M2"
    )
endif()

if (SGL_KERNEL_ENABLE_FP4)
    list(APPEND SGL_KERNEL_CUDA_FLAGS
        "-DENABLE_NVFP4=1"
    )
endif()

string(REPLACE "-D__CUDA_NO_HALF_OPERATORS__"       "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REPLACE "-D__CUDA_NO_HALF_CONVERSIONS__"     "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REPLACE "-D__CUDA_NO_BFLOAT16_CONVERSIONS__" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REPLACE "-D__CUDA_NO_HALF2_OPERATORS__"      "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

set(SOURCES
    "csrc/kernel.cu"
    "csrc/kernel_warp.cu"
    "csrc/kernel_ptx.cu"
    "csrc/kernel_naive.cu"
    "csrc/common_extension.cc"
)

Python_add_library(common_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})

target_compile_options(common_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${SGL_KERNEL_CUDA_FLAGS}>)
target_include_directories(common_ops PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES} c10 cuda cublas cublasLt)

install(TARGETS common_ops LIBRARY DESTINATION "${PROJECT_SOURCE_DIR}/python/my_kernel")