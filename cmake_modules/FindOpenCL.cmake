# - Try to find OpenCL
# This module tries to find an OpenCL implementation on your system. It supports
# AMD / ATI, Apple and NVIDIA implementations, but should work, too.
#
# To set manually the paths, define these environment variables:
# OpenCL_INCPATH    - Include path (e.g. OpenCL_INCPATH=/opt/cuda/4.0/cuda/include)
# OpenCL_LIBPATH    - Library path (e.h. OpenCL_LIBPATH=/usr/lib64/nvidia)
#
# Once done this will define
#  OPENCL_FOUND        - system has OpenCL
#  OPENCL_INCLUDE_DIRS - the OpenCL include directory
#  OPENCL_LIB_DIR      - directory in which OpenCL link libs are located
#  OPENCL_LIBRARIES    - link these to use OpenCL
#  OPENCL_HAS_CPP_BINDINGS - cl.hpp found
#  OPENCL_VENDOR - defined on platforms other than OS X.  set this value if you have
#                  the NVidia and AMD/ATI SDKs installed on platforms other than OS X
#                  in order to cause auto-detection to look for only one or the other

FIND_PACKAGE(PackageHandleStandardArgs)

SET (OPENCL_VERSION_STRING "0.1.0")
SET (OPENCL_VERSION_MAJOR 0)
SET (OPENCL_VERSION_MINOR 1)
SET (OPENCL_VERSION_PATCH 0)

IF (APPLE)

    FIND_LIBRARY(OPENCL_LIBRARIES OpenCL DOC "OpenCL lib for OSX")
    FIND_PATH(OPENCL_INCLUDE_DIRS OpenCL/cl.h DOC "Include for OpenCL on OSX")
    FIND_PATH(_OPENCL_CPP_INCLUDE_DIRS OpenCL/cl.hpp DOC "Include for OpenCL CPP bindings on OSX")

ELSE (APPLE)

    IF (WIN32)

        set(OPENCL_VENDOR "default" CACHE STRING "If you have both the AMD/ATI and NVidia SDKs installed, specify either \"AMDATI\" or \"NVidia\" to force only one or the other to be detected, or, alternatively, specify OpenCL_LIBPATH and OpenCL_INCPATH.")

        if(NOT DEFINED OpenCL_INCPATH AND NOT DEFINED OpenCL_LIBPATH)
            if(${OPENCL_VENDOR} STREQUAL "default" OR ${OPENCL_VENDOR} STREQUAL "AMDATI")
                if(DEFINED ENV{ATISTREAMSDKROOT})
                    message(STATUS "Detected ATISTREAMSDKROOT environment variable ($ENV{ATISTREAMSDKROOT})")
                    set(_OPENCL_SDK_ROOT $ENV{ATISTREAMSDKROOT})
                    set(OPENCL_VENDOR "AMDATI")
                elseif(DEFINED ENV{AMDAPPSDKROOT})
                    message(STATUS "Detected AMDAPPSDKROOT environment variable ($ENV{AMDAPPSDKROOT})")
                    set(_OPENCL_SDK_ROOT $ENV{AMDAPPSDKROOT})
                    set(OPENCL_VENDOR "AMDATI")
                elseif(${OPENCL_VENDOR} STREQUAL "AMDATI")
                    message(FATAL_ERROR "Neither the AMDAPPSDKROOT nor the ATISTREAMSDKROOT environent variables are defined, which suggests the AMD/ATI SDK is not installed.  You may need to fix your environment variables or specify OpenCL_LIBPATH and OpenCL_INCPATH.")
                endif()
                if(NOT DEFINED OpenCL_LIBPATH)
                    # The AMD SDK currently installs both x86 and x86_64 libraries
                    # This is only a hack to find out architecture
                    IF( ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "AMD64" )
                        SET(OPENCL_LIB_DIR "${_OPENCL_SDK_ROOT}/lib/x86_64")
                    ELSE (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "AMD64")
                        SET(OPENCL_LIB_DIR "${_OPENCL_SDK_ROOT}/lib/x86")
                    ENDIF( ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "AMD64" )
                endif()
             endif()
             if((NOT DEFINED _OPENCL_SDK_ROOT AND ${OPENCL_VENDOR} STREQUAL "default") OR ${OPENCL_VENDOR} STREQUAL "NVidia")
                 if(DEFINED ENV{CUDA_PATH})
                     message(STATUS "Detected CUDA_PATH environment variable ($ENV{CUDA_PATH})")
                     set(_OPENCL_SDK_ROOT $ENV{CUDA_PATH})
                     set(OPENCL_VENDOR "NVidia")
                 elseif(${OPENCL_VENDOR} STREQUAL "NVidia")
                     message(FATAL_ERROR "The CUDA_PATH environment variable is not defined, which suggests that the NVidia SDK is not installed.  You may need to fix your environment variables or specify OpenCL_LIBPATH and OpenCL_INCPATH.")
                 endif()
                 if(NOT DEFINED OpenCL_LIBPATH)
                     # The NVidia SDK currently installs both x86 and x86_64 libraries
                     # This is only a hack to find out architecture
                     IF( ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "AMD64" )
                         SET(OPENCL_LIB_DIR "${_OPENCL_SDK_ROOT}/lib/x64")
                     ELSE (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "AMD64")
                         SET(OPENCL_LIB_DIR "${_OPENCL_SDK_ROOT}/lib/Win32")
                     ENDIF( ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "AMD64" )
                 endif()
             endif()
             if(NOT DEFINED _OPENCL_SDK_ROOT)
                 message(FATAL_ERROR "Failed to detect NVidia or ATI SDKs.  You may need to set OpenCL_INCPATH and OpenCL_LIBPATH.")
             endif()
        endif()

        FIND_LIBRARY(OPENCL_LIBRARIES OpenCL.lib PATHS ${OPENCL_LIB_DIR} ENV OpenCL_LIBPATH)

        # On Win32 search relative to the library
        FIND_PATH(OPENCL_INCLUDE_DIRS CL/cl.h PATHS "${_OPENCL_SDK_ROOT}/include" ENV OpenCL_INCPATH)
        FIND_PATH(_OPENCL_CPP_INCLUDE_DIRS CL/cl.hpp PATHS "${_OPENCL_SDK_ROOT}/include" ENV OpenCL_INCPATH)

        unset(_OPENCL_SDK_ROOT)

    ELSE (WIN32)

        # Unix style platforms
        FIND_LIBRARY(OPENCL_LIBRARIES OpenCL
            PATHS ENV LD_LIBRARY_PATH ENV OpenCL_LIBPATH
        )

        GET_FILENAME_COMPONENT(OPENCL_LIB_DIR ${OPENCL_LIBRARIES} PATH)
        GET_FILENAME_COMPONENT(_OPENCL_INC_CAND ${OPENCL_LIB_DIR}/../../include ABSOLUTE)

        # The AMD SDK currently does not place its headers
        # in /usr/include, therefore also search relative
        # to the library
        FIND_PATH(OPENCL_INCLUDE_DIRS CL/cl.h PATHS ${_OPENCL_INC_CAND} "/usr/local/cuda/include" "/opt/AMDAPP/include" ENV OpenCL_INCPATH)
        FIND_PATH(_OPENCL_CPP_INCLUDE_DIRS CL/cl.hpp PATHS ${_OPENCL_INC_CAND} "/usr/local/cuda/include" "/opt/AMDAPP/include" ENV OpenCL_INCPATH)

    ENDIF (WIN32)

ENDIF (APPLE)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(OpenCL DEFAULT_MSG OPENCL_LIBRARIES OPENCL_INCLUDE_DIRS)

IF(_OPENCL_CPP_INCLUDE_DIRS)
    SET( OPENCL_HAS_CPP_BINDINGS TRUE )
    LIST( APPEND OPENCL_INCLUDE_DIRS ${_OPENCL_CPP_INCLUDE_DIRS} )
    # This is often the same, so clean up
    LIST( REMOVE_DUPLICATES OPENCL_INCLUDE_DIRS )
ENDIF(_OPENCL_CPP_INCLUDE_DIRS)

MARK_AS_ADVANCED(
  OPENCL_INCLUDE_DIRS
)

