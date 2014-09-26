# - Try to find FreeImagePlus
# Once done, this will define
#
#  FreeImagePlus_FOUND - system has FreeImagePlus
#  FreeImagePlus_INCLUDE_DIRS - the FreeImagePlus include directories 
#  FreeImagePlus_LIBRARIES - link these to use FreeImagePlus

include(FindPkgMacros)
findpkg_begin(FreeImagePlus)

set(FreeImagePlus_LIBRARY_NAMES freeimageplus)
get_debug_names(FreeImagePlus_LIBRARY_NAMES)

use_pkgconfig(FreeImagePlus_PKGC freeimageplus)

find_path(FreeImagePlus_INCLUDE_DIR NAMES FreeImagePlus.h PATHS ${FreeImagePlus_PKGC_INCLUDE_DIRS})
find_library(FreeImagePlus_LIBRARY_REL NAMES ${FreeImage_LIBRARY_NAMES} PATHS ${FreeImage_PKGC_LIBRARY_DIRS})
find_library(FreeImagePlus_LIBRARY_DBG NAMES ${FreeImage_LIBRARY_NAMES_DBG} PATHS ${FreeImage_PKGC_LIBRARY_DIRS})
make_library_set(FreeImagePlus_LIBRARY)

findpkg_finish(FreeImage)

