#
# Try to find the FreeImage library and include path.
# Once done this will define
#
# FREEIMAGE_FOUND
# FREEIMAGE_INCLUDE_PATH
# FREEIMAGE_LIBRARY
# 

IF (WIN32)
	FIND_PATH( FREEIMAGE_INCLUDE_PATH FreeImage.h
        C:/FreeImage/Dist/x64
		${PROJECT_SOURCE_DIR}/extern/FreeImage
		DOC "The directory where FreeImage.h resides")
	FIND_LIBRARY( FREEIMAGE_LIBRARY
		NAMES FreeImage freeimage
		PATHS
        C:/FreeImage/Dist/x64
		${PROJECT_SOURCE_DIR}/FreeImage
		DOC "The FreeImage library")
ELSE (WIN32)
	FIND_PATH( FREEIMAGE_INCLUDE_PATH FreeImage.h
		/usr/include
		/usr/local/include
		/sw/include
		/opt/local/include
		DOC "The directory where FreeImage.h resides")
	FIND_LIBRARY( FREEIMAGE_LIBRARY
		NAMES FreeImage freeimage
		PATHS
		/usr/lib64
		/usr/lib
		/usr/local/lib64
		/usr/local/lib
		/sw/lib
		/opt/local/lib
		DOC "The FreeImage library")
ENDIF (WIN32)

SET(FREEIMAGE_LIBRARIES ${FREEIMAGE_LIBRARY})

IF (FREEIMAGE_INCLUDE_PATH AND FREEIMAGE_LIBRARY)
	SET( FREEIMAGE_FOUND TRUE CACHE BOOL "Set to TRUE if FreeImage is found, FALSE otherwise")
ELSE (FREEIMAGE_INCLUDE_PATH AND FREEIMAGE_LIBRARY)
	SET( FREEIMAGE_FOUND FALSE CACHE BOOL "Set to TRUE if FreeImage is found, FALSE otherwise")
ENDIF (FREEIMAGE_INCLUDE_PATH AND FREEIMAGE_LIBRARY)

MARK_AS_ADVANCED(
	FREEIMAGE_FOUND 
	FREEIMAGE_LIBRARY
	FREEIMAGE_LIBRARIES
	FREEIMAGE_INCLUDE_PATH)

IF (FREEIMAGE_FOUND)
    MESSAGE(STATUS "FREEIMAGE_INCLUDE_PATH = ${FREEIMAGE_INCLUDE_PATH}")
    MESSAGE(STATUS "FREEIMAGE_LIBRARY = ${FREEIMAGE_LIBRARY}")
ELSE (FREEIMAGE_FOUND)
    IF (FreeImage_FIND_REQUIRED)
      MESSAGE(FATAL_ERROR "Could not find FreeImage")
    ENDIF (FreeImage_FIND_REQUIRED)
ENDIF (FREEIMAGE_FOUND)
