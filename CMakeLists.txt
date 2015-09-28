cmake_minimum_required (VERSION 2.6)
project (StereoSystem)


find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)


set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_PREFIX_PATH "/opt/qt/5.5/gcc_64")
# Widgets finds its own dependencies (QtGui and QtCore).
find_package(Qt5Widgets REQUIRED)
find_package(Qt5OpenGL REQUIRED)

#########################################################
# FIND GLUT
#########################################################
find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
add_definitions(${GLUT_DEFINITIONS})
if(NOT GLUT_FOUND)
    message(ERROR " GLUT not found!")
endif(NOT GLUT_FOUND)
#########################################################
# FIND OPENGL
#########################################################
find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})
if(NOT OPENGL_FOUND)
    message(ERROR " OPENGL not found!")
endif(NOT OPENGL_FOUND)


set(SOURCE_FILES 
	 main.cpp
	 
	 cuda/StereoMatch.cu
	 cuda/ColorConverter.cu
	 cuda/kernels/StereoMatchKernels.cu
	 cuda/kernels/ColorConverterKernels.cu
	 
	 gui/gui.cpp
	 gui/widgets/myGlWidget.cpp
	 
	 ctrl/ctrl.cpp
	 ctrl/procThread.cpp
	 
	 grabber/grabber.cpp
)

QT5_WRAP_UI(UIS_HDRS
    gui/forms/mainform.ui
)

cuda_add_executable(StereoSystem ${SOURCE_FILES} ${UIS_HDRS} )

set(TARGET StereoSystem PROPERTY CMAKE_C_FLAGS_DEBUG "-DDEBUG")

set_target_properties(StereoSystem PROPERTIES AUTOMOC TRUE)
qt5_use_modules(StereoSystem Widgets OpenGL)

target_link_libraries(StereoSystem ${QT_LIBRARIES} ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})


# copy all needed files to binary dir
file(GLOB files_needed
    mystereocalib_0.17.yml
    tsukubaleft.jpg
    tsukubaright.jpg
)
file(COPY ${files_needed} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
