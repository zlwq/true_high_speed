cmake_minimum_required(VERSION 3.21)

project(true_high_speed LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)

find_package(Qt6 6.5 REQUIRED COMPONENTS Core Gui Widgets)
set(OpenCV_DIR
    "D:/QtDeps/opencv/opencv/build"
    CACHE PATH "Directory containing OpenCVConfig.cmake"
    FORCE
)

find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)

set(Torch_DIR "D:/QTDEPS/libtorch/share/cmake/Torch" CACHE PATH
    "Directory containing TorchConfig.cmake")
find_package(Torch REQUIRED)

set(ONNXRUNTIME_ROOT
    "D:/QTDEPS/ONNXRUNTIME-WIN-X64-GPU_CUDA13-1.29.0"
    CACHE PATH "ONNX Runtime GPU package root")
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS "${ONNXRUNTIME_ROOT}/include"
    NO_DEFAULT_PATH
    REQUIRED)
find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED)

qt_standard_project_setup()

qt_add_executable(true_high_speed
    WIN32
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
    inferenceworker.cpp
    inferenceworker.h
    ringinference.cpp
    ringinference.h
    locotrackinference.cpp
    locotrackinference.h
    trackon2inference.cpp
    trackon2inference.h
    tapinference.cpp
    tapinference.h
    physicalanalysis.cpp
    physicalanalysis.h
    reviewdialog.cpp
    reviewdialog.h
    csvmetadata.h
)

target_include_directories(true_high_speed PRIVATE
    "${ONNXRUNTIME_INCLUDE_DIR}"
    "${OpenCV_INCLUDE_DIRS}"
)

target_link_libraries(true_high_speed PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    "${ONNXRUNTIME_LIBRARY}"
)

target_compile_definitions(true_high_speed PRIVATE
    THS_SOURCE_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/model"
    NOMINMAX
)

if(MSVC)
    target_compile_options(true_high_speed PRIVATE /utf-8 /bigobj)
    set_property(TARGET true_high_speed PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

if(TORCH_CXX_FLAGS)
    separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(true_high_speed PRIVATE ${TORCH_CXX_FLAGS_LIST})
endif()

# 将 LibTorch、ONNX Runtime 与 OpenCV 的运行库复制到 exe 目录。Qt 的
# DLL/插件仍由 Qt Creator 部署步骤或下方 install 部署脚本处理。
if(WIN32)
    find_program(FFMPEG_EXECUTABLE NAMES ffmpeg.exe ffmpeg)
    file(GLOB TORCH_RUNTIME_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    file(GLOB ONNXRUNTIME_DLLS "${ONNXRUNTIME_ROOT}/lib/*.dll")
    file(GLOB OPENCV_RUNTIME_DLLS
        "${OpenCV_DIR}/x64/vc16/bin/*.dll")
    add_custom_command(TARGET true_high_speed POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_RUNTIME_DLLS}
                ${ONNXRUNTIME_DLLS}
                ${OPENCV_RUNTIME_DLLS}
                "$<TARGET_FILE_DIR:true_high_speed>"
        COMMAND_EXPAND_LISTS
        VERBATIM)
    if(FFMPEG_EXECUTABLE)
        add_custom_command(TARGET true_high_speed POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${FFMPEG_EXECUTABLE}"
                    "$<TARGET_FILE_DIR:true_high_speed>/ffmpeg.exe"
            VERBATIM)
    endif()
endif()

install(TARGETS true_high_speed
    BUNDLE DESTINATION .
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)

qt_generate_deploy_app_script(
    TARGET true_high_speed
    OUTPUT_SCRIPT deploy_script
    NO_UNSUPPORTED_PLATFORM_ERROR
)
install(SCRIPT "${deploy_script}")
