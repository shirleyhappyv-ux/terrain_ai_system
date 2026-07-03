cmake_minimum_required(VERSION 3.16)
project(GisDynamicClient LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# 包含 QGIS 核心与 GUI 头文件路径
include_directories(/usr/include/qgis)

find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)
find_package(GDAL REQUIRED)

# 显式链接 Ubuntu 系统下的 QGIS 共享动态库
set(QGIS_LIBRARIES qgis_core qgis_gui qgis_analysis)

add_executable(GisDynamicClient
    main.cpp
    mainwindow.cpp
    mainwindow.h
    custom_map_tools.cpp
    custom_map_tools.h
)

target_link_libraries(GisDynamicClient
    Qt5::Widgets
    GDAL
    ${QGIS_LIBRARIES}
)