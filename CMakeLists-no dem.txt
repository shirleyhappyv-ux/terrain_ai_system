cmake_minimum_required(VERSION 3.16)
project(GisDynamicClient LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# 查找 Qt5 核心模块，新加了 Svg 模块
find_package(Qt5 COMPONENTS Core Gui Widgets Xml Svg REQUIRED)

# 查找 GDAL 依赖
find_package(GDAL REQUIRED)


include_directories(
    /usr/include/qgis
    ${GDAL_INCLUDE_DIR}
)
# =============================================================

# QGIS 核心动态库
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
    Qt5::Xml
    Qt5::Svg
    ${GDAL_LIBRARIES}
    ${QGIS_LIBRARIES}
)
# =============================================================