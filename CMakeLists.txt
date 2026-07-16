cmake_minimum_required(VERSION 3.16)
project(GisDynamicClient LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# 1. 查找 Qt 模块 (Core, Gui, Widgets, Xml, Svg, WebSockets)
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core Gui Widgets Xml Svg WebSockets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Gui Widgets Xml Svg WebSockets)

# 2. 查找 GDAL 与 QGIS 依赖
find_package(GDAL REQUIRED)

if(WIN32)
    list(APPEND CMAKE_PREFIX_PATH 
        "C:/OSGeo4W/apps/Qt6/lib/cmake"
        "C:/OSGeo4W/apps/Qt6"
        "C:/OSGeo4W"
    )
    include_directories(
        "C:/OSGeo4W/apps/qgis/include"
        "C:/OSGeo4W/include"
        "C:/OSGeo4W/apps/Qt6/include"
    )
    link_directories(
        "C:/OSGeo4W/apps/qgis/lib"
        "C:/OSGeo4W/lib"
    )
else()
    include_directories(
        "/usr/include/qgis"
        ${GDAL_INCLUDE_DIRS}
    )
    link_directories(
        "/usr/lib"
        "/usr/lib/x86_64-linux-gnu"
    )
endif()

# 🌟 3. 必须先声明可执行文件目标！(add_executable 必须在前)
add_executable(GisDynamicClient
    main.cpp
    mainwindow.cpp
    mainwindow.h
    custom_map_tools.cpp
    custom_map_tools.h
    gisapiserver.cpp
    gisapiserver.h
)

# 🌟 4. 声明目标后再进行头文件目录与库文件的链接！(target_link_libraries 必须在后)
if(WIN32)
    target_include_directories(GisDynamicClient PRIVATE
        "C:/OSGeo4W/apps/qgis/include"
        "C:/OSGeo4W/include"
        "C:/OSGeo4W/apps/Qt6/include"
    )
    target_link_libraries(GisDynamicClient PRIVATE
        qgis_core
        qgis_gui
        qgis_analysis
        GDAL::GDAL
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Widgets
        Qt${QT_VERSION_MAJOR}::Gui
        Qt${QT_VERSION_MAJOR}::Xml
        Qt${QT_VERSION_MAJOR}::Svg
        Qt${QT_VERSION_MAJOR}::WebSockets
    )
else()
    target_include_directories(GisDynamicClient PRIVATE 
        "/usr/include/qgis"
        ${GDAL_INCLUDE_DIRS}
    )
    target_link_libraries(GisDynamicClient PRIVATE
        qgis_core
        qgis_gui
        qgis_analysis
        ${GDAL_LIBRARIES}
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Widgets
        Qt${QT_VERSION_MAJOR}::Gui
        Qt${QT_VERSION_MAJOR}::Xml
        Qt${QT_VERSION_MAJOR}::Svg
        Qt${QT_VERSION_MAJOR}::WebSockets
    )
endif()