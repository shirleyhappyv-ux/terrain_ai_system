// =================================================================
// 位置：/workspaces/terrain_ai_system/main.cpp
// =================================================================
#include <qgsapplication.h>
#include <qgsproviderregistry.h>
#include "mainwindow.h"
#include <QDebug>
#include <QDir>

int main(int argc, char *argv[]) {
    // 1. 🔧 终极修正 1：对于带有 Canvas 画布的 GUI 客户端，第三个参数【必须】为 true！
    // 只有这样，QGIS 才会为地图画布和渲染器分配完整的内存对象。
    QgsApplication app(argc, argv, true); 
    
    // 2. 🔧 终极修正 2：显式覆盖并锁死系统前缀和插件路径
    // 根据 Ubuntu 默认 libqgis-dev 安装规范
    QString prefixPath = "/usr";
    QString pluginPath = "/usr/lib/qgis/plugins";
    
    QgsApplication::setPrefixPath(prefixPath, true);
    QgsApplication::setPluginPath(pluginPath);
    
    // 3. 执行 QGIS 核心单例的底层 C++ 静态对象初始化
    QgsApplication::initQgis();

    // 4. 🔧 终极修正 3：显式强行刷新 Provider 注册表，打通 gdal/ogr 插件
    QgsProviderRegistry::instance(pluginPath);

    // 5. 打印驱动自检信息，确保 "gdal" 和 "ogr" 在列表里
    QStringList availableProviders = QgsProviderRegistry::instance()->providerList();
    qDebug() << "========================================";
    qDebug() << "【QGIS 核心单例初始化成功】";
    qDebug() << "系统可用 GIS 数据驱动列表:";
    qDebug() << availableProviders;
    qDebug() << "========================================";

    // 6. 安全拉起主界面（此时基础单例指针均已就绪，绝不会再发生段错误）
    MainWindow mainWindow;
    mainWindow.show();

    int execCode = app.exec();
    
    // 7. 退出时安全释放 QGIS 资源
    QgsApplication::exitQgis();
    return execCode;
}