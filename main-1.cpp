#include <qgsapplication.h>
#include <qgsproviderregistry.h>
#include "mainwindow.h"
#include <QDebug>

int main(int argc, char *argv[]) {
    // 1. 启动完整的 GUI 支持型应用程序单例，为 Canvas 画布分配完整的渲染上下文
    QgsApplication app(argc, argv, true); 
    
    // 2. 锁死 Linux 容器环境下的 QGIS 核心路径与共享插件驱动路径
    QString prefixPath = "/usr";
    QString pluginPath = "/usr/lib/qgis/plugins";
    
    QgsApplication::setPrefixPath(prefixPath, true);
    QgsApplication::setPluginPath(pluginPath);
    
    // 3. 执行核心单例静态对象链式初始化
    QgsApplication::initQgis();

    // 4. 显式刷新全局 Provider 注册表，激活系统的 gdal 与 ogr 动态库
    QgsProviderRegistry::instance(pluginPath);

    // 调试打印：确认核心驱动加载状态
    qDebug() << "========================================";
    qDebug() << "【QGIS 系统环境初始化成功】";
    qDebug() << "可用 GIS 数据驱动列表:" << QgsProviderRegistry::instance()->providerList();
    qDebug() << "========================================";

    // 5. 拉起主界面
    MainWindow mainWindow;
    mainWindow.show();

    int execCode = app.exec();
    
    // 6. 退出时安全卸载全局 GIS 资源
    QgsApplication::exitQgis();
    return execCode;
}