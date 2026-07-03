#include <qgsapplication.h>
#include "mainwindow.cpp"

int main(int argc, char *argv[]) {
    // 必须要将系统设置为 Headless 无 GUI 渲染环境
    QgsApplication app(argc, argv, true);
    
    // 配置 QGIS 系统提供者路径
    QgsApplication::setPrefixPath("/usr", true);
    QgsApplication::initQgis();

    MainWindow mainWindow;
    mainWindow.show();

    int execCode = app.exec();
    QgsApplication::exitQgis();
    return execCode;
}