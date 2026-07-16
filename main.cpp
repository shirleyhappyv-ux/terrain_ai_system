#include <qgsapplication.h>
#include <qgsproviderregistry.h>
#include "mainwindow.h"
#include <QDir>
#include <QFile>
#include <QProcessEnvironment>

int main(int argc, char *argv[]) {
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);

    // 🌟 自动探测并静默注入 PROJ 数据库路径（兼容 Linux Codespaces 与 Windows 绿色包）
    QStringList possibleProjPaths = {
        "/usr/share/proj",
        "/usr/local/share/proj",
        "/usr/lib/x86_64-linux-gnu/proj",
        QCoreApplication::applicationDirPath() + "/proj"
    };

    for (const QString& path : possibleProjPaths) {
        if (QFile::exists(path + "/proj.db")) {
            qputenv("PROJ_LIB", path.toUtf8());
            break;
        }
    }

    QgsApplication app(argc, argv, true);

    QString appDir = QCoreApplication::applicationDirPath();
    QString qgisPrefix = appDir;                      
    QString pluginPath = appDir + "/qgis_plugins";   

#ifndef _WIN32
    // Linux 环境下的系统级路径配置
    qgisPrefix = "/usr";
    pluginPath = "/usr/lib/qgis/plugins";
#endif

    QgsApplication::setPrefixPath(qgisPrefix, true);
    QgsApplication::setPluginPath(pluginPath);
    
    QgsApplication::initQgis();
    QgsProviderRegistry::instance(pluginPath);

    MainWindow* mainWindow = new MainWindow();
    mainWindow->show();

    int execCode = app.exec();

    delete mainWindow;
    QgsApplication::exitQgis();
    return execCode;
}