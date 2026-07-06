#pragma once

#include <QMainWindow>
#include <QLineEdit>
#include <QPushButton>
#include <QTableWidget>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QMessageBox>
#include <QListWidget>

// QGIS 核心组件
#include <qgsmapcanvas.h>
#include <qgsvectorlayer.h>
#include <qgsrasterlayer.h>
#include <qgsproject.h>

// 自定义工具
#include "custom_map_tools.h"

// 结构体：存储多图层穿透检索到的候选空间要素元数据
struct SearchResultFeature {
    QString featureName;   // 要素真实名称
    QString layerName;     // 所属图层名称
    QgsGeometry geometry;  // 空间几何位置
    QgsFeatureId fid;      // 要素底层的ID
};

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:
    bool eventFilter(QObject *obj, QEvent *event) override;

private slots:
    // 核心业务
    void executeDynamicMCDASearch();
    void generateProfileAndLOS();
    void updateStatusBarDistance(double dist);
    
    // 地图控制与高级多图层穿透检索
    void zoomInMap();
    void zoomOutMap();
    void activateMeasureTool();
    void activatePlotTool();
    void executeAdvancedGlobalSearch(); // 🔧 升级：全图层多结果穿透检索
    void handleSearchResultSelected(QListWidgetItem* item); // 🔧 新增：二次确认列表点选联动
    void insertQuickSearchText();
    void handleTableClicked(int row, int column);
    void handleMapPointPlotted(const QgsPointXY& point);

private:
    void setupUserInterface();
    void loadGisDataStacked();
    void addCandidatePointToMap(const QString& id, double lon, double lat, double score, const QString& grade);

    QgsMapCanvas* mCanvas;
    QgsVectorLayer* mRoadsLayer;
    QgsVectorLayer* mPlacesLayer;   // 🔧 新增图层指针
    QgsVectorLayer* mPoisLayer;     // 🔧 新增图层指针
    QgsVectorLayer* mBuildingsLayer;// 🔧 新增图层指针
    QgsVectorLayer* mPlotLayer;     // 选址落点层
    QgsVectorLayer* mSearchMarkLayer;// 🔧 新增：检索精确定位高亮标记层（黄色星号）
    QList<QgsMapLayer*> mLayerLayers;

    // 工具
    MapMeasureTool* mMeasureTool;
    QgsMapToolEmitPoint* mPlotTool;

    // UI 控件
    QPushButton* btnZoomIn;
    QPushButton* btnZoomOut;
    QPushButton* btnMeasure;
    QPushButton* btnPlot;
    
    QLineEdit* leSearchName;
    QPushButton* btnSearch;
    QListWidget* listWidgetSearchResults; // 🔧 新增：待用户二次确认的类似结果列表框
    
    QLineEdit* leMaxSlope;
    QLineEdit* leIdealElev;
    QLineEdit* leMaxDist;
    QLineEdit* leAhpWeights;
    
    QTableWidget* tableWidget;
    QLabel* lblStatus;

    // 内存数据转换缓存
    QList<SearchResultFeature> mCurrentFoundFeatures;
};