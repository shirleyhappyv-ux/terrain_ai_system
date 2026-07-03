#pragma once
#include <QMainWindow>
#include <qgsmapcanvas.h>
#include <qgsvectorlayer.h>
#include <qgsrasterlayer.h>
#include <QLineEdit>
#include <QDoubleSpinBox>
#include <QTableWidget>
#include <QPushButton>
#include <QLabel>
#include "gdal_priv.h"

struct TargetCandidate {
    int id;
    double x, y;
    double slope;      // 准则1：坡度 (极小型)
    double elevation;  // 准则2：高程 (越接近理想目标越好)
    double distToRoad; // 准则3：距道路距离 (极小型)
    double mcdaScore;  // TOPSIS 贴进度得分
};

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

private slots:
    void activateMeasureMode();
    void activatePlotMode();
    void executeDynamicMCDASearch();
    void generateProfileAndLOS();
    void updateStatusBarDistance(double d);

private:
    QgsMapCanvas* mCanvas;
    QList<QgsMapLayer*> mLayerLayers;
    QgsVectorLayer* mPlotLayer;
    QgsVectorLayer* mRoadsLayer;
    GDALDataset* mDemDataset;

    // 动态交互窗口 UI 控件
    QDoubleSpinBox* sbMaxSlope;
    QDoubleSpinBox* sbTargetEle;
    QDoubleSpinBox* sbMaxRoadDist;
    QLineEdit* leAhpWeights; // 格式: "0.5,0.3,0.2"
    QTableWidget* twResultsTable;
    QLabel* lblStatus;

    void setupUserInterface();
    void loadGisDataStacked();
    double queryElevationGDAL(double x, double y);
};