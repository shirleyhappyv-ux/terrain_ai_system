#pragma once

#include <QMainWindow>
#include <QLineEdit>
#include <QPushButton>
#include <QTableWidget>
#include <QListWidget>        
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>

// QGIS & GDAL 核心头文件
#include <qgsmapcanvas.h>
#include <qgsvectorlayer.h>
#include <qgsrasterlayer.h>
#include <qgsproject.h>
#include <gdal_priv.h>

struct GisSearchTarget {
    QString name;        
    QString details;     
    QgsGeometry geometry;
    bool isMcdaResult = false;   
};

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

private slots:
    void zoomInMap();                   
    void zoomOutMap();                  
    void executeTextSearch();           
    void executeTunnelSiteSelection();  
    void handleTableDoubleClicked(int row, int column); 

private:
    void setupUi();
    void initGisLayers();
    
    // 🌟 补全真实 DEM 采样函数声明
    bool sampleRealDemData(double lon, double lat, double &elevation, double &slope, double &roughness);

    double CalculateMetricScore(double value, double opt_min, double opt_max,
                               double avail_min1, double avail_max1,
                               double avail_min2, double avail_max2,
                               double veto_min, double veto_max, bool has_veto_min, bool has_veto_max);

    QgsMapCanvas* mCanvas = nullptr;
    QgsVectorLayer* mRoadsLayer = nullptr;
    QgsVectorLayer* mPlacesLayer = nullptr;
    QgsVectorLayer* mPoisLayer = nullptr;
    QgsVectorLayer* mMarkLayer = nullptr; 
    
    // 🌟 明确声明 DEM 栅格图层指针
    QgsRasterLayer* mDemRasterLayer = nullptr;
    GDALDataset* mDemDataset = nullptr;

    QLineEdit* leTextSearch = nullptr;
    QLineEdit* leAreaMin = nullptr; QLineEdit* leElevMin = nullptr; QLineEdit* leElevMax = nullptr;
    QLineEdit* leHeightIdeal = nullptr; QLineEdit* leBiGaoIdeal = nullptr; QLineEdit* leSlopeIdeal = nullptr; QLineEdit* leRoughIdeal = nullptr;
    QLineEdit* leRoadDist = nullptr; QLineEdit* leWaterDist = nullptr;

    QTableWidget* tableWidgetConfirm = nullptr; 
    QLabel* lblStatus = nullptr;

    QListWidget* listWidgetSimilarConfirm = nullptr; 
    QLabel* lblSimilarTitle = nullptr;              
    QPushButton* btnSmartSearch = nullptr;          

    QList<GisSearchTarget> mCurrentResults;

protected:
    bool eventFilter(QObject *obj, QEvent *event) override;
};