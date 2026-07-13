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

// QGIS & GDAL 核心指针
#include <qgsmapcanvas.h>
#include <qgsvectorlayer.h>
#include <qgsproject.h>

struct GisSearchTarget {
    QString name;        
    QString details;     
    QgsGeometry geometry;
    bool isMcdaResult;   
};

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow() = default;

private slots:
    void zoomInMap();                   
    void zoomOutMap();                  
    void executeTextSearch();           
    void executeTunnelSiteSelection();  
    void handleTableDoubleClicked(int row, int column); 

private:
    void setupUi();
    void initGisLayers();
    double CalculateMetricScore(double value, double opt_min, double opt_max,
                                 double avail_min1, double avail_max1,
                                 double avail_min2, double avail_max2,
                                 double veto_min, double veto_max, bool has_veto_min, bool has_veto_max);

    QgsMapCanvas* mCanvas;
    QgsVectorLayer* mRoadsLayer;
    QgsVectorLayer* mPlacesLayer;
    QgsVectorLayer* mPoisLayer;
    QgsVectorLayer* mMarkLayer; 

    QLineEdit* leTextSearch;
    
    // UI输入框变量声明：已剔除 NDVI 植被部分
    QLineEdit* leAreaMin; QLineEdit* leElevMin; QLineEdit* leElevMax;
    QLineEdit* leHeightIdeal; QLineEdit* leBiGaoIdeal; QLineEdit* leSlopeIdeal; QLineEdit* leRoughIdeal;
    QLineEdit* leRoadDist; QLineEdit* leWaterDist;

    QTableWidget* tableWidgetConfirm; 
    QLabel* lblStatus;

    QListWidget* listWidgetSimilarConfirm; 
    QLabel* lblSimilarTitle;              
    QPushButton* btnSmartSearch;          

    QList<GisSearchTarget> mCurrentResults;

protected:
    bool eventFilter(QObject *obj, QEvent *event) override;
};
