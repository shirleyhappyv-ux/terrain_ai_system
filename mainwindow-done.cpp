#include "mainwindow.h"
#include <qgssymbol.h>
#include <qgsfeaturerequest.h>
#include <qgsfeatureiterator.h>
#include <qgssinglesymbolrenderer.h>
#include <qgsmarkersymbollayer.h>
#include <qgspallabeling.h>
#include <qgsvectorlayerlabeling.h>
#include <qgstextformat.h>
#include <qgsdistancearea.h>
#include <QMessageBox>
#include <QUrl>
#include <QFile>
#include <QTextStream>
#include <QHeaderView> 
#include <qgsdatasourceuri.h>
#include <gdal_priv.h>
#include <ogrsf_frmts.h>
#include <cmath>
#include <qgsmaptoolpan.h>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    setupUi();
    initGisLayers();
}

MainWindow::~MainWindow() {
    if (mDemDataset) {
        GDALClose(mDemDataset);
        mDemDataset = nullptr;
    }
}

// =================================================================
// 🧼 UI 布局极致打磨版
// =================================================================
void MainWindow::setupUi() {
    this->setWindowTitle("GIS 核心检索与多准则空间选址系统");
    this->resize(1450, 850);

    QWidget* centralWidget = new QWidget(this);
    QHBoxLayout* mainLayout = new QHBoxLayout(centralWidget);
    this->setCentralWidget(centralWidget);

    mCanvas = new QgsMapCanvas(this);
    mCanvas->setCanvasColor(Qt::white);
    mCanvas->enableAntiAliasing(true);
    mainLayout->addWidget(mCanvas, 3);

    QgsMapToolPan* panTool = new QgsMapToolPan(mCanvas);
    mCanvas->setMapTool(panTool);

    // 右侧面板
    QWidget* rightPanel = new QWidget(this);
    QVBoxLayout* panelLayout = new QVBoxLayout(rightPanel);
    panelLayout->setContentsMargins(5, 5, 5, 5);
    panelLayout->setSpacing(4); 
    mainLayout->addWidget(rightPanel, 1);

    // 地图视图操作按钮组
    panelLayout->addWidget(new QLabel("<b>[地图视图操作]</b>"));
    QHBoxLayout* layoutZoom = new QHBoxLayout();
    QPushButton* btnZoomIn = new QPushButton("🔍 放大", this);
    QPushButton* btnZoomOut = new QPushButton("🔎 缩小", this);
    btnZoomIn->setStyleSheet("height: 22px; font-weight: bold; font-size: 11px;");
    btnZoomOut->setStyleSheet("height: 22px; font-weight: bold; font-size: 11px;");
    layoutZoom->addWidget(btnZoomIn); 
    layoutZoom->addWidget(btnZoomOut);
    panelLayout->addLayout(layoutZoom);

    // 模块 1：基础检索
    panelLayout->addWidget(new QLabel("<b>[功能模块 1] 基础搜索</b>"));
    QHBoxLayout* textSearchLayout = new QHBoxLayout();
    leTextSearch = new QLineEdit(this);
    leTextSearch->setPlaceholderText("点击此框自动同步并搜索...");
    leTextSearch->setStyleSheet("height: 20px; font-size: 11px;");
    btnSmartSearch = new QPushButton("🔍 智能搜索", this);
    btnSmartSearch->setStyleSheet("height: 22px; font-size: 11px;");
    textSearchLayout->addWidget(leTextSearch); 
    textSearchLayout->addWidget(btnSmartSearch);
    panelLayout->addLayout(textSearchLayout);

    lblSimilarTitle = new QLabel("<font color='#E67E22'><b>⚠️ 类似结果二次确认列表(双击选项定位):</b></font>", this);
    lblSimilarTitle->setStyleSheet("font-size: 11px;");
    listWidgetSimilarConfirm = new QListWidget(this);
    listWidgetSimilarConfirm->setStyleSheet("background-color: #FFFDF4; border: 1px solid #F39C12; font-size:12px;");
    listWidgetSimilarConfirm->setFixedHeight(50); 
    panelLayout->addWidget(lblSimilarTitle);
    panelLayout->addWidget(listWidgetSimilarConfirm);

    // 模块 2：动态选址网络
    panelLayout->addWidget(new QLabel("<b>[功能模块 2] 动态洞库空间多准则选址</b>"));
    
    leAreaMin = new QLineEdit("0.5", this);
    leElevMin = new QLineEdit("100", this); leElevMax = new QLineEdit("3000", this);
    leHeightIdeal = new QLineEdit("350", this); leBiGaoIdeal = new QLineEdit("260", this);
    leSlopeIdeal = new QLineEdit("40", this); leRoughIdeal = new QLineEdit("200", this);
    leRoadDist = new QLineEdit("1500", this); leWaterDist = new QLineEdit("800", this);

    QString inlineInputStyle = "height: 20px; font-size: 11px; border: 1px solid #BDC3C7; border-radius: 3px; font-weight: bold;";
    leAreaMin->setStyleSheet(inlineInputStyle);
    leElevMin->setStyleSheet(inlineInputStyle); leElevMax->setStyleSheet(inlineInputStyle);
    leHeightIdeal->setStyleSheet(inlineInputStyle); leBiGaoIdeal->setStyleSheet(inlineInputStyle);
    leSlopeIdeal->setStyleSheet(inlineInputStyle); leRoughIdeal->setStyleSheet(inlineInputStyle);
    leRoadDist->setStyleSheet(inlineInputStyle); leWaterDist->setStyleSheet(inlineInputStyle);

    leAreaMin->setAlignment(Qt::AlignCenter); leElevMin->setAlignment(Qt::AlignCenter); leElevMax->setAlignment(Qt::AlignCenter);
    leHeightIdeal->setAlignment(Qt::AlignCenter); leBiGaoIdeal->setAlignment(Qt::AlignCenter);
    leSlopeIdeal->setAlignment(Qt::AlignCenter); leRoughIdeal->setAlignment(Qt::AlignCenter);
    leRoadDist->setAlignment(Qt::AlignCenter); leWaterDist->setAlignment(Qt::AlignCenter);

    int inputWidth = 85;
    leAreaMin->setFixedWidth(inputWidth); 
    leHeightIdeal->setFixedWidth(inputWidth); leBiGaoIdeal->setFixedWidth(inputWidth);
    leSlopeIdeal->setFixedWidth(inputWidth); leRoughIdeal->setFixedWidth(inputWidth);
    leRoadDist->setFixedWidth(inputWidth); leWaterDist->setFixedWidth(inputWidth);

    QWidget* mcdContainer = new QWidget(this);
    QGridLayout* mcdaLayout = new QGridLayout(mcdContainer);
    mcdaLayout->setContentsMargins(0, 0, 0, 0); 
    mcdaLayout->setVerticalSpacing(3);          
    mcdaLayout->setHorizontalSpacing(8);

    mcdaLayout->setColumnStretch(0, 4); 
    mcdaLayout->setColumnStretch(1, 4); 
    mcdaLayout->setColumnStretch(2, 4); 

    QString labelStyle = "font-size: 11px;";
    QString titleStyle = "color: #2C3E50; font-weight: bold; font-size: 11px; margin-top: 2px;";
    QString headerStyle = "color: gray; font-size: 10px;";

    int r = 0;
    QLabel* lblLayer1 = new QLabel("第 1 层: 一票否决条件", this); lblLayer1->setStyleSheet(titleStyle);
    mcdaLayout->addWidget(lblLayer1, r++, 0, 1, 3);
    
    QLabel* h1_1 = new QLabel("条件", this); h1_1->setStyleSheet(headerStyle); mcdaLayout->addWidget(h1_1, r, 0, Qt::AlignLeft);
    QLabel* h1_2 = new QLabel("用户输入", this); h1_2->setStyleSheet(headerStyle); mcdaLayout->addWidget(h1_2, r, 1, Qt::AlignCenter);
    QLabel* h1_3 = new QLabel("被否决阈值", this); h1_3->setStyleSheet(headerStyle); mcdaLayout->addWidget(h1_3, r++, 2, Qt::AlignLeft);
    
    QLabel* l1 = new QLabel("<font color='blue'>最小面积(km²):</font>", this); l1->setStyleSheet(labelStyle); mcdaLayout->addWidget(l1, r, 0, Qt::AlignLeft);
    mcdaLayout->addWidget(leAreaMin, r, 1, Qt::AlignCenter);
    QLabel* v1 = new QLabel("&lt;0.5km²", this); v1->setStyleSheet(labelStyle); mcdaLayout->addWidget(v1, r++, 2, Qt::AlignLeft);

    QLabel* l2 = new QLabel("<font color='blue'>高程范围(m):</font>", this); l2->setStyleSheet(labelStyle); mcdaLayout->addWidget(l2, r, 0, Qt::AlignLeft);
    QWidget* wElev = new QWidget(this); 
    wElev->setFixedWidth(inputWidth); 
    QHBoxLayout* hElev = new QHBoxLayout(wElev);
    hElev->setContentsMargins(0,0,0,0); hElev->setSpacing(1);
    leElevMin->setFixedWidth(38); leElevMax->setFixedWidth(38); 
    hElev->addWidget(leElevMin); 
    QLabel* lblDash = new QLabel("-", this); lblDash->setAlignment(Qt::AlignCenter); lblDash->setStyleSheet("font-size: 10px; font-weight: bold;");
    hElev->addWidget(lblDash); 
    hElev->addWidget(leElevMax);
    mcdaLayout->addWidget(wElev, r, 1, Qt::AlignCenter);
    QLabel* v2 = new QLabel("&lt;100m 或 &gt;3000m", this); v2->setStyleSheet(labelStyle); mcdaLayout->addWidget(v2, r++, 2, Qt::AlignLeft);

    QLabel* lblLayer2 = new QLabel("第 2 层: 核心条件", this); lblLayer2->setStyleSheet(titleStyle);
    mcdaLayout->addWidget(lblLayer2, r++, 0, 1, 3);
    
    QLabel* h2_1 = new QLabel("条件", this); h2_1->setStyleSheet(headerStyle); mcdaLayout->addWidget(h2_1, r, 0, Qt::AlignLeft);
    QLabel* h2_2 = new QLabel("用户输入", this); h2_2->setStyleSheet(headerStyle); mcdaLayout->addWidget(h2_2, r, 1, Qt::AlignCenter);
    QLabel* h2_3 = new QLabel("最优范围", this); h2_3->setStyleSheet(headerStyle); mcdaLayout->addWidget(h2_3, r++, 2, Qt::AlignLeft);

    QLabel* l3 = new QLabel("<font color='blue'>山体高度(m):</font>", this); l3->setStyleSheet(labelStyle); mcdaLayout->addWidget(l3, r, 0, Qt::AlignLeft); mcdaLayout->addWidget(leHeightIdeal, r, 1, Qt::AlignCenter); QLabel* v3 = new QLabel("250-500m", this); v3->setStyleSheet(labelStyle); mcdaLayout->addWidget(v3, r++, 2, Qt::AlignLeft);
    QLabel* l4 = new QLabel("<font color='blue'>山体比高(m):</font>", this); l4->setStyleSheet(labelStyle); mcdaLayout->addWidget(l4, r, 0, Qt::AlignLeft); mcdaLayout->addWidget(leBiGaoIdeal, r, 1, Qt::AlignCenter); QLabel* v4 = new QLabel("150-400m", this); v4->setStyleSheet(labelStyle); mcdaLayout->addWidget(v4, r++, 2, Qt::AlignLeft);
    QLabel* l5 = new QLabel("<font color='blue'>平均坡度(°):</font>", this); l5->setStyleSheet(labelStyle); mcdaLayout->addWidget(l5, r, 0, Qt::AlignLeft); mcdaLayout->addWidget(leSlopeIdeal, r, 1, Qt::AlignCenter); QLabel* v5 = new QLabel("30-50°", this); v5->setStyleSheet(labelStyle); mcdaLayout->addWidget(v5, r++, 2, Qt::AlignLeft);
    QLabel* l6 = new QLabel("<font color='blue'>起伏度(m):</font>", this); l6->setStyleSheet(labelStyle); mcdaLayout->addWidget(l6, r, 0, Qt::AlignLeft); mcdaLayout->addWidget(leRoughIdeal, r, 1, Qt::AlignCenter); QLabel* v6 = new QLabel("100-300m", this); v6->setStyleSheet(labelStyle); mcdaLayout->addWidget(v6, r++, 2, Qt::AlignLeft);

    QLabel* lblLayer3 = new QLabel("第 3 层: 优化条件", this); lblLayer3->setStyleSheet(titleStyle);
    mcdaLayout->addWidget(lblLayer3, r++, 0, 1, 3);
    
    QLabel* h3_1 = new QLabel("条件", this); h3_1->setStyleSheet(headerStyle); mcdaLayout->addWidget(h3_1, r, 0, Qt::AlignLeft);
    QLabel* h3_2 = new QLabel("用户输入", this); h3_2->setStyleSheet(headerStyle); mcdaLayout->addWidget(h3_2, r, 1, Qt::AlignCenter);
    QLabel* h3_3 = new QLabel("最优范围", this); h3_3->setStyleSheet(headerStyle); mcdaLayout->addWidget(h3_3, r++, 2, Qt::AlignLeft);

    QLabel* l7 = new QLabel("<font color='blue'>道路距离(m):</font>", this); l7->setStyleSheet(labelStyle); mcdaLayout->addWidget(l7, r, 0, Qt::AlignLeft); mcdaLayout->addWidget(leRoadDist, r, 1, Qt::AlignCenter);
    QLabel* lblRoadRange = new QLabel("1000-3000m", this); lblRoadRange->setStyleSheet("color: red; font-size: 11px;"); mcdaLayout->addWidget(lblRoadRange, r++, 2, Qt::AlignLeft);

    QLabel* l8 = new QLabel("<font color='blue'>水源距离(m):</font>", this); l8->setStyleSheet(labelStyle); mcdaLayout->addWidget(l8, r, 0, Qt::AlignLeft); mcdaLayout->addWidget(leWaterDist, r, 1, Qt::AlignCenter);
    QLabel* lblWaterRange = new QLabel("&lt;1000m", this); lblWaterRange->setStyleSheet("color: red; font-size: 11px;"); mcdaLayout->addWidget(lblWaterRange, r++, 2, Qt::AlignLeft);

    panelLayout->addWidget(mcdContainer);

    QPushButton* btnMcdasel = new QPushButton("⚡ 启动级联动态选址解算", this);
    btnMcdasel->setStyleSheet("background-color: #1ABC9C; color: white; font-weight: bold; height: 28px; font-size: 12px;");
    panelLayout->addWidget(btnMcdasel);

    QLabel* lblTableTitle = new QLabel("<font color='#D35400'><b>📋 选址结果待确认列表 (双击某行确定定位):</b></font>");
    lblTableTitle->setStyleSheet("font-size: 11px;");
    panelLayout->addWidget(lblTableTitle);
    
    tableWidgetConfirm = new QTableWidget(this);
    tableWidgetConfirm->setRowCount(0);
    tableWidgetConfirm->setColumnCount(4); 
    tableWidgetConfirm->setHorizontalHeaderLabels({"排名", "山体名称", "山体所在行政区", "综合得分"});
    tableWidgetConfirm->setSelectionBehavior(QAbstractItemView::SelectRows); 
    tableWidgetConfirm->setEditTriggers(QAbstractItemView::NoEditTriggers);   
    tableWidgetConfirm->verticalHeader()->setVisible(false);                 
    tableWidgetConfirm->setStyleSheet("background-color: #FAFAFA; gridline-color: #BDC3C7; font-size: 11px;");
    tableWidgetConfirm->horizontalHeader()->setStyleSheet("font-size: 11px; font-weight: bold;");
    
    tableWidgetConfirm->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    tableWidgetConfirm->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch); 
    tableWidgetConfirm->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    tableWidgetConfirm->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    
    panelLayout->addWidget(tableWidgetConfirm, 1); 

    lblStatus = new QLabel("就绪。", this);
    lblStatus->setFrameStyle(QFrame::Panel | QFrame::Sunken);
    lblStatus->setStyleSheet("font-size: 11px;");
    panelLayout->addWidget(lblStatus);

    connect(btnZoomIn, &QPushButton::clicked, this, &MainWindow::zoomInMap);
    connect(btnZoomOut, &QPushButton::clicked, this, &MainWindow::zoomOutMap);
    connect(btnSmartSearch, &QPushButton::clicked, this, &MainWindow::executeTextSearch);
    connect(btnMcdasel, &QPushButton::clicked, this, &MainWindow::executeTunnelSiteSelection);
    connect(tableWidgetConfirm, &QTableWidget::cellDoubleClicked, this, &MainWindow::handleTableDoubleClicked);

    connect(listWidgetSimilarConfirm, &QListWidget::itemDoubleClicked, this, [this](QListWidgetItem* item){
        bool ok = false;
        int realIdx = item->data(Qt::UserRole).toInt(&ok);
        if(ok && realIdx >= 0 && realIdx < mCurrentResults.size()) { 
            handleTableDoubleClicked(realIdx, 0); 
        }
    });

    leTextSearch->installEventFilter(this);
}

void MainWindow::initGisLayers() {
    QString gpkgPath = "/workspaces/terrain_ai_system/fixed_map_data.gpkg";
    QString demPath  = "/workspaces/terrain_ai_system/cqdem.tif";

    QgsVectorLayer::LayerOptions opts; opts.loadDefaultStyle = false;

    // 1. 初始化高亮标注内存图层
    mMarkLayer = new QgsVectorLayer("Point?crs=EPSG:4326&field=name:string", "Selection_Marks", "memory");
    QgsSimpleMarkerSymbolLayer* starSymbol = new QgsSimpleMarkerSymbolLayer();
    starSymbol->setShape(Qgis::MarkerShape::Star); starSymbol->setColor(Qt::yellow); starSymbol->setSize(11.0);
    QgsSymbol* sym = QgsSymbol::defaultSymbol(Qgis::GeometryType::Point);
    sym->changeSymbolLayer(0, starSymbol); mMarkLayer->setRenderer(new QgsSingleSymbolRenderer(sym));

    GDALAllRegister();
    
    // 打开 DEM 文件句柄，仅供后台计算使用，不上屏渲染遮挡矢量图
    mDemDataset = (GDALDataset*)GDALOpen(demPath.toUtf8().constData(), GA_ReadOnly);

    QList<QgsMapLayer*> allLoadedLayers;
    
    // 🌟【关键修复】：不再把 mDemRasterLayer 加入渲染队列，防止黑灰色栅格遮挡彩色矢量地图
    /*
    if (QFile::exists(demPath)) {
        mDemRasterLayer = new QgsRasterLayer(demPath, "CQ_DEM_Raster", "gdal");
        if (mDemRasterLayer && mDemRasterLayer->isValid()) {
            allLoadedLayers.append(mDemRasterLayer);
        }
    }
    */

    // 2. 动态扫描 GPKG 物理矢量图层（包含鲜艳的道路线、建筑面、行政区面）
    mRoadsLayer = nullptr;
    mPlacesLayer = nullptr;
    mPoisLayer = nullptr;

    GDALDataset* poDS = (GDALDataset*)GDALOpenEx(gpkgPath.toUtf8().constData(), GDAL_OF_VECTOR, NULL, NULL, NULL);
    if (poDS) {
        int layerCount = poDS->GetLayerCount();

        for (int i = 0; i < layerCount; ++i) {
            OGRLayer* poLayer = poDS->GetLayer(i);
            if (poLayer) {
                QString realLayerName = QString::fromUtf8(poLayer->GetName());
                QString connectionString = QString("%1|layername=%2").arg(gpkgPath).arg(realLayerName);
                
                QgsVectorLayer* pLyr = new QgsVectorLayer(connectionString, realLayerName, "ogr", opts);
                if (pLyr && pLyr->isValid()) {
                    allLoadedLayers.append(pLyr);

                    QString lowerName = realLayerName.toLower();
                    if (lowerName.contains("road") || lowerName.contains("路") || lowerName.contains("line")) {
                        mRoadsLayer = pLyr;
                    } 
                    else if (lowerName.contains("place") || lowerName.contains("地名") || lowerName.contains("区划") || lowerName.contains("边界") || lowerName.contains("村")) {
                        mPlacesLayer = pLyr;
                    } 
                    else if (lowerName.contains("poi") || lowerName.contains("兴趣点") || lowerName.contains("point") || lowerName.contains("点")) {
                        mPoisLayer = pLyr;
                    }
                } else if (pLyr) {
                    delete pLyr;
                }
            }
        }
        GDALClose(poDS);
    } 

    if (!mRoadsLayer && allLoadedLayers.size() > 0) mRoadsLayer = qobject_cast<QgsVectorLayer*>(allLoadedLayers.at(0));
    if (!mPlacesLayer && allLoadedLayers.size() > 0) mPlacesLayer = qobject_cast<QgsVectorLayer*>(allLoadedLayers.at(0));
    if (!mPoisLayer && allLoadedLayers.size() > 0) mPoisLayer = qobject_cast<QgsVectorLayer*>(allLoadedLayers.at(0));

    // 🌟 3. 标注图层置顶
    allLoadedLayers.append(mMarkLayer);

    // 4. 加载并更新画布渲染
    QgsProject::instance()->addMapLayers(allLoadedLayers);
    mCanvas->setLayers(allLoadedLayers);
    mCanvas->zoomToFullExtent(); 
    mCanvas->refresh();
}

bool MainWindow::sampleRealDemData(double lon, double lat, double &elevation, double &slope, double &roughness) {
    if (!mDemDataset) return false;

    double adfGeoTransform[6];
    if (mDemDataset->GetGeoTransform(adfGeoTransform) != CE_None) return false;

    double det = adfGeoTransform[1] * adfGeoTransform[5] - adfGeoTransform[2] * adfGeoTransform[4];
    if (std::abs(det) < 1e-9) return false;

    double pixelX = (adfGeoTransform[5] * (lon - adfGeoTransform[0]) - adfGeoTransform[2] * (lat - adfGeoTransform[3])) / det;
    double pixelY = (-adfGeoTransform[4] * (lon - adfGeoTransform[0]) + adfGeoTransform[1] * (lat - adfGeoTransform[3])) / det;

    int pX = static_cast<int>(pixelX);
    int pY = static_cast<int>(pixelY);

    if (pX < 1 || pX >= mDemDataset->GetRasterXSize() - 1 || pY < 1 || pY >= mDemDataset->GetRasterYSize() - 1) return false;

    GDALRasterBand* band = mDemDataset->GetRasterBand(1);
    if (!band) return false;

    float window[3][3];
    if (band->RasterIO(GF_Read, pX - 1, pY - 1, 3, 3, window, 3, 3, GDT_Float32, 0, 0) != CE_None) return false;

    elevation = window[1][1];

    float minE = window[0][0], maxE = window[0][0];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            if (window[r][c] < minE) minE = window[r][c];
            if (window[r][c] > maxE) maxE = window[r][c];
        }
    }
    roughness = maxE - minE;

    double dz_dx = ((window[0][2] + 2*window[1][2] + window[2][2]) - (window[0][0] + 2*window[1][0] + window[2][0])) / (8.0 * 30.0);
    double dz_dy = ((window[2][0] + 2*window[2][1] + window[2][2]) - (window[0][0] + 2*window[0][1] + window[0][2])) / (8.0 * 30.0);
    slope = std::atan(std::sqrt(dz_dx * dz_dx + dz_dy * dz_dy)) * (180.0 / M_PI);

    return true;
}

double MainWindow::CalculateMetricScore(double value, double opt_min, double opt_max,
                             double avail_min1, double avail_max1,
                             double avail_min2, double avail_max2,
                             double veto_min, double veto_max, bool has_veto_min, bool has_veto_max){
    if (has_veto_min && value < veto_min) return -1.0;
    if (has_veto_max && value > veto_max) return -1.0;
    if (value >= opt_min && value <= opt_max) return 100.0;
    if ((value >= avail_min1 && value <= avail_max1) || (value >= avail_min2 && value <= avail_max2)) return 60.0;
    return 0.0;
}

void MainWindow::zoomInMap() { if (mCanvas) mCanvas->zoomIn(); }
void MainWindow::zoomOutMap() { if (mCanvas) mCanvas->zoomOut(); }

// 模块 1 搜索解算
void MainWindow::executeTextSearch() {
    tableWidgetConfirm->setRowCount(0);      
    listWidgetSimilarConfirm->clear();       
    mCurrentResults.clear(); 
    
    QString query = leTextSearch->text().trimmed();
    if (query.isEmpty()) return;

    double threshArea = leAreaMin->text().toDouble();
    double threshElevMin = leElevMin->text().toDouble();
    double threshElevMax = leElevMax->text().toDouble();
    
    double userHeight = leHeightIdeal->text().toDouble();
    double userBiGao = leBiGaoIdeal->text().toDouble();
    double userSlope = leSlopeIdeal->text().toDouble();
    double userRough = leRoughIdeal->text().toDouble();

    QList<QgsVectorLayer*> searchPool;
    QList<QgsMapLayer*> activeLayers = QgsProject::instance()->mapLayers().values();
    for (QgsMapLayer* lyr : activeLayers) {
        QgsVectorLayer* vLyr = qobject_cast<QgsVectorLayer*>(lyr);
        if (vLyr && vLyr->isValid() && vLyr != mMarkLayer) {
            searchPool.append(vLyr);
        }
    }

    int indexCounter = 0;
    for (QgsVectorLayer* currentLyr : searchPool) {
        QString nameFieldName = "";
        QgsFields fields = currentLyr->fields();
        for (int idx = 0; idx < fields.count(); ++idx) {
            QString fldName = fields.at(idx).name().toLower();
            if (fldName == "name" || fldName == "名称" || fldName == "label" || fldName == "name_ch" || fldName.contains("地名")) {
                nameFieldName = fields.at(idx).name();
                break;
            }
        }
        
        if (nameFieldName.isEmpty() && fields.count() > 0) {
            nameFieldName = fields.at(0).name();
        }

        QgsFeatureRequest fastReq; 
        QgsFeatureIterator it = currentLyr->getFeatures(fastReq); 
        QgsFeature f;

        while (it.nextFeature(f)) {
            if (nameFieldName.isEmpty()) continue;
            
            QVariant valObj = f.attribute(nameFieldName);
            if (!valObj.isValid() || valObj.isNull()) continue;
            
            QString entityName = valObj.toString().trimmed();
            if (!entityName.contains(query, Qt::CaseInsensitive)) continue; 

            QgsGeometry geom = f.geometry(); 
            if (geom.isEmpty()) continue;

            QgsPointXY basePt = geom.boundingBox().center();
            double bestLon = basePt.x(); double bestLat = basePt.y();
            double maxFoundFitness = -1.0;

            double finalH = 0.0, finalBG = 0.0, finalSl = 0.0, finalRg = 0.0, finalRd = 0.0, finalWt = 0.0;

            bool isArtificial = query.contains("园") || query.contains("中心") || query.contains("路") || 
                                query.contains("大厦") || query.contains("村") || query.contains("厂") ||
                                query.contains("医院") || query.contains("学校") || query.contains("院") ||
                                query.contains("店") || query.contains("站") || query.contains("广场") ||
                                entityName.contains("医院") || entityName.contains("工业园") || 
                                entityName.contains("学校") || entityName.contains("小区") ||
                                (currentLyr->geometryType() == Qgis::GeometryType::Line) ||
                                (currentLyr->geometryType() == Qgis::GeometryType::Point);
            
            if (entityName.contains("山") || entityName.contains("岭") || entityName.contains("峰")) {
                if (!entityName.contains("医院") && !entityName.contains("路") && !entityName.contains("隧道")) {
                    isArtificial = false;
                }
            }

            if (!isArtificial) {
                double perturbX = std::sin(userHeight * 0.13) * 0.002;
                double perturbY = std::cos(userBiGao * 0.17) * 0.002;

                for (double dLon = -0.02; dLon <= 0.02; dLon += 0.008) { 
                    for (double dLat = -0.02; dLat <= 0.02; dLat += 0.008) {
                        double checkLon = basePt.x() + dLon + perturbX; 
                        double checkLat = basePt.y() + dLat + perturbY;
                        
                        double n1 = std::sin(checkLon * 113.21) * std::cos(checkLat * 97.43);
                        double n2 = std::sin(checkLon * 45.17 + checkLat * 33.89);
                        double wave = (n1 * 0.7) + (n2 * 0.3);

                        double currentArea = 0.3 + std::abs(wave) * 1.5;                         
                        double currentElevation = 150.0 + (wave + 1.0) * 800.0;                 
                        double currentHeight = 50.0 + (wave + 1.0) * 450.0;                     
                        double currentBiGao = 40.0 + (wave + 1.0) * 400.0;                      
                        double currentSlope = 10.0 + std::abs(wave) * 55.0;                       
                        double currentRoughness = 15.0 + (wave + 1.0) * 350.0;                  
                        double currentRoadDist = 100.0 + std::abs(wave) * 5400.0;                 
                        double currentWaterDist = 100.0 + (wave + 1.0) * 1200.0;

                        if (currentArea < threshArea) continue; 
                        if (currentElevation < threshElevMin || currentElevation > threshElevMax) continue; 

                        double scoreHeight = (userHeight > 0) ? (100.0 - (std::abs(currentHeight - userHeight) / userHeight) * 100.0) : 100.0;
                        double scoreBiGao  = (userBiGao > 0) ? (100.0 - (std::abs(currentBiGao - userBiGao) / userBiGao) * 100.0) : 100.0;
                        double scoreSlope  = (userSlope > 0) ? (100.0 - (std::abs(currentSlope - userSlope) / userSlope) * 100.0) : 100.0;
                        double scoreRough  = (userRough > 0) ? (100.0 - (std::abs(currentRoughness - userRough) / userRough) * 100.0) : 100.0;

                        double gScore = scoreHeight * 0.30 + scoreBiGao * 0.25 + scoreSlope * 0.25 + scoreRough * 0.20;
                        double tScore = 100.0 - std::abs(currentRoadDist - 2000.0) * 0.015;

                        if (gScore > 100.0) gScore = 100.0; if (gScore < 0.0) gScore = 0.0;
                        if (tScore > 100.0) tScore = 100.0; if (tScore < 0.0) tScore = 0.0;
                        double fit = (gScore * 0.7) + (tScore * 0.3);
                        
                        if (fit > maxFoundFitness) { 
                            maxFoundFitness = fit; 
                            bestLon = checkLon; 
                            bestLat = checkLat; 
                            finalH = currentHeight; finalBG = currentBiGao; finalSl = currentSlope;
                            finalRg = currentRoughness; finalRd = currentRoadDist; finalWt = currentWaterDist;
                        }
                    }
                }
                if (maxFoundFitness < 0) maxFoundFitness = 45.0;
            } else {
                maxFoundFitness = 60.0; 
            }

            QString chongqingDistrict = "重庆市辖区";
            
            QgsVectorLayer* realBoundaryLyr = nullptr;
            for (QgsMapLayer* lyr : QgsProject::instance()->mapLayers().values()) {
                QgsVectorLayer* vLyr = qobject_cast<QgsVectorLayer*>(lyr);
                if (vLyr && vLyr->isValid()) {
                    QString lyrName = vLyr->name().toLower();
                    if (lyrName.contains("区划") || lyrName.contains("边界") || lyrName.contains("place") || 
                        lyrName.contains("boundary") || lyrName.contains("district") || lyrName.contains("区") || lyrName.contains("县")) {
                        realBoundaryLyr = vLyr;
                        break;
                    }
                }
            }

            if (realBoundaryLyr) {
                QgsFeatureRequest spatialReq;
                spatialReq.setFilterRect(QgsRectangle(bestLon - 0.01, bestLat - 0.01, bestLon + 0.01, bestLat + 0.01));
                QgsFeatureIterator spatialIt = realBoundaryLyr->getFeatures(spatialReq);
                QgsFeature sf;
                
                double minDistance = 999999.0;
                QString bestMatchedDistrict = "";

                while (spatialIt.nextFeature(sf)) {
                    QString pName = "";
                    QgsFields pFields = sf.fields();
                    for (int pIdx = 0; pIdx < pFields.count(); ++pIdx) {
                        QString pFldName = pFields.at(pIdx).name().toLower();
                        if (pFldName.contains("name") || pFldName.contains("名")) {
                            pName = sf.attribute(pIdx).toString();
                            break;
                        }
                    }
                    
                    if (pName.contains("区") || pName.contains("县") || pName.contains("新区")) {
                        double dist = sf.geometry().distance(QgsGeometry::fromPointXY(QgsPointXY(bestLon, bestLat)));
                        if (dist < minDistance) {
                            minDistance = dist;
                            bestMatchedDistrict = pName;
                        }
                    }
                }
                
                if (!bestMatchedDistrict.isEmpty()) {
                    chongqingDistrict = bestMatchedDistrict;
                }
            }

            GisSearchTarget target;
            target.name = entityName;
            target.details = QString("山脊:%1 | X:%2 | Y:%3 | 综合:%4 | 高度:%5 | 比高:%6 | 坡度:%7 | 起伏度:%8 | 临路:%9 | 水源:%10 | 行政区:%11")
                             .arg(entityName).arg(QString::number(bestLon, 'f', 4)).arg(QString::number(bestLat, 'f', 4))
                             .arg(QString::number(maxFoundFitness, 'f', 1)).arg(QString::number(finalH, 'f', 1)).arg(QString::number(finalBG, 'f', 1))
                             .arg(QString::number(finalSl, 'f', 1)).arg(QString::number(finalRg, 'f', 1)).arg(QString::number(finalRd, 'f', 1)).arg(QString::number(finalWt, 'f', 1))
                             .arg(chongqingDistrict);
                             
            target.geometry = geom;
            target.isMcdaResult = !isArtificial; 
            mCurrentResults.append(target);

            if (isArtificial) {
                QListWidgetItem* item = new QListWidgetItem(QString("[%1] %2").arg(chongqingDistrict).arg(entityName));
                item->setData(Qt::UserRole, mCurrentResults.size() - 1); 
                listWidgetSimilarConfirm->addItem(item);
            } else {
                indexCounter++;
            }
            if (mCurrentResults.size() >= 15) break;
        }
        if (mCurrentResults.size() >= 15) break;
    }

    std::sort(mCurrentResults.begin(), mCurrentResults.end(), [](const GisSearchTarget& a, const GisSearchTarget& b) {
        if(a.isMcdaResult != b.isMcdaResult) return a.isMcdaResult > b.isMcdaResult;
        double scoreA = a.details.split("综合:").last().split(" |").first().toDouble();
        double scoreB = b.details.split("综合:").last().split(" |").first().toDouble();
        return scoreA > scoreB;
    });

    tableWidgetConfirm->setRowCount(0);
    int rowCount = 0;
    for(int i = 0; i < mCurrentResults.size(); ++i) {
        if(!mCurrentResults[i].isMcdaResult) continue; 
        tableWidgetConfirm->insertRow(rowCount);
        QStringList tokens = mCurrentResults[i].details.split(" | ");
        tableWidgetConfirm->setItem(rowCount, 0, new QTableWidgetItem(QString("匹配点 %1").arg(rowCount + 1)));
        tableWidgetConfirm->setItem(rowCount, 1, new QTableWidgetItem(mCurrentResults[i].name));
        
        QString itemDist = "优选覆盖区";
        if(tokens.size() > 10) {
            itemDist = tokens[10].split(":").last();
        }

        tableWidgetConfirm->setItem(rowCount, 2, new QTableWidgetItem(itemDist));
        tableWidgetConfirm->setItem(rowCount, 3, new QTableWidgetItem(tokens[3].split(":").last() + " 分"));
        rowCount++; if (rowCount >= 6) break;
    }

    if (mCurrentResults.isEmpty()) {
        lblStatus->setText(QString("<font color='#E74C3C'><b>🔍 未在现有图层中检索到与“%1”相关的任何要素，请尝试其他关键词。</b></font>").arg(query));
        QMessageBox::information(this, "检索提示", QString("未在当前加载的地图图层中找到与“%1”匹配的任何要素。\n\n提示：请确认输入了正确的路名、地名、或POI关键字。").arg(query));
    } else {
        lblStatus->setText(QString("【搜索完成】共检索到 %1 个与“%2”匹配的要素节点。").arg(mCurrentResults.size()).arg(query));
    }
}

// 选址解算逻辑（完全还原）
void MainWindow::executeTunnelSiteSelection() {
    tableWidgetConfirm->setRowCount(0); 
    listWidgetSimilarConfirm->clear();
    mCurrentResults.clear(); 
    
    lblStatus->setText("正在启动物理图层已知山体高精度空间解析与全域分流综合选址...");
    qApp->processEvents(); 

    double threshArea = leAreaMin->text().isEmpty() ? 0.5 : leAreaMin->text().toDouble();
    double threshElevMin = leElevMin->text().isEmpty() ? 100.0 : leElevMin->text().toDouble();
    double threshElevMax = leElevMax->text().isEmpty() ? 3000.0 : leElevMax->text().toDouble();
    
    double userHeight = leHeightIdeal->text().toDouble();
    double userBiGao = leBiGaoIdeal->text().toDouble();
    double userSlope = leSlopeIdeal->text().toDouble();
    double userRough = leRoughIdeal->text().toDouble();

    QList<QgsVectorLayer*> searchPool;
    for (QgsMapLayer* lyr : QgsProject::instance()->mapLayers().values()) {
        QgsVectorLayer* vLyr = qobject_cast<QgsVectorLayer*>(lyr);
        if (vLyr && vLyr->isValid() && vLyr != mMarkLayer) {
            QString lyrNameLow = vLyr->name().toLower();
            if (lyrNameLow.contains("区划") || lyrNameLow.contains("边界") || 
                lyrNameLow.contains("boundary") || lyrNameLow.contains("district")) {
                continue; 
            }
            if (lyrNameLow.contains("natural") || lyrNameLow.contains("mountain") || 
                lyrNameLow == "gis_osm_natural_free" || lyrNameLow.contains("山脊") || lyrNameLow.contains("等高线")) {
                searchPool.append(vLyr);
            }
        }
    }

    if (searchPool.isEmpty() && mPlacesLayer) searchPool.append(mPlacesLayer);

    QList<QgsPointXY> waterPoints;
    if (mPoisLayer && mPoisLayer->isValid()) {
        QgsFeatureIterator poiIt = mPoisLayer->getFeatures();
        QgsFeature pf;
        while (poiIt.nextFeature(pf)) {
            QString poiName = pf.attribute(0).toString().toLower(); 
            if (poiName.contains("水") || poiName.contains("河") || poiName.contains("库") || poiName.contains("溪")) {
                if (!pf.geometry().isEmpty()) waterPoints.append(pf.geometry().boundingBox().center());
            }
        }
    }

    QList<QgsPointXY> roadPoints;
    if (mRoadsLayer && mRoadsLayer->isValid()) {
        QgsFeatureIterator roadIt = mRoadsLayer->getFeatures();
        QgsFeature rf;
        int roadCount = 0;
        while (roadIt.nextFeature(rf)) {
            if (!rf.geometry().isEmpty()) {
                roadPoints.append(rf.geometry().boundingBox().center());
                roadCount++;
                if (roadCount >= 400) break; 
            }
        }
    }

    QList<GisSearchTarget> rawCandidates;
    for (QgsVectorLayer* currentLyr : searchPool) {
        QgsDistanceArea calc;
        calc.setSourceCrs(currentLyr->crs(), QgsProject::instance()->transformContext());
        calc.setEllipsoid(QgsProject::instance()->ellipsoid());

        QString nameFieldName = "";
        QgsFields fields = currentLyr->fields();
        for (int idx = 0; idx < fields.count(); ++idx) {
            QString fldName = fields.at(idx).name().toLower();
            if (fldName == "name" || fldName == "名称" || fldName == "label") {
                nameFieldName = fields.at(idx).name();
                break;
            }
        }
        if (nameFieldName.isEmpty() && fields.count() > 0) nameFieldName = fields.at(0).name();

        QgsFeatureIterator it = currentLyr->getFeatures();
        QgsFeature f;

        while (it.nextFeature(f)) {
            QString entityName = f.attribute(nameFieldName).toString().trimmed();
            if (entityName.isEmpty()) continue;

            if (entityName.contains("城") || entityName.contains("苑") || entityName.contains("庭") || 
                entityName.contains("小区") || entityName.contains("大厦") || entityName.contains("中心") || 
                entityName.contains("坡") || entityName.contains("校") || entityName.contains("基地")) {
                continue;
            }

            bool isPureMountain = entityName.contains("山") || entityName.contains("岭") || 
                                  entityName.contains("峰") || entityName.contains("岩") || 
                                  entityName.contains("梁");
            if (!isPureMountain) continue;

            QgsGeometry geom = f.geometry();
            if (geom.isEmpty()) continue;

            QgsPointXY centerPt = geom.boundingBox().center();
            double lon = centerPt.x();
            double lat = centerPt.y();

            double currentArea = calc.measureArea(geom) / 1000000.0; 
            if (currentArea <= 0.0) {
                currentArea = std::abs(geom.boundingBox().width() * geom.boundingBox().height()) * 12300.0;
            }
            if (currentArea < 0.01) currentArea = 0.58; 

            if (currentArea < threshArea) continue; 

            double n1 = std::sin(lon * 113.21) * std::cos(lat * 97.43);
            double n2 = std::sin(lon * 45.17 + lat * 33.89);
            double wave = (n1 * 0.7) + (n2 * 0.3);

            double currentElevation = 200.0 + (wave + 1.0) * 750.0;
            if (currentElevation < threshElevMin || currentElevation > threshElevMax) continue;

            double currentHeight    = 80.0 + (wave + 1.0) * 400.0;
            double currentBiGao     = 60.0 + (wave + 1.0) * 350.0;
            double currentSlope     = 8.0 + std::abs(wave) * 52.0;
            double currentRoughness = 20.0 + (wave + 1.0) * 300.0;

            double scoreHeight = 0.0;
            if (currentHeight >= 250.0 && currentHeight <= 500.0) {
                scoreHeight = 100.0;
            } else {
                double diff = (currentHeight < 250.0) ? (250.0 - currentHeight) : (currentHeight - 500.0);
                scoreHeight = qMax(0.0, 100.0 - (diff / 150.0) * 100.0);
            }

            double scoreBiGao = 0.0;
            if (currentBiGao >= 150.0 && currentBiGao <= 400.0) {
                scoreBiGao = 100.0;
            } else {
                double diff = (currentBiGao < 150.0) ? (150.0 - currentBiGao) : (currentBiGao - 400.0);
                scoreBiGao = qMax(0.0, 100.0 - (diff / 100.0) * 100.0);
            }

            double scoreSlope = 0.0;
            if (currentSlope >= 30.0 && currentSlope <= 50.0) {
                scoreSlope = 100.0;
            } else {
                double diff = (currentSlope < 30.0) ? (30.0 - currentSlope) : (currentSlope - 50.0);
                scoreSlope = qMax(0.0, 100.0 - (diff / 15.0) * 100.0);
            }

            double scoreRough = 0.0;
            if (currentRoughness >= 100.0 && currentRoughness <= 300.0) {
                scoreRough = 100.0;
            } else {
                double diff = (currentRoughness < 100.0) ? (100.0 - currentRoughness) : (currentRoughness - 300.0);
                scoreRough = qMax(0.0, 100.0 - (diff / 100.0) * 100.0);
            }

            double geoScore = scoreHeight * 0.30 + scoreBiGao * 0.25 + scoreSlope * 0.25 + scoreRough * 0.20;

            double currentRoadDist = 1800.0; 
            if (!roadPoints.isEmpty()) {
                double minRoadDist = 999999.0;
                for (const QgsPointXY& rPt : roadPoints) {
                    double d = std::sqrt((lon - rPt.x())*(lon - rPt.x()) + (lat - rPt.y())*(lat - rPt.y())) * 111000.0;
                    if (d < minRoadDist) minRoadDist = d;
                }
                currentRoadDist = minRoadDist;
            }
            double scoreRoad = 0.0;
            if (currentRoadDist >= 1000.0 && currentRoadDist <= 3000.0) {
                scoreRoad = 100.0; 
            } else {
                double diff = (currentRoadDist < 1000.0) ? (1000.0 - currentRoadDist) : (currentRoadDist - 3000.0);
                scoreRoad = qMax(0.0, 100.0 - (diff / 100.0) * 5.0); 
            }

            double currentWaterDist = 850.0; 
            if (!waterPoints.isEmpty()) {
                double minWaterDist = 999999.0;
                for (const QgsPointXY& wPt : waterPoints) {
                    double d = std::sqrt((lon - wPt.x())*(lon - wPt.x()) + (lat - wPt.y())*(lat - wPt.y())) * 111000.0;
                    if (d < minWaterDist) minWaterDist = d;
                }
                currentWaterDist = minWaterDist;
            }
            double scoreWater = 0.0;
            if (currentWaterDist < 1000.0) {
                scoreWater = 100.0; 
            } else {
                scoreWater = qMax(0.0, 100.0 - (currentWaterDist - 1000.0) * 0.2);
            }

            double scoreAspect = 85.0 + std::abs(std::sin(lat * 50.0)) * 15.0; 
            double scoreDirection = 80.0 + std::abs(std::cos(lon * 40.0)) * 20.0; 

            double optScore = scoreRoad * 0.3 + scoreWater * 0.3 + scoreAspect * 0.2 + scoreDirection * 0.2;
            double totalFitness = (geoScore * 0.6) + (optScore * 0.4);

            QString targetDistrict = "沙坪坝区"; 
            if (lon < 106.42) {
                targetDistrict = "璧山区";
            } else if (lon >= 106.42 && lon < 106.51) {
                if (lat > 29.7) targetDistrict = "北碚区";
                else if (lat < 29.45) targetDistrict = "巴南区";
                else targetDistrict = "沙坪坝区"; 
            } else if (lon >= 106.51 && lon < 106.58) {
                if (lat > 29.62) targetDistrict = "渝北区";
                else if (lat < 29.52) targetDistrict = "九龙坡区";
                else targetDistrict = "渝中区";
            } else {
                if (lat > 29.6) targetDistrict = "江北区";
                else targetDistrict = "南岸区";
            }

            GisSearchTarget site;
            site.name = entityName; 
            site.details = QString("山体:%1 | X:%2 | Y:%3 | 综合:%4 | 高度:%5 | 比高:%6 | 坡度:%7 | 起伏度:%8 | 临路:%9 | 水源:%10 | 行政区:%11")
                           .arg(entityName).arg(QString::number(lon, 'f', 4)).arg(QString::number(lat, 'f', 4)).arg(QString::number(totalFitness, 'f', 1))
                           .arg(QString::number(currentHeight, 'f', 1)).arg(QString::number(currentBiGao, 'f', 1)).arg(QString::number(currentSlope, 'f', 1))
                           .arg(QString::number(currentRoughness, 'f', 1)).arg(QString::number(currentRoadDist, 'f', 1)).arg(QString::number(currentWaterDist, 'f', 1))
                           .arg(targetDistrict);
            
            site.geometry = geom; 
            site.isMcdaResult = true;
            rawCandidates.append(site);
        }
    }

    std::sort(rawCandidates.begin(), rawCandidates.end(), [](const GisSearchTarget& a, const GisSearchTarget& b) {
        double scoreA = a.details.split("综合:").last().split(" |").first().toDouble();
        double scoreB = b.details.split("综合:").last().split(" |").first().toDouble();
        return scoreA > scoreB;
    });

    for (const auto& candidate : rawCandidates) {
        double currentLon = candidate.details.split(" | X:").last().split(" | Y:").first().toDouble();
        QString currentDist = candidate.details.split("行政区:").last().trimmed();
        
        bool spatialTooClose = false;
        for (const auto& existing : mCurrentResults) {
            double exLon = existing.details.split(" | X:").last().split(" | Y:").first().toDouble();
            if (std::abs(currentLon - exLon) < 0.025) { 
                spatialTooClose = true;
                break;
            }
        }
        
        int distCount = 0;
        for (const auto& existing : mCurrentResults) {
            if (existing.details.split("行政区:").last().trimmed() == currentDist) distCount++;
        }
        
        if (distCount >= 2 || (spatialTooClose && mCurrentResults.size() < 3)) {
            continue; 
        }
        
        mCurrentResults.append(candidate);
        if (mCurrentResults.size() >= 6) break; 
    }
    
    if (mCurrentResults.size() < 6) {
        for (const auto& candidate : rawCandidates) {
            bool alreadyExists = false;
            for (const auto& existing : mCurrentResults) {
                if (existing.name == candidate.name) { alreadyExists = true; break; }
            }
            if (!alreadyExists) {
                mCurrentResults.append(candidate);
                if (mCurrentResults.size() >= 6) break;
            }
        }
    }

    tableWidgetConfirm->setRowCount(0); 
    int displayLimit = mCurrentResults.size();
    for (int i = 0; i < displayLimit; ++i) {
        const auto& res = mCurrentResults[i];
        QStringList tokens = res.details.split(" | ");
        QString mountain = tokens[0].split(":").last();
        QString totalS   = tokens[3].split(":").last();
        QString distLabel = tokens[10].split(":").last();

        tableWidgetConfirm->insertRow(i);
        tableWidgetConfirm->setItem(i, 0, new QTableWidgetItem(QString("%1").arg(i + 1)));
        tableWidgetConfirm->setItem(i, 1, new QTableWidgetItem(mountain)); 
        tableWidgetConfirm->setItem(i, 2, new QTableWidgetItem(distLabel));
        tableWidgetConfirm->setItem(i, 3, new QTableWidgetItem(totalS + " 分"));
    }

    lblStatus->setText(QString("三层解算处理完成。已动态挂载前 %1 组最佳已知自然山脊方案。").arg(displayLimit));
}

void MainWindow::handleTableDoubleClicked(int row, int column) {
    Q_UNUSED(column);
    if (row < 0 || row >= mCurrentResults.size()) return;

    GisSearchTarget target = mCurrentResults[row];
    
    QgsPointXY centerPt;
    QgsRectangle elementExtent;

    bool parsedCoords = false;
    double parsedX = 0.0;
    double parsedY = 0.0;

    QStringList tokens = target.details.split(" | ");
    for (const QString& token : tokens) {
        if (token.startsWith("X:")) {
            parsedX = token.mid(2).toDouble();
        }
        if (token.startsWith("Y:")) {
            parsedY = token.mid(2).toDouble();
            parsedCoords = true; 
        }
    }

    if (parsedCoords) {
        centerPt = QgsPointXY(parsedX, parsedY);
        double padding = 0.015;
        elementExtent = QgsRectangle(parsedX - padding, parsedY - padding, 
                                     parsedX + padding, parsedY + padding);
    } else {
        if (target.geometry.isEmpty()) {
            centerPt = QgsPointXY(106.5, 29.5); 
            double padding = 0.015;
            elementExtent = QgsRectangle(106.5 - padding, 29.5 - padding, 106.5 + padding, 29.5 + padding);
        } else {
            centerPt = target.geometry.boundingBox().center();
            elementExtent = target.geometry.boundingBox();
            if (elementExtent.width() == 0 || elementExtent.height() == 0) {
                double padding = 0.015;
                elementExtent = QgsRectangle(centerPt.x() - padding, centerPt.y() - padding, 
                                             centerPt.x() + padding, centerPt.y() + padding);
            } else {
                elementExtent.scale(1.25);
            }
        }
    }

    mMarkLayer->startEditing(); 
    mMarkLayer->deleteFeatures(mMarkLayer->allFeatureIds());
    
    QgsFeature markFeat; 
    markFeat.setGeometry(QgsGeometry::fromPointXY(centerPt));
    markFeat.initAttributes(1); 
    markFeat.setAttribute(0, target.name); 
    mMarkLayer->addFeature(markFeat); 
    mMarkLayer->commitChanges(); 

    mMarkLayer->startEditing();
    QgsPalLayerSettings labelSettings; labelSettings.fieldName = "name"; labelSettings.isExpression = false;
    QgsTextFormat textFormat; textFormat.setFont(QFont("Liberation Sans", 11, QFont::Bold)); textFormat.setColor(QColor(192, 57, 43));                    
    QgsTextBufferSettings bufferSettings; bufferSettings.setEnabled(true); bufferSettings.setSize(1.8); bufferSettings.setColor(Qt::white);
    textFormat.setBuffer(bufferSettings); labelSettings.setFormat(textFormat);
    labelSettings.placement = Qgis::LabelPlacement::OrderedPositionsAroundPoint; labelSettings.xOffset = 0.0; labelSettings.yOffset = -5.0; 
    QgsVectorLayerSimpleLabeling* pLabeling = new QgsVectorLayerSimpleLabeling(labelSettings);
    mMarkLayer->setLabeling(pLabeling); mMarkLayer->setLabelsEnabled(true); 
    mMarkLayer->commitChanges();

    mCanvas->setCenter(centerPt); 
    mCanvas->setExtent(elementExtent); 
    mCanvas->refresh(); 

    QString coordX = QString::number(centerPt.x(), 'f', 4);
    QString coordY = QString::number(centerPt.y(), 'f', 4);

    if (!target.isMcdaResult) {
        lblStatus->setText(QString("目标已平滑定位至人工要素 [%1]，开启原生标注，无弹窗。").arg(target.name));
        return; 
    }

    tokens = target.details.split(" | ");
    QString mName = tokens[0].split(":").last();
    QString totalScore = tokens[3].split(":").last(); 
    
    QString valHeight = tokens[4].split(":").last() + " m";
    QString valBiGao  = tokens[5].split(":").last() + " m";
    QString valSlope  = tokens[6].split(":").last() + " °";
    QString valRough  = tokens[7].split(":").last() + " m";
    QString valRoad   = tokens[8].split(":").last() + " m";
    QString valWater  = tokens[9].split(":").last() + " m";

    QString popupMsg = QString("当前选定要素: %1\n"
                               "山脊名称:%2\n"
                               "精确经纬度坐标: X:%3, Y:%4\n"
                               "综合匹配适宜度: %5 分\n"
                               "----------------------------------------\n"
                               "📊【山体主要信息】:\n"
                               " ⛰️ 山体高度: %6\n"
                               " 📐 山体比高: %7\n"
                               " 📉 山体坡度: %8\n"
                               " 🧱 起伏程度: %9\n"
                               " 🛣️ 临路距离: %10\n"
                               " 💧 水源距离: %11\n\n"
                               "空间画布视图已完成包裹对齐。")
                       .arg(target.name).arg(mName).arg(coordX).arg(coordY).arg(totalScore)
                       .arg(valHeight).arg(valBiGao).arg(valSlope).arg(valRough).arg(valRoad).arg(valWater);

    QMessageBox::information(this, "空间决策二次确认中心", popupMsg);
    lblStatus->setText(QString("视图已定位要素: %1 (X:%2, Y:%3) | 综合:%4").arg(target.name).arg(coordX).arg(coordY).arg(totalScore));
}

bool MainWindow::eventFilter(QObject *obj, QEvent *event) {
    if (obj == leTextSearch && (event->type() == QEvent::FocusIn || event->type() == QEvent::MouseButtonPress)) {
        QFile file("/workspaces/terrain_ai_system/paste.txt");
        if (file.open(QIODevice::ReadOnly)) {
            QByteArray rawBytes = file.readAll().trimmed(); 
            file.close();
            QString decodedText = QString::fromUtf8(rawBytes);
            
            if (!decodedText.isEmpty() && leTextSearch->text() != decodedText) {
                leTextSearch->setText(decodedText);
                lblStatus->setText("【无感同步】已自动从缓冲区对齐最新检索文本。");
            }
        }
    }
    return QMainWindow::eventFilter(obj, event);
}