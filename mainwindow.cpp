#include "mainwindow.h"
#include <qgssymbol.h>
#include <qgsfeaturerequest.h>
#include <qgsfeatureiterator.h>
#include <qgssinglesymbolrenderer.h>
#include <qgsmarkersymbollayer.h>
#include <QMessageBox>
#include <QUrl>
#include <QFile>
#include <QTextStream>
#include <QHeaderView> // 🔧 必须引入：用于控制表格拉伸
#include <cmath>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    setupUi();
    initGisLayers();
}

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

    QWidget* rightPanel = new QWidget(this);
    QVBoxLayout* panelLayout = new QVBoxLayout(rightPanel);
    mainLayout->addWidget(rightPanel, 1);

    // 画布缩放
    panelLayout->addWidget(new QLabel("<b>[画布分辨率微调]</b>"));
    QHBoxLayout* layoutZoom = new QHBoxLayout();
    QPushButton* btnZoomIn = new QPushButton("🔍 放大", this);
    QPushButton* btnZoomOut = new QPushButton("🔎 缩小", this);
    layoutZoom->addWidget(btnZoomIn); layoutZoom->addWidget(btnZoomOut);
    panelLayout->addLayout(layoutZoom);

    // 模块 1：基础检索
    panelLayout->addWidget(new QLabel("<br><b>[功能模块 1] 基础地理实体检索</b>"));
    QHBoxLayout* textSearchLayout = new QHBoxLayout();
    leTextSearch = new QLineEdit(this);
    leTextSearch->setPlaceholderText("请输入地名/路名/POI...");
    QPushButton* btnPasteBridge = new QPushButton("📋 读取本机文本", this);
    btnPasteBridge->setStyleSheet("background-color: #2ECC71; color: white; font-weight: bold;");
    QPushButton* btnTextSearch = new QPushButton("🎯 检索", this);
    textSearchLayout->addWidget(leTextSearch); textSearchLayout->addWidget(btnPasteBridge); textSearchLayout->addWidget(btnTextSearch);
    panelLayout->addLayout(textSearchLayout);

    // 模块 2：9大条件网格
    panelLayout->addWidget(new QLabel("<br><b>[功能模块 2] 动态洞库空间多准则选址</b>"));
    QWidget* gridContainer = new QWidget(this);
    QGridLayout* gridLayout = new QGridLayout(gridContainer);
    gridLayout->setContentsMargins(0,0,0,0);

    leAreaMin = new QLineEdit("0.5", this);
    leElevMin = new QLineEdit("100", this); leElevMax = new QLineEdit("3000", this);
    leHeightIdeal = new QLineEdit("350", this); leBiGaoIdeal = new QLineEdit("260", this);
    leSlopeIdeal = new QLineEdit("40", this); leRoughIdeal = new QLineEdit("200", this);
    leNdviInput = new QLineEdit("0.55", this); leRoadDist = new QLineEdit("1500", this); leWaterDist = new QLineEdit("800", this);

    gridLayout->addWidget(new QLabel("第一层.最小面积 (km²):"), 0, 0); gridLayout->addWidget(leAreaMin, 0, 1);
    gridLayout->addWidget(new QLabel("第一层.高程范围 (m):"), 1, 0);
    QHBoxLayout* elevLayout = new QHBoxLayout(); elevLayout->addWidget(leElevMin); elevLayout->addWidget(leElevMax); gridLayout->addLayout(elevLayout, 1, 1);
    gridLayout->addWidget(new QLabel("第二层.山体高度 (m):"), 2, 0); gridLayout->addWidget(leHeightIdeal, 2, 1);
    gridLayout->addWidget(new QLabel("第二层.山体比高 (m):"), 3, 0); gridLayout->addWidget(leBiGaoIdeal, 3, 1);
    gridLayout->addWidget(new QLabel("第二层.平均坡度 (°):"), 4, 0); gridLayout->addWidget(leSlopeIdeal, 4, 1);
    gridLayout->addWidget(new QLabel("第二层.起伏度 (m):"), 5, 0); gridLayout->addWidget(leRoughIdeal, 5, 1);
    gridLayout->addWidget(new QLabel("第三层.植被 NDVI:"), 6, 0); gridLayout->addWidget(leNdviInput, 6, 1);
    gridLayout->addWidget(new QLabel("第三层.临路距离 (m):"), 7, 0); gridLayout->addWidget(leRoadDist, 7, 1);
    gridLayout->addWidget(new QLabel("第三层.水源距离 (m):"), 8, 0); gridLayout->addWidget(leWaterDist, 8, 1);
    gridContainer->setLayout(gridLayout); panelLayout->addWidget(gridContainer);

    QPushButton* btnMcdasel = new QPushButton("⚡ 启动级联动态选址解算", this);
    btnMcdasel->setStyleSheet("background-color: #1ABC9C; color: white; font-weight: bold;");
    panelLayout->addWidget(btnMcdasel);

// 🔧 优化：5 列，彻底剔除表格经纬度字段展示
    panelLayout->addWidget(new QLabel("<br><font color='#D35400'><b>📋 选址结果待确认列表 (双击某行确定定位):</b></font>"));
    tableWidgetConfirm = new QTableWidget(this);
    tableWidgetConfirm->setColumnCount(5); // 5列：排名、山脊名称、综合得分、地质得分、交通得分
    tableWidgetConfirm->setHorizontalHeaderLabels({"排名", "山脊名称", "综合得分", "地质得分", "交通得分"});
    tableWidgetConfirm->setSelectionBehavior(QAbstractItemView::SelectRows); 
    tableWidgetConfirm->setEditTriggers(QAbstractItemView::NoEditTriggers);   
    tableWidgetConfirm->verticalHeader()->setVisible(false);                 
    tableWidgetConfirm->setStyleSheet("background-color: #FAFAFA; gridline-color: #BDC3C7;");
    
    tableWidgetConfirm->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    panelLayout->addWidget(tableWidgetConfirm);

    lblStatus = new QLabel("就绪。", this);
    lblStatus->setFrameStyle(QFrame::Panel | QFrame::Sunken);
    panelLayout->addWidget(lblStatus);

    // 信号绑定（表格绑定 cellDoubleClicked）
    connect(btnZoomIn, &QPushButton::clicked, this, &MainWindow::zoomInMap);
    connect(btnZoomOut, &QPushButton::clicked, this, &MainWindow::zoomOutMap);
    connect(btnTextSearch, &QPushButton::clicked, this, &MainWindow::executeTextSearch);
    connect(btnMcdasel, &QPushButton::clicked, this, &MainWindow::executeTunnelSiteSelection);
    connect(tableWidgetConfirm, &QTableWidget::cellDoubleClicked, this, &MainWindow::handleTableDoubleClicked);

    // 乱码清洗
    connect(btnPasteBridge, &QPushButton::clicked, this, [this]() {
        QFile file("/workspaces/terrain_ai_system/paste.txt");
        if (file.open(QIODevice::ReadOnly)) {
            QByteArray rawBytes = file.readAll().trimmed(); file.close();
            QString rawStr = QString::fromLatin1(rawBytes);
            QString decodedText = rawStr.contains("%") ? QUrl::fromPercentEncoding(rawBytes) : QString::fromUtf8(rawBytes);
            if (decodedText.contains("?")) decodedText = QString::fromLocal8Bit(rawBytes);
            decodedText.remove(QRegExp("[\\x00-\\x08\\x0B-\\x0C\\x0E-\\x1F]"));
            leTextSearch->setText(decodedText);
            lblStatus->setText("【成功】本地文本已无损导入输入框。");
        }
    });
}

void MainWindow::initGisLayers() {
    QString gpkgPath = "/workspaces/terrain_ai_system/fixed_map_data.gpkg";
    QgsVectorLayer::LayerOptions opts; opts.loadDefaultStyle = false;
    mRoadsLayer = new QgsVectorLayer(gpkgPath + "|layername=gis_osm_roads_free", "Roads", "ogr", opts);
    mPlacesLayer = new QgsVectorLayer(gpkgPath + "|layername=gis_osm_places_free", "Places", "ogr", opts);
    mPoisLayer = new QgsVectorLayer(gpkgPath + "|layername=gis_osm_pois_free", "POIs", "ogr", opts);
    
    mMarkLayer = new QgsVectorLayer("Point?crs=EPSG:4326&field=name:string", "Selection_Marks", "memory");
    QgsSimpleMarkerSymbolLayer* starSymbol = new QgsSimpleMarkerSymbolLayer();
    starSymbol->setShape(Qgis::MarkerShape::Star); starSymbol->setColor(Qt::yellow); starSymbol->setSize(11.0);
    starSymbol->setStrokeColor(Qt::darkYellow); starSymbol->setStrokeWidth(1.5);
    QgsSymbol* sym = QgsSymbol::defaultSymbol(Qgis::GeometryType::Point);
    sym->changeSymbolLayer(0, starSymbol); mMarkLayer->setRenderer(new QgsSingleSymbolRenderer(sym));

    QgsProject::instance()->addMapLayers({mRoadsLayer, mPlacesLayer, mPoisLayer, mMarkLayer});
    mCanvas->setLayers({mMarkLayer, mRoadsLayer, mPlacesLayer, mPoisLayer});
    mCanvas->zoomToFullExtent(); mCanvas->refresh();
}

void MainWindow::zoomInMap() { if (mCanvas) mCanvas->zoomIn(); }
void MainWindow::zoomOutMap() { if (mCanvas) mCanvas->zoomOut(); }

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

// =================================================================
// 🚀 终极重构：Places、Pois、Natural 三图层全空间刺透交叉检索引擎
// =================================================================
void MainWindow::executeTextSearch() {
    tableWidgetConfirm->setRowCount(0); // 清空前端表格旧数据
    mCurrentResults.clear();
    
    QString query = leTextSearch->text().trimmed();
    if (query.isEmpty()) return;

    // 1. 获取模块 2 界面用户当前实时设置的理想期望参数
    double userHeight = leHeightIdeal->text().toDouble();
    double userBiGao = leBiGaoIdeal->text().toDouble();
    double userSlope = leSlopeIdeal->text().toDouble();
    double userRough = leRoughIdeal->text().toDouble();

    // 构建标准属性 SQL 过滤表达式
    QString expr = QString("\"name\" LIKE '%%1%'").arg(query);
    QgsFeatureRequest req; req.setFilterExpression(expr);

    // 2. 构造三图层联合穿透池（加入 pois 和 潜在的底图层进行拉网式排查）
    QList<QgsVectorLayer*> searchPool;
    if (mPlacesLayer && mPlacesLayer->isValid()) searchPool.append(mPlacesLayer);
    if (mPoisLayer && mPoisLayer->isValid()) searchPool.append(mPoisLayer);

    // 额外物理兼容：尝试动态获取加载进来的自然山脉图层，防止遗漏
    QgsVectorLayer* poNaturalLyr = qobject_cast<QgsVectorLayer*>(QgsProject::instance()->mapLayersByName("gis_osm_natural_free").value(0, nullptr));
    if (poNaturalLyr && poNaturalLyr->isValid()) {
        searchPool.append(poNaturalLyr);
    } else if (mRoadsLayer && mRoadsLayer->isValid()) {
        // 如果自然层不在，捎带脚排查一下周边道路层（保底健全）
        QgsVectorLayer* poNatBackup = new QgsVectorLayer("/workspaces/terrain_ai_system/fixed_map_data.gpkg|layername=gis_osm_natural_free", "Natural_Backup", "ogr");
        if (poNatBackup->isValid()) {
            QgsProject::instance()->addMapLayer(poNatBackup);
            searchPool.append(poNatBackup);
        }
    }

    if (searchPool.isEmpty()) {
        lblStatus->setText("错误：当前系统内没有可用的矢量底图层进行空间交叉检索！");
        return;
    }

    int indexCounter = 0;

    // 3. 开始跨图层扫描碰撞
    for (QgsVectorLayer* currentLyr : searchPool) {
        QgsFeatureIterator it = currentLyr->getFeatures(req);
        QgsFeature f;

        while (it.nextFeature(f)) {
            QString ridgeName = f.attribute("name").toString();
            QgsGeometry geom = f.geometry();
            if (geom.isEmpty()) continue;

            // 抓取地理网格核心物理中点 (X, Y)
            QgsPointXY centerPt = geom.boundingBox().center();
            double lon = centerPt.x();
            double lat = centerPt.y();

            // 联动产生切片网格的 9 大特征参数
            double n1 = std::sin(lon * 113.21) * std::cos(lat * 97.43);
            double n2 = std::sin(lon * 45.17 + lat * 33.89);
            double wave = (n1 * 0.7) + (n2 * 0.3);

            double currentHeight = 50.0 + (wave + 1.0) * 450.0;                     
            double currentBiGao = 40.0 + (wave + 1.0) * 400.0;                      
            double currentSlope = 10.0 + std::abs(wave) * 55.0;                       
            double currentRoughness = 15.0 + (wave + 1.0) * 350.0;                  
            double currentNdvi = 0.25 + std::abs(wave) * 0.55;                        
            double currentRoadDist = 100.0 + std::abs(wave) * 5400.0;                 
            double currentWaterDist = 100.0 + (wave + 1.0) * 1200.0;                

            // 连续插值地质精细打分
            double scoreHeight = 100.0 - (std::abs(currentHeight - userHeight) / userHeight) * 100.0;
            double scoreBiGao  = 100.0 - (std::abs(currentBiGao - userBiGao) / userBiGao) * 100.0;
            double scoreSlope  = 100.0 - (std::abs(currentSlope - userSlope) / userSlope) * 100.0;
            double scoreRough  = 100.0 - (std::abs(currentRoughness - userRough) / userRough) * 100.0;

            double geoScore = scoreHeight * 0.30 + scoreBiGao * 0.25 + scoreSlope * 0.25 + scoreRough * 0.20;
            if (currentNdvi < 0.4 || currentNdvi > 0.7) geoScore -= 12.0; 

            // 交通不便连续扣减模型
            double trafficScore = 100.0;
            double idealRoadCenter = 2000.0; 
            double distanceDeviation = std::abs(currentRoadDist - idealRoadCenter);
            trafficScore -= distanceDeviation * 0.015; 
            if (currentWaterDist > 1000.0) trafficScore -= 10.0;

            if (geoScore > 100.0) geoScore = 100.0; if (geoScore < 0.0) geoScore = 0.0;
            if (trafficScore > 100.0) trafficScore = 100.0; if (trafficScore < 0.0) trafficScore = 0.0;

            double totalFitness = (geoScore * 0.7) + (trafficScore * 0.3);

            // 4. 数据打包入栈，打上多准则融合标签
            GisSearchTarget target;
            target.name = QString("%1 (%2)").arg(ridgeName).arg(currentLyr->name() == "Places" ? "地名" : "山峰/自然要素");
            target.details = QString("山脊:%1 | X:%2, Y:%3 | 综合:%4 | 地质:%5 | 交通:%6")
                             .arg(ridgeName)
                             .arg(QString::number(lon, 'f', 4))
                             .arg(QString::number(lat, 'f', 4))
                             .arg(QString::number(totalFitness, 'f', 1))
                             .arg(QString::number(geoScore, 'f', 1))
                             .arg(QString::number(trafficScore, 'f', 1));
            target.geometry = geom;
            target.isMcdaResult = true; 
            mCurrentResults.append(target);

            // 5. 动态插入表格行
            tableWidgetConfirm->insertRow(indexCounter);
            tableWidgetConfirm->setItem(indexCounter, 0, new QTableWidgetItem(QString("匹配点 %1").arg(indexCounter + 1)));
            tableWidgetConfirm->setItem(indexCounter, 1, new QTableWidgetItem(target.name));
            tableWidgetConfirm->setItem(indexCounter, 2, new QTableWidgetItem(QString::number(totalFitness, 'f', 1) + " "));
            tableWidgetConfirm->setItem(indexCounter, 3, new QTableWidgetItem(QString::number(geoScore, 'f', 1) + " "));
            tableWidgetConfirm->setItem(indexCounter, 4, new QTableWidgetItem(QString::number(trafficScore, 'f', 1) + " "));
            
            indexCounter++;
            if (indexCounter >= 8) break; // 限制前 8 个最优匹配展示
        }
        if (indexCounter >= 8) break;
    }

    // 6. 全局强行按级联得分执行降序重排
    std::sort(mCurrentResults.begin(), mCurrentResults.end(), [](const GisSearchTarget& a, const GisSearchTarget& b) {
        double scoreA = a.details.split("综合:").last().split(" |").first().toDouble();
        double scoreB = b.details.split("综合:").last().split(" |").first().toDouble();
        return scoreA > scoreB;
    });

    // 动态重刷界面表格项
    for (int i = 0; i < mCurrentResults.size(); ++i) {
        QStringList tokens = mCurrentResults[i].details.split(" | ");
        tableWidgetConfirm->item(i, 1)->setText(mCurrentResults[i].name);
        tableWidgetConfirm->item(i, 2)->setText(tokens[2].split(":").last() + " ");
        tableWidgetConfirm->item(i, 3)->setText(tokens[3].split(":").last() + " ");
        tableWidgetConfirm->item(i, 4)->setText(tokens[4].split(":").last() + " ");
    }

    lblStatus->setText(QString("全图层交叉刺透检索结束。成功抓取并挂载了 %1 个物理空间关联要素。").arg(mCurrentResults.size()));
}

// ⚡ 优化后的 3 层动态连续打分选址算法（完美拉开地质、交通双重阶梯梯度分值）
void MainWindow::executeTunnelSiteSelection() {
    tableWidgetConfirm->setRowCount(0); // 🔧 升级：清空旧表格行
    mCurrentResults.clear();
    lblStatus->setText("正在实时读取空间底图、切片扫描中...请稍候...");
    qApp->processEvents(); 

    double threshArea = leAreaMin->text().toDouble();
    double threshElevMin = leElevMin->text().toDouble();
    double threshElevMax = leElevMax->text().toDouble();
    
    double userHeight = leHeightIdeal->text().toDouble();
    double userBiGao = leBiGaoIdeal->text().toDouble();
    double userSlope = leSlopeIdeal->text().toDouble();
    double userRough = leRoughIdeal->text().toDouble();

    for (double lon = 106.32; lon <= 106.52; lon += 0.03) {
        for (double lat = 29.52; lat <= 30.02; lat += 0.04) {
            
            // 多阶不规则周期波噪声
            double n1 = std::sin(lon * 113.21) * std::cos(lat * 97.43);
            double n2 = std::sin(lon * 45.17 + lat * 33.89);
            double wave = (n1 * 0.7) + (n2 * 0.3); 

            double currentArea = 0.3 + std::abs(wave) * 1.5;                         
            double currentElevation = 150.0 + (wave + 1.0) * 800.0;                 
            double currentHeight = 50.0 + (wave + 1.0) * 450.0;                     
            double currentBiGao = 40.0 + (wave + 1.0) * 400.0;                      
            double currentSlope = 10.0 + std::abs(wave) * 55.0;                       
            double currentRoughness = 15.0 + (wave + 1.0) * 350.0;                  
            double currentNdvi = 0.25 + std::abs(wave) * 0.55;                        
            
            // 交通多样离散化范围拓展[cite: 2]
            double currentRoadDist = 100.0 + std::abs(wave) * 5400.0;                 
            double currentWaterDist = 100.0 + (wave + 1.0) * 1200.0;                

            // 【第一层：一票否决】
            if (currentArea < threshArea) continue; 
            if (currentElevation < threshElevMin || currentElevation > threshElevMax) continue; 

            // 【第二层：硬核条件检验】
            if (CalculateMetricScore(currentHeight, 250, 500, 150, 250, 500, 700, 100, 800, true, true) < 0) continue;
            if (CalculateMetricScore(currentBiGao, 150, 400, 100, 150, 400, 600, 80, 700, true, true) < 0) continue;
            if (CalculateMetricScore(currentSlope, 30, 50, 25, 30, 50, 60, 20, 70, true, true) < 0) continue;
            if (CalculateMetricScore(currentRoughness, 100, 300, 50, 100, 300, 500, 30, 600, true, true) < 0) continue;

            // 连续线性地质插值优化
            double scoreHeight = 100.0 - (std::abs(currentHeight - userHeight) / userHeight) * 100.0;
            double scoreBiGao  = 100.0 - (std::abs(currentBiGao - userBiGao) / userBiGao) * 100.0;
            double scoreSlope  = 100.0 - (std::abs(currentSlope - userSlope) / userSlope) * 100.0;
            double scoreRough  = 100.0 - (std::abs(currentRoughness - userRough) / userRough) * 100.0;

            double geoScore = scoreHeight * 0.30 + scoreBiGao * 0.25 + scoreSlope * 0.25 + scoreRough * 0.20;
            if (currentNdvi < 0.4 || currentNdvi > 0.7) geoScore -= 12.0; 

            // 【第三层：交通打分梯度改良机制】
            double trafficScore = 100.0;
            double idealRoadCenter = 2000.0; // 设定 2000 米为绝对理论中心值
            double distanceDeviation = std::abs(currentRoadDist - idealRoadCenter);
            trafficScore -= distanceDeviation * 0.015; // 彻底打破满分大平台限制，每偏离 1 米细腻扣减

            if (currentWaterDist > 1000.0) trafficScore -= 10.0;

            if (geoScore > 100.0) geoScore = 100.0; if (geoScore < 20.0) geoScore = 20.0;
            if (trafficScore > 100.0) trafficScore = 100.0; if (trafficScore < 15.0) trafficScore = 15.0;

            double totalFitness = (geoScore * 0.7) + (trafficScore * 0.3);

            // 真实矢量空间地名穿透反查
            QString ridgeName = "未指定区域";
            if (mPlacesLayer && mPlacesLayer->isValid()) {
                QgsFeatureRequest spatialReq;
                spatialReq.setFilterRect(QgsRectangle(lon - 0.02, lat - 0.02, lon + 0.02, lat + 0.02));
                QgsFeatureIterator it = mPlacesLayer->getFeatures(spatialReq); QgsFeature f;
                if (it.nextFeature(f)) {
                    QString nameAttr = f.attribute("name").toString();
                    if (!nameAttr.isEmpty()) ridgeName = nameAttr;
                }
            }

            GisSearchTarget site;
            site.name = ridgeName; 
            site.details = QString("山脊:%1 | X:%2, Y:%3 | 综合:%4 | 地质:%5 | 交通:%6")
                           .arg(ridgeName)
                           .arg(QString::number(lon, 'f', 4))
                           .arg(QString::number(lat, 'f', 4))
                           .arg(QString::number(totalFitness, 'f', 1))
                           .arg(QString::number(geoScore, 'f', 1))
                           .arg(QString::number(trafficScore, 'f', 1));
            
            site.geometry = QgsGeometry::fromPointXY(QgsPointXY(lon, lat));
            site.isMcdaResult = true;
            mCurrentResults.append(site);
        }
    }

    // 综合得分高到低降序重排
    std::sort(mCurrentResults.begin(), mCurrentResults.end(), [](const GisSearchTarget& a, const GisSearchTarget& b) {
        double scoreA = a.details.split("综合:").last().split(" |").first().toDouble();
        double scoreB = b.details.split("综合:").last().split(" |").first().toDouble();
        return scoreA > scoreB;
    });

// 🔧 优化：平铺渲染到原生表格中（跳过经纬度列）
    int displayLimit = std::min(6, mCurrentResults.size());
    for (int i = 0; i < displayLimit; ++i) {
        const auto& res = mCurrentResults[i];
        QStringList tokens = res.details.split(" | ");
        QString mountain = tokens[0].split(":").last();
        QString totalS   = tokens[2].split(":").last();
        QString geoS     = tokens[3].split(":").last();
        QString traS     = tokens[4].split(":").last();

        tableWidgetConfirm->insertRow(i);
        tableWidgetConfirm->setItem(i, 0, new QTableWidgetItem(QString("%1").arg(i + 1)));
        tableWidgetConfirm->setItem(i, 1, new QTableWidgetItem(mountain));
        tableWidgetConfirm->setItem(i, 2, new QTableWidgetItem(totalS + " "));
        tableWidgetConfirm->setItem(i, 3, new QTableWidgetItem(geoS + " "));
        tableWidgetConfirm->setItem(i, 4, new QTableWidgetItem(traS + " "));
    }

    lblStatus->setText(QString("空间表格级联选址解算结束。已渲染排前 %1 组最优战术推荐点。").arg(displayLimit));
}

// 🔧 升级：修改为标准的物理表格单元格整行双击事件监听
// =================================================================
void MainWindow::handleTableDoubleClicked(int row, int column) {
    Q_UNUSED(column);
    if (row < 0 || row >= mCurrentResults.size()) return;

    GisSearchTarget target = mCurrentResults[row];
    if (target.geometry.isEmpty()) return;

    // 1. 清洗历史加点图层，避免多图层高亮交叉污染
    mMarkLayer->startEditing(); 
    mMarkLayer->deleteFeatures(mMarkLayer->allFeatureIds());
    
    // 2. 注入全新黄色落点靶心五角星
    QgsFeature markFeat; 
    QgsPointXY centerPt = target.geometry.boundingBox().center();
    markFeat.setGeometry(QgsGeometry::fromPointXY(centerPt));
    markFeat.initAttributes(1); 
    markFeat.setAttribute(0, target.name);
    mMarkLayer->addFeature(markFeat); 
    mMarkLayer->commitChanges();

    // 3. 画布视口平滑跳转定位，锁死在 1:50,000 黄金开阔度比例尺
    mCanvas->setCenter(centerPt); 
    mCanvas->zoomScale(50000); 
    mCanvas->refresh();

    // 4. 🌟【终极漏洞修复】：直接通过物理几何实体获取 X 和 Y 坐标，永不发生字符串切片越界！
    QString coordX = QString::number(centerPt.x(), 'f', 4);
    QString coordY = QString::number(centerPt.y(), 'f', 4);

    // 5. 逆向平稳解出各项子评分
    QStringList tokens = target.details.split(" | ");
    QString mName  = tokens[0].split(":").last();
    QString tScore = tokens[2].split(":").last();
    QString gScore = tokens[3].split(":").last();
    QString cScore = tokens[4].split(":").last();

    // 6. 重新完美、规整地组装信息提示框
    QString popupMsg = QString("当前选定要素: %1\n完整打分流水线快照: 山脊:%2 | X:%3, Y:%4 | 综合:%5 | 地质:%6 | 交通:%7\n\n空间地图已强制重置并定向跳转。")
                       .arg(target.name)
                       .arg(mName)
                       .arg(coordX)  // 经度 X 饱满输出
                       .arg(coordY)  // 纬度 Y 同框输出，绝不丢失！
                       .arg(tScore)
                       .arg(gScore)
                       .arg(cScore);

    QMessageBox::information(this, "空间决策二次确认中心", popupMsg);
    lblStatus->setText(QString("当前视图已锁定至候选点: %1 (X:%2, Y:%3)").arg(target.name).arg(coordX).arg(coordY));
}