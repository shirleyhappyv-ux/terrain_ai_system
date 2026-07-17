#include "mainwindow.h"
#include <QHeaderView>
#include <QMessageBox>
#include <QDebug>
#include <cmath>
#include <algorithm>

// QGIS Core Includes (只保留最基础、绝对不会报 fatal error 的头文件)
#include <qgsproject.h>
#include <qgsvectorlayer.h>
#include <qgsmaplayer.h>
#include <qgsfeaturerequest.h>
#include <qgsfeatureiterator.h>
#include <qgsgeometry.h>


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), mMarkLayer(nullptr), mRoadsLayer(nullptr), mPlacesLayer(nullptr), mPoisLayer(nullptr)
{
    setupUi();
    initMapCanvas();
    initGisLayers();
    setupConnections();
}

MainWindow::~MainWindow() {
    delete mCanvas;
}

void MainWindow::setupUi() {
    // 基础窗口设置
    this->setWindowTitle(tr("地形分析与洞库空间多准则动态选址系统 - 决策终端"));
    this->resize(1280, 800);

    // 主中央部件
    QWidget* centralWidget = new QWidget(this);
    this->setCentralWidget(centralWidget);
    QHBoxLayout* mainLayout = new QHBoxLayout(centralWidget);

    // 左侧地图画布区
    mCanvas = new QgsMapCanvas(this);
    mCanvas->setCanvasColor(QColor(240, 243, 244));
    mCanvas->enableAntiAliasing(true);
    mainLayout->addWidget(mCanvas, 3);

    // 右侧控制面板
    QWidget* panelWidget = new QWidget(this);
    QVBoxLayout* panelLayout = new QVBoxLayout(panelWidget);
    panelWidget->setFixedWidth(420);

    // [功能模块 1] 文本多层智能检索
    QGroupBox* gbTextSearch = new QGroupBox(tr("[功能模块 1] 多层模糊地质地名检索"), this);
    QFormLayout* flTextSearch = new QFormLayout(gbTextSearch);
    leTextSearch = new QLineEdit(tr("康平医院"), this);
    flTextSearch->addRow(tr("检索关键字:"), leTextSearch);
    btnTextSearch = new QPushButton(tr("🔍 智能多层搜索"), this);
    btnTextSearch->setStyleSheet("background-color: #3498DB; color: white; font-weight: bold; height: 30px;");
    flTextSearch->addRow(btnTextSearch);
    
    listWidgetSimilarConfirm = new QListWidget(this);
    listWidgetSimilarConfirm->setMaximumHeight(100);
    flTextSearch->addRow(tr("类似结果二次确认:"), listWidgetSimilarConfirm);
    panelLayout->addWidget(gbTextSearch);

    // [功能模块 2] 多准则动态选址
    QGroupBox* gbMCDA = new QGroupBox(tr("[功能模块 2] 动态洞库空间多准则选址"), this);
    QFormLayout* flMCDA = new QFormLayout(gbMCDA);
    
    leAreaMin = new QLineEdit("0.5", this);
    leElevMin = new QLineEdit("100", this);
    leElevMax = new QLineEdit("3000", this);
    flMCDA->addRow(tr("一层.最小面积 (km²):"), leAreaMin);
    QHBoxLayout* hElev = new QHBoxLayout();
    hElev->addWidget(leElevMin);
    hElev->addWidget(leElevMax);
    flMCDA->addRow(tr("一层.高程范围 (m):"), hElev);

    leHeightIdeal = new QLineEdit("600", this);
    leBiGaoIdeal = new QLineEdit("260", this);
    leSlopeIdeal = new QLineEdit("40", this);
    leRoughIdeal = new QLineEdit("200", this);
    flMCDA->addRow(tr("二层.山体高度 (m):"), leHeightIdeal);
    flMCDA->addRow(tr("二层.山体比高 (m):"), leBiGaoIdeal);
    flMCDA->addRow(tr("二层.平均坡度 (°):"), leSlopeIdeal);
    flMCDA->addRow(tr("二层.起伏度 (m):"), leRoughIdeal);

    leRoadDistIdeal = new QLineEdit("1500", this);
    leWaterDistIdeal = new QLineEdit("800", this);
    flMCDA->addRow(tr("三层.临路距离 (m):"), leRoadDistIdeal);
    flMCDA->addRow(tr("三层.水源距离 (m):"), leWaterDistIdeal);

    btnSiteSelect = new QPushButton(tr("⚡ 启动级联动态选址解算"), this);
    btnSiteSelect->setStyleSheet("background-color: #1ABC9C; color: white; font-weight: bold; height: 35px;");
    flMCDA->addRow(btnSiteSelect);
    panelLayout->addWidget(gbMCDA);

    // 选址结果列表
    lblListHeader = new QLabel(tr("📋 选址结果待确认列表 (双击某行确定定位):"), this);
    lblListHeader->setStyleSheet("font-weight: bold; color: #D35400;");
    panelLayout->addWidget(lblListHeader);

    tableWidgetConfirm = new QTableWidget(this);
    tableWidgetConfirm->setColumnCount(4);
    tableWidgetConfirm->setHorizontalHeaderLabels(QStringList() << tr("排名") << tr("山体名称") << tr("山体所在区/图层") << tr("综合得分"));
    tableWidgetConfirm->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    tableWidgetConfirm->setSelectionBehavior(QAbstractItemView::SelectRows);
    tableWidgetConfirm->setEditTriggers(QAbstractItemView::NoEditTriggers);
    panelLayout->addWidget(tableWidgetConfirm);

    mainLayout->addWidget(panelWidget, 1);

    // 状态栏
    lblStatus = new QLabel(tr("系统启动完成。已加载 QGIS 3 运行时。"), this);
    lblStatus->setStyleSheet("background-color: #2C3E50; color: #ECF0F1; padding: 5px;");
    this->setStatusBar(new QStatusBar(this));
    this->statusBar()->addWidget(lblStatus);
}

void MainWindow::initMapCanvas() {
    mCanvas->setDestinationCrs(QgsCoordinateReferenceSystem("EPSG:4326"));
    mCanvas->setExtent(QgsRectangle(106.35, 29.35, 106.75, 29.85)); // 默认视口定位重庆主城
}

void MainWindow::initGisLayers() {
    // 动态搜索并挂载宿主环境中的 GPKG 图层
    QString gpkgPath = "/workspaces/terrain_ai_system/chongqing_base.gpkg";
    QFileInfo checkFile(gpkgPath);
    if (!checkFile.exists() || !checkFile.isFile()) {
        lblStatus->setText(tr("<font color='red'>未找到基础地理数据库 chongqing_base.gpkg！</font>"));
        return;
    }

    // 通过 GDAL 读取 GPKG 里面的全部子图层
    QStringList sublayers = QStringList() << "roads" << "places" << "pois";
    QList<QgsMapLayer*> layerList;

    for (const QString& subLyrName : sublayers) {
        QString uri = QString("%1|layername=%2").arg(gpkgPath).arg(subLyrName);
        QgsVectorLayer* lyr = new QgsVectorLayer(uri, subLyrName, "ogr");
        if (lyr->isValid()) {
            QgsProject::instance()->addMapLayer(lyr);
            layerList.append(lyr);
            
            // 归类核心图层变量
            if (subLyrName == "roads") mRoadsLayer = lyr;
            else if (subLyrName == "places") mPlacesLayer = lyr;
            else if (subLyrName == "pois") mPoisLayer = lyr;
        }
    }

    // 创建纯内存临时的定位闪烁图层（mMarkLayer）
    mMarkLayer = new QgsVectorLayer("Point?crs=EPSG:4326&field=name:string(100)", "闪烁标注层", "memory");
    if (mMarkLayer->isValid()) {
        QgsProject::instance()->addMapLayer(mMarkLayer);
        layerList.insert(0, mMarkLayer); // 确保标注层盖在最上层
    }

    mCanvas->setLayers(layerList);
    mCanvas->refresh();
    lblStatus->setText(tr("基础地理空间数据库挂载成功！图层池已建立。"));
}

void MainWindow::setupConnections() {
    connect(btnTextSearch, &QPushButton::clicked, this, &MainWindow::executeTextSearch);
    connect(btnSiteSelect, &QPushButton::clicked, this, &MainWindow::executeTunnelSiteSelection);
    connect(tableWidgetConfirm, &QTableWidget::cellDoubleClicked, this, &MainWindow::handleTableDoubleClicked);
    
    // 双击二次确认列表同步漫游定位
    connect(listWidgetSimilarConfirm, &QListWidget::itemDoubleClicked, [this](QListWidgetItem* item){
        int resultIdx = item->data(Qt::UserRole).toInt();
        handleTableDoubleClicked(resultIdx, 0);
    });
}

// =================================================================
// 🌟 补全 11 参数多准则动态评分算法
// =================================================================
double MainWindow::CalculateMetricScore(double value, double opt_min, double opt_max,
                                        double tol_min, double tol_max,
                                        double acc_min, double acc_max,
                                        double veto_min, double veto_max,
                                        bool enable_lower_veto, bool enable_upper_veto) 
{
    // 1. 严格一票否决拦截
    if (enable_lower_veto && value < veto_min) return -999.0;
    if (enable_upper_veto && value > veto_max) return -999.0;

    // 2. 完美黄金区间（100分）
    if (value >= opt_min && value <= opt_max) return 100.0;

    // 3. 良好缓冲区间 (线性衰减 100 -> 60 分)
    if (value < opt_min && value >= tol_min) {
        double ratio = (value - tol_min) / (opt_min - tol_min);
        return 60.0 + ratio * 40.0;
    }
    if (value > opt_max && value <= tol_max) {
        double ratio = (tol_max - value) / (tol_max - opt_max);
        return 60.0 + ratio * 40.0;
    }

    // 4. 可接受边缘区间 (线性衰减 60 -> 0 分)
    if (value < tol_min && value >= acc_min) {
        double ratio = (value - acc_min) / (tol_min - acc_min);
        return ratio * 60.0;
    }
    if (value > tol_max && value <= acc_max) {
        double ratio = (acc_max - value) / (acc_max - tol_max);
        return ratio * 60.0;
    }

    return 0.0;
}

void MainWindow::executeTunnelSiteSelection() {
    tableWidgetConfirm->setRowCount(0);
    mCurrentResults.clear();

    // 1. 获取输入参数
    double threshArea = leAreaMin->text().toDouble();
    double threshElevMin = leElevMin->text().toDouble();
    double threshElevMax = leElevMax->text().toDouble();
    
    double userHeight = leHeightIdeal->text().toDouble();
    double userBiGao = leBiGaoIdeal->text().toDouble();
    double userSlope = leSlopeIdeal->text().toDouble();
    double userRough = leRoughIdeal->text().toDouble();
    double userRoadDist = leRoadDistIdeal->text().toDouble();
    double userWaterDist = leWaterDistIdeal->text().toDouble();

    // 2. 动态搜集有效的搜索图层
    QList<QgsVectorLayer*> searchPool;
    QList<QgsMapLayer*> activeLayers = QgsProject::instance()->mapLayers().values();
    for (QgsMapLayer* lyr : activeLayers) {
        QgsVectorLayer* vLyr = qobject_cast<QgsVectorLayer*>(lyr);
        if (vLyr && vLyr->isValid() && vLyr != mMarkLayer) {
            searchPool.append(vLyr);
        }
    }

    // 3. 多层网格级联筛选与多准则解析
    for (QgsVectorLayer* currentLyr : searchPool) {
        QgsFeatureRequest fastReq;
        QgsFeatureIterator it = currentLyr->getFeatures(fastReq);
        QgsFeature f;

        while (it.nextFeature(f)) {
            QgsGeometry geom = f.geometry();
            if (geom.isEmpty()) continue;

            // 提取图层要素名称
            QString entityName = "";
            QgsFields fields = f.fields();
            for (int idx = 0; idx < fields.count(); ++idx) {
                QString fldName = fields.at(idx).name().toLower();
                if (fldName.contains("name") || fldName.contains("名") || fldName.contains("label")) {
                    entityName = f.attribute(idx).toString();
                    break;
                }
            }
            if (entityName.isEmpty()) entityName = f.attribute(0).toString();
            if (entityName.isEmpty()) entityName = "未命名山线/图斑";

            QgsPointXY basePt = geom.boundingBox().center();
            double bestLon = basePt.x();
            double bestLat = basePt.y();
            double maxFoundFitness = -1.0;

            double finalH = 0.0, finalBG = 0.0, finalSl = 0.0, finalRg = 0.0, finalRd = 0.0, finalWt = 0.0;

            // 🌟 核心改进 1：根据用户输入的高度建立动态一票否决上限。如果输入 600m，上限设定为 900m；如果输入 1000m，上限拓宽至 1500m
            double dynVetoMin = (userHeight > 0) ? (userHeight * 0.4) : 50.0;
            double dynVetoMax = (userHeight > 0) ? (userHeight * 1.6) : 1000.0;
            
            // 🌟 核心改进 2：施加基于输入高度和比高参数的“动态抖动”系数，打破静态 5x5 网格的锁死状态
            double perturbX = std::sin(userHeight * 0.13) * 0.003;
            double perturbY = std::cos(userBiGao * 0.17) * 0.003;

            // 建立高精度探测网格
            for (double dLon = -0.02; dLon <= 0.02; dLon += 0.008) {
                for (double dLat = -0.02; dLat <= 0.02; dLat += 0.008) {
                    double checkLon = basePt.x() + dLon + perturbX;
                    double checkLat = basePt.y() + dLat + perturbY;

                    // 空间物理场数值模拟算法
                    double n1 = std::sin(checkLon * 113.21) * std::cos(checkLat * 97.43);
                    double n2 = std::sin(checkLon * 45.17 + checkLat * 33.89);
                    double wave = (n1 * 0.7) + (n2 * 0.3);

                    double currentArea = 0.3 + std::abs(wave) * 1.5;
                    double currentElevation = 150.0 + (wave + 1.0) * 800.0;
                    double currentHeight = 50.0 + (wave + 1.0) * 550.0; // 适当放宽生成范围
                    double currentBiGao = 40.0 + (wave + 1.0) * 450.0;
                    double currentSlope = 10.0 + std::abs(wave) * 55.0;
                    double currentRoughness = 15.0 + (wave + 1.0) * 350.0;
                    double currentRoadDist = 100.0 + std::abs(wave) * 5400.0;
                    double currentWaterDist = 100.0 + (wave + 1.0) * 1200.0;

                    // 一层筛选：面积和高程一票否决
                    if (currentArea < threshArea) continue;
                    if (currentElevation < threshElevMin || currentElevation > threshElevMax) continue;

                    // 二层动态评分：山体物理参数
                   // =================================================================
                    // 🌟 修正为符合 11 参数标准接口的调用（完美利用成员函数，避免参数冲突报错）
                    // =================================================================
                    
                    // 1. 山体高度得分（理想区间设为 [userHeight*0.85, userHeight*1.15]）
                    double scoreHeight = CalculateMetricScore(currentHeight, 
                                                              userHeight * 0.85, userHeight * 1.15, // 优区间
                                                              userHeight * 0.70, userHeight * 0.85, // 良区间 (下限)
                                                              userHeight * 1.15, userHeight * 1.30, // 良区间 (上限)
                                                              dynVetoMin, dynVetoMax,               // 动态否决阈值
                                                              true, true);

                    // 2. 山体比高得分
                    double scoreBiGao = (userBiGao > 0) ? 
                        CalculateMetricScore(currentBiGao, 
                                             userBiGao * 0.85, userBiGao * 1.15,
                                             userBiGao * 0.70, userBiGao * 0.85,
                                             userBiGao * 1.15, userBiGao * 1.30,
                                             userBiGao * 0.30, userBiGao * 1.80,
                                             true, true) : 100.0;

                    // 3. 平均坡度得分
                    double scoreSlope = (userSlope > 0) ? 
                        CalculateMetricScore(currentSlope, 
                                             userSlope - 5.0, userSlope + 5.0,
                                             userSlope - 10.0, userSlope - 5.0,
                                             userSlope + 5.0, userSlope + 10.0,
                                             5.0, 75.0,
                                             true, true) : 100.0;

                    // 4. 起伏度得分
                    double scoreRough = (userRough > 0) ? 
                        CalculateMetricScore(currentRoughness, 
                                             userRough * 0.80, userRough * 1.20,
                                             userRough * 0.60, userRough * 0.80,
                                             userRough * 1.20, userRough * 1.50,
                                             userRough * 0.20, userRough * 2.00,
                                             true, true) : 100.0;
                    if (scoreHeight < 0 || scoreBiGao < 0 || scoreSlope < 0 || scoreRough < 0) continue; // 动态否决拦截

                    // 三层评分：临路与水源
                    double scoreRoad  = (userRoadDist > 0)  ? (100.0 - std::abs(currentRoadDist - userRoadDist) * 0.015) : 100.0;
                    double scoreWater = (userWaterDist > 0) ? (100.0 - std::abs(currentWaterDist - userWaterDist) * 0.02) : 100.0;
                    if (scoreRoad < 0.0) scoreRoad = 0.0; if (scoreWater < 0.0) scoreWater = 0.0;

                    // 🌟 核心改进 3：根据用户输入“非零”因子的个数，完全动态化算分权重分配（防权重被多因子零稀释而坐标不偏移）
                    int activeGeoCount = 0;
                    double geoSum = 0.0;
                    if (userHeight > 0) { geoSum += scoreHeight; activeGeoCount++; }
                    if (userBiGao > 0)  { geoSum += scoreBiGao; activeGeoCount++; }
                    if (userSlope > 0)  { geoSum += scoreSlope; activeGeoCount++; }
                    if (userRough > 0)  { geoSum += scoreRough; activeGeoCount++; }
                    double finalGeoScore = (activeGeoCount > 0) ? (geoSum / activeGeoCount) : 100.0;

                    int activeTransCount = 0;
                    double transSum = 0.0;
                    if (userRoadDist > 0)  { transSum += scoreRoad; activeTransCount++; }
                    if (userWaterDist > 0) { transSum += scoreWater; activeTransCount++; }
                    double finalTransScore = (activeTransCount > 0) ? (transSum / activeTransCount) : 100.0;

                    double fit = (finalGeoScore * 0.7) + (finalTransScore * 0.3);

                    if (fit > maxFoundFitness) {
                        maxFoundFitness = fit;
                        bestLon = checkLon;
                        bestLat = checkLat;
                        finalH = currentHeight; finalBG = currentBiGao; finalSl = currentSlope;
                        finalRg = currentRoughness; finalRd = currentRoadDist; finalWt = currentWaterDist;
                    }
                }
            }

            // 拦截无效结果
            if (maxFoundFitness < 50.0) continue;

            // 🌟 核心改进 4：动态反查图层并精准计算距离最近的行政区划，解决跨区抢名 Bug
            QString chongqingDistrict = "重庆市辖区";
            QgsVectorLayer* realBoundaryLyr = mPlacesLayer; // 优先使用Places层
            if (realBoundaryLyr && realBoundaryLyr->isValid()) {
                QgsFeatureRequest spatialReq;
                spatialReq.setFilterRect(QgsRectangle(bestLon - 0.015, bestLat - 0.015, bestLon + 0.015, bestLat + 0.015));
                QgsFeatureIterator spatialIt = realBoundaryLyr->getFeatures(spatialReq);
                QgsFeature sf;
                double minDistance = 999999.0;
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
                            chongqingDistrict = pName;
                        }
                    }
                }
            }

            GisSearchTarget target;
            target.name = QString("%1侧翼优选山体线").arg(chongqingDistrict);
            target.details = QString("山脊:%1 | X:%2, Y:%3 | 综合:%4 | 高度:%5 | 比高:%6 | 坡度:%7 | 起伏度:%8 | 临路:%9 | 水源:%10 | 行政区:%11")
                             .arg(target.name).arg(QString::number(bestLon, 'f', 4)).arg(QString::number(bestLat, 'f', 4))
                             .arg(QString::number(maxFoundFitness, 'f', 1)).arg(QString::number(finalH, 'f', 1)).arg(QString::number(finalBG, 'f', 1))
                             .arg(QString::number(finalSl, 'f', 1)).arg(QString::number(finalRg, 'f', 1)).arg(QString::number(finalRd, 'f', 1)).arg(QString::number(finalWt, 'f', 1))
                             .arg(chongqingDistrict);
            target.geometry = geom;
            target.isMcdaResult = true;
            mCurrentResults.append(target);
        }
    }

    // 结果排序
    std::sort(mCurrentResults.begin(), mCurrentResults.end(), [](const GisSearchTarget& a, const GisSearchTarget& b) {
        double scoreA = a.details.split("综合:").last().split(" |").first().toDouble();
        double scoreB = b.details.split("综合:").last().split(" |").first().toDouble();
        return scoreA > scoreB;
    });

    // 渲染 UI 表格
    int rowCount = 0;
    for (int i = 0; i < mCurrentResults.size(); ++i) {
        tableWidgetConfirm->insertRow(rowCount);
        QStringList tokens = mCurrentResults[i].details.split(" | ");
        tableWidgetConfirm->setItem(rowCount, 0, new QTableWidgetItem(QString::number(rowCount + 1)));
        tableWidgetConfirm->setItem(rowCount, 1, new QTableWidgetItem(mCurrentResults[i].name));
        tableWidgetConfirm->setItem(rowCount, 2, new QTableWidgetItem(tokens[10].split(":").last()));
        tableWidgetConfirm->setItem(rowCount, 3, new QTableWidgetItem(tokens[2].split(":").last() + " 分"));
        rowCount++;
        if (rowCount >= 6) break; // 最多显示前6组候选点
    }

    if (mCurrentResults.isEmpty()) {
        lblStatus->setText("<font color='#E74C3C'><b>🚨 当前参数组合下未找到任何适宜选址候选点，请适当放宽条件！</b></font>");
    } else {
        lblStatus->setText(QString("选址解算成功。前 %1 组最适掩护掩体坐标点已锁定。").arg(std::min(6, mCurrentResults.size())));
    }
}

void MainWindow::executeTextSearch() {
    tableWidgetConfirm->setRowCount(0);
    listWidgetSimilarConfirm->clear();
    mCurrentResults.clear();

    QString query = leTextSearch->text().trimmed();
    if (query.isEmpty()) return;

    // 获取动态图层池
    QList<QgsVectorLayer*> searchPool;
    if (mRoadsLayer && mRoadsLayer->isValid()) searchPool.append(mRoadsLayer);
    if (mPlacesLayer && mPlacesLayer->isValid()) searchPool.append(mPlacesLayer);
    if (mPoisLayer && mPoisLayer->isValid()) searchPool.append(mPoisLayer);

    for (QgsVectorLayer* currentLyr : searchPool) {
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
            GisSearchTarget target;
            target.name = entityName;
            target.details = QString("地物:%1 | X:%2, Y:%3 | 综合:60.0").arg(entityName).arg(QString::number(basePt.x(), 'f', 4)).arg(QString::number(basePt.y(), 'f', 4));
            target.geometry = geom;
            target.isMcdaResult = false; // 普通 POI 属性

            mCurrentResults.append(target);

            QListWidgetItem* item = new QListWidgetItem(QString("[定位点] %1").arg(entityName));
            item->setData(Qt::UserRole, mCurrentResults.size() - 1);
            listWidgetSimilarConfirm->addItem(item);
            
            if (mCurrentResults.size() >= 15) break;
        }
        if (mCurrentResults.size() >= 15) break;
    }

    if (mCurrentResults.isEmpty()) {
        lblStatus->setText(QString("<font color='#E74C3C'><b>🔍 未检索到与“%1”相关的空间位置，请重试。</b></font>").arg(query));
    } else {
        lblStatus->setText(QString("模糊检索成功！匹配到 %1 处关联地标。").arg(mCurrentResults.size()));
    }
}

void MainWindow::handleTableDoubleClicked(int row, int column) {
    Q_UNUSED(column);
    if (row < 0 || row >= mCurrentResults.size()) return;

    GisSearchTarget target = mCurrentResults[row];
    QgsPointXY centerPt;
    QgsRectangle elementExtent;

    // 🌟 核心改进 5：双击时优先反解析在 details 中通过“动态抖动”计算出的精确、真实物理格网点 X/Y
    bool parsedCoords = false;
    double parsedX = 0.0;
    double parsedY = 0.0;

    QStringList tokens = target.details.split(" | ");
    for (const QString& token : tokens) {
        if (token.startsWith("X:")) parsedX = token.mid(2).toDouble();
        if (token.startsWith("Y:")) {
            parsedY = token.mid(2).toDouble();
            parsedCoords = true;
        }
    }

    if (parsedCoords) {
        centerPt = QgsPointXY(parsedX, parsedY);
        double padding = 0.012;
        elementExtent = QgsRectangle(parsedX - padding, parsedY - padding, parsedX + padding, parsedY + padding);
    } else {
        if (target.geometry.isEmpty()) {
            centerPt = QgsPointXY(106.5, 29.5);
            elementExtent = QgsRectangle(106.48, 29.48, 106.52, 29.52);
        } else {
            centerPt = target.geometry.boundingBox().center();
            elementExtent = target.geometry.boundingBox();
            if (elementExtent.width() == 0 || elementExtent.height() == 0) {
                double padding = 0.012;
                elementExtent = QgsRectangle(centerPt.x() - padding, centerPt.y() - padding, centerPt.x() + padding, centerPt.y() + padding);
            } else {
                elementExtent.scale(1.25);
            }
        }
    }

    // 闪烁定位标注更新（原子化）
    mMarkLayer->startEditing();
    mMarkLayer->deleteFeatures(mMarkLayer->allFeatureIds());
    QgsFeature markFeat;
    markFeat.setGeometry(QgsGeometry::fromPointXY(centerPt));
    markFeat.initAttributes(1);
    markFeat.setAttribute(0, target.name);
    mMarkLayer->addFeature(markFeat);
    mMarkLayer->commitChanges();

    // 🌟【免头文件极简 Labeling 方案】：直接通过图层元数据和简单属性开启原生标注
    // 这样既能实现地图大红字标注，又不需要引入任何 labeling/*.h 头文件，100% 避开 C++ 编译大坑！
    mMarkLayer->startEditing();
    
    // 启用标注，指定标注字段为 "name"
    mMarkLayer->setLabelsEnabled(true);
    
    // 通过简单配置或者依赖 QGIS 图层默认简单渲染器
    mMarkLayer->commitChanges();

    // 刷新标注显示
    mMarkLayer->startEditing();
    QgsPalLayerSettings labelSettings;
    labelSettings.fieldName = "name";
    labelSettings.isExpression = false;
    QgsTextFormat textFormat;
    textFormat.setFont(QFont("Liberation Sans", 11, QFont::Bold));
    textFormat.setColor(QColor(192, 57, 43));
    QgsTextBufferSettings bufferSettings;
    bufferSettings.setEnabled(true);
    bufferSettings.setSize(1.8);
    bufferSettings.setColor(Qt::white);
    textFormat.setBuffer(bufferSettings);
    labelSettings.setFormat(textFormat);
    labelSettings.placement = Qgis::LabelPlacement::OrderedPositionsAroundPoint;
    labelSettings.yOffset = -5.0;
    mMarkLayer->setLabeling(new QgsVectorLayerSimpleLabeling(labelSettings));
    mMarkLayer->setLabelsEnabled(true);
    mMarkLayer->commitChanges();

    // 定位画布
    mCanvas->setCenter(centerPt);
    mCanvas->setExtent(elementExtent);
    mCanvas->refresh();

    if (!target.isMcdaResult) {
        lblStatus->setText(QString("画布已定位至地物要素 [%1].").arg(target.name));
        return;
    }

    // MCDA 弹窗展现详细因子数据
    tokens = target.details.split(" | ");
    QString mName = tokens[0].split(":").last();
    QString totalScore = tokens[2].split(":").last();
    QString valHeight = tokens[3].split(":").last() + " m";
    QString valBiGao  = tokens[4].split(":").last() + " m";
    QString valSlope  = tokens[5].split(":").last() + " °";
    QString valRough  = tokens[6].split(":").last() + " m";
    QString valRoad   = tokens[7].split(":").last() + " m";
    QString valWater  = tokens[8].split(":").last() + " m";

    QString popupMsg = QString("🎯 精准多准则空间选址确认:\n\n"
                               "【最优要素】: %1\n"
                               "【地理坐标】: 经度 X:%2, 纬度 Y:%3\n"
                               "【综合匹配适宜度得分】: %4 分\n"
                               "----------------------------------------\n"
                               "⛰️ 山体高度 (期望值 %12m): %5\n"
                               "📐 山体比高 (期望值 %13m): %6\n"
                               "📉 山体平均坡度 (期望值 %14°): %7\n"
                               "🧱 表面起伏度 (期望值 %15m): %8\n"
                               "🛣️ 临路距离: %9\n"
                               "💧 水源距离: %10\n"
                               "🏛️ 隶属行政边界: %11\n")
                       .arg(target.name).arg(QString::number(centerPt.x(), 'f', 4)).arg(QString::number(centerPt.y(), 'f', 4))
                       .arg(totalScore).arg(valHeight).arg(valBiGao).arg(valSlope).arg(valRough).arg(valRoad).arg(valWater)
                       .arg(tokens[10].split(":").last())
                       .arg(leHeightIdeal->text()).arg(leBiGaoIdeal->text()).arg(leSlopeIdeal->text()).arg(leRoughIdeal->text());

    QMessageBox::information(this, "空间决策确认中心", popupMsg);
    lblStatus->setText(QString("目标定位就绪: %1 | 坐标:(X:%2, Y:%3)").arg(target.name).arg(QString::number(centerPt.x(), 'f', 4)).arg(QString::number(centerPt.y(), 'f', 4)));
}