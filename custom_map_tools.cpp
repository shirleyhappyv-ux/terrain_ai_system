#include "custom_map_tools.h"
#include <qgsproject.h>
#include <qgsfields.h>
#include <qgsfeature.h>

// --- 量测工具实现 ---
MapMeasureTool::MapMeasureTool(QgsMapCanvas* canvas) : QgsMapTool(canvas) {
    mRubberBand = new QgsRubberBand(canvas, QgsWkbTypes::LineGeometry);
    mRubberBand->setColor(Qt::red);
    mRubberBand->setWidth(2);
}

MapMeasureTool::~MapMeasureTool() { delete mRubberBand; }

void MapMeasureTool::canvasMoveEvent(QgsMapMouseEvent* e) {
    if (mPoints.isEmpty()) return;
    mRubberBand->removeLastPoint();
    mRubberBand->addPoint(e->mapPoint());
}

void MapMeasureTool::canvasReleaseEvent(QgsMapMouseEvent* e) {
    if (e->button() == Qt::LeftButton) {
        QgsPointXY pt = e->mapPoint();
        mPoints.append(pt);
        mRubberBand->addPoint(pt);
        // 如果是多段线，计算总投影物理长度
        double totalDist = 0.0;
        for (int i = 0; i < mPoints.size() - 1; ++i) {
            totalDist += mPoints[i].distance(mPoints[i+1]); // 基于CRS地图单位（度或米）
        }
        emit distanceChanged(totalDist);
    } else if (e->button() == Qt::RightButton && !mPoints.isEmpty()) {
        mPoints.clear();
        mRubberBand->reset(QgsWkbTypes::LineGeometry);
    }
}
void MapMeasureTool::deactivate() { mRubberBand->reset(QgsWkbTypes::LineGeometry); QgsMapTool::deactivate(); }

// --- 标绘工具实现 ---
MapPlotTool::MapPlotTool(QgsMapCanvas* canvas, QgsVectorLayer* plotLayer) 
    : QgsMapToolEmitPoint(canvas), mPlotLayer(plotLayer) {}

void MapPlotTool::canvasReleaseEvent(QgsMapMouseEvent* e) {
    if (!mPlotLayer || e->button() != Qt::LeftButton) return;
    
    QgsPointXY mapPt = e->mapPoint();
    QgsFeature feature(mPlotLayer->fields());
    feature.setGeometry(QgsGeometry::fromPointXY(mapPt));
    feature.setAttribute("id", mNextFeatureId++);
    feature.setAttribute("name", QString("PlotMark_%1").arg(mNextFeatureId));
    
    mPlotLayer->startEditing();
    mPlotLayer->addFeature(feature);
    mPlotLayer->commitChanges();
    mCanvas()->refresh();
}