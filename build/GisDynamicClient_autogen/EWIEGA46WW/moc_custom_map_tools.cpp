/****************************************************************************
** Meta object code from reading C++ file 'custom_map_tools.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.13)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../../custom_map_tools.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'custom_map_tools.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.13. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_MapMeasureTool_t {
    QByteArrayData data[4];
    char stringdata0[46];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_MapMeasureTool_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_MapMeasureTool_t qt_meta_stringdata_MapMeasureTool = {
    {
QT_MOC_LITERAL(0, 0, 14), // "MapMeasureTool"
QT_MOC_LITERAL(1, 15, 15), // "distanceChanged"
QT_MOC_LITERAL(2, 31, 0), // ""
QT_MOC_LITERAL(3, 32, 13) // "totalDistance"

    },
    "MapMeasureTool\0distanceChanged\0\0"
    "totalDistance"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_MapMeasureTool[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   19,    2, 0x06 /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::Double,    3,

       0        // eod
};

void MapMeasureTool::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<MapMeasureTool *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->distanceChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (MapMeasureTool::*)(double );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&MapMeasureTool::distanceChanged)) {
                *result = 0;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject MapMeasureTool::staticMetaObject = { {
    QMetaObject::SuperData::link<QgsMapTool::staticMetaObject>(),
    qt_meta_stringdata_MapMeasureTool.data,
    qt_meta_data_MapMeasureTool,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *MapMeasureTool::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MapMeasureTool::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_MapMeasureTool.stringdata0))
        return static_cast<void*>(this);
    return QgsMapTool::qt_metacast(_clname);
}

int MapMeasureTool::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QgsMapTool::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 1)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 1;
    }
    return _id;
}

// SIGNAL 0
void MapMeasureTool::distanceChanged(double _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
struct qt_meta_stringdata_MapPlotTool_t {
    QByteArrayData data[1];
    char stringdata0[12];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_MapPlotTool_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_MapPlotTool_t qt_meta_stringdata_MapPlotTool = {
    {
QT_MOC_LITERAL(0, 0, 11) // "MapPlotTool"

    },
    "MapPlotTool"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_MapPlotTool[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

void MapPlotTool::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    (void)_o;
    (void)_id;
    (void)_c;
    (void)_a;
}

QT_INIT_METAOBJECT const QMetaObject MapPlotTool::staticMetaObject = { {
    QMetaObject::SuperData::link<QgsMapToolEmitPoint::staticMetaObject>(),
    qt_meta_stringdata_MapPlotTool.data,
    qt_meta_data_MapPlotTool,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *MapPlotTool::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MapPlotTool::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_MapPlotTool.stringdata0))
        return static_cast<void*>(this);
    return QgsMapToolEmitPoint::qt_metacast(_clname);
}

int MapPlotTool::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QgsMapToolEmitPoint::qt_metacall(_c, _id, _a);
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
