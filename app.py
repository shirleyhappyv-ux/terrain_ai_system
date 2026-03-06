import streamlit as st
import cv2
import numpy as np
from PIL import Image
import folium
from streamlit_folium import st_folium
from skimage.feature import canny
import random

st.set_page_config(layout="wide")

st.title("AI关键地形要素自动识别系统（演示版）")

st.sidebar.title("图层控制")

show_river = st.sidebar.checkbox("河流", True)
show_ridge = st.sidebar.checkbox("山脊", True)
show_road = st.sidebar.checkbox("道路", True)
show_keypoints = st.sidebar.checkbox("战术关键点", True)

uploaded = st.file_uploader("上传地图影像", type=["jpg", "png", "jpeg"])

if uploaded:

    image = Image.open(uploaded)
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    st.subheader("原始地图")

    st.image(image, width=500)

    st.subheader("AI识别过程")

    progress = st.progress(0)

    progress.progress(10)

    # -------------------
    # 河流识别（蓝色检测）
    # -------------------

    lower_blue = np.array([0,0,100])
    upper_blue = np.array([120,120,255])

    river_mask = cv2.inRange(img, lower_blue, upper_blue)

    contours,_ = cv2.findContours(
        river_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    progress.progress(40)

    # -------------------
    # 山脊检测
    # -------------------

    edges = canny(gray, sigma=2)

    ridge_points = np.where(edges)

    progress.progress(60)

    # -------------------
    # 道路检测
    # -------------------

    edges2 = cv2.Canny(gray,50,150)

    lines = cv2.HoughLinesP(
        edges2,
        1,
        np.pi/180,
        threshold=100,
        minLineLength=80,
        maxLineGap=10
    )

    road_lines = []

    if lines is not None:
        for l in lines:
            x1,y1,x2,y2 = l[0]
            road_lines.append((x1,y1,x2,y2))

    progress.progress(80)

    # -------------------
    # 自动生成战术关键点
    # -------------------

    h,w = gray.shape

    keypoints = []

    for i in range(5):

        y = random.randint(0,h)
        x = random.randint(0,w)

        keypoints.append((y,x))

    progress.progress(100)

    st.success("AI识别完成")

    # -------------------
    # 构建 GIS 地图
    # -------------------

    m = folium.Map(
        location=[0,0],
        zoom_start=2,
        crs="Simple"
    )

    # 把上传图片作为地图底图
    folium.raster_layers.ImageOverlay(
        image=img,
        bounds=[[0,0],[h,w]],
        opacity=1
    ).add_to(m)

    m.fit_bounds([[0,0],[h,w]])

    # -------------------
    # 河流图层
    # -------------------

    if show_river:

        for cnt in contours:

            for p in cnt[::20]:

                x = int(p[0][0])
                y = int(p[0][1])

                folium.CircleMarker(
                    location=[y,x],
                    radius=1,
                    color="blue"
                ).add_to(m)

    # -------------------
    # 山脊图层
    # -------------------

    if show_ridge:

        ys = ridge_points[0]
        xs = ridge_points[1]

        for i in range(0,len(xs),50):

            folium.CircleMarker(
                location=[ys[i],xs[i]],
                radius=1,
                color="red"
            ).add_to(m)

    # -------------------
    # 道路图层
    # -------------------

    if show_road:

        for r in road_lines:

            x1,y1,x2,y2 = r

            folium.PolyLine(
                locations=[
                    [y1,x1],
                    [y2,x2]
                ],
                color="yellow",
                weight=3
            ).add_to(m)

    # -------------------
    # 战术关键点
    # -------------------

    if show_keypoints:

        for p in keypoints:

            folium.Marker(
                location=[p[0],p[1]],
                icon=folium.Icon(color="red")
            ).add_to(m)

    st.subheader("GIS地图识别结果")

    st_folium(m,width=1100,height=650)

    # -------------------
    # 统计信息
    # -------------------

    st.subheader("识别统计")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("河流",len(contours))
    col2.metric("山脊",len(ridge_points[0]))
    col3.metric("道路",len(road_lines))
    col4.metric("关键点",len(keypoints))

else:

    st.info("请上传地图开始AI识别")
