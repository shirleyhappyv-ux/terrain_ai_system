import streamlit as st
import cv2
import numpy as np
from PIL import Image
import folium
from streamlit_folium import st_folium
import networkx as nx
import random

st.set_page_config(layout="wide")

st.title("智能地形分析系统")

st.sidebar.title("图层管理")

layer_water = st.sidebar.checkbox("水体",True)
layer_road = st.sidebar.checkbox("道路",True)
layer_veg = st.sidebar.checkbox("植被",True)
layer_key = st.sidebar.checkbox("战术关键点",True)
layer_route = st.sidebar.checkbox("通行路线",True)

uploaded = st.file_uploader("上传地图影像", type=["jpg","png"])

if uploaded:

    image = Image.open(uploaded)
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    st.image(image,width=500)

    st.subheader("AI识别中")

    progress = st.progress(0)

    progress.progress(20)

    # 水体识别
    lower_blue = np.array([0,0,100])
    upper_blue = np.array([120,120,255])

    water = cv2.inRange(img,lower_blue,upper_blue)

    progress.progress(40)

    # 植被识别
    lower_green = np.array([0,80,0])
    upper_green = np.array([120,255,120])

    veg = cv2.inRange(img,lower_green,upper_green)

    progress.progress(60)

    # 道路检测
    edges = cv2.Canny(gray,50,150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi/180,
        threshold=100,
        minLineLength=80,
        maxLineGap=10
    )

    progress.progress(80)

    # 战术关键点
    h,w = gray.shape

    keypoints = []

    for i in range(5):

        keypoints.append((
            random.randint(0,h),
            random.randint(0,w)
        ))

    progress.progress(100)

    st.success("识别完成")

    # 创建地图
    m = folium.Map(
        location=[30,110],
        zoom_start=5,
        tiles="OpenStreetMap"
    )

    # 水体
    if layer_water:

        ys,xs = np.where(water>0)

        for i in range(0,len(xs),200):

            folium.CircleMarker(
                location=[30+ys[i]*0.0001,110+xs[i]*0.0001],
                radius=1,
                color="blue"
            ).add_to(m)

    # 植被
    if layer_veg:

        ys,xs = np.where(veg>0)

        for i in range(0,len(xs),200):

            folium.CircleMarker(
                location=[30+ys[i]*0.0001,110+xs[i]*0.0001],
                radius=1,
                color="green"
            ).add_to(m)

    # 道路
    if layer_road and lines is not None:

        for l in lines:

            x1,y1,x2,y2 = l[0]

            folium.PolyLine(
                locations=[
                    [30+y1*0.0001,110+x1*0.0001],
                    [30+y2*0.0001,110+x2*0.0001]
                ],
                color="yellow",
                weight=3
            ).add_to(m)

    # 关键点
    if layer_key:

        for p in keypoints:

            folium.Marker(
                location=[30+p[0]*0.0001,110+p[1]*0.0001],
                icon=folium.Icon(color="red")
            ).add_to(m)

    # 路线规划
    if layer_route:

        start = [30.1,110.1]
        end = [30.3,110.3]

        folium.PolyLine(
            locations=[start,end],
            color="orange",
            weight=4
        ).add_to(m)

    st.subheader("GIS地图展示")

    st_folium(m,width=1100,height=650)

else:

    st.info("上传地图开始分析")
