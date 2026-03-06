import streamlit as st
import cv2
import numpy as np
from PIL import Image
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="AI地形识别Demo", layout="wide")

st.title("AI辅助关键地形识别系统 Demo")

# ----------------------------
# 上传图片
# ----------------------------

uploaded_file = st.file_uploader("上传侦察图像", type=["jpg","png","jpeg"])

# 防止重复识别
if "processed" not in st.session_state:
    st.session_state.processed = False

if uploaded_file and not st.session_state.processed:

    image = Image.open(uploaded_file)
    image_np = np.array(image)

    st.subheader("原始图像")
    st.image(image, use_column_width=True)

    # ----------------------------
    # 模拟AI识别
    # ----------------------------

    h, w, _ = image_np.shape

    detections = [
        {"name":"山地","box":[w*0.2,h*0.2,w*0.35,h*0.35]},
        {"name":"河流","box":[w*0.6,h*0.5,w*0.8,h*0.6]},
        {"name":"道路","box":[w*0.4,h*0.7,w*0.6,h*0.8]},
        {"name":"建筑","box":[w*0.7,h*0.2,w*0.85,h*0.35]}
    ]

    result_img = image_np.copy()

    locations = []

    for d in detections:

        x1,y1,x2,y2 = map(int,d["box"])

        cv2.rectangle(result_img,(x1,y1),(x2,y2),(255,0,0),2)

        cv2.putText(
            result_img,
            d["name"],
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,0,0),
            2
        )

        # 模拟GIS坐标
        lat = 34.0 + np.random.uniform(-0.01,0.01)
        lon = 108.0 + np.random.uniform(-0.01,0.01)

        locations.append((d["name"],lat,lon))

    # ----------------------------
    # 显示识别图像
    # ----------------------------

    st.subheader("AI识别结果")

    st.image(result_img, use_column_width=True)

    # ----------------------------
    # GIS地图
    # ----------------------------

    st.subheader("GIS地图展示")

    m = folium.Map(location=[34.0,108.0],zoom_start=12)

    for name,lat,lon in locations:

        folium.Marker(
            [lat,lon],
            popup=name,
            tooltip=name,
            icon=folium.Icon(color="red")
        ).add_to(m)

    st_data = st_folium(m,width=700,height=500)

    # ----------------------------
    # 识别列表
    # ----------------------------

    st.subheader("识别到的地形要素")

    for name,lat,lon in locations:
        st.write(f"{name} 位置: {lat:.4f}, {lon:.4f}")

    # ----------------------------
    # 自动停止
    # ----------------------------

    st.session_state.processed = True

    st.success("AI识别完成，系统已停止自动推理")
