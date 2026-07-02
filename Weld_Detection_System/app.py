import os
import sys

# ==========================================
# 🛑 终极云端环境急救补丁 (必须放在最顶端)
# 强行卸载 YOLO 偷偷夹带的带界面版 OpenCV，强制换成无头版！
# ==========================================
try:
    import cv2
except ImportError:
    os.system(f"{sys.executable} -m pip uninstall -y opencv-python opencv-contrib-python")
    os.system(f"{sys.executable} -m pip install opencv-python-headless")
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tempfile
import urllib.request
from ultralytics import YOLO

# 核心导入：我们的算法工具箱
from weld_utils import GeometricQuantifier, WeldTrajectoryMerger

# ==========================================
# 0. 引擎缓存初始化与状态管理 (含自动下载逻辑)
# ==========================================
@st.cache_resource
def load_ai_engine(weight_path):
    download_url = "https://github.com/fangjw2002/Zhongshi_FLH/releases/download/v1.0.0/best.pt"

    if weight_path == "best.pt" and not os.path.exists("best.pt"):
        with st.spinner("首次启动环境，正在加载核心 AI 模型..."):
            try:
                urllib.request.urlretrieve(download_url, "best.pt")
                st.success("核心 AI 模型加载成功！系统已就绪。")
            except Exception as e:
                st.error(f"模型自动下载失败，请检查网络或链接: {str(e)}")
                st.warning("已降级使用轻量级基础模型进行 UI 演示")
                return YOLO("yolov8n-seg.pt")

    if not os.path.exists(weight_path):
        st.warning(f"未找到 {weight_path}，已降级使用预训练底座演示")
        return YOLO("yolov8n-seg.pt")

    return YOLO(weight_path)

if 'final_dedup_report' not in st.session_state:
    st.session_state['final_dedup_report'] = None

# ==========================================
# 1. 极致单屏级页面配置
# ==========================================
st.set_page_config(
    page_title="焊缝动态缺陷识别系统",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .block-container { padding: 0.5rem 1.5rem 0rem 1.5rem !important; }
    .head-container { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
    .main-title { font-size: 24px !important; color: #1E90FF; font-weight: bold; margin: 0; padding: 0; }
    .sub-title { font-size: 13px !important; color: #888; margin: 0; padding: 0; }
    div[data-testid="stMetric"] { padding: 4px 12px !important; background-color: #f8f9fa; border: 1px solid #e6e9ef; border-radius: 4px; }
    div[data-testid="stMetricValue"] { font-size: 18px !important; font-weight: bold !important; }
    div[data-testid="stMetricLabel"] { font-size: 11px !important; color: #555 !important; margin-bottom: 0px !important; }
    div[data-testid="stImage"] img { max-height: 320px !important; width: 100% !important; object-fit: contain !important; background-color: #000000 !important; border-radius: 4px; }
    .stButton>button { height: 38px !important; font-size: 14px !important; margin-top: 16px !important; margin-bottom: 12px !important; width: 100% !important; }
    hr { margin: 8px 0 !important; padding: 0 !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 侧边栏与顶层面板
# ==========================================
st.sidebar.markdown("### 核心算法调优")
model_type = st.sidebar.selectbox("识别算法", ["Weld-YOLOv8-seg (动态优化)", "Weld-YOLOv11s-seg (高速实时)"])
mm_per_pixel = st.sidebar.number_input("物理比例尺 (mm/px)", value=0.12, step=0.01)

st.markdown("""
    <div class="head-container">
        <div class="main-title">能源装备焊缝动态智能检测系统</div>
        <div class="sub-title">赛题31：基于边缘计算与跨帧追踪的缺陷定量分析平台</div>
    </div>
""", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

WEIGHT_PATH = "best.pt"
model = load_ai_engine(WEIGHT_PATH)

# ==========================================
# 3. 核心骨架布局
# ==========================================
tab1, tab2, tab3 = st.tabs(["动态流式实时探伤", "全局去重报告与导出", "系统操作日志"])

with tab1:
    col_video, col_data = st.columns([0.75, 1])

    with col_video:
        uploaded_video = st.file_uploader("导入动态视频流", type=['mp4', 'avi'], label_visibility="collapsed")
        video_player = st.empty()
        start_btn = st.button("启动边缘多任务探伤流水线")

    with col_data:
        metrics_placeholder = st.empty()
        with metrics_placeholder.container():
            m_count, m_area, m_len = st.columns(3)
            m_count.metric("屏幕活跃缺陷数", "0 个")
            m_area.metric("历史峰值面积", "0.00 mm²")
            m_len.metric("最长骨架极值", "0.00 mm")

        st.markdown("<div style='font-size:13px; font-weight:bold; margin: 3px 0;'>📋 实时检出事件日志流</div>",
                    unsafe_allow_html=True)
        table_placeholder = st.empty()
        table_placeholder.dataframe(pd.DataFrame(columns=["追踪ID", "缺陷类型", "实时动态面积", "实时动态长度"]),
                                    width='stretch', height=180)

    # ==========================================
    # 4. 核心流式循环与骨架注水逻辑
    # ==========================================
    if start_btn and uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 25.0

        st.markdown("<br>", unsafe_allow_html=True)
        progress_text = st.empty()
        progress_bar = st.progress(0)

        raw_tracks_for_dedup = {}
        global_max_area = 0.0
        global_max_len = 0.0
        frame_idx = 0
        AI_PROCESS_SKIP = 4

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1

            if total_frames > 0:
                current_percent = min(frame_idx / total_frames, 1.0)
                current_second = frame_idx / video_fps
                progress_bar.progress(current_percent)
                progress_text.markdown(f"**AI 边缘计算实时处理进度:** `{frame_idx} / {total_frames}` 帧 (时间轴: `{current_second:.2f}s`) 🚀 `{current_percent * 100:.1f}%`")

            if frame_idx % AI_PROCESS_SKIP != 0:
                continue

            frame = cv2.resize(frame, (640, 480))
            results = model.track(source=frame, persist=True, verbose=False, conf=0.25)
            res = results[0]

            current_display_data = []

            if res.boxes is not None and res.boxes.id is not None:
                boxes_xyxy = res.boxes.xyxy.cpu().numpy()
                classes_idx = res.boxes.cls.cpu().numpy()
                track_ids = res.boxes.id.cpu().numpy().astype(int)
                masks_xy = res.masks.xy if res.masks is not None else [None] * len(boxes_xyxy)

                for i in range(len(track_ids)):
                    cls_name = model.names[int(classes_idx[i])]
                    if cls_name in ["Welding line", "Welding_line"]:
                        continue

                    tr_id = track_ids[i]
                    area_px = 0.0
                    centroid_cX, centroid_cY = -1, -1

                    if masks_xy[i] is not None and len(masks_xy[i]) >= 3:
                        pts = np.array(masks_xy[i], dtype=np.int32).reshape((-1, 1, 2))
                        area_px = float(cv2.contourArea(pts))
                        M = cv2.moments(pts)
                        if M["m00"] != 0:
                            centroid_cX = int(M["m10"] / M["m00"])
                            centroid_cY = int(M["m01"] / M["m00"])
                    else:
                        area_px = float((boxes_xyxy[i][2] - boxes_xyxy[i][0]) * (boxes_xyxy[i][3] - boxes_xyxy[i][1]))
                        centroid_cX = int((boxes_xyxy[i][0] + boxes_xyxy[i][2]) / 2)
                        centroid_cY = int((boxes_xyxy[i][1] + boxes_xyxy[i][3]) / 2)

                    pixel_length_skel = 0.0
                    is_new_record = False
                    if tr_id not in raw_tracks_for_dedup or area_px > raw_tracks_for_dedup[tr_id]["max_area"]:
                        is_new_record = True

                    if masks_xy[i] is not None:
                        if is_new_record:
                            pixel_length_skel = GeometricQuantifier.compute_skeleton_length(masks_xy[i])
                        else:
                            pixel_length_skel = raw_tracks_for_dedup[tr_id]["max_len"]

                    current_area_mm2 = area_px * (mm_per_pixel ** 2)
                    current_len_mm = pixel_length_skel * mm_per_pixel

                    if current_area_mm2 > global_max_area: global_max_area = current_area_mm2
                    if current_len_mm > global_max_len: global_max_len = current_len_mm

                    current_display_data.append({
                        "追踪ID": f"ID_{tr_id} 🟢",
                        "缺陷类型": cls_name,
                        "实时动态面积": f"{current_area_mm2:.2f} mm²",
                        "实时动态长度": f"{current_len_mm:.2f} mm"
                    })

                    if tr_id not in raw_tracks_for_dedup:
                        raw_tracks_for_dedup[tr_id] = {
                            "id": tr_id, "class": cls_name,
                            "max_area": area_px, "max_len": pixel_length_skel,
                            "centroid_history": [(centroid_cX, centroid_cY)],
                            "first_seen_frame": frame_idx,
                            "last_seen_frame": frame_idx
                        }
                    else:
                        raw_tracks_for_dedup[tr_id]["max_area"] = max(raw_tracks_for_dedup[tr_id]["max_area"], area_px)
                        raw_tracks_for_dedup[tr_id]["max_len"] = max(raw_tracks_for_dedup[tr_id]["max_len"], pixel_length_skel)
                        raw_tracks_for_dedup[tr_id]["centroid_history"].append((centroid_cX, centroid_cY))
                        raw_tracks_for_dedup[tr_id]["last_seen_frame"] = frame_idx

            annotated_frame = res.plot()
            annotated_frame = cv2.resize(annotated_frame, (800, 600))
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_player.image(frame_rgb, channels="RGB")

            with metrics_placeholder.container():
                m_count, m_area, m_len = st.columns(3)
                m_count.metric("屏幕活跃缺陷数", f"{len(current_display_data)} 个")
                m_area.metric("历史峰值面积", f"{global_max_area:.2f} mm²")
                m_len.metric("最长骨架极值", f"{global_max_len:.2f} mm")

            if current_display_data:
                table_placeholder.dataframe(pd.DataFrame(current_display_data), width='stretch', height=180, hide_index=True)

        cap.release()
        progress_bar.empty()
        progress_text.empty()

        # ==========================================
        # 5. 视频播放完毕 -> 执行深度去重
        # ==========================================
        with st.spinner("视频流处理完毕，正在进行 AI 时空轨迹特征二次去重分析..."):
            raw_tracks_list = list(raw_tracks_for_dedup.values())
            if len(raw_tracks_list) > 0:
                final_merged_tracks, merge_id_map = WeldTrajectoryMerger.merge_trajectories(
                    raw_tracks_list, max_gap_frames=60, max_spatial_distance=80.0
                )
            else:
                final_merged_tracks = []

            final_table = []
            for tr in final_merged_tracks:
                start_second = round(tr["first_seen_frame"] / video_fps, 2)
                end_second = round(tr.get("last_seen_frame", tr["first_seen_frame"]) / video_fps, 2)
                phys_area = round(tr["max_area"] * (mm_per_pixel ** 2), 2)
                phys_len = round(tr["max_len"] * mm_per_pixel, 2)
                phys_width = round(phys_area / phys_len, 2) if phys_len > 0 else round(np.sqrt(phys_area), 2)

                history_pts = tr.get("centroid_history", [])
                if history_pts:
                    x_coords = [p[0] for p in history_pts if p[0] != -1]
                    y_coords = [p[1] for p in history_pts if p[1] != -1]
                    loc_distribution = f"X:[{min(x_coords)}-{max(x_coords)}]px, Y:[{min(y_coords)}-{max(y_coords)}]px" if x_coords else "边缘区域分布"
                else:
                    loc_distribution = "未知区域"

                risk_level = "🟢 轻微 (放行)"
                if tr["class"] in ["裂纹", "未熔合", "crack", "lack of fusion", "Crack"]:
                    risk_level = "🔴 极高危 (建议停机返修)"
                elif tr["class"] in ["气孔", "夹渣", "pore", "slag", "Porosity"]:
                    if phys_len > 3.0 or phys_area > 10.0:
                        risk_level = "🟡 中风险 (打磨复检)"

                final_table.append({
                    "去重后独立ID": f"Unified_ID_{tr['id']}",
                    "缺陷类别": tr["class"],
                    "检出时段": f"{start_second}s -> {end_second}s",
                    "最大物理面积 (mm²)": phys_area,
                    "骨架真实极长 (mm)": phys_len,
                    "计算平均宽度 (mm)": phys_width,
                    "焊缝位置分布空间轴": loc_distribution,
                    "工程安全评级": risk_level
                })

            st.session_state['final_dedup_report'] = pd.DataFrame(final_table)

        st.success(f"探伤闭警全线闭环！确认为 **{len(final_merged_tracks)}** 个独立核心缺陷。")

with tab2:
    st.markdown("### 深度时空去重分析总表")
    if st.session_state['final_dedup_report'] is not None:
        df_report = st.session_state['final_dedup_report']
        if not df_report.empty:
            st.dataframe(df_report, width='stretch')
            csv_data = df_report.to_csv(index=False).encode('utf-8-sig')
            st.download_button("导出探伤结算报告 (CSV)", data=csv_data, file_name="Weld_Inspection_Final_Report.csv", mime="text/csv", type="primary")
        else:
            st.success("本次探伤未发现任何有效缺陷，该管段合格！")
    else:
        st.info("请先在第一个标签页中运行动态视频流探伤。")

with tab3:
    st.markdown("### 核心检测引擎·离线标定白皮书")
    st.info("**工业级合规提示：** 本面板展示的数据为该检测引擎在入库部署前跑出的离线标定成绩单。")
    st.markdown("---")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("目标检测查准率 Precision", "92.2%")
    col_m2.metric("缺陷捕获查全率 Recall", "89.3%")
    col_m3.metric("边界框平均精度 mAP50", "91.5%")
    col_m4.metric("像素实例分割 Mask mAP50", "83.6%")

    st.markdown("<br>", unsafe_allow_html=True)
    col_chart, col_raw_table = st.columns([1.2, 0.8])
    with col_chart:
        chart_data = pd.DataFrame({
            "缺陷类别": ["裂纹 (Crack)", "气孔 (Porosity)", "飞溅 (Spatters)", "焊缝 (Welding line)"],
            "画框定位精度": [93.5, 87.6, 87.9, 97.1],
            "像素轮廓精度": [90.7, 75.5, 72.1, 96.1]
        })
        st.bar_chart(chart_data, x="缺陷类别", height=240)
    with col_raw_table:
        st.dataframe(chart_data, hide_index=True, use_container_width=True)
