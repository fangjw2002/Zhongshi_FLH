import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tempfile
import os
from ultralytics import YOLO

# 核心导入：我们的算法工具箱
from weld_utils import GeometricQuantifier, WeldTrajectoryMerger


# ==========================================
# 0. 引擎缓存初始化与状态管理
# ==========================================
@st.cache_resource
def load_ai_engine(weight_path):
    if not os.path.exists(weight_path):
        st.warning(f"⚠️ 未找到 {weight_path}，已降级使用预训练底座演示")
        return YOLO("yolov8n-seg.pt")
    return YOLO(weight_path)


# 初始化全局会话状态，用于跨 Tab 传递去重报告
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
st.sidebar.markdown("### 🛠️ 核心算法调优")
model_type = st.sidebar.selectbox("识别算法", ["Weld-YOLOv8-seg (动态优化)", "Weld-YOLOv11s-seg (高速实时)"])
mm_per_pixel = st.sidebar.number_input("物理比例尺 (mm/px)", value=0.12, step=0.01)

st.markdown("""
    <div class="head-container">
        <div class="main-title">⚙️ 能源装备焊缝动态智能检测系统</div>
        <div class="sub-title">赛题31：基于边缘计算与跨帧追踪的缺陷定量分析平台</div>
    </div>
""", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# 加载模型 (注意：本地演示请把 best.pt 放在当前目录，并修改此处路径)
WEIGHT_PATH = "best.pt" if os.path.exists("best.pt") else "yolov8l-seg.pt"
model = load_ai_engine(WEIGHT_PATH)

# ==========================================
# 3. 核心骨架布局
# ==========================================
tab1, tab2, tab3 = st.tabs(["🎥 动态流式实时探伤", "📊 全局去重报告与导出", "📜 系统操作日志"])

with tab1:
    col_video, col_data = st.columns([0.75, 1])

    with col_video:
        uploaded_video = st.file_uploader("导入动态视频流", type=['mp4', 'avi'], label_visibility="collapsed")
        video_player = st.empty()
        start_btn = st.button("🚀 启动流式多任务探伤流水线")

    with col_data:
        metrics_placeholder = st.empty()
        with metrics_placeholder.container():
            m_count, m_area, m_len = st.columns(3)
            m_count.metric("当前屏缺陷数", "0 个")
            m_area.metric("历史最大面积", "0.00 mm²")
            m_len.metric("最长骨架裂纹", "0.00 mm")

        st.markdown("<div style='font-size:13px; font-weight:bold; margin: 3px 0;'>📋 实时检出事件日志流</div>",
                    unsafe_allow_html=True)
        table_placeholder = st.empty()
        table_placeholder.dataframe(pd.DataFrame(columns=["追踪ID", "缺陷类型", "物理面积", "骨架长度"]),
                                    width='stretch', height=180)

    # ==========================================
    # 4. 核心流式循环与骨架注水逻辑
    # ==========================================
    if start_btn and uploaded_video:
        # A. 准备视频流
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        # B. 初始化“二次去重器”收集器（核心改动处 A）
        raw_tracks_for_dedup = {}

        global_max_area = 0.0
        global_max_len = 0.0
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1

            results = model.track(source=frame, persist=True, verbose=False, conf=0.25)
            res = results[0]

            annotated_frame = res.plot()
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_player.image(frame_rgb, channels="RGB")

            current_frame_data = []

            if res.boxes is not None and res.boxes.id is not None:
                boxes_xyxy = res.boxes.xyxy.cpu().numpy()
                classes_idx = res.boxes.cls.cpu().numpy()
                track_ids = res.boxes.id.cpu().numpy().astype(int)
                masks_xy = res.masks.xy if res.masks is not None else [None] * len(boxes_xyxy)

                # C. 注入骨架算法与数据收集 (核心改动处 B)
                for i in range(len(track_ids)):
                    cls_name = model.names[int(classes_idx[i])]
                    tr_id = track_ids[i]

                    # 1. 面积计算
                    area_px = 0.0
                    centroid_cX, centroid_cY = -1, -1
                    if masks_xy[i] is not None and len(masks_xy[i]) >= 3:
                        pts = np.array(masks_xy[i], dtype=np.int32).reshape((-1, 1, 2))
                        area_px = float(cv2.contourArea(pts))
                        # 计算重心供去重算法使用
                        M = cv2.moments(pts)
                        if M["m00"] != 0:
                            centroid_cX = int(M["m10"] / M["m00"])
                            centroid_cY = int(M["m01"] / M["m00"])
                    else:
                        area_px = float((boxes_xyxy[i][2] - boxes_xyxy[i][0]) * (boxes_xyxy[i][3] - boxes_xyxy[i][1]))
                        centroid_cX = int((boxes_xyxy[i][0] + boxes_xyxy[i][2]) / 2)
                        centroid_cY = int((boxes_xyxy[i][1] + boxes_xyxy[i][3]) / 2)

                    # 2. 调用骨架化算法算出真实路径长度 (高阶指标)
                    pixel_length_skel = 0.0
                    if masks_xy[i] is not None:
                        pixel_length_skel = GeometricQuantifier.compute_skeleton_length(masks_xy[i])

                    # 3. 换算物理比例尺
                    area_mm2 = area_px * (mm_per_pixel ** 2)
                    len_mm = pixel_length_skel * mm_per_pixel

                    if area_mm2 > global_max_area: global_max_area = area_mm2
                    if len_mm > global_max_len: global_max_len = len_mm

                    # 4. 前端大屏日志
                    current_frame_data.append({
                        "追踪ID": f"ID_{tr_id}",
                        "缺陷类型": cls_name,
                        "物理面积": f"{area_mm2:.2f} mm²",
                        "骨架长度": f"{len_mm:.2f} mm"
                    })

                    # 5. 存储至收集器，准备视频结束后的深度去重
                    if tr_id not in raw_tracks_for_dedup:
                        raw_tracks_for_dedup[tr_id] = {
                            "id": tr_id, "class": cls_name,
                            "max_area": area_px, "max_len": pixel_length_skel,
                            "centroid_history": [(centroid_cX, centroid_cY)],
                            "first_seen_frame": frame_idx
                        }
                    else:
                        tr = raw_tracks_for_dedup[tr_id]
                        tr["max_area"] = max(tr["max_area"], area_px)
                        tr["max_len"] = max(tr["max_len"], pixel_length_skel)
                        tr["centroid_history"].append((centroid_cX, centroid_cY))

            # 动态更新前端指标
            with metrics_placeholder.container():
                m_count, m_area, m_len = st.columns(3)
                m_count.metric("当前屏缺陷数", f"{len(current_frame_data)} 个")
                m_area.metric("历史最大面积", f"{global_max_area:.2f} mm²")
                m_len.metric("最长骨架裂纹", f"{global_max_len:.2f} mm")

            if current_frame_data:
                table_placeholder.dataframe(pd.DataFrame(current_frame_data), width='stretch', height=180,
                                            hide_index=True)

        cap.release()

        # ==========================================
        # 5. 视频播放完毕 -> 执行深度去重 (高分项)
        # ==========================================
        with st.spinner("🔄 视频流处理完毕，正在进行 AI 时空轨迹特征二次去重分析..."):
            raw_tracks_list = list(raw_tracks_for_dedup.values())
            final_merged_tracks, merge_id_map = WeldTrajectoryMerger.merge_trajectories(
                raw_tracks_list, max_gap_frames=60, max_spatial_distance=80.0
            )

            # 整理出最终的闭环大账表
            final_table = []
            for tr in final_merged_tracks:
                final_table.append({
                    "去重后独立ID": f"Unified_ID_{tr['id']}",
                    "缺陷类别": tr["class"],
                    "生命周期全览帧": f"帧 {tr['first_seen_frame']} -> {tr.get('last_seen_frame', tr['first_seen_frame'])}",
                    "最大物理面积 (mm²)": round(tr["max_area"] * (mm_per_pixel ** 2), 2),
                    "骨架真实极长 (mm)": round(tr["max_len"] * mm_per_pixel, 2)
                })

            st.session_state['final_dedup_report'] = pd.DataFrame(final_table)

        st.success(
            f"🎉 探伤闭环完成！原始追踪到 {len(raw_tracks_list)} 个散点 ID，AI 时空去重后确认真实缺陷为 **{len(final_merged_tracks)}** 个。请前往【TAB 2】下载详尽报告。")

# ==========================================
# TAB 2：导出报告中心 (满足闭环需求)
# ==========================================
with tab2:
    st.markdown("### 📊 深度时空去重分析总表")
    if st.session_state['final_dedup_report'] is not None:
        df_report = st.session_state['final_dedup_report']
        st.dataframe(df_report, width='stretch')

        # 将 DataFrame 转换为 CSV 以供下载
        csv_data = df_report.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 导出工业级探伤结算报告 (CSV)",
            data=csv_data,
            file_name="Weld_Inspection_Final_Report.csv",
            mime="text/csv",
            type="primary"
        )
    else:
        st.info("💡 请先在第一个标签页中运行动态视频流探伤，算法结束后将在此生成去重报告。")

with tab3:
    st.info("系统心跳正常。就绪。")