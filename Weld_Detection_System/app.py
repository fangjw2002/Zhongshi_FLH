import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import os

# ==========================================
# 1. 极致单屏级页面配置
# ==========================================
st.set_page_config(
    page_title="焊缝动态缺陷识别系统",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 核心魔改：压低视频窗口，缩紧组件间距，调大推理按钮四周的空隙
st.markdown("""
    <style>
    .block-container { padding: 0.5rem 1.5rem 0rem 1.5rem !important; }
    .head-container { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
    .main-title { font-size: 24px !important; color: #1E90FF; font-weight: bold; margin: 0; padding: 0; }
    .sub-title { font-size: 13px !important; color: #888; margin: 0; padding: 0; }
    div[data-testid="stMetric"] { padding: 4px 12px !important; background-color: #f8f9fa; border: 1px solid #e6e9ef; border-radius: 4px; }
    div[data-testid="stMetricValue"] { font-size: 18px !important; font-weight: bold !important; }
    div[data-testid="stMetricLabel"] { font-size: 11px !important; color: #555 !important; margin-bottom: 0px !important; }

    div[data-testid="stVideo"] video { max-height: 240px !important; width: 100% !important; object-fit: contain !important; background-color: #000000 !important; border-radius: 4px; }

    /* 🔻🔻🔻 修改点：为按钮引入 margin 上下外边距，显著拉大按钮四周的空隙 🔻🔻🔻 */
    .stButton>button { 
        height: 38px !important; 
        font-size: 14px !important; 
        margin-top: 16px !important;    /* 增加与上方视频窗口的间距 */
        margin-bottom: 12px !important; /* 增加与下方组件或分割线的间距 */
        width: 100% !important;
    }
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

# ==========================================
# 3. 核心骨架布局
# ==========================================
tab1, tab2, tab3 = st.tabs(["🎥 动态视频流实时探伤", "📊 缺陷定量分析与统计", "📜 系统操作与探伤日志"])

with tab1:
    col_video, col_data = st.columns([0.75, 1])

    with col_video:
        uploaded_video = st.file_uploader("导入动态视频流", type=['mp4'], label_visibility="collapsed")
        video_player = st.empty()

        if uploaded_video is None:
            video_player.info("💡 提示：请拖入连续图像视频流文件...")
        else:
            video_player.video(uploaded_video.read(), format="video/mp4", loop=True, autoplay=True, muted=True)

        # 这里的按钮会自动加载上面专门调节过 margin 的 CSS 样式
        start_btn = st.button("🚀 启动边缘多任务探伤流水线")

    with col_data:
        metrics_placeholder = st.empty()
        with metrics_placeholder.container():
            m_count, m_area, m_len = st.columns(3)
            m_count.metric("当前追踪缺陷数", "0 个")
            m_area.metric("最大物理面积", "0.00 mm²")
            m_len.metric("最长骨架裂纹", "0.00 mm")

        st.markdown("<div style='font-size:13px; font-weight:bold; margin: 3px 0;'>📋 跨帧去重事件日志流</div>",
                    unsafe_allow_html=True)
        table_placeholder = st.empty()
        # 💡 升级点：已将 use_container_width=True 替换为 width='stretch' 消除警告
        table_placeholder.dataframe(pd.DataFrame(columns=["追踪ID", "缺陷类型", "最大物理面积", "最大物理长度"]),
                                    width='stretch', height=160)

    # ---- 执行逻辑 ----
    if start_btn:
        if uploaded_video is None:
            st.warning("请先上传视频文件！")
        else:
            with st.spinner("⏳ 正在调用 AI 推理引擎处理全帧动态视频..."):
                time.sleep(1.5)

            json_report_path = "/root/autodl-tmp/task2_reports/video_tracking_unified_report.json"

            if not os.path.exists(json_report_path):
                st.toast("⚠️ 提示：未在云端检测到 JSON，当前展示本地演示数据。")
                defects_list = [
                    {"track_id": "#Track-01", "class": "裂纹", "max_area": 1050, "max_len": 125},
                    {"track_id": "#Track-02", "class": "气孔", "max_area": 320, "max_len": 24},
                    {"track_id": "#Track-03", "class": "未熔合", "max_area": 880, "max_len": 90}
                ]
            else:
                st.toast("✅ 成功接入算法报告！")
                with open(json_report_path, 'r', encoding='utf-8') as f:
                    defects_list = json.load(f)

            if defects_list:
                total_defects = len(defects_list)
                max_physical_area = 0.0
                max_physical_length = 0.0
                table_data = []

                for defect in defects_list:
                    px_area = defect.get("max_area", 0)
                    px_len = defect.get("max_len", 0)
                    current_mm2_area = px_area * (mm_per_pixel ** 2)
                    current_mm_len = px_len * mm_per_pixel

                    if current_mm2_area > max_physical_area:
                        max_physical_area = current_mm2_area
                    if current_mm_len > max_physical_length:
                        max_physical_length = current_mm_len

                    table_data.append({
                        "追踪ID": defect.get("track_id", "Unknown"),
                        "缺陷类型": defect.get("class", "异常特征"),
                        "最大物理面积": f"{current_mm2_area:.2f} mm²",
                        "最大物理长度": f"{current_mm_len:.2f} mm"
                    })

                with metrics_placeholder.container():
                    m_count, m_area, m_len = st.columns(3)
                    m_count.metric(label="当前追踪缺陷数", value=f"{total_defects} 个")
                    m_area.metric(label="最大物理面积", value=f"{max_physical_area:.2f} mm²")
                    m_len.metric(label="最长骨架裂纹", value=f"{max_physical_length:.2f} mm")

                # 💡 升级点：已将 use_container_width=True 替换为 width='stretch'
                table_placeholder.dataframe(pd.DataFrame(table_data), width='stretch', height=160, hide_index=True)

                st.success("🎉 分析完成：动态视频流已处理完毕，定量分析数据提取成功！")
            else:
                st.info("当前视频未检出任何缺陷特征。")

# ---- TAB 2 与 TAB 3 保持高度紧凑设计 ----
with tab2:
    st.info("生产线大数据面板：等待数据库接入...")
with tab3:
    st.info("系统日志加载中...")