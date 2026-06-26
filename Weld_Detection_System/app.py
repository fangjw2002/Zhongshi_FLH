import streamlit as st
import pandas as pd
import numpy as np
import time

# ==========================================
# 1. 页面全局配置 (打造工业科技感)
# ==========================================
st.set_page_config(
    page_title="焊缝动态缺陷识别系统",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 自定义一些简单的 CSS 让界面看起来更专业
st.markdown("""
    <style>
    .main-title { font-size: 42px; color: #1E90FF; font-weight: bold; text-align: center; margin-bottom: 0px; }
    .sub-title { font-size: 18px; color: #888; text-align: center; margin-bottom: 30px; }
    .stButton>button { width: 100%; height: 50px; font-size: 20px; font-weight: bold; background-color: #1E90FF; color: white; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">⚙️ 能源装备焊缝动态智能检测系统</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">赛题31：基于边缘计算与跨帧追踪的缺陷定量分析平台</p>', unsafe_allow_html=True)
st.divider()

# ==========================================
# 2. 核心骨架布局：左边视频，右边数据
# ==========================================
col_video, col_data = st.columns([1.2, 1])  # 左侧稍微宽一点

with col_video:
    st.subheader("📺 动态实时监测视图")

    # 【修复点 1】：限定最安全的 mp4 格式，提示用户注意编码
    uploaded_video = st.file_uploader(
        "导入由同学C合成的“测试用动态视频流” (请确保为 H.264 编码的标准 .mp4 格式)",
        type=['mp4']
    )

    # 视频播放点占位符
    video_player = st.empty()

    if uploaded_video is None:
        video_player.info("等待接入工业摄像头视频流或本地测试文件...")
    else:
        # 【修复点 2】：将文件读取为 Bytes 字节流，并利用原生参数开启自动循环播放
        video_bytes = uploaded_video.read()
        video_player.video(video_bytes, format="video/mp4", loop=True, autoplay=True, muted=True)

with col_data:
    st.subheader("📊 缺陷定量分析数据看板")

    # 指标卡片占位符
    kpi1, kpi2, kpi3 = st.columns(3)
    defect_count_ui = kpi1.empty()
    max_area_ui = kpi2.empty()
    status_ui = kpi3.empty()

    # 初始状态下的指标展示
    defect_count_ui.metric("当前缺陷总数", "0 个")
    max_area_ui.metric("最大缺陷面积", "0.00 mm²")
    status_ui.metric("系统状态", "⚪ 待机中", "FPS: 0")

    # 统计图表占位符
    st.markdown("#### 📈 实时缺陷面积走势 (mm²)")
    chart_ui = st.empty()

    # 假数据面板
    st.markdown("#### 📋 跨帧检测事件日志")
    log_ui = st.empty()

# ==========================================
# 3. 伪造动态逻辑 (死代码 Dummy Code)
# ==========================================
st.divider()
start_btn = st.button("🚀 启动自动化探伤流水线")

if start_btn:
    if uploaded_video is None:
        st.warning("请先在左侧上传一段普通视频占位，再启动系统！")
    else:
        # 模拟系统运行状态
        status_ui.metric("系统状态", "🟢 正在分析", "FPS: 25")

        # 准备假图表的数据容器
        chart_data = []

        # 【修复点 3】：循环次数和停顿微调，给予浏览器足够的视频解码缓冲时间
        for i in range(1, 21):
            # 写死核心假数据：检测到 3 个气孔
            current_defects = 3
            defect_count_ui.metric("当前缺陷总数", f"{current_defects} 个气孔")

            # 伪造最大缺陷面积的跳动数据
            simulated_area = 15.4 + np.random.uniform(-0.6, 0.6)
            max_area_ui.metric("最大缺陷面积", f"{simulated_area:.2f} mm²", f"{simulated_area - 15.4:+.2f} mm²")

            # 伪造折线图数据
            chart_data.append(simulated_area)
            chart_ui.line_chart(pd.DataFrame(chart_data, columns=["缺陷面积走势"]))

            # 伪造右侧的文字日志面板
            fake_log = pd.DataFrame({
                "时间戳": [f"00:00:{i:02d}", f"00:00:{max(0, i - 1):02d}", f"00:00:{max(0, i - 2):02d}"],
                "追踪ID": ["#Track-01", "#Track-02", "#Track-03"],
                "缺陷类型": ["气孔 (Porosity)"] * 3,
                "定量面积": [f"{simulated_area:.2f} mm²", "14.21 mm²", "12.88 mm²"]
            })
            log_ui.dataframe(fake_log, use_container_width=True, hide_index=True)

            # 稍微延长到 0.25 秒，缓解前端 WebSocket 渲染压力
            time.sleep(0.25)

        status_ui.metric("系统状态", "✅ 分析完成", "全帧去重完毕")
        st.success("🎉 动态视频流全帧深度探伤完成！这是目前的【空壳界面】，下周接入模型即可合体！")