import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import time

# 1. 页面基本配置（设置宽屏模式和标题）
st.set_page_config(
    page_title="能源装备智能焊缝检测系统",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 侧边栏 (Sidebar) - 用于控制和模型参数配置
st.sidebar.markdown("## 🛠️ 控制面板")
st.sidebar.markdown("---")

st.sidebar.subheader("🤖 模型配置")
model_type = st.sidebar.selectbox(
    "选择核心识别算法",
    ["Weld-YOLOv11s (高速实时)", "Weld-YOLOv11x (高精全面)", "RT-DETR-Weld (端到端变压器)"]
)

# 阈值滑动条
conf_threshold = st.sidebar.slider("置信度阈值 (Confidence)", 0.0, 1.0, 0.50, 0.05)
iou_threshold = st.sidebar.slider("非极大值抑制 (IoU) 阈值", 0.0, 1.0, 0.45, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 硬件与环境")
device = st.sidebar.radio("计算加速设备", ["GPU (CUDA 12.1)", "CPU (本地集群)"])
auto_save = st.sidebar.checkbox("自动保存检测报告到云端", value=True)

# 3. 主界面顶部 - 系统标题与工业 KPI 指标看板
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>⚡ 能源装备智能焊缝检测系统 v1.0</h1>",
            unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>2026能源装备创新设计大赛项目成果展示平台</p>",
            unsafe_allow_html=True)
st.divider()

# 四栏实时数据看板
kp1, kp2, kp3, kp4 = st.columns(4)
kp1.metric(label="📊 今日累计已检测焊缝", value="1,428 道", delta="+15% (较昨日)")
kp2.metric(label="⏱️ AI 平均检测耗时", value="38 ms", delta="-4 ms (算法加速)")
kp3.metric(label="🎯 异常缺陷检出率", value="99.2%", delta="+0.4% (增量训练)")
kp4.metric(label="🟢 边缘设备运行状态", value="正常 (Running)", delta="温度: 41.5 °C")

st.divider()

# 4. 主界面核心 - 多功能标签页 (Tabs) 切换
tab1, tab2, tab3 = st.tabs(["🔍 单张/批量图像检测", "📊 缺陷数据统计分析", "📜 系统操作与探伤日志"])

# ---- TAB 1: 实时检测界面 ----
with tab1:
    st.subheader("📸 焊缝图像在线智能探伤")

    # 文件上传器
    uploaded_file = st.file_uploader(
        "请上传待检测的焊缝图像 (支持 JPG, PNG, TIFF 格式，例如射线、超声、或外观微距图)...",
        type=["jpg", "png", "jpeg", "tiff"]
    )

    if uploaded_file is not None:
        # 读取图片
        image = Image.open(uploaded_file)

        # 模拟 AI 模型的推理延迟
        with st.spinner('🚀 核心算法正在深度解析焊缝图像，识别缺陷特征...'):
            time.sleep(1.2)  # 模拟1.2秒的检测过程

        # 左右两栏对比显示
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(image, caption="原始焊缝图像 (Original)", use_container_width=True)

        with col_img2:
            # 模拟 AI 绘制边界框（在真实场景中，你会用 YOLO 的 results.plot() 替换这部分）
            detected_image = image.copy()
            draw = ImageDraw.Draw(detected_image)
            w, h = detected_image.size
            # 在图中央模拟画一个红色的缺陷框
            draw.rectangle([w * 0.35, h * 0.4, w * 0.65, h * 0.6], outline="red", width=max(4, int(w * 0.01)))
            st.image(detected_image, caption="💡 AI 缺陷识别结果 (Detected)", use_container_width=True)

        # 缺陷报告数据表
        st.subheader("📋 缺陷分析结果明细")
        # 模拟数据表，置信度会根据左侧滑动条动态变化
        mock_defect_data = pd.DataFrame({
            "缺陷编号": ["#WELD-2026-001"],
            "缺陷类型": ["未熔合 (Lack of Fusion)"],
            "置信度 (Confidence)": [f"{conf_threshold + (1 - conf_threshold) * 0.76:.2%}"],
            "几何像素尺寸 (BBox)": [f"[X:{int(w * 0.35)}, Y:{int(h * 0.4)}, W:{int(w * 0.3)}, H:{int(h * 0.2)}]"],
            "工业安全评级": ["严重缺陷 (High Risk - 建议返修)"]
        })
        st.dataframe(mock_defect_data, use_container_width=True)

        # 导出交互
        st.success("🎉 检测完成！报告已自动生成。")
        st.button("📥 导出该焊缝国家标准探伤检测报告 (PDF)")

    else:
        st.info("💡 提示：请在上方上传一张工业焊缝图片，系统将自动基于您在左侧配置的算法和阈值进行毫秒级缺陷捕获。")

# ---- TAB 2: 缺陷统计分析 ----
with tab2:
    st.subheader("📈 生产线焊缝质量大数据看板")
    st.write("此页面展示系统在历史运行中累积的缺陷宏观统计，适合大赛答辩时向评委展示系统的工程落地能力。")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### 🔄 近30天常见缺陷类型趋势 (条数)")
        # 模拟缺陷趋势数据
        trend_data = pd.DataFrame(
            np.random.randint(2, 15, size=(15, 3)),
            columns=['气孔 (Porosity)', '裂纹 (Crack)', '夹渣 (Slag)']
        )
        st.line_chart(trend_data)

    with col_chart2:
        st.markdown("#### 📊 各锅炉/压力容器批次焊缝合格率 (%)")
        # 模拟合格率数据
        bar_data = pd.DataFrame({
            '装备批次': ['集箱管件A批次', '主蒸汽管B批次', '压力容器C批次', '重组反应器D批次'],
            '合格率 (%)': [98.5, 94.2, 99.1, 97.6]
        })
        st.bar_chart(bar_data.set_index('装备批次'))

# ---- TAB 3: 系统历史日志 ----
with tab3:
    st.subheader("📜 工业现场探伤日志审计")
    # 模拟系统后台真实日志
    log_df = pd.DataFrame({
        "事件时间 (Timestamp)": ["2026-06-25 18:30:12", "2026-06-25 17:45:30", "2026-06-25 17:22:54"],
        "操作员/节点": ["Admin_Operator", "Edge_Scanner_02", "System_Init"],
        "执行动作/事件说明": ["完成主蒸汽管B批次#04号焊缝检测，发现“严重未熔合”缺陷",
                              "批量拉取射线图像 50 张，触发并行推理", "系统成功初始化，GPU 加速引擎 (TensorRT) 挂载成功"],
        "运行日志级别": ["WARNING", "INFO", "SUCCESS"]
    })
    st.table(log_df)

# 页脚致谢
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #aaa; font-size: 12px;'>© 2026 能源装备智能焊缝检测系统研发团队 | 助力高端装备制造无损检测智能化</p>",
    unsafe_allow_html=True)