import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time

# 1. 页面基本配置（设置宽屏模式和标题）
st.set_page_config(
    page_title="能源装备智能焊缝动态检测与定量分析系统",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 侧边栏 (Sidebar) - 紧扣赛题31的算法与跟踪参数配置
st.sidebar.markdown("## 🛠️ 工业级控制面板")
st.sidebar.markdown("---")

st.sidebar.subheader("🤖 核心多任务模型")
model_type = st.sidebar.selectbox(
    "选择识别与分割算法",
    ["Weld-YOLOv8-seg (动态优化版)", "Weld-YOLOv11s-seg (高速实时)", "RT-DETR-Weld (Transformer架构)"]
)

st.sidebar.subheader("🔄 跨帧跟踪与去重引擎")
tracker_type = st.sidebar.selectbox(
    "选择目标跟踪算法",
    ["ByteTrack (高帧率推荐)", "BoT-SORT (高精度运动补偿)"]
)
max_lost_frames = st.sidebar.slider("跟踪容错最大丢失帧数 (Max Lost)", 5, 50, 15, 5)

# 阈值滑动条
conf_threshold = st.sidebar.slider("缺陷置信度阈值 (Confidence)", 0.0, 1.0, 0.50, 0.05)
iou_threshold = st.sidebar.slider("重叠度 (IoU) 过滤阈值", 0.0, 1.0, 0.45, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 硬件与云端对接")
device = st.sidebar.radio("计算加速设备", ["GPU (CUDA 12.1)", "CPU (本地集群)"])
auto_save = st.sidebar.checkbox("自动同步定量分析报告至云端", value=True)

# 3. 主界面顶部 - 系统标题与工业 KPI 指标看板
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>⚡ 能源装备智能焊缝动态检测与定量分析系统</h1>",
            unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #666;'>2026能源装备创新设计大赛项目成果展示平台 —— 赛题31专项研发</p>",
    unsafe_allow_html=True)
st.divider()

# 四栏实时数据看板 - 突出“跨帧去重”和“定量分析”指标
kp1, kp2, kp3, kp4 = st.columns(4)
kp1.metric(label="🎥 今日已分析视频流总长", value="324.5 分钟", delta="+12.5%")
kp2.metric(label="🎯 动态跟踪准确率 (MOTA)", value="96.8%", delta="+0.4% (算法优化)")
kp3.metric(label="🔄 跨帧去重后缺陷总数", value="42 处", delta="-150 处伪重复 (去重率78%)")
kp4.metric(label="🟢 边缘计算节点状态", value="正常 (Running)", delta="温度: 43.2 °C")

st.divider()

# 4. 主界面核心 - 多功能标签页 (Tabs) 切换
tab1, tab2, tab3 = st.tabs(["🎥 动态视频流实时探伤", "📊 缺陷定量分析与统计", "📜 系统操作与探伤日志"])

# ---- TAB 1: 动态视频流实时探伤界面 ----
with tab1:
    st.subheader("📹 焊缝连续图像/视频流在线智能检测")
    st.write("支持实时导入工业摄像头视频流或连续图像序列，自动定位焊缝并进行跨帧缺陷去重统计。")

    # 文件上传器：升级为支持视频格式
    uploaded_video = st.file_uploader(
        "请上传待检测的焊缝动态视频流 (支持 MP4, AVI, MOV 格式)...",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:
        st.success("✅ 视频流加载成功！等待触发实时流水线。")

        # 预设交互控制键
        start_detection = st.button("▶️ 开始动态检测与跨帧去重分析")

        # 创建用于动态刷新的空容器（模拟视频流连续播放的核心逻辑）
        video_col, data_col = st.columns([3, 2])

        with video_col:
            video_placeholder = st.empty()  # 视频画面暂位符
        with data_col:
            metrics_placeholder = st.empty()  # 实时动态数据看板暂位符
            table_placeholder = st.empty()  # 实时表格暂位符

        if start_detection:
            # 建立模拟高保真视频流循环（共40帧，模拟焊缝滚动的动态效果）
            total_frames = 40

            for frame_idx in range(1, total_frames + 1):
                # 1. 动态生成一张模拟焊缝图像 (灰度底色 + 中间一条焊缝区域)
                img = Image.new('RGB', (640, 480), color=(60, 60, 62))
                draw = ImageDraw.Draw(img)

                # 画出“焊缝定位区域”
                draw.rectangle([180, 0, 460, 480], fill=(90, 92, 98))
                draw.line([(320, 0), (320, 480)], fill=(120, 125, 130), width=2, joint="round")

                # 2. 模拟缺陷动态出现、跨帧跟踪与移出画面
                current_defects = []
                if 5 <= frame_idx <= 32:
                    y_offset = (frame_idx - 5) * 12  # 让缺陷随着视频帧向下滚动

                    # 模拟 YOLO-seg 的不规则彩色边缘缺陷分割（绘制多边形）
                    poly_points = [
                        (280, 80 + y_offset), (310, 75 + y_offset),
                        (340, 95 + y_offset), (300, 120 + y_offset),
                        (270, 100 + y_offset)
                    ]
                    draw.polygon(poly_points, fill=(255, 0, 0, 100), outline="red")

                    # 绘制跟踪 BBox 和 ID 标签
                    draw.rectangle([265, 70 + y_offset, 345, 125 + y_offset], outline=(255, 215, 0), width=3)
                    draw.text((270, 50 + y_offset), "ID: 01 | Crack", fill=(255, 215, 0))

                    # 缺陷定量计算估算（赛题要求测算长度、宽度、面积等）
                    sim_length = 12.4 + np.sin(frame_idx) * 0.2
                    sim_width = 3.5 + np.cos(frame_idx) * 0.1
                    sim_area = 35.4 + np.sin(frame_idx) * 0.5

                    current_defects.append({
                        "当前帧号": frame_idx,
                        "目标跟踪ID": "#01",
                        "缺陷类别": "裂纹 (Crack)",
                        "定量长度 (mm)": f"{sim_length:.2f}",
                        "定量宽度 (mm)": f"{sim_width:.2f}",
                        "估算面积 (mm²)": f"{sim_area:.2f}",
                        "置信度": f"{conf_threshold + 0.38:.2%}"
                    })

                # 3. 将绘制好的帧推送到前端界面
                video_placeholder.image(img,
                                        caption=f"⚡ 动态边缘计算视频流 | 帧率: 25 FPS | 当前第 {frame_idx}/{total_frames} 帧",
                                        use_container_width=True)

                # 4. 右侧实时指标动态刷新
                with metrics_placeholder.container():
                    st.markdown("#### 📊 当前帧智能分析追踪")
                    m1, m2 = st.columns(2)
                    m1.metric("当前帧捕获目标数", f"{len(current_defects)} 个")
                    m2.metric("当前全局唯一缺陷数", "1 个" if frame_idx >= 5 else "0 个")

                # 5. 右侧表格动态刷新
                if current_defects:
                    df_current = pd.DataFrame(current_defects)
                    table_placeholder.dataframe(df_current, use_container_width=True)
                else:
                    table_placeholder.info("🔍 当前帧焊缝区域未见明显异常特征...")

                # 控制播放速度（模拟真实视频流延迟）
                time.sleep(0.08)

            # ---- 视频播放完毕，展示最终的“跨帧去重”与“定量分析报告”结果 ----
            st.success("🎉 动态视频流全帧深度探伤完成！已启动跨帧去重机制生成最终定量报告。")

            st.markdown("### 📋 最终成果交付分析报告（已去重合并）")
            # 展示去重后、包含完整定量分析参数的最终报告表格
            final_report_data = pd.DataFrame({
                "全域唯一缺陷ID": ["#WELD-TRACK-01"],
                "缺陷类别": ["裂纹 (Crack)"],
                "累计捕获帧数": ["28 帧 (第5-32帧)"],
                "最大几何长度": ["12.61 mm"],
                "最大几何宽度": ["3.62 mm"],
                "平均定量面积": ["35.42 mm²"],
                "位置分布": ["焊缝中心区下游 (Y:70~425 px)"],
                "置信度": [f"{conf_threshold + 0.38:.2%}"]
            })
            st.dataframe(final_report_data, use_container_width=True)
            st.button("📥 导出该焊缝动态视频国家标准探伤检测报告 (PDF)")

    else:
        st.info(
            "💡 团队协同提示：请在上方上传一段连续焊缝图像视频。系统将自动基于您在左侧配置的检测算法与 ByteTrack 跟踪器进行全帧级缺陷捕获与定量化分析。")

# ---- TAB 2: 缺陷数据统计分析（对应定量分析要求） ----
with tab2:
    st.subheader("📈 焊缝质量历史大数据看板（跨帧去重后）")
    st.write("此页面展示生产线历史运行中累积的缺陷宏观定量统计，直观呈现空间位置分布等关键参数。")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### 📐 被测缺陷几何面积分布特征 (mm²)")
        # 模拟缺陷面积定量数据
        area_distribution = pd.DataFrame(
            np.random.normal(25, 8, size=(100, 1)),
            columns=['历史缺陷定量面积 (mm²)']
        )
        st.bar_chart(area_distribution)

    with col_chart2:
        st.markdown("#### 📍 沿轴向焊缝缺陷空间位置分布概率 (%)")
        # 模拟缺陷位置分布数据
        position_data = pd.DataFrame({
            '位置分布': ['起始端 (0-20%)', '中前段 (21-50%)', '中后段 (51-80%)', '末尾端 (81-100%)'],
            '发生概率 (%)': [12.5, 44.2, 31.1, 12.2]
        })
        st.bar_chart(position_data.set_index('位置分布'))

# ---- TAB 3: 系统操作与探伤日志（保留审计日志能力） ----
with tab3:
    st.subheader("📜 工业现场多任务流日志审计")
    log_df = pd.DataFrame({
        "事件时间 (Timestamp)": ["2026-06-25 18:30:12", "2026-06-25 17:45:30", "2026-06-25 17:22:54"],
        "操作员/计算节点": ["Edge_Tracker_Node_01", "Edge_Scanner_02", "System_Init"],
        "执行动作/事件说明": [
            "完成连续视频流检测：共解析1200帧，经跨帧去重算法融合为1个独立裂纹事件",
            "接收连续图像序列 150 张，触发目标跟踪队列挂载",
            "系统初始化，GPU 加速引擎及实例分割边缘对齐算法加载成功"
        ],
        "日志级别": ["WARNING", "INFO", "SUCCESS"]
    })
    st.table(log_df)

# 页脚致谢
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #aaa; font-size: 12px;'>© 2026 能源装备智能焊缝检测系统研发团队 | 助力高端装备制造无损检测智能化</p>",
    unsafe_allow_html=True)