# 导入依赖库
import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import io
import zipfile

# -------------------------- 页面配置 --------------------------
st.set_page_config(
    page_title="YOLO11 目标检测可视化平台",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 页面标题（贴合你的毕业设计内容）
st.title("🎓 目标检测可视化平台")
st.markdown("### YOLO11n-SDMB | YOLO11n-SDMB-tiny")
st.markdown("支持：批量图片检测 | 自定义检测类别 | 检测框样式自定义 | 视频/摄像头检测")
st.divider()

# -------------------------- 缓存加载自定义模型（核心：你的两个自研模型） --------------------------
@st.cache_resource
def load_custom_model(model_name):
    """加载模型  缓存机制避免重复加载"""
    try:
        model = YOLO(model_name)
        return model
    except ModuleNotFoundError as e:
        st.error(f"❌ 模型加载失败：缺少模块 {e}")
        st.warning("请检查：1. 安装和训练模型时一致的ultralytics版本；2. 复制缺失的自定义模块文件（如modules_addition.py）到ultralytics/nn/目录")
        raise  # 抛出异常，方便你看详细报错
    except Exception as e:
        st.error(f"❌ 模型加载失败：{e}")
        st.warning("请检查模型文件路径是否正确，或模型文件是否损坏")
        raise

# -------------------------- 侧边栏：核心参数配置 --------------------------
with st.sidebar:
    st.header("⚙️ 检测参数设置")

    # ====================== 核心新增：你的自研双模型选择 ======================
    st.subheader("🧠 选择模型")
    model_option = st.selectbox(
        "切换检测模型",
        options=[
            "YOLO11n-SDMB",
            "YOLO11n-SDMB-tiny"
        ],
        index=0,
        help="模型：YOLO11n-SDMB改进模型 / YOLO11n-SDMB-tiny轻量化模型"
    )
    # 映射模型文件名（将你的模型文件放在项目根目录即可）
    model_mapping = {
        "YOLO11n-SDMB": "yolo11n_sdmb.pt",
        "YOLO11n-SDMB-tiny": "yolo11n_sdmb_tiny.pt"
    }
    selected_model_file = model_mapping.get(model_option, "yolo11n_sdmb.pt")

    # 置信度阈值
    conf_threshold = st.slider(
        "置信度阈值",
        min_value=0.1, max_value=1.0, value=0.25, step=0.05,
        help="过滤低置信度检测结果"
    )

    # 检测框样式自定义（保留原有功能）
    st.subheader("🎨 检测框样式设置")
    line_width = st.slider("框体线条粗细", 1, 10, 2, 1)
    box_color = st.color_picker("检测框颜色", "#ff7f0e")
    box_rgb = tuple(int(box_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    # 加载你的自定义模型
    model = load_custom_model(selected_model_file)
    # 自动获取模型的类别（兼容你训练的数据集类别）
    all_classes = list(model.names.values())
    all_class_ids = list(model.names.keys())

    # 自定义检测类别（保留原有功能）
    st.subheader("📋 自定义检测类别")
    selected_classes = st.multiselect(
        "勾选需要检测的目标（默认检测所有类别）",
        options=all_classes,
        default=[],
        help="只检测你需要的目标类别"
    )
    filter_class_ids = [all_class_ids[all_classes.index(c)] for c in selected_classes] if selected_classes else None

    # 检测模式（4种模式：单图+批量+视频+摄像头）
    st.subheader("📂 选择检测模式")
    detect_mode = st.radio(
        "",
        ["📸 单张图片检测", "🖼️ 批量图片检测", "🎥 视频检测", "📹 实时摄像头检测"],
        index=0
    )

    st.divider()
    st.info(f"✅ 当前加载模型：{model_option}\n💡 首次加载模型后会缓存")

# -------------------------- 统一绘制检测结果（应用自定义样式） --------------------------
def plot_detection(result):
    return result.plot(
        line_width=line_width,
        colors=[box_rgb],
        labels=True,
        conf=True
    )

# -------------------------- 1. 单张图片检测 --------------------------
def image_detection():
    import cv2
    st.subheader("📸 单张图片目标检测")
    uploaded_img = st.file_uploader("上传单张图片", type=["jpg", "jpeg", "png"])
    
    if uploaded_img is not None:
        img = Image.open(uploaded_img)
        img_np = np.array(img)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 原图")
            st.image(img, use_column_width=True)
        
        # 自研模型推理
        with st.spinner("🔍 模型检测中..."):
            results = model.predict(
                source=img_np,
                conf=conf_threshold,
                classes=filter_class_ids,
                save=False
            )
        
        result_img = plot_detection(results[0])
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        with col2:
            st.markdown("#### 检测结果")
            st.image(result_img_rgb, use_column_width=True)
        
        # 检测统计
        detect_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]
        st.markdown(f"✅ 检测到目标：**{len(detect_classes)}** 个 | 类别：**{set(detect_classes)}**")
        
        # 下载结果
        buf = io.BytesIO()
        Image.fromarray(result_img_rgb).save(buf, format="PNG")
        st.download_button("💾 下载结果图", buf.getvalue(), "detection_result.png", "image/png")

# -------------------------- 2. 批量图片检测（毕业设计核心功能） --------------------------
def batch_image_detection():
    import cv2
    st.subheader("🖼️ 批量图片检测（多图上传+打包下载）")
    uploaded_files = st.file_uploader(
        "上传多张图片（Ctrl/Shift多选）",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"✅ 成功上传 {len(uploaded_files)} 张图片")
        st.divider()

        result_images = []
        result_filenames = []
        grid_cols = st.columns(3)

        for idx, img_file in enumerate(uploaded_files):
            img = Image.open(img_file)
            img_np = np.array(img)

            # 自研模型批量推理
            results = model.predict(
                source=img_np,
                conf=conf_threshold,
                classes=filter_class_ids,
                save=False
            )

            res_img = plot_detection(results[0])
            res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
            result_images.append(res_img_rgb)
            result_filenames.append(f"YOLO11n-SDMB检测_{img_file.name}")

            # 网格展示
            with grid_cols[idx % 3]:
                st.image(res_img_rgb, caption=f"结果 {idx+1}", use_column_width=True)

        # ZIP批量下载
        st.divider()
        st.subheader("📦 批量下载所有检测结果")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for img_data, fname in zip(result_images, result_filenames):
                img_buf = io.BytesIO()
                Image.fromarray(img_data).save(img_buf, format="PNG")
                zf.writestr(fname, img_buf.getvalue())

        st.download_button(
            label="💾 下载全部结果（ZIP打包）",
            data=zip_buffer.getvalue(),
            file_name="YOLO11n-SDMB批量检测结果.zip",
            mime="application/zip"
        )

# -------------------------- 3. 视频检测 --------------------------
def video_detection():
    import cv2
    st.subheader("🎥 视频目标检测")
    uploaded_video = st.file_uploader("上传视频", type=["mp4", "avi", "mov"])
    
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        st.markdown("#### 原视频")
        st.video(video_path)
        st.markdown("#### 实时检测过程")

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w, h = int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        placeholder = st.empty()

        with st.spinner("视频处理中..."):
            results = model.predict(
                source=video_path,
                conf=conf_threshold,
                classes=filter_class_ids,
                stream=True
            )

            for res in results:
                frame = plot_detection(res)
                out.write(frame)
                placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

        cap.release()
        out.release()
        st.success("✅ 视频检测完成！")
        st.video(output_path)

        # 下载视频
        with open(output_path, "rb") as f:
            st.download_button("💾 下载检测视频", f, "YOLO11n-SDMB_video_result.mp4", "video/mp4")

        os.unlink(video_path)
        os.unlink(output_path)

# -------------------------- 4. 实时摄像头检测 --------------------------
def camera_detection():
    import cv2
    st.subheader("📹 实时摄像头检测")
    camera = st.camera_input("点击开启摄像头")
    if camera:
        img = Image.open(camera)
        img_np = np.array(img)
        results = model.predict(source=img_np, conf=conf_threshold, classes=filter_class_ids)
        res_img = plot_detection(results[0])
        st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), use_column_width=True)

# -------------------------- 主程序 --------------------------
if __name__ == "__main__":
    if detect_mode == "📸 单张图片检测":
        image_detection()
    elif detect_mode == "🖼️ 批量图片检测":
        batch_image_detection()
    elif detect_mode == "🎥 视频检测":
        video_detection()
    elif detect_mode == "📹 实时摄像头检测":
        camera_detection()

import streamlit.web.cli as stcli
import sys
if __name__ == "__main__":
    sys.argv = ["streamlit", "run", __file__]
    stcli.main()
    
