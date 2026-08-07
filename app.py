import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ============================================================
# Page Configuration & Custom CSS Styling
# ============================================================
st.set_page_config(
    page_title="AI Rice Identity Inspection Terminal",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS สำหรับธีมเข้มสไตล์ Futuristic
st.markdown("""
<style>
    .stApp {
        background-color: #070A11;
        color: #E2E8F0;
    }
    .header-box {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        font-weight: 700;
        font-size: 16px;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    div[data-testid="stButton"] > button[kind="secondary"] {
        background: #1E293B;
        color: #94A3B8;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    div[data-testid="stButton"] > button[kind="secondary"]:hover {
        background: #334155;
        color: #FFFFFF;
        border-color: #38BDF8;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 1. ฟังก์ชันสกัดฟีเจอร์ Voronoi (X1 - X6)
# ============================================================
def extract_voronoi_features(points):
    try:
        vor = Voronoi(points)
        areas = []
        shape_indices = []
        neighbor_counts = [0] * len(points)

        for p1, p2 in vor.ridge_points:
            neighbor_counts[p1] += 1
            neighbor_counts[p2] += 1

        for region_index in vor.point_region:
            region = vor.regions[region_index]
            if -1 not in region and len(region) > 2:
                polygon = vor.vertices[region]
                area = 0.5 * np.abs(
                    np.dot(polygon[:, 0], np.roll(polygon[:, 1], 1)) -
                    np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1))
                )
                perimeter = np.sum(
                    np.sqrt(
                        np.sum(
                            np.diff(np.vstack([polygon, polygon[0]]), axis=0)**2,
                            axis=1
                        )
                    )
                )
                if area > 0:
                    areas.append(area)
                    shape_indices.append(perimeter / np.sqrt(area))

        if len(areas) < 3:
            return None, None

        x1 = float(np.mean(areas))
        x2 = float(np.mean(neighbor_counts))
        x3 = float(len(areas) / np.sum(areas))
        x4 = float(np.mean(shape_indices))
        x5 = float(len(points))
        x6 = float(np.max(areas) / np.min(areas)) if np.min(areas) > 0 else 1.0

        return [x1, x2, x3, x4, x5, x6], vor
    except Exception:
        return None, None


# ============================================================
# 2. ฟังก์ชันประมวลผลภาพถ่ายข้าว
# ============================================================
def process_image(img):
    if img is None:
        return None, None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    points = cv2.goodFeaturesToTrack(
        enhanced,
        maxCorners=250,
        qualityLevel=0.005,
        minDistance=8
    )

    if points is None or len(points) < 4:
        return img, None, None

    points = points.reshape(-1, 2)
    features, vor = extract_voronoi_features(points)

    return img, features, vor


# ============================================================
# 3. โหลดและเทรน AI Model (GI vs Non-GI)
# ============================================================
@st.cache_resource
def load_ai_model():
    np.random.seed(42)
    X_nongi = np.random.normal([381, 4.73, 0.0038, 4.64, 26, 17], [50, 0.2, 0.001, 0.5, 5, 10], (100, 6))
    X_gi = np.random.normal([350, 5.08, 0.0037, 4.47, 44, 22], [50, 0.2, 0.001, 0.5, 5, 10], (100, 6))

    X_train = np.vstack([X_nongi, X_gi])
    y_train = np.array([0] * 100 + [1] * 100)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    pca = PCA(n_components=2)
    pca.fit(X_train_scaled)

    # Calibrate RSII Direction
    X_gi_scaled = scaler.transform(X_gi)
    pc_gi = pca.transform(X_gi_scaled)
    rsii_gi_raw = 0.7706 * pc_gi[:, 0] + 0.2294 * pc_gi[:, 1]
    rsii_direction = -1.0 if np.mean(rsii_gi_raw) > 0 else 1.0

    return scaler, model, pca, rsii_direction

scaler, model, pca, rsii_direction = load_ai_model()


# ============================================================
# 4. ฟังก์ชันเรนเดอร์ HTML Dashboard
# ============================================================
def render_dashboard(features):
    X_scaled = scaler.transform([features])
    probability = model.predict_proba(X_scaled)[0]

    classes = list(model.classes_)
    gi_index = classes.index(1) if 1 in classes else 1
    nongi_index = classes.index(0) if 0 in classes else 0

    gi_prob = probability[gi_index] * 100
    nongi_prob = probability[nongi_index] * 100

    pc_scores = pca.transform(X_scaled)[0]
    pc1, pc2 = pc_scores[0], pc_scores[1]
    rsii_score = (0.7706 * pc1 + 0.2294 * pc2) * rsii_direction

    if gi_prob >= 75:
        result_title = "🌾 ข้าวหอมมะลิ GI"
        status_badge = "VERIFIED GI AUTHENTIC"
        primary_color = "#00E676"
        accent_glow = "rgba(0, 230, 118, 0.25)"
        recommendation = f"<b>[อนุมัติการรับรอง]</b> ค่า RSII เท่ากับ <b>{rsii_score:+.3f}</b> และความน่าจะเป็น GI เท่ากับ <b>{gi_prob:.1f}%</b> มีลักษณะทางเรขาคณิตสอดคล้องกับมาตรฐานข้าวหอมมะลิ GI อย่างมีนัยสำคัญ สามารถออกใบรับรองสายพันธุ์ได้"
    elif gi_prob >= 50:
        result_title = "🌾 มีแนวโน้มเป็นข้าวหอมมะลิ GI"
        status_badge = "MODERATE MATCH"
        primary_color = "#FFD600"
        accent_glow = "rgba(255, 214, 0, 0.25)"
        recommendation = f"<b>[ควรตรวจสอบเพิ่ม]</b> ค่า RSII เท่ากับ <b>{rsii_score:+.3f}</b> และความน่าจะเป็น <b>{gi_prob:.1f}%</b> อยู่ในระดับก้ำกึ่ง แนะนำให้สุ่มถ่ายภาพเมล็ดอื่นเพิ่มเติมเพื่อยืนยัน"
    else:
        result_title = "🌾 ข้าวหอมมะลิ Non-GI"
        status_badge = "NON-GI DETECTED"
        primary_color = "#00E5FF"
        accent_glow = "rgba(0, 229, 255, 0.25)"
        recommendation = f"<b>[ไม่ผ่านเกณฑ์ GI]</b> ค่า RSII เท่ากับ <b>{rsii_score:+.3f}</b> และความน่าจะเป็น GI เพียง <b>{gi_prob:.1f}%</b> จัดอยู่ในกลุ่มข้าวหอมมะลิทั่วไป (Non-GI)"

    html_code = f"""
    <div style="width: 100%; font-family: 'Segoe UI', system-ui, sans-serif; background: #0B0F19; border-radius: 20px; box-shadow: 0 20px 50px rgba(0,0,0,0.6); border: 1px solid rgba(255, 255, 255, 0.08); overflow: hidden; color: #E2E8F0; padding: 24px; box-sizing: border-box;">
        
        <!-- Top Tech Bar -->
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 16px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="width: 10px; height: 10px; background: {primary_color}; border-radius: 50%; box-shadow: 0 0 10px {primary_color};"></span>
                <span style="font-size: 12px; font-weight: 700; letter-spacing: 1.5px; color: #94A3B8; text-transform: uppercase;">AI Rice Inspection Terminal</span>
            </div>
            <span style="background: {accent_glow}; color: {primary_color}; border: 1px solid {primary_color}; font-size: 11px; font-weight: 800; padding: 4px 14px; border-radius: 30px; letter-spacing: 0.5px;">
                {status_badge}
            </span>
        </div>

        <!-- Main HUD Grid -->
        <div style="display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 16px; margin-bottom: 20px;">
            <!-- Left Card: Classification Verdict -->
            <div style="background: linear-gradient(145deg, #131B2E, #1A243B); border-radius: 16px; padding: 20px; border: 1px solid rgba(255,255,255,0.05); display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div style="font-size: 11px; color: #64748B; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">Classification Verdict</div>
                    <div style="font-size: 20px; font-weight: 800; color: #FFFFFF; margin-bottom: 16px;">{result_title}</div>
                    <div style="font-size: 48px; font-weight: 900; color: {primary_color}; line-height: 1; text-shadow: 0 0 20px {accent_glow};">
                        {gi_prob:.1f}<span style="font-size: 24px;">%</span>
                    </div>
                    <div style="font-size: 12px; color: #94A3B8; font-weight: 600; margin-top: 6px;">GI Probability Score</div>
                </div>

                <div style="margin-top: 20px;">
                    <div style="display: flex; justify-content: space-between; font-size: 11px; font-weight: 700; margin-bottom: 8px;">
                        <span style="color: {primary_color};">GI: {gi_prob:.1f}%</span>
                        <span style="color: #00E5FF;">Non-GI: {nongi_prob:.1f}%</span>
                    </div>
                    <div style="height: 8px; background: #0F172A; border-radius: 6px; overflow: hidden; display: flex; border: 1px solid rgba(255,255,255,0.05);">
                        <div style="width: {gi_prob}%; background: {primary_color}; box-shadow: 0 0 10px {primary_color};"></div>
                        <div style="width: {nongi_prob}%; background: #00E5FF;"></div>
                    </div>
                </div>
            </div>

            <!-- Right Card: RSII Index -->
            <div style="background: linear-gradient(145deg, #131B2E, #1A243B); border-radius: 16px; padding: 20px; border: 1px solid rgba(255,255,255,0.05); display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div style="font-size: 11px; color: #64748B; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">Structural Identity Index</div>
                    <div style="font-size: 13px; font-weight: 700; color: #38BDF8;">RSII Score</div>
                    <div style="font-size: 38px; font-weight: 900; color: #F8FAFC; margin-top: 8px; line-height: 1;">
                        {rsii_score:+.3f}
                    </div>
                </div>

                <div style="background: rgba(15, 23, 42, 0.8); padding: 12px; border-radius: 10px; border: 1px solid rgba(56, 189, 248, 0.2); margin-top: 12px;">
                    <div style="font-size: 10px; color: #94A3B8; text-transform: uppercase; font-weight: 700;">Structural Pattern Density</div>
                    <div style="font-size: 14px; font-weight: 800; color: #38BDF8; margin-top: 2px;">
                        {"โครงสร้างแน่น (GI Typical Pattern)" if rsii_score > 0 else "โครงสร้างโปร่ง (Non-GI Pattern)"}
                    </div>
                    <div style="font-size: 10px; color: #64748B; margin-top: 2px;">คำนวณจาก Voronoi PCA Projection</div>
                </div>
            </div>
        </div>

        <!-- Geometric Features Breakdown (X1 - X4) -->
        <div style="background: linear-gradient(145deg, #131B2E, #1A243B); border-radius: 16px; padding: 16px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 20px;">
            <div style="font-size: 11px; color: #64748B; font-weight: 700; text-transform: uppercase; margin-bottom: 12px;">📐 Geometric Features Breakdown (X1 - X4)</div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
                <div style="background: rgba(15, 23, 42, 0.6); padding: 10px 8px; border-radius: 8px; text-align: center; border-bottom: 2px solid #38BDF8;">
                    <div style="font-size: 10px; color: #94A3B8; font-weight: 600; margin-bottom: 4px;">X1 พื้นที่เซลล์เฉลี่ย</div>
                    <div style="font-size: 15px; font-weight: 800; color: #F8FAFC;">{features[0]:.1f}</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); padding: 10px 8px; border-radius: 8px; text-align: center; border-bottom: 2px solid #818CF8;">
                    <div style="font-size: 10px; color: #94A3B8; font-weight: 600; margin-bottom: 4px;">X2 เพื่อนบ้านเฉลี่ย</div>
                    <div style="font-size: 15px; font-weight: 800; color: #F8FAFC;">{features[1]:.2f}</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); padding: 10px 8px; border-radius: 8px; text-align: center; border-bottom: 2px solid #C084FC;">
                    <div style="font-size: 10px; color: #94A3B8; font-weight: 600; margin-bottom: 4px;">X3 ความหนาแน่น</div>
                    <div style="font-size: 15px; font-weight: 800; color: #F8FAFC;">{features[2]:.4f}</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); padding: 10px 8px; border-radius: 8px; text-align: center; border-bottom: 2px solid #F472B6;">
                    <div style="font-size: 10px; color: #94A3B8; font-weight: 600; margin-bottom: 4px;">X4 ดัชนีรูปร่าง</div>
                    <div style="font-size: 15px; font-weight: 800; color: #F8FAFC;">{features[3]:.2f}</div>
                </div>
            </div>
        </div>

        <!-- Executive Recommendation Box -->
        <div style="background: rgba(15, 23, 42, 0.9); border-left: 4px solid {primary_color}; padding: 14px 18px; border-radius: 10px; font-size: 12px; color: #CBD5E1; line-height: 1.6;">
            <span style="color: {primary_color}; font-weight: 800; margin-right: 6px;">💡 AI Decision Recommendation:</span> {recommendation}
        </div>
    </div>
    """
    st.html(html_code)


# ============================================================
# 5. UI Application Flow Control
# ============================================================

# Header Section
st.markdown("""
<div class="header-box">
    <h1 style="margin: 0; font-size: 28px; font-weight: 900; color: #FFFFFF; display: flex; align-items: center; gap: 12px;">
        🌾 AI Rice Identity Inspection Terminal
    </h1>
    <p style="margin: 6px 0 0 0; font-size: 14px; color: #94A3B8;">
        ระบบจำแนกอัตลักษณ์ข้าวหอมมะลิ GI และ Non-GI ด้วย Voronoi Tessellation & RSII Index
    </p>
</div>
""", unsafe_allow_html=True)

if "processed" not in st.session_state:
    st.session_state.processed = False

col_upload, col_preview = st.columns([1, 1.2], gap="large")

with col_upload:
    st.markdown("### 📷 1. เลือก/ถ่ายภาพเมล็ดข้าว")
    uploaded_file = st.file_uploader(
        "อัปโหลดภาพหน้าตัดเมล็ดข้าว (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        key="file_uploader"
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.markdown("<br>", unsafe_allow_html=True)
        btn_col1, btn_col2 = st.columns([1, 1])

        with btn_col1:
            if st.button("🔍 เริ่มวิเคราะห์ด้วย AI", type="primary"):
                st.session_state.processed = True

        with btn_col2:
            if st.button("🔄 ล้างค่า / ภาพใหม่", type="secondary"):
                st.session_state.processed = False
                st.rerun()

with col_preview:
    if uploaded_file is not None:
        st.markdown("### 🖼️ 2. ภาพตัวอย่าง")
        if not st.session_state.processed:
            st.image(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), caption="ภาพหน้าตัดเมล็ดข้าวที่อัปโหลด", use_container_width=True)
        else:
            with st.spinner("⚡ กำลังสกัดฟีเจอร์ Voronoi และประมวลผลด้วย AI Model..."):
                img_out, features, vor = process_image(raw_img)

                if features is not None and vor is not None:
                    fig, ax = plt.subplots(figsize=(5, 5), facecolor='#0B0F19')
                    ax.set_facecolor('#0B0F19')
                    ax.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
                    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_width=1.5, line_colors='#00E676', point_size=5)
                    ax.set_title("Voronoi Endosperm Tessellation", fontsize=11, color='#00E676', fontweight='bold', loc='left', pad=12)
                    ax.axis("off")
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.error("❌ ไม่พบจุดโครงสร้างบนเมล็ดข้าวที่เพียงพอ กรุณาถ่ายภาพหน้าตัดให้คมชัดยิ่งขึ้น")

# Dashboard Output
if uploaded_file is not None and st.session_state.processed:
    if features is not None and vor is not None:
        st.markdown("---")
        st.markdown("### 📊 3. ผลการวิเคราะห์และตรวจสอบอัตลักษณ์ (AI Inspection Dashboard)")
        render_dashboard(features)
