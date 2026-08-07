import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="AI Rice Identity Detection", layout="centered")
st.title("🌾 AI Rice Identity Inspection")

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

    X_gi_scaled = scaler.transform(X_gi)
    pc_gi = pca.transform(X_gi_scaled)
    rsii_gi_raw = 0.7706 * pc_gi[:, 0] + 0.2294 * pc_gi[:, 1]
    rsii_direction = -1.0 if np.mean(rsii_gi_raw) > 0 else 1.0

    return scaler, model, pca, rsii_direction

scaler, model, pca, rsii_direction = load_ai_model()

PROVINCE_RSII_REF = {
    "เชียงราย": {"mean": -0.81, "sd": 0.52},
    "ร้อยเอ็ด": {"mean": -0.77, "sd": 0.44},
    "พะเยา": {"mean": 0.18, "sd": 0.60},
    "อุบลราชธานี": {"mean": 0.44, "sd": 1.06},
    "สุรินทร์": {"mean": 0.96, "sd": 2.07}
}

def match_origin_province(rsii_val):
    closest_province = None
    min_distance = float('inf')

    for prov, stat in PROVINCE_RSII_REF.items():
        dist = abs(rsii_val - stat["mean"])
        if dist < min_distance:
            min_distance = dist
            closest_province = prov

    ref_mean = PROVINCE_RSII_REF[closest_province]["mean"]
    ref_sd = PROVINCE_RSII_REF[closest_province]["sd"]
    return closest_province, ref_mean, ref_sd

def render_dashboard(features):
    X_scaled = scaler.transform([features])
    probability = model.predict_proba(X_scaled)[0]

    nongi_prob = probability[0] * 100
    gi_prob = probability[1] * 100

    pc_scores = pca.transform(X_scaled)[0]
    pc1, pc2 = pc_scores[0], pc_scores[1]
    rsii_score = (0.7706 * pc1 + 0.2294 * pc2) * rsii_direction

    matched_prov, ref_mean, ref_sd = match_origin_province(rsii_score)

    if gi_prob >= 75:
        result_title = "🌾 ข้าวหอมมะลิ GI"
        status_badge = "VERIFIED GI AUTHENTIC"
        primary_color = "#00E676"
        accent_glow = "rgba(0, 230, 118, 0.25)"
        recommendation = f"<b>[อนุมัติการรับรอง]</b> ค่า RSII เท่ากับ <b>{rsii_score:+.3f}</b> มีลักษณะทางเรขาคณิตสอดคล้องกับกลุ่มฐานข้อมูลอัตลักษณ์จังหวัด <b>{matched_prov}</b> (ค่าเฉลี่ยอ้างอิง {ref_mean:+.2f} ± {ref_sd:.2f}) สามารถออกใบรับรองสายพันธุ์ GI ได้"
    elif gi_prob >= 50:
        result_title = "🌾 มีแนวโน้มเป็นข้าวหอมมะลิ GI"
        status_badge = "MODERATE MATCH"
        primary_color = "#FFD600"
        accent_glow = "rgba(255, 214, 0, 0.25)"
        recommendation = f"<b>[ควรตรวจสอบเพิ่ม]</b> ค่า RSII เท่ากับ <b>{rsii_score:+.3f}</b> ใกล้เคียงกับกลุ่มอัตลักษณ์จังหวัด <b>{matched_prov}</b> แนะนำให้สุ่มถ่ายภาพเมล็ดอื่นเพิ่มเติมเพื่อยืนยัน"
    else:
        result_title = "🌾 ข้าวหอมมะลิ Non-GI"
        status_badge = "NON-GI DETECTED"
        primary_color = "#00E5FF"
        accent_glow = "rgba(0, 229, 255, 0.25)"
        recommendation = f"<b>[ไม่ผ่านเกณฑ์ GI]</b> ค่า RSII เท่ากับ <b>{rsii_score:+.3f}</b> มีรูปแบบเรขาคณิตสอดคล้องกับกลุ่มอัตลักษณ์แหล่งปลูก <b>{matched_prov}</b> (ค่าเฉลี่ยอ้างอิง {ref_mean:+.2f} ± {ref_sd:.2f}) ซึ่งจัดอยู่ในกลุ่มข้าวหอมมะลิทั่วไป (Non-GI)"

    html_code = f"""
    <div style="width: 100%; font-family: sans-serif; background: #0B0F19; border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.08); overflow: hidden; color: #E2E8F0; padding: 20px; box-sizing: border-box;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 12px; margin-bottom: 16px;">
            <span style="font-size: 11px; font-weight: 700; color: #94A3B8; text-transform: uppercase;">AI Rice Inspection Terminal</span>
            <span style="background: {accent_glow}; color: {primary_color}; border: 1px solid {primary_color}; font-size: 10px; font-weight: 800; padding: 3px 10px; border-radius: 20px;">{status_badge}</span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px;">
            <div style="background: #131B2E; border-radius: 12px; padding: 16px;">
                <div style="font-size: 10px; color: #64748B; font-weight: 700;">CLASSIFICATION VERDICT</div>
                <div style="font-size: 16px; font-weight: 800; color: #FFFFFF; margin-top: 4px;">{result_title}</div>
                <div style="font-size: 36px; font-weight: 900; color: {primary_color}; margin-top: 8px;">{gi_prob:.1f}%</div>
                <div style="font-size: 11px; color: #94A3B8;">GI Probability Score</div>
            </div>
            <div style="background: #131B2E; border-radius: 12px; padding: 16px;">
                <div style="font-size: 10px; color: #64748B; font-weight: 700;">RSII SCORE</div>
                <div style="font-size: 30px; font-weight: 900; color: #F8FAFC; margin-top: 8px;">{rsii_score:+.3f}</div>
                <div style="font-size: 12px; font-weight: 800; color: #38BDF8; margin-top: 8px;">📍 จ.{matched_prov}</div>
            </div>
        </div>
        <div style="background: rgba(15, 23, 42, 0.9); border-left: 4px solid {primary_color}; padding: 12px; border-radius: 8px; font-size: 11px; color: #CBD5E1; line-height: 1.5;">
            <span style="color: {primary_color}; font-weight: 800;">💡 Recommendation:</span> {recommendation}
        </div>
    </div>
    """
    st.html(html_code)

uploaded_file = st.file_uploader("📷 เลือกรูปภาพหน้าตัดเมล็ดข้าว", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img, features, vor = process_image(img)

    if features is not None and vor is not None:
        fig, ax = plt.subplots(figsize=(4, 4), facecolor='#0B0F19')
        ax.set_facecolor('#0B0F19')
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_width=1.2, line_colors='#00E676', point_size=4)
        ax.set_title("Voronoi Endosperm Tessellation", fontsize=10, color='#00E676', fontweight='bold', loc='left')
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)

        render_dashboard(features)
    else:
        st.error("❌ ไม่พบจุดโครงสร้างบนเมล็ดข้าวที่เพียงพอ กรุณาถ่ายภาพหน้าตัดให้คมชัดยิ่งขึ้น")