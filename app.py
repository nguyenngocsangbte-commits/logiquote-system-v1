# app.py
from __future__ import annotations
import json
import requests
import pandas as pd
import streamlit as st

# =======================
# Cấu hình
# =======================
API_URL = "http://127.0.0.1:8000"   # đổi nếu bạn chạy API ở host/port khác

ROUTES = ["USA", "JAPAN", "TW", "KOREA"]
SERVICES = {"Hàng thường (normal)": "normal", "Hàng đông lạnh (frozen)": "frozen"}

st.set_page_config(
    page_title="Hệ thống báo giá Logistics",
    page_icon="📦",
    layout="centered",
)

# =======================
# UI — Nhập liệu
# =======================
st.title("📦 Hệ thống báo giá Logistics")

with st.expander("🔧 Cấu hình (tùy chọn)", expanded=False):
    api_input = st.text_input("API URL", value=API_URL, help="Ví dụ: http://127.0.0.1:8000")
    API_URL = api_input.strip() or API_URL

st.subheader("Nhập thông tin để tính báo giá")

c1, c2 = st.columns(2)
route = c1.selectbox("Tuyến quốc tế", ROUTES, index=1)           # mặc định JAPAN
service_label = c2.selectbox("Loại hàng", list(SERVICES.keys()), index=0)
service = SERVICES[service_label]

c3, c4 = st.columns(2)
gw = c3.number_input("GW — Cân nặng thực tế (kg)", min_value=0.0, step=0.5, format="%.2f", value=12.0)
length_cm = c4.number_input("D — Dài (cm)", min_value=0.0, step=1.0, value=50.0)

c5, c6 = st.columns(2)
width_cm  = c5.number_input("R — Rộng (cm)", min_value=0.0, step=1.0, value=40.0)
height_cm = c6.number_input("C — Cao (cm)",  min_value=0.0, step=1.0, value=30.0)

items_text = st.text_input("Mặt hàng (phân tách dấu phẩy)", value="mỹ phẩm, điện thoại")

btn = st.button("🚀 Tính báo giá", type="primary")

# =======================
# Gọi API & Hiển thị
# =======================
def vnd(x):
    if x is None:
        return "—"
    try:
        return f"{float(x):,.0f} VND"
    except Exception:
        return str(x)

if btn:
    # kiểm tra input
    if gw <= 0 or length_cm <= 0 or width_cm <= 0 or height_cm <= 0:
        st.error("Vui lòng nhập GW và D/R/C > 0.")
        st.stop()

    payload = {
        "route": route,
        "service": service,
        "gw": gw,
        "length_cm": length_cm,
        "width_cm": width_cm,
        "height_cm": height_cm,
        "items": [s.strip() for s in items_text.split(",") if s.strip()],
    }

    try:
        resp = requests.post(f"{API_URL}/quote", json=payload, timeout=30)
    except requests.exceptions.ConnectionError:
        st.error("Không kết nối được API. Hãy kiểm tra xem bạn đã chạy `uvicorn api:app --reload --port 8000` chưa.")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi gọi API: {e}")
        st.stop()

    if resp.status_code != 200:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        st.error(f"Lỗi từ API ({resp.status_code}): {detail}")
        st.stop()

    data = resp.json()

    # --- Thông tin tổng quan
    st.success("✅ Đã tính xong báo giá")
    meta_cols = st.columns(4)
    meta_cols[0].metric("Tuyến", data.get("route_resolved", route))
    meta_cols[1].metric("Loại hàng", data.get("service", service))
    meta_cols[2].metric("GW (kg)", f"{data.get('gw', 0):,.2f}")
    meta_cols[3].metric("CW (kg)", f"{data.get('cw', 0):,.2f}")

    # DW & lưu ý
    dw = data.get("dw", 0)
    ignore_dw = data.get("ignore_dw", False)
    info_cols = st.columns(2)
    info_cols[0].metric("DW (kg)", f"{dw:,.2f}")
    if ignore_dw:
        info_cols[1].write(":orange[Quốc gia này bỏ DW → tính theo GW.]")

    # --- Giá quốc tế
    st.subheader("💸 Cước quốc tế")
    price_cols = st.columns(2)
    price_cols[0].metric("Đơn giá (VND/kg)", vnd(data.get("unit_price_vnd_per_kg")))
    price_cols[1].metric("Tổng cước quốc tế", vnd(data.get("intl_total_vnd")))

    # --- Phụ phí theo từng mặt hàng
    st.subheader("🧾 Phụ phí theo mặt hàng")
    surcharge_items = data.get("surcharge_items", []) or []
    if surcharge_items:
        df = pd.DataFrame(surcharge_items)
        # đổi tên cột, format số
        rename_map = {
            "item_display": "Mặt hàng",
            "display_text": "Phụ thu (hiển thị)",
            "min_vnd": "Min (VND)",
            "max_vnd": "Max (VND)",
            "unit": "Đơn vị"
        }
        df_show = df.rename(columns=rename_map)
        if "Min (VND)" in df_show:
            df_show["Min (VND)"] = df_show["Min (VND)"].apply(lambda x: f"{x:,}" if pd.notna(x) else "—")
        if "Max (VND)" in df_show:
            df_show["Max (VND)"] = df_show["Max (VND)"].apply(lambda x: f"{x:,}" if pd.notna(x) else "—")
        st.dataframe(df_show[["Mặt hàng","Phụ thu (hiển thị)","Min (VND)","Max (VND)","Đơn vị"]], use_container_width=True)
    else:
        st.info("Không có danh sách phụ phí (hoặc bạn để trống trường Mặt hàng).")

    # Tổng phụ phí (text mô tả do API kết luận)
    st.write(f"**Tổng phụ thu:** {data.get('surcharge_text','—')}")

    # --- Tổng cộng
    st.subheader("🧮 Tổng cộng (ước tính)")
    grand_min = data.get("grand_total_min")
    grand_max = data.get("grand_total_max")

    if grand_min is None and grand_max is None:
        st.warning("Chưa đủ dữ liệu để tính tổng cộng (có thể đơn giá quốc tế hoặc phụ phí là 'Liên hệ').")
    else:
        if grand_max is None or abs(grand_min - grand_max) < 1e-6:
            st.success(f"**Tổng giá:** {vnd(grand_min)}")
        else:
            st.success(f"**Tổng giá:** {vnd(grand_min)} – {vnd(grand_max)}")
    st.divider()
    st.header("🚚 Tối ưu phương thức lấy hàng (T3)")
    
    with st.expander("Nhập thông tin lấy hàng", expanded=True):
        c1, c2, c3, c4 = st.columns([1,1,1,1])
        with c1:
            lat = st.number_input("LAT", value=10.77, step=0.0001, format="%.6f")
        with c2:
            lon = st.number_input("LON", value=106.70, step=0.0001, format="%.6f")
        with c3:
            preferred_branch = st.selectbox("CN ưu tiên", ["", "HCM", "CanTho"], index=0)
        with c4:
            svc_pick = st.selectbox("Loại hàng (T3)", ["normal","frozen"])
    
        if st.button("Tính phương án lấy hàng", type="primary"):
            try:
                payload_pick = {
                    "lat": lat, "lon": lon,
                    "weight": gw,                # dùng GW người dùng đã nhập
                    "service": svc_pick,
                    "preferred_branch": preferred_branch or None,
                }
                r = requests.post(f"{api_url}/pickup", json=payload_pick, timeout=30)
                if r.status_code != 200:
                    st.error(f"Lỗi API pickup: {r.text}")
                else:
                    pick = r.json()
                    if pick["options"]:
                        dfp = pd.DataFrame(pick["options"])
                        st.dataframe(dfp, use_container_width=True, hide_index=True)
                    best = pick.get("best")
                    if best:
                        st.success(f"✅ Gợi ý: **{best['provider']}** ({best['service_group']}) • "
                                   f"~{best['distance_km']:.1f} km • **{best['cost_vnd']:,} đ** — {best['note']}")
                    else:
                        st.warning("Không có phương án khả thi. Hãy cân nhắc tự mang ra nhà xe gần nhất.")
            except requests.exceptions.RequestException as e:
                st.error(f"Lỗi gọi API pickup: {e}")
    
    st.divider()
    st.header("✈️ Lịch bay gần nhất")
    
    with st.expander("Xem lịch bay theo tuyến", expanded=True):
        route_f = st.selectbox("Tuyến", ["USA","JAPAN","TW","KOREA"])
        if st.button("Tải lịch bay"):
            try:
                rr = requests.post(f"{api_url}/flight", json={"route": route_f}, timeout=30)
                if rr.status_code != 200:
                    st.error(f"Lỗi API flight: {rr.text}")
                else:
                    fl = rr.json()
                    rows = fl.get("rows", [])
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    best = fl.get("best")
                    if best:
                        st.success(
                            f"✅ **Chuyến gần nhất**: {best['day']} • "
                            f"Cắt hàng: **{best['cutoff_date']}** • Bay: **{best['flight_date']}**"
                            + (f" • Nhận sớm: {best['receive_earliest']}" if best.get("receive_earliest") else "")
                            + (f" • Nhận muộn: {best['receive_latest']}" if best.get("receive_latest") else "")
                        )
                    else:
                        st.warning("Chưa xác định được chuyến phù hợp.")
            except requests.exceptions.RequestException as e:
                st.error(f"Lỗi gọi API flight: {e}")

    # --- Tải dữ liệu JSON kết quả
    st.download_button(
        "⬇️ Tải JSON kết quả",
        data=json.dumps(data, ensure_ascii=False, indent=2),
        file_name="quote_result.json",
        mime="application/json",
    )

    # Gợi ý hành động tiếp theo
    st.caption("💡 Mẹo: Cập nhật bảng phụ phí (Google Sheets) để hiển thị chi tiết hơn cho từng mặt hàng.")
