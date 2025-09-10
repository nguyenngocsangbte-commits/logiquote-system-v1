# api.py
from __future__ import annotations
import re, string, unicodedata
from typing import List, Optional, Tuple, Dict, Literal   # <— thêm Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field                      # <— bỏ conlist

# ========== CẤU HÌNH NGUỒN DỮ LIỆU ==========
SHEET_ID_WEIGHT = "13RsE8YeENnq0DMkby_mpAIfl_5LhsiHRF0BxfYWAioI"

def _csv_url_from(sheet_id: str, gid: int) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

COUNTRIES: Dict[str, dict] = {
    "USA":   {"keywords": ["usa","us","mỹ","my","hoa kỳ","america","united states","u.s.a"], "weight": {"gid": 0},         "ignore_dw": False},
    "JAPAN": {"keywords": ["japan","jp","jpn","nhật","nhat","nhật bản","nhat ban","nihon","nb"], "weight": {"gid": 94619048}, "ignore_dw": False},
    "TW":    {"keywords": ["tw","taiwan","đài loan","dai loan","dai-loan","dailoan"],           "weight": {"gid": 296455518}, "ignore_dw": True},
    "KOREA": {"keywords": ["korea","kr","south korea","hàn quốc","han quoc","hq"],              "weight": {"gid": 276997218}, "ignore_dw": True},
}
WEIGHT_SHEET_MAP = {c: _csv_url_from(v.get("weight", {}).get("sheet_id", SHEET_ID_WEIGHT), int(v["weight"]["gid"])) for c,v in COUNTRIES.items()}
IGNORE_DW_COUNTRIES = {c for c, cfg in COUNTRIES.items() if cfg.get("ignore_dw")}

# Phụ thu (Publish CSV)
SURCHARGE_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRzTLDx0m0cQUs0IsZSIlK9P_u5pRiS4JJFnwNlUAcjI3tiiofXq7OhZVQJQiK492hAV0Bf6dh8kibZ/pub?gid=1895925525&single=true&output=csv"

# ==== Pickup (T3) & Flight configs ====
from dataclasses import dataclass
from datetime import datetime, timedelta, date

# Bảng giá/điểm gửi nội địa (publish CSV)
URL_PRICING               = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRzTLDx0m0cQUs0IsZSIlK9P_u5pRiS4JJFnwNlUAcjI3tiiofXq7OhZVQJQiK492hAV0Bf6dh8kibZ/pub?gid=320384519&single=true&output=csv"
URL_MAI_LINH_STATIONS     = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRzTLDx0m0cQUs0IsZSIlK9P_u5pRiS4JJFnwNlUAcjI3tiiofXq7OhZVQJQiK492hAV0Bf6dh8kibZ/pub?gid=1196694517&single=true&output=csv"
URL_PHUONG_TRANG_STATIONS = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRzTLDx0m0cQUs0IsZSIlK9P_u5pRiS4JJFnwNlUAcjI3tiiofXq7OhZVQJQiK492hAV0Bf6dh8kibZ/pub?gid=576608299&single=true&output=csv"

# Chi nhánh nội bộ
BRANCHES = {
    "HCM":    (10.806982, 106.629517),
    "CanTho": (10.008958, 105.778090),
}

REQUIRED_PRICING_KEYS = [
    "FREE_RADIUS_KM", "TIER2_RADIUS_KM", "TIER2_RATE_VND_PER_KG",
    "VIETTELPOST_RATE_VND_PER_KG", "MAX_DISTANCE_TO_DEPOT_KM",
    "MAI_LINH_RATE_VND_PER_KG", "PHUONG_TRANG_RATE_VND_PER_KG",
]

def _valid_latlon(lat, lon):
    try:
        lat = float(lat); lon = float(lon)
    except Exception:
        return False
    return -90 <= lat <= 90 and -180 <= lon <= 180

def load_pricing() -> Dict[str, float]:
    df = read_csv_url(URL_PRICING, ["key","value"])
    kv = {str(k).strip(): str(v).strip() for k,v in zip(df["key"], df["value"])}
    missing = [k for k in REQUIRED_PRICING_KEYS if k not in kv]
    if missing: raise ValueError(f"Thiếu khóa pricing: {missing}")
    def to_float(name): return float(kv[name])
    return {
        "FREE_RADIUS_KM":               to_float("FREE_RADIUS_KM"),
        "TIER2_RADIUS_KM":              to_float("TIER2_RADIUS_KM"),
        "TIER2_RATE_VND_PER_KG":        int(to_float("TIER2_RATE_VND_PER_KG")),
        "VIETTELPOST_RATE_VND_PER_KG":  int(to_float("VIETTELPOST_RATE_VND_PER_KG")),
        "MAX_DISTANCE_TO_DEPOT_KM":     to_float("MAX_DISTANCE_TO_DEPOT_KM"),
        "MAI_LINH_RATE_VND_PER_KG":     int(to_float("MAI_LINH_RATE_VND_PER_KG")),
        "PHUONG_TRANG_RATE_VND_PER_KG": int(to_float("PHUONG_TRANG_RATE_VND_PER_KG")),
    }

def load_stations_by_url(url: str) -> pd.DataFrame:
    df = read_csv_url(url, ["name","lat","lon"]).copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat","lon"])
    df = df[df.apply(lambda r: _valid_latlon(r["lat"], r["lon"]), axis=1)]
    if df.empty: raise ValueError("Danh sách trạm trống hoặc tọa độ không hợp lệ.")
    return df

# ——— Haversine & phương án lấy hàng ———
def haversine_km(lat1, lon1, lat2, lon2):
    R=6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2-lat1); dlmb = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))

def nearest_branch(lat, lon):
    best_name, best_dist = None, float("inf")
    for name, (b_lat,b_lon) in BRANCHES.items():
        d = haversine_km(lat, lon, b_lat, b_lon)
        if d < best_dist: best_name, best_dist = name, d
    return best_name, best_dist

@dataclass(frozen=True)
class QuoteOption:
    service_group: str   # "In-house" / "ViettelPost" / "Chanh"
    provider: str        # Tên NCC
    branch: str          # CN áp dụng (nếu có)
    distance_km: float
    cost_vnd: int
    feasible: bool
    note: str

def inhouse_quote(lat, lon, weight, pricing, preferred_branch=None) -> QuoteOption:
    if weight <= 0:
        return QuoteOption("In-house","NV Công ty","-", float("inf"), -1, False, "Khối lượng không hợp lệ")
    FREE = float(pricing["FREE_RADIUS_KM"])
    T2   = float(pricing["TIER2_RADIUS_KM"])
    RATE = int(pricing["TIER2_RATE_VND_PER_KG"])
    if preferred_branch and preferred_branch in BRANCHES:
        branch = preferred_branch; b_lat,b_lon = BRANCHES[branch]; dist = haversine_km(lat,lon,b_lat,b_lon)
    else:
        branch, dist = nearest_branch(lat, lon)
    if dist <= FREE:
        return QuoteOption("In-house","NV Công ty", branch, dist, 0, True, f"≤{int(FREE)} km: miễn phí")
    if dist <= T2:
        return QuoteOption("In-house","NV Công ty", branch, dist, int(RATE*weight), True, f">{int(FREE)}–{int(T2)} km: {RATE:,} đ/kg")
    return QuoteOption("In-house","NV Công ty", branch, dist, -1, False, f">{int(T2)} km: phải thuê ngoài")

def viettelpost_quote(weight, is_frozen, pricing) -> QuoteOption:
    if weight <= 0:
        return QuoteOption("ViettelPost","Viettel Post","-", 0.0, -1, False, "Khối lượng không hợp lệ")
    if is_frozen:
        return QuoteOption("ViettelPost","Viettel Post","-", 0.0, -1, False, "Không nhận hàng đông lạnh")
    rate = int(pricing["VIETTELPOST_RATE_VND_PER_KG"])
    return QuoteOption("ViettelPost","Viettel Post","-", 0.0, int(rate*weight), True, "Lấy hàng tận nhà toàn quốc")

def chanh_quote(label, stations, lat, lon, weight, rate, pricing) -> QuoteOption:
    if weight <= 0: return QuoteOption("Chanh", label, "-", float("inf"), -1, False, "Khối lượng không hợp lệ")
    if stations.empty: return QuoteOption("Chanh", label, "-", float("inf"), -1, False, "Chưa có điểm gửi")
    s = stations.copy()
    s["dist_km"] = s.apply(lambda r: haversine_km(lat, lon, float(r["lat"]), float(r["lon"])), axis=1)
    row = s.sort_values("dist_km").iloc[0]
    nearest_dist = float(row["dist_km"])
    max_km = float(pricing["MAX_DISTANCE_TO_DEPOT_KM"])
    if nearest_dist > max_km:
        return QuoteOption("Chanh", label, "-", nearest_dist, -1, False, f"Điểm gửi gần nhất {nearest_dist:.1f} km (> {max_km} km)")
    cost = int(rate*weight)
    note = f"Mang đến điểm '{row.get('name','?')}' (~{nearest_dist:.1f} km)"
    return QuoteOption("Chanh", label, "-", nearest_dist, cost, True, note)

def build_all_pickup_quotes(lat, lon, weight, is_frozen, preferred_branch, pricing, ml, pt):
    return [
        inhouse_quote(lat, lon, weight, pricing, preferred_branch),
        viettelpost_quote(weight, is_frozen, pricing),
        chanh_quote("Mai Linh", ml, lat, lon, weight, int(pricing["MAI_LINH_RATE_VND_PER_KG"]), pricing),
        chanh_quote("Phuong Trang", pt, lat, lon, weight, int(pricing["PHUONG_TRANG_RATE_VND_PER_KG"]), pricing),
    ]

def choose_best_quote(quotes: List[QuoteOption]) -> Optional[QuoteOption]:
    feasible = [q for q in quotes if q.feasible and q.cost_vnd >= 0]
    return min(feasible, key=lambda q: (q.cost_vnd, q.distance_km, q.service_group)) if feasible else None

# ========== TIỆN ÍCH CHUẨN HOÁ ==========
def _norm_vn(s: str) -> str:
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def _norm_key(s: str) -> str:
    base = _norm_vn(s)
    table = str.maketrans({ch: " " for ch in string.punctuation + "’‘“”…"})
    base = base.translate(table)
    base = re.sub(r"[^a-z0-9\s]", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    return base

def resolve_sheet_name(user_input: str) -> Optional[str]:
    t = _norm_vn(user_input)
    for code, cfg in COUNTRIES.items():
        if any(_norm_vn(k) in t for k in cfg.get("keywords", [])):
            return code
    up = user_input.strip().upper()
    return up if up in WEIGHT_SHEET_MAP else None

def read_csv_url(url: str, required_cols: Optional[List[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(url, engine="python", on_bad_lines="skip")
    df.columns = [str(c).strip() for c in df.columns]
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Thiếu cột: {missing}")
    if df.empty:
        raise ValueError("Sheet rỗng hoặc không truy cập được.")
    return df

# ========== CHUẨN HOÁ BẢNG GIÁ & TÍNH CƯỚC ==========
class NoFrozenPricingError(Exception): pass

def normalize_price_sheet_from_df(df: pd.DataFrame, service: str = "normal") -> pd.DataFrame:
    s = str(service).strip().lower()
    if s in {"normal","n"}:
        df = df.iloc[:, [0,1]]
    elif s in {"frozen","f"}:
        if df.shape[1] < 3: raise NoFrozenPricingError()
        if df.iloc[:,2].dropna().empty: raise NoFrozenPricingError()
        df = df.iloc[:, [0,2]]
    else:
        raise ValueError("Service không hợp lệ (normal/frozen)")
    df = df.rename(columns={df.columns[0]:"KG", df.columns[1]:"PRICE"})

    def parse_range(val):
        if pd.isna(val): return (np.nan, np.nan)
        txt = re.sub(r'[^\d\.\-\+\><]+', ' ', str(val).strip().lower())
        nums = re.findall(r'(\d+(?:\.\d+)?)', txt)
        if len(nums) >= 2: return (float(nums[0]), float(nums[1]))
        if len(nums) == 1:
            v = float(nums[0]); return (v, np.inf) if ('+' in txt or '>' in txt) else (v, v)
        return (np.nan, np.nan)

    rngs = df["KG"].apply(parse_range)
    df[["min_kg","max_kg"]] = pd.DataFrame(rngs.tolist(), index=df.index)
    df = df[pd.to_numeric(df["min_kg"], errors="coerce").notna()].copy()
    df["PRICE"] = pd.to_numeric(df["PRICE"].astype(str).str.replace(r"[^\d\.-]","", regex=True), errors="coerce")
    if df["PRICE"].dropna().empty: raise ValueError("PRICE không hợp lệ.")
    return df.sort_values("min_kg").reset_index(drop=True)

def find_price_for_cw(df_norm: pd.DataFrame, cw: float):
    cond = (df_norm['min_kg'] <= cw) & (df_norm['max_kg'] >= cw)
    matched = df_norm[cond]
    if not matched.empty:
        price = matched.iloc[0]['PRICE']; return price, price * cw
    inf_match = df_norm[df_norm['max_kg'] == np.inf]
    if not inf_match.empty:
        price = inf_match.iloc[0]['PRICE']; return price, price * cw
    finite_max = df_norm['max_kg'].replace(np.inf, np.nan)
    if cw > finite_max.max():
        last = df_norm.iloc[-1]; price = last['PRICE']; return price, price * cw
    return None, None

def read_weight_sheet(sheet_name: str) -> pd.DataFrame:
    if sheet_name not in WEIGHT_SHEET_MAP:
        raise ValueError(f"Tuyến không hợp lệ. Chọn: {list(WEIGHT_SHEET_MAP.keys())}")
    return pd.read_csv(WEIGHT_SHEET_MAP[sheet_name], engine="python", on_bad_lines="skip")

# ========== PHỤ THU ==========
_VN_DASHES = r"[\-–—]"

def _parse_money_token(tok: str) -> Optional[int]:
    t = re.sub(r"[^\d,\.]", "", str(tok).strip()).replace(",", "").replace(".", "")
    return int(t) if t.isdigit() else None

def _split_amount_and_unit(s: str) -> Tuple[str, str]:
    s = str(s)
    if "/" in s:
        left, unit = s.rsplit("/", 1); return left.strip(), unit.strip().lower()
    return s.strip(), ""

def _parse_range_amount(s: str) -> Tuple[Optional[int], Optional[int]]:
    core = re.sub(r"\(.*?\)", "", s).strip()
    parts = [p.strip() for p in re.split(_VN_DASHES, core) if p.strip()]
    if len(parts) == 1:
        v = _parse_money_token(parts[0]); return v, v
    if len(parts) >= 2:
        return _parse_money_token(parts[0]), _parse_money_token(parts[1])
    return None, None

def normalize_surcharge_sheet(df: pd.DataFrame) -> pd.DataFrame:
    req = ["item", "surcharge"]
    for c in req:
        if c not in df.columns: raise ValueError("Thiếu cột 'item'/'surcharge'")
    out = df.copy()
    out["item_norm"] = out["item"].astype(str).str.strip().str.lower()
    out["item_key"]  = out["item"].astype(str).apply(_norm_key)
    mins, maxs, units, cleans = [], [], [], []
    for raw in out["surcharge"].astype(str):
        left, unit = _split_amount_and_unit(raw)
        a_min, a_max = _parse_range_amount(left)
        mins.append(a_min); maxs.append(a_max); units.append(unit); cleans.append(left)
    out["min_vnd"]=mins; out["max_vnd"]=maxs; out["unit"]=units; out["surcharge_clean"]=cleans
    return out

def load_surcharges() -> pd.DataFrame:
    return normalize_surcharge_sheet(read_csv_url(SURCHARGE_URL, ["item","surcharge"]))

def lookup_surcharge(items: List[str], surcharge_df: pd.DataFrame) -> List[dict]:
    res = []
    if surcharge_df.empty:
        return res
    for it in items:
        key = _norm_key(it)
        hit = surcharge_df[surcharge_df["item_key"] == key]
        row = hit.iloc[0] if not hit.empty else None
        if row is None:
            res.append({"item": it, "item_display": it, "display_text": "Liên hệ",
                        "min_vnd": None, "max_vnd": None, "unit": ""})
            continue
        display_name = str(row["item"]).strip() or it
        if pd.notna(row["min_vnd"]):
            if pd.notna(row["max_vnd"]) and row["max_vnd"] != row["min_vnd"]:
                range_txt = f"{int(row['min_vnd']):,} – {int(row['max_vnd']):,}"
            else:
                range_txt = f"{int(row['min_vnd']):,}"
            unit_txt = f"/{row['unit']}" if row["unit"] else ""
            disp = range_txt + unit_txt
        else:
            disp = "Liên hệ"
        res.append({"item": it, "item_display": display_name, "display_text": disp,
                    "min_vnd": row["min_vnd"], "max_vnd": row["max_vnd"], "unit": row["unit"]})
    return res

# ========== FASTAPI ==========
app = FastAPI(title="LogiQuote API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuoteRequest(BaseModel):
    route: str = Field(..., description="Tuyến quốc tế (ví dụ: USA/TW/JAPAN/KOREA)")
    # Dùng Literal để ràng buộc hợp lệ (v2-friendly)
    service: Literal["normal","frozen"] = "normal"
    gw: float = Field(..., gt=0, description="Cân nặng thực tế (kg)")
    length_cm: float = Field(..., gt=0)
    width_cm: float  = Field(..., gt=0)
    height_cm: float = Field(..., gt=0)
    # Pydantic v2: dùng List[str] + default_factory
    items: List[str] = Field(default_factory=list, description="Danh sách mặt hàng")

class SurchargeItem(BaseModel):
    item: str
    item_display: str
    display_text: str
    min_vnd: Optional[int] = None
    max_vnd: Optional[int] = None
    unit: str = ""

class QuoteResponse(BaseModel):
    route_input: str
    route_resolved: str
    service: str
    gw: float
    dw: float
    cw: float
    ignore_dw: bool
    unit_price_vnd_per_kg: Optional[float] = None
    intl_total_vnd: Optional[float] = None
    surcharge_total_min: Optional[int] = 0
    surcharge_total_max: Optional[int] = 0
    surcharge_text: str
    surcharge_items: List[SurchargeItem]
    grand_total_min: Optional[float] = None
    grand_total_max: Optional[float] = None

# === Pydantic models (đặt cạnh các model hiện có) ===
class PickupRequest(BaseModel):
    lat: float = Field(..., description="Vĩ độ")
    lon: float = Field(..., description="Kinh độ")
    weight: float = Field(..., gt=0)
    service: str = Field("normal", pattern="^(normal|frozen)$")
    preferred_branch: Optional[str] = Field(None, description="HCM/CanTho (tuỳ chọn)")

class PickupOptionModel(BaseModel):
    service_group: str
    provider: str
    branch: str
    distance_km: float
    cost_vnd: int
    feasible: bool
    note: str

class PickupResponse(BaseModel):
    options: List[PickupOptionModel]
    best: Optional[PickupOptionModel] = None

# === Endpoint tối ưu lấy hàng ===
@app.post("/pickup", response_model=PickupResponse)
def pickup(req: PickupRequest):
    if not _valid_latlon(req.lat, req.lon):
        raise HTTPException(status_code=400, detail="Toạ độ không hợp lệ.")
    pricing = load_pricing()
    ml = load_stations_by_url(URL_MAI_LINH_STATIONS)
    pt = load_stations_by_url(URL_PHUONG_TRANG_STATIONS)
    quotes = build_all_pickup_quotes(
        req.lat, req.lon, req.weight,
        req.service == "frozen",
        req.preferred_branch, pricing, ml, pt
    )
    best = choose_best_quote(quotes)
    to_model = lambda q: PickupOptionModel(
        service_group=q.service_group, provider=q.provider, branch=q.branch or "-",
        distance_km=float(q.distance_km), cost_vnd=int(q.cost_vnd),
        feasible=bool(q.feasible), note=q.note
    )
    return PickupResponse(
        options=[to_model(q) for q in sorted(quotes, key=lambda x: (0 if (x.feasible and x.cost_vnd>=0) else 1, x.cost_vnd if x.cost_vnd>=0 else 1e18, x.distance_km))],
        best=(to_model(best) if best else None)
    )
# ==== Flight (lịch bay) ====
CSV_URL_FLIGHT = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRzTLDx0m0cQUs0IsZSIlK9P_u5pRiS4JJFnwNlUAcjI3tiiofXq7OhZVQJQiK492hAV0Bf6dh8kibZ/pub?gid=1603620150&single=true&output=csv"
COL_AREA="Khu vực"; COL_DAY="Lịch Bay"; COL_CUT="Thời Gian Cắt Hàng"
THU_VI=['Thứ 2','Thứ 3','Thứ 4','Thứ 5','Thứ 6','Thứ 7','CN']
WEEKDAY_MAP={"thứ 2":0,"thứ 3":1,"thứ 4":2,"thứ 5":3,"thứ 6":4,"thứ 7":5,"cn":6}

def _area_match(text: str, sheet_name: str) -> bool:
    t = _norm_vn(text)
    return any(_norm_vn(k) in t for k in COUNTRIES.get(sheet_name, {}).get("keywords", []))

def parse_delivery_days_range(text: str):
    if not isinstance(text, str): return (0,0)
    t = text.strip().lower().replace('ngày','').replace('–','-')
    m = re.match(r'^\s*(\d+)\s*-\s*(\d+)\s*$', t)
    if m: a,b=int(m.group(1)),int(m.group(2)); return (min(a,b),max(a,b))
    m2 = re.match(r'^\s*(\d+)\s*$', t); return (int(m2.group(1)), int(m2.group(1))) if m2 else (0,0)

def add_business_days(start_dt: datetime, days: int) -> datetime:
    d = start_dt; added = 0
    while added < days:
        d += timedelta(days=1)
        if d.weekday() < 5: added += 1
    return d

def parse_cutoff(text: str, fallback_wd: int):
    hour = 12; wd=None
    if isinstance(text,str):
        low = text.lower().strip()
        mhr = re.search(r'(\d{1,2})\s*h', low)
        if mhr: hour = max(0, min(23, int(mhr.group(1))))
        if re.search(r'\bcn\b|chủ nhật', low): wd = 6
        mthu = re.search(r'thứ\s*([2-7])', low)
        if mthu: wd = int(mthu.group(1)) - 2
        mt = re.search(r'\bt\s*([2-7])\b', low)
        if mt: wd = int(mt.group(1)) - 2
    if wd is None: wd = fallback_wd
    return hour, wd

def next_cutoff_and_flight(now: datetime, flight_wd: int, cutoff_hour: int, cutoff_wd: int):
    days_to_cut = (cutoff_wd - now.weekday()) % 7
    cutoff_dt = (now + timedelta(days=days_to_cut)).replace(hour=cutoff_hour, minute=0, second=0, microsecond=0)
    if cutoff_dt < now: cutoff_dt += timedelta(days=7)
    diff = (flight_wd - cutoff_wd) % 7
    flight_dt = cutoff_dt + timedelta(days=diff)
    return cutoff_dt, flight_dt

class FlightRow(BaseModel):
    area: str
    day: str
    cutoff: str
    deliver_text: str

class FlightBest(BaseModel):
    area: str
    day: str
    cutoff_date: str
    flight_date: str
    receive_earliest: Optional[str] = None
    receive_latest: Optional[str] = None

class FlightRequest(BaseModel):
    route: str

class FlightResponse(BaseModel):
    rows: List[FlightRow]
    best: Optional[FlightBest] = None

@app.post("/flight", response_model=FlightResponse)
def flight(req: FlightRequest):
    sheet_name = resolve_sheet_name(req.route) or req.route.strip().upper()
    df = read_csv_url(CSV_URL_FLIGHT)
    df.columns = df.columns.str.strip()
    col_del = next((c for c in ["Thời Gian Phát Hàng (không tính T7, CN)", "Thời Gian Phát Hàng"] if c in df.columns), None)
    data = df[df[COL_AREA].astype(str).apply(lambda x: _area_match(x, sheet_name))].copy()
    rows = []
    if not data.empty and col_del:
        for _, r in data.iterrows():
            rows.append(FlightRow(area=str(r[COL_AREA]), day=str(r[COL_DAY]), cutoff=str(r[COL_CUT]), deliver_text=str(r[col_del])))
    # tìm chuyến gần nhất
    best = None; best_flight = None
    now = datetime.now()
    if not data.empty and col_del:
        for _, row in data.iterrows():
            flight_wd = WEEKDAY_MAP.get(str(row[COL_DAY]).strip().lower()); 
            if flight_wd is None: continue
            cutoff_hour, cutoff_wd = parse_cutoff(row[COL_CUT], flight_wd)
            cutoff_dt, flight_dt = next_cutoff_and_flight(now, flight_wd, cutoff_hour, cutoff_wd)
            dmin, dmax = parse_delivery_days_range(row[col_del])
            recv_early = add_business_days(flight_dt, dmin) if dmin else None
            recv_late  = add_business_days(flight_dt, dmax) if dmax else None
            if (best is None) or (flight_dt < best_flight):
                best = FlightBest(
                    area=str(row[COL_AREA]),
                    day=str(row[COL_DAY]),
                    cutoff_date=cutoff_dt.date().isoformat(),
                    flight_date=flight_dt.date().isoformat(),
                    receive_earliest=(recv_early.date().isoformat() if recv_early else None),
                    receive_latest=(recv_late.date().isoformat() if recv_late else None),
                )
                best_flight = flight_dt
    return FlightResponse(rows=rows, best=best)


@app.get("/")
def root():
    return {"message": "Chào mừng đến hệ thống báo giá logistics!"}

@app.post("/quote", response_model=QuoteResponse)
def quote(req: QuoteRequest):
    # 1) Resolve tuyến & tải bảng giá
    sheet_name = resolve_sheet_name(req.route) or req.route.strip().upper()
    if sheet_name not in WEIGHT_SHEET_MAP:
        raise HTTPException(status_code=400, detail=f"Tuyến không hợp lệ. Chọn: {list(WEIGHT_SHEET_MAP.keys())}")

    try:
        sheet_df = read_weight_sheet(sheet_name)
        df_norm = normalize_price_sheet_from_df(sheet_df, req.service)
    except NoFrozenPricingError:
        raise HTTPException(status_code=400, detail=f"Tuyến {sheet_name} không có giá 'frozen'.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi bảng giá: {e}")

    # 2) DW/CW
    dw = (req.length_cm * req.width_cm * req.height_cm) / 5000.0
    ignore_dw = sheet_name in IGNORE_DW_COUNTRIES
    cw_base = req.gw if ignore_dw else max(req.gw, dw)
    cw = np.ceil(cw_base * 2) / 2  # làm tròn 0.5 kg

    unit_price, intl_total = find_price_for_cw(df_norm, float(cw))

    # 3) Phụ thu
    surcharge_items: List[SurchargeItem] = []
    s_min = s_max = 0
    has_contact = False
    surcharge_text = "0 đ"

    try:
        if req.items:
            s_df = load_surcharges()
            found = lookup_surcharge(req.items, s_df)
            for r in found:
                surcharge_items.append(SurchargeItem(**r))
                if r["min_vnd"] is None:
                    has_contact = True
                else:
                    s_min += int(r["min_vnd"])
                    s_max += int(r["max_vnd"] if r["max_vnd"] is not None else r["min_vnd"])
            if s_min == 0 and has_contact:
                surcharge_text = "Liên hệ"
            elif s_min == s_max:
                surcharge_text = f"{s_min:,} đ" + (" (+ mục cần liên hệ)" if has_contact else "")
            else:
                surcharge_text = f"{s_min:,} – {s_max:,} đ" + (" (+ mục cần liên hệ)" if has_contact else "")
    except Exception as e:
        surcharge_text = f"Liên hệ (lỗi đọc phụ thu: {e})"
        has_contact = True

    # 4) Tổng cộng
    grand_min = grand_max = None
    if intl_total is not None:
        base = float(intl_total)
        if s_min == 0 and s_max == 0 and "Liên hệ" in surcharge_text:
            grand_min = base
            grand_max = None
        elif s_min == s_max:
            grand_min = base + s_min
            grand_max = base + s_max
        else:
            grand_min = base + s_min
            grand_max = base + s_max

    return QuoteResponse(
        route_input=req.route,
        route_resolved=sheet_name,
        service=req.service,
        gw=req.gw,
        dw=round(dw, 2),
        cw=float(cw),
        ignore_dw=ignore_dw,
        unit_price_vnd_per_kg=None if unit_price is None else float(unit_price),
        intl_total_vnd=None if intl_total is None else float(intl_total),
        surcharge_total_min=None if s_min == 0 and "Liên hệ" in surcharge_text else int(s_min),
        surcharge_total_max=None if s_max == 0 and "Liên hệ" in surcharge_text else int(s_max),
        surcharge_text=surcharge_text,
        surcharge_items=surcharge_items,
        grand_total_min=None if grand_min is None else float(grand_min),
        grand_total_max=None if grand_max is None else float(grand_max),
    )
