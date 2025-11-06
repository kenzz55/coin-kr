import argparse
import os
import cv2
import sys
import numpy as np
import math

# ======= 조정 지점(필요할 때만 바꿔도 됨) =======
BLUR_K = 11        # 가우시안 커널 (홀수)
SIGMA = 0.33       # auto-canny 민감도 (0.25~0.5 사이 튜닝)
CLOSE_K = 3        # 형태학 닫힘 커널 크기(홀수 권장)
MIN_COIN_AREA = 2000
# ============================================

# 허프 고정 파라미터(필요하면 여기서만 바꿔)
HOUGH_DP = 1.2
HOUGH_PARAM1 = 120.0
HOUGH_PARAM2 = 22.0
TARGET_LONG_SIDE = 1200  # 자동 스케일 시 긴 변 기준 픽셀(과제용은 1000~1400 추천)

_CLASS_SIGMA = {"10": 0.09, "50": 0.07, "100": 0.055, "500": 0.050}

def parse_args():
    p = argparse.ArgumentParser("CV assignment runner")
    p.add_argument("--input", required=True, type=str, help="path to input image")
    p.add_argument("--scale", type=float, default=0.0, help="전체 처리 스케일(예: 0.6). 0이면 원본 크기")
    return p.parse_args()

# ----------------- 유틸 -----------------
def resize_to_fit(img, scale=None):
    if not scale or scale == 1.0:
        return img, 1.0
    h, w = img.shape[:2]
    nh, nw = int(h * scale), int(w * scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA), scale

def auto_canny_thresholds(gray, sigma=0.33):
    v = np.median(gray)
    lo = int(max(0, (1.0 - sigma) * v))
    hi = int(min(255, (1.0 + sigma) * v))
    return lo, hi

def center_in_mask(mask, cx, cy, pad=0):
    H, W = mask.shape[:2]
    if not (0 <= cx < W and 0 <= cy < H): return False
    if pad <= 0: return mask[cy, cx] > 0
    x1, y1 = max(0, cx - pad), max(0, cy - pad)
    x2, y2 = min(W, cx + pad + 1), min(H, cy + pad + 1)
    return (mask[y1:y2, x1:x2] > 0).any()

def radius_bounds_from_bbox(w, h, r_est, tight=(0.85, 1.15)):
    r_box = 0.5 * min(w, h)
    r_min = max(6, min(tight[0] * r_est, r_box * 1.05))
    r_max = max(7, min(tight[1] * r_est, r_box * 1.25))
    return int(r_min), int(r_max)

# ------------- Step 1: 전처리 -------------
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    k = BLUR_K if BLUR_K % 2 == 1 else BLUR_K + 1
    gray_blur = cv2.GaussianBlur(gray_eq, (k, k), 0)
    lo, hi = auto_canny_thresholds(gray_blur, SIGMA)
    edges = cv2.Canny(gray_blur, lo, hi)
    kk = CLOSE_K if CLOSE_K % 2 == 1 else CLOSE_K + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, ker, iterations=1)
    # (출력 제거) 디버그 프린트 없음
    return gray_blur, edges_closed

# ------------- Step 2: 마스크 -------------
def build_clean_mask(gray_blur):
    bin_mask = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 35, 5
    )
    ker5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, ker5, iterations=2)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, 8)
    clean = np.zeros_like(bin_mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_COIN_AREA:
            clean[labels == i] = 255
    return clean

def build_mask_adaptive(gray_blur):
    bin_mask = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 35, 5
    )
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, k, iterations=2)
    return bin_mask

def build_mask_otsu(gray_blur, guide=None, guide_dilate=17):
    _, bin_mask = cv2.threshold(
        gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, k, iterations=2)
    if guide is not None:
        gate = cv2.dilate(guide, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(guide_dilate,guide_dilate)), 1)
        bin_mask = cv2.bitwise_and(bin_mask, gate)
    return bin_mask

def detect_with_mask(img, gray_blur, mask):
    clean = mask.copy()
    if (clean > 0).sum() > 0:
        clean = split_touching_by_watershed(clean)
    touching, ok = classify_components(clean)
    contour_circles = circles_from_ok_components(clean, ok)
    hough_circles = hough_on_touching(img, clean, gray_blur, touching)
    return dedup_merge(contour_circles + hough_circles, center_dist_ratio=0.5)

def split_touching_by_watershed(clean_mask):
    dist = cv2.distanceTransform(clean_mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.48 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(clean_mask, kernel, iterations=1)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    color = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color, markers)
    sep = np.zeros_like(clean_mask)
    sep[markers > 1] = 255
    return sep

# ------------- Step 3: 붙음/안붙음 분류 -------------
def classify_components(clean_mask):
    m = ((clean_mask > 0).astype(np.uint8)) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    touching, ok = [], []
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < MIN_COIN_AREA:
            continue
        roi = (labels == i).astype(np.uint8) * 255
        roi = roi[y:y + h, x:x + w]
        roi_smooth = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, k3, iterations=1)
        cnts, _ = cv2.findContours(roi_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = cnts[0]
        per = max(1e-6, cv2.arcLength(c, True))
        c_area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        hull_area = max(1.0, cv2.contourArea(hull))
        circularity = 4.0 * np.pi * c_area / (per * per)
        solidity    = c_area / hull_area
        er = cv2.erode(roi_smooth, k3, 1)
        dist = cv2.distanceTransform(er, cv2.DIST_L2, 5)
        dist = cv2.GaussianBlur(dist, (0, 0), 1.0)
        _, peak_bin = cv2.threshold(dist, 0.55 * (dist.max() or 1), 255, cv2.THRESH_BINARY)
        peak_bin = peak_bin.astype(np.uint8)
        peak_bin = cv2.morphologyEx(peak_bin, cv2.MORPH_OPEN, k3, iterations=1)
        n_peaks, _ = cv2.connectedComponents(peak_bin)
        peaks = n_peaks - 1
        very_round = (circularity >= 0.72 and solidity >= 0.90)
        is_touching = (peaks >= 2) and (not very_round)
        (touching if is_touching else ok).append((x, y, w, h, area))
    # (출력 제거)
    return touching, ok

# ------------- Step 4A: 컨투어 -------------
def circles_from_ok_components(clean_mask, bbox_list):
    circles = []
    for (x, y, w, h, _) in bbox_list:
        roi = (clean_mask[y:y + h, x:x + w] > 0).astype(np.uint8)
        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        (cx, cy), r = cv2.minEnclosingCircle(c)
        circles.append((int(x + cx), int(y + cy), int(r)))
    # (출력 제거)
    return circles

# ------------- Step 4B: 허프 -------------
def ring_support_score(edges_ring, cx, cy, r, step_deg=4, band=4):
    H, W = edges_ring.shape[:2]
    hits = total = 0
    for a in range(0, 360, step_deg):
        t = math.radians(a)
        x = int(round(cx + r * math.cos(t)))
        y = int(round(cy + r * math.sin(t)))
        if 0 <= x < W and 0 <= y < H:
            total += 1
            if edges_ring[max(0, y - band):min(H, y + band + 1),
                          max(0, x - band):min(W, x + band + 1)].any():
                hits += 1
    return hits / max(1, total)

def mask_support_score(comp_mask, cx, cy, r, step_deg=4):
    H, W = comp_mask.shape[:2]
    hit = tot = 0
    for a in range(0, 360, step_deg):
        t = math.radians(a)
        x = int(round(cx + r * math.cos(t)))
        y = int(round(cy + r * math.sin(t)))
        if 0 <= x < W and 0 <= y < H:
            tot += 1
            if comp_mask[y, x] > 0:
                hit += 1
    return hit / max(1, tot)

def _guess_radius_from_ring(ring_roi):
    H, W = ring_roi.shape[:2]
    A = H * W
    cnts, _ = cv2.findContours(ring_roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    radii = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 0.00005 * A or a > 0.01 * A:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if r < 6 or r > min(H, W) * 0.6:
            continue
        radii.append(r)
    if len(radii) >= 4:
        return float(np.median(np.array(radii, dtype=np.float32)))
    return None

def count_peaks(mask_bin):
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    er = cv2.erode(mask_bin, k3, 1)
    dist = cv2.distanceTransform(er, cv2.DIST_L2, 5)
    dist = cv2.GaussianBlur(dist, (0, 0), 1.0)
    thr = 0.55 * (dist.max() if dist.max() > 0 else 1.0)
    _, peak = cv2.threshold(dist, thr, 255, cv2.THRESH_BINARY)
    peak = peak.astype(np.uint8)
    peak = cv2.morphologyEx(peak, cv2.MORPH_OPEN, k3, iterations=1)
    n, _ = cv2.connectedComponents(peak)
    return max(0, n - 1)

def estimate_coins_in_bbox(w, h, r_est, peaks_hint=0):
    k = 0.88
    dx = max(1.0, 2.0 * r_est * k)
    dy = max(1.0, 2.0 * r_est * k)
    nx = max(1, int(round(w / dx)))
    ny = max(1, int(round(h / dy)))
    est = nx * ny
    if peaks_hint > 0:
        est = int(round(0.5 * est + 0.5 * peaks_hint))
    return max(1, est)

def refine_circle(gray_blur, ring_full, mask_full, cx, cy, r,
                  r_sweep=6, step_deg=3, band=3, center_win=2, iters=2):
    H, W = gray_blur.shape[:2]
    def score(x, y, rr):
        if rr < 6: return -1e9
        if not (0 <= x < W and 0 <= y < H): return -1e9
        s_ring = ring_support_score(ring_full, x, y, rr, step_deg=step_deg, band=band)
        s_mask = mask_support_score(mask_full,  x, y, rr, step_deg=step_deg)
        return 0.6 * s_ring + 0.4 * s_mask

    best_r, best_s = r, score(cx, cy, r)
    r_lo, r_hi = max(6, r - r_sweep), r + r_sweep
    for rr in range(int(r_lo), int(r_hi) + 1):
        s = score(cx, cy, rr)
        if s > best_s:
            best_r, best_s = rr, s

    best_x, best_y = int(cx), int(cy)
    for _ in range(iters):
        improved = False
        bx, by = best_x, best_y
        for dy in range(-center_win, center_win + 1):
            for dx in range(-center_win, center_win + 1):
                x2, y2 = bx + dx, by + dy
                s = score(x2, y2, best_r)
                if s > best_s:
                    best_s, best_x, best_y = s, x2, y2
                    improved = True
        if not improved:
            break

    return best_x, best_y, int(best_r)

def hough_on_touching(img_bgr, clean_mask, gray_blur, touching_boxes, scale=1.0):
    ring_full = cv2.morphologyEx(
        clean_mask, cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    found = []
    MIN_RADIUS_FLOOR = 6
    for (x, y, w, h, area) in touching_boxes:
        roi_g = gray_blur[y:y + h, x:x + w].copy()
        roi_m = (clean_mask[y:y + h, x:x + w] > 0).astype(np.uint8)

        L = float(max(w, h))
        minR = MIN_RADIUS_FLOOR
        maxR = int(max(minR + 1, 0.55 * L))
        minDist_base = int(max(8, 0.20 * L))
        roi_blur = cv2.GaussianBlur(roi_g, (15, 15), 1.6)

        dr = max(8, int(0.10 * L))
        bins = []
        rL = minR
        while rL < maxR:
            rH = min(maxR, rL + dr)
            bins.append((rL, rH))
            rL = rH

        cands = []
        for (rL, rH) in bins:
            r_mid = 0.5 * (rL + rH)
            minDist = int(max(minDist_base, 0.45 * r_mid))
            for p2 in (HOUGH_PARAM2 + 10, HOUGH_PARAM2 + 6, HOUGH_PARAM2 + 2):
                cs = cv2.HoughCircles(
                    roi_blur, cv2.HOUGH_GRADIENT,
                    dp=HOUGH_DP, minDist=minDist,
                    param1=HOUGH_PARAM1, param2=int(p2),
                    minRadius=int(rL), maxRadius=int(rH)
                )
                if cs is None:
                    continue
                for (cx, cy, r) in cs[0]:
                    CX, CY, R = int(round(cx)) + x, int(round(cy)) + y, int(round(r))
                    if not center_in_mask(roi_m, int(cx), int(cy), pad=1):
                        continue
                    s_ring = ring_support_score(ring_full, CX, CY, R, step_deg=3, band=2)
                    s_mask = mask_support_score(clean_mask, CX, CY, R, step_deg=3)
                    if 360.0 * s_ring < 135.0:
                        continue
                    score = 0.6 * s_ring + 0.4 * s_mask
                    cands.append((CX, CY, R, score))

        if not cands:
            for (rL, rH) in bins:
                r_mid = 0.5 * (rL + rH)
                minDist = int(max(minDist_base, 0.45 * r_mid))
                for p2 in (HOUGH_PARAM2 - 2, HOUGH_PARAM2):
                    cs = cv2.HoughCircles(
                        roi_blur, cv2.HOUGH_GRADIENT,
                        dp=HOUGH_DP, minDist=minDist,
                        param1=HOUGH_PARAM1, param2=int(p2),
                        minRadius=int(rL), maxRadius=int(rH)
                    )
                    if cs is None:
                        continue
                    for (cx, cy, r) in cs[0]:
                        CX, CY, R = int(round(cx)) + x, int(round(cy)) + y, int(round(r))
                        if not center_in_mask(roi_m, int(cx), int(cy), pad=1):
                            continue
                        s_ring = ring_support_score(ring_full, CX, CY, R, step_deg=3, band=2)
                        s_mask = mask_support_score(clean_mask, CX, CY, R, step_deg=3)
                        if 360.0 * s_ring < 130.0:
                            continue
                        score = 0.58 * s_ring + 0.42 * s_mask
                        cands.append((CX, CY, R, score))

        cands.sort(key=lambda t: (-t[3], -t[2]))
        keep = []
        for cx, cy, r, s in cands:
            drop = False
            for kx, ky, kr, ks in keep:
                if ((cx - kx)**2 + (cy - ky)**2) ** 0.5 < max(r, kr) * 0.50:
                    drop = True
                    break
            if not drop:
                keep.append((cx, cy, r, s))

        # (출력 제거) ROI 로그 제거
        for cx, cy, r, _ in keep:
            rx, ry, rr = refine_circle(gray_blur, ring_full, clean_mask, cx, cy, r)
            found.append((rx, ry, rr))

    found = dedup_merge(found, center_dist_ratio=0.5)
    # (출력 제거)
    return found

def score_circle(gray_blur, mask_full, cx, cy, r):
    ring_full = cv2.morphologyEx(
        mask_full, cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    )
    s_ring = ring_support_score(ring_full, cx, cy, r, step_deg=3, band=3)
    s_mask = mask_support_score(mask_full,  cx, cy, r, step_deg=3)
    return 0.6*s_ring + 0.4*s_mask

def merge_detections(gray_blur, maskA, maskB, circlesA, circlesB):
    mask_union = cv2.bitwise_or(maskA, maskB)
    cand = []
    for (cx,cy,r) in circlesA: cand.append((cx,cy,r, score_circle(gray_blur, mask_union, cx,cy,r)))
    for (cx,cy,r) in circlesB: cand.append((cx,cy,r, score_circle(gray_blur, mask_union, cx,cy,r)))
    cand.sort(key=lambda t: t[3], reverse=True)
    kept = []
    for cx,cy,r,s in cand:
        drop = False
        for kx,ky,kr,ks in kept:
            if ((cx-kx)**2 + (cy-ky)**2)**0.5 < max(r,kr)*0.45:
                drop = True; break
        if not drop:
            kept.append((cx,cy,r,s))
    return [(cx,cy,r) for (cx,cy,r,_) in kept]

# ------------- Step 5: 병합/표시 -------------
def dedup_merge(circles, center_dist_ratio=0.5):
    out = []
    for (cx, cy, r) in sorted(circles, key=lambda t: -t[2]):
        drop = False
        for (kx, ky, kr) in out:
            if ((cx - kx) ** 2 + (cy - ky) ** 2) ** 0.5 < max(r, kr) * center_dist_ratio:
                drop = True
                break
        if not drop:
            out.append((cx, cy, r))
    return out

def drop_center_inside(circles, ratio=0.85):
    circles = sorted(circles, key=lambda t: t[2], reverse=True)
    kept = []
    for (cx, cy, r) in circles:
        contained = False
        for (kx, ky, kr) in kept:
            dx, dy = cx - kx, cy - ky
            if dx*dx + dy*dy < (kr * kr) and r <= ratio * kr:
                contained = True
                break
        if not contained:
            kept.append((cx, cy, r))
    return kept

# ====================== 1단계: 10원 전용 색(구리) 스코어 ======================
C10_dA_dB_MIN = 5.0
C10_S_MIN     = 14.0
C10_H_RANGE   = (8, 26)
C10_SCORE_THR = 0.52

def _mask_annulus(shape, cx, cy, r, inner, outer):
    H, W = shape[:2]
    Y, X = np.ogrid[:H, :W]
    d2 = (X - cx)**2 + (Y - cy)**2
    return (((inner*r)**2 <= d2) & (d2 <= (outer*r)**2)).astype(np.uint8) * 255

def _mask_disk(shape, cx, cy, r, ratio):
    H, W = shape[:2]
    Y, X = np.ogrid[:H, :W]
    d2 = (X - cx)**2 + (Y - cy)**2
    return (d2 <= (ratio*r)**2).astype(np.uint8) * 255

def _robust_percentile(arr, mask, p):
    m = mask > 0
    vals = arr[m]
    if vals.size == 0: return 0.0
    return float(np.percentile(vals, p))

def _robust_median(arr, mask):
    m = mask > 0
    vals = arr[m]
    if vals.size == 0: return 0.0
    return float(np.median(vals))

def _bg_norm_feats(img_bgr, cx, cy, r):
    def _mask_disk_local(shape, cx, cy, r, ratio):
        H, W = shape[:2]; Y, X = np.ogrid[:H, :W]
        d2 = (X - cx)**2 + (Y - cy)**2
        return (d2 <= (ratio*r)**2).astype(np.uint8) * 255
    def _mask_annulus_local(shape, cx, cy, r, inner, outer):
        H, W = shape[:2]; Y, X = np.ogrid[:H, :W]
        d2 = (X - cx)**2 + (Y - cy)**2
        return (((inner*r)**2 <= d2) & (d2 <= (outer*r)**2)).astype(np.uint8) * 255

    disk = _mask_disk_local(img_bgr.shape, cx, cy, r, 0.78)
    bg   = _mask_annulus_local(img_bgr.shape, cx, cy, r, 1.05, 1.30)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    Hc, Sc, Vc = hsv[...,0], hsv[...,1], hsv[...,2]
    Lc, Ac, Bc = lab[...,0], lab[...,1], lab[...,2]

    v80 = _robust_percentile(Vc, disk, 80.0)
    disk_mask = ((disk>0) & (Vc < v80) & (Sc >= 20)).astype(np.uint8)*255
    v90_bg = _robust_percentile(Vc, bg, 90.0)
    bg_mask   = ((bg>0) & (Vc < v90_bg)).astype(np.uint8)*255

    L_in = _robust_median(Lc, disk_mask);  L_bg = _robust_median(Lc, bg_mask)
    A_in = _robust_median(Ac, disk_mask);  A_bg = _robust_median(Ac, bg_mask)
    B_in = _robust_median(Bc, disk_mask);  B_bg = _robust_median(Bc, bg_mask)
    S_in = _robust_median(Sc, disk_mask);  S_bg = _robust_median(Sc, bg_mask)
    H_in = _robust_median(Hc, disk_mask)

    feat = {
        "dA": A_in - A_bg,
        "dB": B_in - B_bg,
        "dS": S_in - S_bg,
        "H":  H_in,
        "S_in": S_in,
        "dL": L_in - L_bg,
    }
    lowS   = 1.0 if S_in < 26 else 0.0
    flatAB = 1.0 if (abs(feat["dA"]) + abs(feat["dB"]) < 5.5) else 0.0
    feat["silverness_hint"] = 0.7*lowS + 0.3*flatAB
    return feat

def _copper_score(feat):
    dA, dB, dS, H = feat["dA"], feat["dB"], feat["dS"], feat["H"]
    silver_hint = feat.get("silverness_hint", 0.0)
    t1 = np.clip(((dA - dB) - C10_dA_dB_MIN) / 8.0, 0, 1)
    t2 = np.clip((dS - C10_S_MIN) / 25.0, 0, 1)
    h_lo, h_hi = C10_H_RANGE
    if H < h_lo:
        t3 = np.clip(1.0 - (h_lo - H)/10.0, 0, 1)
    elif H > h_hi:
        t3 = np.clip(1.0 - (H - h_hi)/10.0, 0, 1)
    else:
        t3 = 1.0
    raw = 0.52*t1 + 0.33*t2 + 0.15*t3
    penalty = np.clip(0.30 * silver_hint, 0, 0.30)
    return float(np.clip(raw - penalty, 0.0, 1.0))

def find_c10_only(img_bgr, circles, debug=False):
    if not circles: return [], []
    feats, scores = [], []
    for (x,y,r) in circles:
        f = _bg_norm_feats(img_bgr, x,y,r)
        s = _copper_score(f)
        feats.append(f); scores.append(s)
    scores = np.array(scores, dtype=np.float32)
    p90 = float(np.percentile(scores, 90)) if len(scores) >= 2 else float(scores.max().item())
    thr = max(C10_SCORE_THR, min(0.85, p90*0.85))
    c10_idx = [i for i,s in enumerate(scores) if s >= thr]
    non_idx = [i for i,_ in enumerate(scores) if i not in c10_idx]
    c10  = [circles[i] for i in c10_idx]
    rest = [circles[i] for i in non_idx]
    return c10, rest

# ===================== [템플릿 매칭 모듈] =====================
class CoinTemplate:
    def __init__(self, label, img_gray, mask=None):
        self.label = label
        self.img = img_gray
        self.mask = mask
        self.kp = None
        self.des = None

def _make_center_mask(shape, inner_ratio=0.10, outer_ratio=0.95):
    H, W = shape[:2]
    Y, X = np.ogrid[:H, :W]
    cy, cx = H//2, W//2
    r2 = (X-cx)**2 + (Y-cy)**2
    R = min(cx, cy)
    inner = (inner_ratio*R)**2
    outer = (outer_ratio*R)**2
    m = ((inner <= r2) & (r2 <= outer)).astype(np.uint8)*255
    return m

def _prep_gray(g):
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8,8))
    ge = clahe.apply(g)
    ge = cv2.GaussianBlur(ge, (3,3), 0)
    return ge

def load_coin_templates(root_dir="assets/templates", size=320, use_akaze=False):
    labels = []
    templs = []
    for lb in ("10","50","100","500"):
        d = os.path.join(root_dir, lb)
        if not os.path.isdir(d): continue
        labels.append(lb)
        for fname in sorted(os.listdir(d)):
            path = os.path.join(d, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            s = max(img.shape[:2])
            pad_y = s - img.shape[0]; pad_x = s - img.shape[1]
            img_sq = cv2.copyMakeBorder(img, pad_y//2, pad_y - pad_y//2,
                                        pad_x//2, pad_x - pad_x//2, cv2.BORDER_REPLICATE)
            img_sq = cv2.resize(img_sq, (size, size), interpolation=cv2.INTER_AREA)
            g = _prep_gray(img_sq)
            mask = _make_center_mask(g.shape, inner_ratio=0.15, outer_ratio=0.98)
            templs.append(CoinTemplate(lb, g, mask))

    if not templs:
        return [], (None, None)

    if use_akaze:
        fe = cv2.AKAZE_create(); norm = cv2.NORM_HAMMING
    else:
        fe = cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=8,
                            edgeThreshold=15, patchSize=31, fastThreshold=10)
        norm = cv2.NORM_HAMMING

    good = []
    for T in templs:
        kp, des = fe.detectAndCompute(T.img, T.mask)
        if des is None or len(kp) < 8: continue
        T.kp, T.des = kp, des
        good.append(T)

    return good, (fe, norm)

def _roi_from_circle(img_bgr, x, y, r, out_size=320):
    H, W = img_bgr.shape[:2]
    x1, y1 = max(0, x - r - 4), max(0, y - r - 4)
    x2, y2 = min(W, x + r + 4), min(H, y + r + 4)
    roi = img_bgr[y1:y2, x1:x2]
    if roi.size == 0: return None, None
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    s = max(g.shape[:2])
    pad_y = s - g.shape[0]; pad_x = s - g.shape[1]
    g = cv2.copyMakeBorder(g, pad_y//2, pad_y - pad_y//2,
                           pad_x//2, pad_x - pad_x//2, cv2.BORDER_REPLICATE)
    g = cv2.resize(g, (out_size, out_size), interpolation=cv2.INTER_AREA)
    g = _prep_gray(g)
    m = np.zeros_like(g, np.uint8)
    cv2.circle(m, (out_size//2, out_size//2), int(0.98*(out_size//2)), 255, -1)
    return g, m

def _match_one(roi_g, roi_mask, templs, fe, norm, ratio=0.75, ransac_thr=3.0):
    if fe is None: return None
    kp2, des2 = fe.detectAndCompute(roi_g, roi_mask)
    if des2 is None or len(kp2) < 8: return None
    bf = cv2.BFMatcher(norm, crossCheck=False)

    def score_TMPL(T):
        matches = bf.knnMatch(T.des, des2, k=2)
        good = []
        for ab in matches:
            if len(ab) < 2: continue
            a, b = ab
            if a.distance < ratio * b.distance:
                good.append(a)
        if len(good) < 8: return None
        src = np.float32([T.kp[m.queryIdx].pt for m in good])
        dst = np.float32([kp2[m.trainIdx].pt for m in good])
        Hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thr)
        if Hm is None or mask is None: return None
        inliers = int(mask.sum())
        denom = max(20, min(len(T.kp), len(kp2)))
        inlier_ratio = inliers / float(denom)
        inlier_dists = [m.distance for m, keep in zip(good, mask.ravel().tolist()) if keep]
        if len(inlier_dists) == 0: return None
        mean_dist = np.mean(inlier_dists)
        dist_term = max(0.0, 1.0 - (mean_dist / 80.0))
        s = 0.7 * inlier_ratio + 0.3 * dist_term
        return s, inliers, len(good)

    by_label = {}
    for T in templs:
        r = score_TMPL(T)
        if r is None: continue
        s, inl, gnum = r
        prev = by_label.get(T.label)
        if prev is None or s > prev[0]:
            by_label[T.label] = (s, inl, gnum)

    if not by_label: return None
    label, (s, inl, gnum) = max(by_label.items(), key=lambda kv: kv[1][0])
    return {"label": label, "score": float(s), "inliers": int(inl), "goods": int(gnum)}

def template_label_coins(img_bgr, circles, templates_cache=None):
    if not circles: return []
    if templates_cache is None or templates_cache[0] == []:
        return []
    templs, (fe, norm) = templates_cache
    results = []
    for (x,y,r) in circles:
        roi_g, roi_m = _roi_from_circle(img_bgr, x,y,r, out_size=320)
        if roi_g is None: continue
        m = _match_one(roi_g, roi_m, templs, fe, norm, ratio=0.75, ransac_thr=3.0)
        if m is None: continue
        results.append((m["label"], m["score"], (x,y,r)))
    return results

# ======= 분류 공통(사이즈/색/템플릿) =======
LABELS = ["10", "50", "100", "500"]
REL_MAP = {"10": 0.75, "50": 0.90, "100": 1.00, "500": 1.104}
REL_ORDER = ["10","50","100","500"]

def _silver_score_from_copper(copper_score):
    return max(0.0, 1.0 - copper_score)

def _gauss_score(x, mu, sigma=0.08):
    d = (x - mu)
    return float(math.exp(-(d*d)/(2*sigma*sigma)))

def group_radii(r_list, tol_ratio=0.07):
    if not r_list: return []
    idx_sorted = np.argsort(r_list)
    groups = [[int(idx_sorted[0])]]
    for idx in idx_sorted[1:]:
        r = r_list[idx]
        placed = False
        for g in groups:
            r_ref = np.median([r_list[i] for i in g])
            if abs(r - r_ref) / max(1.0, r_ref) <= tol_ratio:
                g.append(int(idx)); placed = True; break
        if not placed:
            groups.append([int(idx)])
    groups.sort(key=lambda G: np.median([r_list[i] for i in G]))
    return groups

def estimate_scale_from_pairs(r_list, candidate_labels=None):
    if not r_list: return 1.0
    if candidate_labels is None:
        candidate_labels = LABELS
    ratios = np.array([REL_MAP[lab] for lab in candidate_labels], dtype=np.float32)
    candidates = []
    for r in r_list:
        for rr in ratios:
            candidates.append(r / rr)
    candidates = np.array(candidates, dtype=np.float32)

    def total_L1_err(s):
        exp = s * ratios
        errs = [np.min(np.abs(exp - r)) for r in r_list]
        return float(np.sum(errs))

    best_s, best_err = None, 1e18
    for s in np.percentile(candidates, [10, 25, 50, 75, 90]):
        err = total_L1_err(s)
        if err < best_err:
            best_s, best_err = float(s), float(err)
    return best_s if best_s is not None else 1.0

def size_score_for_class(r, s, lab):
    exp   = s * REL_MAP[lab]
    sigma = _CLASS_SIGMA.get(lab, 0.08)
    e     = abs(r - exp) / max(1.0, exp)
    score = float(math.exp(-(e / sigma) ** 2))
    if lab == "500":
        mid = s * (REL_MAP["100"] + REL_MAP["500"]) * 0.5
        if r < mid * 0.97:
            score *= 0.25
    return score

def color_prior_for_class(copper_score, lab):
    silver_score = _silver_score_from_copper(copper_score)
    if lab == "10":  return 0.9 * copper_score + 0.1 * (1.0 - copper_score)
    if lab == "50":  return 0.6 * silver_score
    if lab == "100": return 0.8 * silver_score
    if lab == "500": return 0.8 * silver_score
    return 0.5

def _cluster_copper_strength(img_bgr, group):
    scores=[]
    for (x,y,r,_) in group:
        scores.append(_copper_score(_bg_norm_feats(img_bgr, x,y,r)))
    return float(np.mean(scores)) if scores else 0.0

def decide_cluster_material(img_bgr, group, thr=C10_SCORE_THR):
    scores = []
    strong = 0
    for (x,y,r,_) in group:
        s = _copper_score(_bg_norm_feats(img_bgr, x,y,r))
        scores.append(s)
        if s >= (thr + 0.08):
            strong += 1
    if not scores:
        return "SILVER"
    avg = float(np.mean(scores))
    med = float(np.median(scores))
    ratio_strong = strong / max(1, len(scores))
    if (avg >= thr) or (med >= thr) or (ratio_strong >= 0.4):
        return "COPPER"
    Hs=[]; Ss=[]
    for (x,y,r,_) in group:
        f = _bg_norm_feats(img_bgr, x,y,r)
        Hs.append(f["H"]); Ss.append(f["S_in"])
    if (6 <= np.median(Hs) <= 28) and (np.median(Ss) >= 30) and (avg >= thr-0.05):
        return "COPPER"
    return "SILVER"



def assign_silver_3groups_direct(silver_groups):
    tmp=[]
    for g in silver_groups:
        mean_r = float(np.mean([r for (_,_,r,_) in g]))
        tmp.append((mean_r, g))
    tmp.sort(key=lambda t:t[0])
    labs = ["50","100","500"]
    return [(labs[i], tmp[i][1]) for i in range(3)]

def label_silver_with_templates(img_bgr, silver_groups, s, templates_cache, tmpl_thr=0.42):
    coins=[]
    for g in silver_groups:
        for (x,y,r,ratio) in g:
            coins.append((x,y,r,ratio))
    if not coins:
        return []
    res = template_label_coins(img_bgr, [(x,y,r) for (x,y,r,_) in coins], templates_cache=templates_cache) if (templates_cache and templates_cache[0] != []) else []
    best = {}
    for (lab, sc, trip) in res:
        if lab not in ("50","100","500"):
            continue
        if (trip not in best) or (sc > best[trip][1]):
            best[trip]=(lab,float(sc))
    labeled=[]
    if len(silver_groups)==1:
        (g,) = silver_groups
        for (x,y,r,ratio) in g:
            lab_sc = best.get((x,y,r))
            if lab_sc and lab_sc[1] >= tmpl_thr:
                lab = lab_sc[0]
            else:
                d100 = abs(r - s*REL_MAP["100"])
                d500 = abs(r - s*REL_MAP["500"])
                d50  = abs(r - s*REL_MAP["50"])
                lab  = min([("50",d50),("100",d100),("500",d500)], key=lambda t:t[1])[0]
            labeled.append((lab,(x,y,r)))
        return labeled
    if len(silver_groups)==2:
        def group_mean_r(g): return float(np.mean([r for (_,_,r,_) in g]))
        mean0 = group_mean_r(silver_groups[0])
        mean1 = group_mean_r(silver_groups[1])
        combos = [
            (("50","100"),  abs(mean0 - s*REL_MAP["50"]) + abs(mean1 - s*REL_MAP["100"])),
            (("50","500"),  abs(mean0 - s*REL_MAP["50"]) + abs(mean1 - s*REL_MAP["500"])),
            (("100","500"), abs(mean0 - s*REL_MAP["100"])+ abs(mean1 - s*REL_MAP["500"])),
            (("100","50"),  abs(mean0 - s*REL_MAP["100"])+ abs(mean1 - s*REL_MAP["50"])),
            (("500","50"),  abs(mean0 - s*REL_MAP["500"])+ abs(mean1 - s*REL_MAP["50"])),
            (("500","100"), abs(mean0 - s*REL_MAP["500"])+ abs(mean1 - s*REL_MAP["100"])),
        ]
        labs_pair = min(combos, key=lambda t:t[1])[0]
        mapped = [(labs_pair[0], silver_groups[0]), (labs_pair[1], silver_groups[1])]
        labeled=[]
        for lab, g in mapped:
            for (x,y,r,ratio) in g:
                labeled.append((lab,(x,y,r)))
        return labeled
    return []

# ======= EASY 모드: 서로 다른 반지름 타입 ≥ 4 → 크기만으로 =======
def classify_easy_mode(img_bgr, circles):
    if not circles:
        return [], {"10":0, "50":0, "100":0, "500":0}
    radii = [r for (_,_,r) in circles]
    groups = group_radii(radii, tol_ratio=0.07)
    base_groups = groups[:4]
    g_meds = [np.median([radii[i] for i in g]) for g in base_groups]
    ordered_labels = ["10","50","100","500"]
    s_list = []
    for i, lab in enumerate(ordered_labels):
        if i < len(g_meds):
            s_list.append(g_meds[i] / REL_MAP[lab])
    s = float(np.median(s_list)) if s_list else estimate_scale_from_pairs(radii)
    expected = {lab: s * REL_MAP[lab] for lab in LABELS}
    labeled = []
    for (x,y,r) in circles:
        best_lab = min(LABELS, key=lambda L: abs(r - expected[L]))
        labeled.append((best_lab, (x,y,r)))
    counts = {"10":0, "50":0, "100":0, "500":0}
    for lab,_ in labeled:
        counts[lab]+=1
    return labeled, counts

# ======= HARD 모드: 타입 < 4 → 템플릿 + 색 + 상대크기 융합 =======
def estimate_scale_100_500_pref(img_bgr, circles, tol=0.06):
    if not circles: return 1.0
    cand = []
    for (x,y,r) in circles:
        feat   = _bg_norm_feats(img_bgr, x,y,r)
        if _copper_score(feat) < 0.55:
            cand.append(r)
    if not cand:
        cand = [r for (_,_,r) in circles]
    s_cands = []
    for r in cand:
        s_cands.append(r / REL_MAP["100"])
        s_cands.append(r / REL_MAP["500"])
    def votes(s):
        v=0
        for r in cand:
            e100 = abs(r - s*REL_MAP["100"]) / max(1.0, s*REL_MAP["100"])
            e500 = abs(r - s*REL_MAP["500"]) / max(1.0, s*REL_MAP["500"])
            if min(e100,e500) <= tol: v+=1
        return v
    scr = [(s, votes(s)) for s in s_cands]
    scr.sort(key=lambda t:(-t[1], t[0]))
    top = [s for (s,v) in scr if v==scr[0][1]][:7] or [scr[0][0]]
    s = float(np.median(np.array(top, np.float32)))
    if not np.isfinite(s) or s<=0:
        s = float(np.median(np.array([r/REL_MAP["100"] for r in cand]+[r/REL_MAP["500"] for r in cand], np.float32)))
    return s

def cluster_by_ratio(circles, s, tol=0.075):
    if not circles: return []
    items = [(x,y,r, float(r)/max(1e-6,s)) for (x,y,r) in circles]
    items.sort(key=lambda t:t[3])
    clusters, cur = [], [items[0]]
    for it in items[1:]:
        if abs(it[3]-cur[-1][3]) <= tol*max(1.0, cur[-1][3]):
            cur.append(it)
        else:
            clusters.append(cur); cur=[it]
    clusters.append(cur)
    return clusters



def classify_hard_mode_cluster_template(img_bgr, circles, templates_cache=None):
    if not circles:
        return [], {"10":0,"50":0,"100":0,"500":0}
    s = estimate_scale_100_500_pref(img_bgr, circles, tol=0.06)
    clusters = cluster_by_ratio(circles, s, tol=0.075)
    copper_groups, silver_groups = [], []
    for g in clusters:
        (copper_groups if decide_cluster_material(img_bgr, g)=="COPPER" else silver_groups).append(g)
    labeled=[]
    for g in copper_groups:
        for (x,y,r,ratio) in g:
            labeled.append(("10",(x,y,r)))
    if len(silver_groups)==3:
        mapped = assign_silver_3groups_direct(silver_groups)
        for lab, g in mapped:
            for (x,y,r,ratio) in g:
                labeled.append((lab,(x,y,r)))
    else:
        labeled += label_silver_with_templates(img_bgr, silver_groups, s, templates_cache, tmpl_thr=0.42)
    counts={"10":0,"50":0,"100":0,"500":0}
    for lab,_ in labeled: counts[lab]+=1
    return labeled, counts

# ======= 최종 분기 엔트리 =======
def classify_coins_v2(img_bgr, circles, templates_cache=None):
    radii = [r for (_,_,r) in circles]
    groups = group_radii(radii, tol_ratio=0.07)
    if len(groups) >= 4:
        return classify_easy_mode(img_bgr, circles)
    if templates_cache is None:
        try:
            templates_cache = load_coin_templates(
                root_dir="assets/templates", size=320, use_akaze=False
            )
        except Exception:
            templates_cache = None
    return classify_hard_mode_cluster_template(
        img_bgr, circles, templates_cache=templates_cache
    )

# ======= main =======
def main():
    args = parse_args()

    img = cv2.imread(args.input)
    if img is None:
        # 과제 요건: 불필요한 출력 금지 → 조용히 비정상 종료
        return 1

    # (0) 처리용 리사이즈
    h, w = img.shape[:2]
    if args.scale > 0:
        scale = args.scale
    else:
        scale = (TARGET_LONG_SIDE / max(h, w)) if max(h, w) > TARGET_LONG_SIDE else 1.0
    if scale != 1.0:
        img, _ = resize_to_fit(img, scale=scale)

    # (1) 전처리
    gray_blur, edges_closed = preprocess(img)

    # (2) 마스크 2종
    maskA = build_mask_adaptive(gray_blur)
    maskB = build_mask_otsu(gray_blur, guide=maskA, guide_dilate=21)

    # (3) 각 마스크별 검출
    circlesA = detect_with_mask(img, gray_blur, maskA)
    circlesB = detect_with_mask(img, gray_blur, maskB)

    # (4) 듀얼 결과 병합 + 내부원 제거
    final_circles = merge_detections(gray_blur, maskA, maskB, circlesA, circlesB)
    final_circles = drop_center_inside(final_circles, ratio=0.85)

    # (5) 템플릿 로드 (폴더 없으면 자동 생략, 메시지 없음)
    templates_cache = None
    try:
        templates_cache = load_coin_templates(root_dir="assets/templates", size=320, use_akaze=False)
    except Exception:
        templates_cache = None

    # (6) 최종 분류
    labeled, counts = classify_coins_v2(img, final_circles, templates_cache=templates_cache)

    # 누락 방지용 기본값
    c10  = int(counts.get("10", 0))
    c50  = int(counts.get("50", 0))
    c100 = int(counts.get("100", 0))
    c500 = int(counts.get("500", 0))

    total = 10 * c10 + 50 * c50 + 100 * c100 + 500 * c500

    # ✅ 출력 형식 (정확히 이 5줄, 여분 공백/문자 금지)
    print(f"500:{c500}")
    print(f"100:{c100}")
    print(f"50:{c50}")
    print(f"10:{c10}")
    print(total)

    return 0

if __name__ == "__main__":
    sys.exit(main())
