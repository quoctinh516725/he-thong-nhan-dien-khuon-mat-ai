"""
Threshold Evaluation Script for UTH Face Detection System
==========================================================
Dataset  : faces94 (~5.2MB, 153 subjects)
Pipeline : YOLOv8-Face -> MediaPipe Selfie Segmenter -> FaceNet (InceptionResnetV1)
Goal     : Measure FAR / FRR across cosine-similarity thresholds, find the optimal
           Equal Error Rate (EER) point, and export confusion matrix + report.

Usage:
    cd d:/HocTap/CV/Code/uth_face_detection_ai
    python test/uth_threshold_test.py
"""

import os
import sys
import zipfile
import urllib.request
import shutil
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# ---- 1. Bootstrap project paths ----
AI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "AI"))
TEST_DIR = os.path.dirname(os.path.abspath(__file__))

os.environ.setdefault("YOLO_FACE_ONNX_PATH", os.path.join(AI_DIR, "models", "yolov8n-face-lindevs.onnx"))
os.environ.setdefault("FACE_MIN_CONFIDENCE", "0.45")

os.chdir(AI_DIR)
sys.path.insert(0, AI_DIR)

from app.services.ai import (
    detect_faces_with_fallback,
    extract_face_crop_tensor,
    get_embedding,
    device,
)

# ---- 2. Dataset paths ----
DATA_DIR = os.path.join(TEST_DIR, "data")
DATASET_ZIP = os.path.join(DATA_DIR, "faces94.zip")
DATASET_DIR = os.path.join(DATA_DIR, "faces94")

DOWNLOAD_URLS = [
    "https://github.com/RiverGao/Face-Recognition/raw/master/faces94.zip",
    "https://github.com/RiverGao/Face-Recognition/raw/main/faces94.zip",
    "https://github.com/chenyijia1997/Face-datasets/raw/master/faces94.zip",
    "https://github.com/chenyijia1997/Face-datasets/raw/main/faces94.zip",
]


# ---- 3. Dataset download and extraction ----
def download_and_extract():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATASET_ZIP):
        for url in DOWNLOAD_URLS:
            try:
                print(f"[Download] Trying: {url}")
                req = urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                )
                with urllib.request.urlopen(req, timeout=60) as resp, open(DATASET_ZIP, "wb") as f:
                    shutil.copyfileobj(resp, f)
                print("[Download] Success.")
                break
            except Exception as e:
                print(f"[Download] Failed ({e}), trying next mirror...")
        else:
            raise RuntimeError(
                f"All download mirrors failed. Please manually place faces94.zip in: {DATASET_ZIP}"
            )
    if not os.path.exists(DATASET_DIR):
        print(f"[Extract] Extracting to {DATASET_DIR} ...")
        with zipfile.ZipFile(DATASET_ZIP, "r") as z:
            z.extractall(DATASET_DIR)
        print("[Extract] Done.")


def get_subject_paths(min_images=10):
    subject_dirs = []
    for root, _dirs, files in os.walk(DATASET_DIR):
        jpegs = [f for f in files if f.lower().endswith(".jpg")]
        if len(jpegs) >= min_images:
            subject_dirs.append(root)
    subject_dirs.sort()
    return subject_dirs


# ---- 4. Per-image embedding using the FULL project pipeline ----
def extract_embedding_from_path(image_path):
    """
    Full pipeline: YOLOv8-Face -> MediaPipe Selfie Segmenter -> FaceNet.
    Returns L2-normalised float32 numpy array of shape (512,), or None on failure.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        return None
    try:
        detections, _ = detect_faces_with_fallback(frame)
        if not detections:
            return None
        # Largest bounding-box face is the primary subject
        primary = max(detections, key=lambda d: d.area)
        face_tensor, *_ = extract_face_crop_tensor(frame, primary.box)
        emb = get_embedding(face_tensor)
        norm = np.linalg.norm(emb)
        return (emb / norm).astype("float32") if norm > 0 else None
    except Exception as exc:
        print(f"    [WARN] {os.path.basename(image_path)}: {exc}")
        return None


# ---- 5. Cosine -> Similarity% conversion (mirrors simplified_faces.py) ----
def cosine_to_sim_percent(c):
    return round(c * 100, 2)



# ---- 6. Main evaluation ----
def run_evaluation():
    print(f"[Info] Running on device: {device}")
    print(f"[Info] AI_DIR   = {AI_DIR}")
    print(f"[Info] TEST_DIR = {TEST_DIR}\n")

    download_and_extract()
    subject_dirs = get_subject_paths(min_images=10)
    print(f"[Dataset] Total subjects with >=10 images: {len(subject_dirs)}")
    if len(subject_dirs) < 20:
        print("[Error] Need at least 20 subjects. Aborting.")
        return

    n_known = max(20, len(subject_dirs) * 2 // 3)
    known_dirs = subject_dirs[:n_known]
    unknown_dirs = subject_dirs[n_known:]
    print(f"[Dataset] Known (enrolled): {len(known_dirs)} subjects")
    print(f"[Dataset] Unknown (impostors): {len(unknown_dirs)} subjects\n")

    # ---- Step 1: Registration templates ----
    print("[Step 1/3] Extracting registration templates ...")
    ref_embeddings = []
    ref_labels = []
    for idx, sdir in enumerate(known_dirs):
        files = sorted(f for f in os.listdir(sdir) if f.lower().endswith(".jpg"))
        emb = extract_embedding_from_path(os.path.join(sdir, files[0]))
        if emb is not None:
            ref_embeddings.append(emb)
            ref_labels.append(idx)
        else:
            ref_embeddings.append(np.zeros(512, dtype="float32"))
            ref_labels.append(idx)
        if (idx + 1) % 10 == 0 or idx == len(known_dirs) - 1:
            print(f"  {idx + 1}/{len(known_dirs)} templates done...")
    ref_matrix = np.stack(ref_embeddings).astype("float32")

    # ---- Step 2: Authorised (genuine) queries ----
    print("\n[Step 2/3] Extracting authorised queries (genuine tests) ...")
    auth_embeddings = []
    auth_labels = []
    for idx, sdir in enumerate(known_dirs):
        files = sorted(f for f in os.listdir(sdir) if f.lower().endswith(".jpg"))
        for q_offset in range(1, 3):
            if q_offset >= len(files):
                continue
            emb = extract_embedding_from_path(os.path.join(sdir, files[q_offset]))
            if emb is not None:
                auth_embeddings.append(emb)
                auth_labels.append(idx)
        if (idx + 1) % 10 == 0 or idx == len(known_dirs) - 1:
            print(f"  {idx + 1}/{len(known_dirs)} authorised folders done...")

    # ---- Step 3: Unauthorised (impostor) queries ----
    print("\n[Step 3/3] Extracting unauthorised queries (impostor tests) ...")
    unauth_embeddings = []
    for idx, sdir in enumerate(unknown_dirs):
        files = sorted(f for f in os.listdir(sdir) if f.lower().endswith(".jpg"))
        for q_offset in range(0, 2):
            if q_offset >= len(files):
                continue
            emb = extract_embedding_from_path(os.path.join(sdir, files[q_offset]))
            if emb is not None:
                unauth_embeddings.append(emb)
        if (idx + 1) % 10 == 0 or idx == len(unknown_dirs) - 1:
            print(f"  {idx + 1}/{len(unknown_dirs)} impostor folders done...")

    auth_matrix = np.stack(auth_embeddings).astype("float32")
    unauth_matrix = np.stack(unauth_embeddings).astype("float32")
    auth_labels_np = np.array(auth_labels)

    print(f"\n[Extraction complete]")
    print(f"  Reference templates : {ref_matrix.shape[0]}")
    print(f"  Authorised queries  : {auth_matrix.shape[0]}")
    print(f"  Unauthorised queries: {unauth_matrix.shape[0]}")

    # ---- Step 4: Cosine similarity (dot product of L2-normalised vectors) ----
    auth_scores = auth_matrix @ ref_matrix.T            # (n_auth, n_known)
    unauth_scores = unauth_matrix @ ref_matrix.T        # (n_unauth, n_known)
    auth_best_sim = auth_scores.max(axis=1)
    auth_best_match = auth_scores.argmax(axis=1)
    unauth_best_sim = unauth_scores.max(axis=1)

    # ---- Step 5: Sweep thresholds ----
    thresholds = np.round(np.arange(0.20, 0.99, 0.02), 4)
    far_list = []
    frr_list = []

    print("\n=== SWEEPING THRESHOLDS (Cosine Similarity) ===")
    print(f"{'Threshold':>10}  {'FAR (%)':>8}  {'FRR (%)':>8}")
    print("-" * 35)
    for t in thresholds:
        false_rejections = int(((auth_best_sim < t) | (auth_best_match != auth_labels_np)).sum())
        frr = false_rejections / len(auth_embeddings) * 100
        false_acceptances = int((unauth_best_sim >= t).sum())
        far = false_acceptances / len(unauth_embeddings) * 100
        far_list.append(far)
        frr_list.append(frr)
        print(f"{t:>10.2f}  {far:>8.2f}  {frr:>8.2f}")

    differences = np.abs(np.array(far_list) - np.array(frr_list))
    eer_idx = int(np.argmin(differences))
    eer_threshold = float(thresholds[eer_idx])
    eer_value = (far_list[eer_idx] + frr_list[eer_idx]) / 2.0
    eer_sim_percent = cosine_to_sim_percent(eer_threshold)

    print(f"\n>>> Optimal EER Threshold (Cosine): {eer_threshold:.2f}  |  EER: {eer_value:.2f}%")
    print(f">>> Equivalent Similarity%: {eer_sim_percent:.1f}%")

    # ---- Confusion matrix at EER threshold ----
    t_opt = eer_threshold
    tp = fn = fp = tn = 0
    for i in range(len(auth_embeddings)):
        if auth_best_sim[i] >= t_opt and auth_best_match[i] == auth_labels[i]:
            tp += 1
        elif auth_best_sim[i] >= t_opt:
            fp += 1
        else:
            fn += 1
    for i in range(len(unauth_embeddings)):
        if unauth_best_sim[i] >= t_opt:
            fp += 1
        else:
            tn += 1

    total = len(auth_embeddings) + len(unauth_embeddings)
    accuracy = (tp + tn) / total * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n=== CLASSIFICATION METRICS AT EER THRESHOLD ({t_opt:.2f}) ===")
    print(f"  Accuracy  : {accuracy:.2f}%")
    print(f"  Precision : {precision:.2f}%")
    print(f"  Recall    : {recall:.2f}%")
    print(f"  F1-Score  : {f1:.2f}%")
    print(f"  TP={tp}  FN={fn}  FP={fp}  TN={tn}")

    # ---- Plot 1: FAR / FRR Curve (Cosine Similarity axis only) ----
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(thresholds, far_list, "r-o", ms=4, label="FAR - False Acceptance Rate")
    ax.plot(thresholds, frr_list, "b-s", ms=4, label="FRR - False Rejection Rate")
    ax.axvline(t_opt, color="green", ls="--", label=f"EER Threshold (cosine={t_opt:.2f}  |  Similarity={eer_sim_percent:.1f}%)")
    ax.axvline(0.75, color="purple", ls=":", alpha=0.8, label="Applied threshold (cosine=0.75  |  Similarity=85.0%)")
    ax.scatter([t_opt], [eer_value], color="black", zorder=6, s=120)
    ax.annotate(f"EER ~{eer_value:.1f}%", (t_opt + 0.015, eer_value + 1.5), fontsize=10)
    ax.set_xlabel("Cosine Similarity Threshold", fontsize=12)
    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_title("FAR vs FRR Curve — Cosine Similarity Threshold", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, ls=":", alpha=0.6)

    plt.tight_layout()
    chart_path = os.path.join(TEST_DIR, "threshold_chart.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n[Chart] Saved to: {chart_path}")

    # ---- Plot 2: Binary Confusion Matrix Heatmap ----
    binary_cm = np.array([[tp, fn], [fp, tn]])
    fig2, ax3 = plt.subplots(figsize=(7, 6))
    im = ax3.imshow(binary_cm, interpolation="nearest", cmap=plt.cm.Greens)
    plt.colorbar(im, ax=ax3)
    ax3.set_title(
        f"Confusion Matrix\n(cosine={t_opt:.2f} | Similarity={eer_sim_percent:.1f}%)"
    )
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["Match (Known)", "Reject (Unknown)"], fontsize=10)
    ax3.set_yticklabels(["True Known", "True Unknown"], fontsize=10)
    ax3.set_xlabel("Predicted Class")
    ax3.set_ylabel("True Class")
    cm_labels = [["TP", "FN"], ["FP", "TN"]]
    for r in range(2):
        for c in range(2):
            ax3.text(
                c, r, f"{cm_labels[r][c]}\n{binary_cm[r, c]}",
                ha="center", va="center", fontsize=13, weight="bold", color="black",
            )
    plt.tight_layout()
    cm_path = os.path.join(TEST_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[Chart] Confusion matrix saved to: {cm_path}")

    # ---- Comparison indices for report table ----
    idx_65 = int(np.argmin(np.abs(thresholds - 0.55)))   # cosine ~= 0.55 -> Similarity% ~= 65
    idx_85 = int(np.argmin(np.abs(thresholds - 0.75)))   # cosine ~= 0.75 -> Similarity% ~= 85

    # ---- Markdown report ----
    sep = "---"
    report_path = os.path.join(TEST_DIR, "evaluation_report.md")
    lines = []
    lines.append("# Bao cao danh gia nguong quyet dinh he thong CCTV nhan dien khuon mat\n")
    lines.append("## Pipeline su dung")
    lines.append("**YOLOv8-Face (ONNX)** -> **MediaPipe Selfie Segmenter** -> **FaceNet InceptionResnetV1** (VGGFace2)\n")
    lines.append("Tuc la pipeline day du, giong het voi luong /identify API dang chay trong san pham thuc te.\n")
    lines.append(sep + "\n")
    lines.append("## Bo du lieu kiem thu: `faces94`")
    lines.append("* **Nguon:** Essex University Faces94 Dataset (anh chan dung trong nha, 153 doi tuong)")
    lines.append(f"* **Doi tuong da dang ky (Known):** {len(known_dirs)} nguoi (moi nguoi dang ky 1 anh mau)")
    lines.append(f"* **Truy van hop le (Genuine queries):** {auth_matrix.shape[0]} anh (anh thu 2 va 3 cua nguoi da dang ky)")
    lines.append(f"* **Truy van gia mao (Impostor queries):** {unauth_matrix.shape[0]} anh ({len(unknown_dirs)} nguoi la x 2 anh)\n")
    lines.append(sep + "\n")
    lines.append("## Ket qua tai nguong EER toi uu\n")
    lines.append("| Chi so | Gia tri |")
    lines.append("|:---|:---:|")
    lines.append(f"| **Nguong Cosine Similarity toi uu (EER)** | `{t_opt:.2f}` |")
    lines.append(f"| **Nguong Similarity% tuong duong (hien thi UI)** | `{eer_sim_percent:.1f}%` |")
    lines.append(f"| **Equal Error Rate (EER)** | **{eer_value:.2f}%** |")
    lines.append(f"| **Accuracy** | {accuracy:.2f}% |")
    lines.append(f"| **Precision** | {precision:.2f}% |")
    lines.append(f"| **Recall** | {recall:.2f}% |")
    lines.append(f"| **F1-Score** | {f1:.2f}% |\n")
    lines.append(sep + "\n")
    lines.append("## Ma tran nham lan (Confusion Matrix)\n")
    lines.append("| | Du doan la Cung nguoi (Match) | Du doan la Nguoi la (Reject) |")
    lines.append("|:---|:---:|:---:|")
    lines.append(f"| **Thuc te la Nguoi da dang ky** | **TP = {tp}** | FN = {fn} |")
    lines.append(f"| **Thuc te la Nguoi la** | FP = {fp} | **TN = {tn}** |\n")
    lines.append(f"* **True Positive (TP):** {tp} - nguoi da dang ky duoc nhan dien dung danh tinh")
    lines.append(f"* **False Negative (FN):** {fn} - nguoi da dang ky bi tu choi hoac nhan nham sang nguoi khac")
    lines.append(f"* **False Positive (FP):** {fp} - nguoi la bi nhan nham la nguoi da dang ky")
    lines.append(f"* **True Negative (TN):** {tn} - nguoi la bi tu choi dung\n")
    lines.append(sep + "\n")
    lines.append("## So sanh nguong\n")
    lines.append("| Nguong | Cosine Similarity | Similarity% (UI) | FAR | FRR |")
    lines.append("|:---|:---:|:---:|:---:|:---:|")
    lines.append(f"| Ma nguon cu (truoc khi sua) | 0.55 | 65.0% | {far_list[idx_65]:.2f}% | {frr_list[idx_65]:.2f}% |")
    lines.append(f"| Nguong EER toi uu | {t_opt:.2f} | {eer_sim_percent:.1f}% | {far_list[eer_idx]:.2f}% | {frr_list[eer_idx]:.2f}% |")
    lines.append(f"| **Nguong dang ap dung (85%)** | **0.75** | **85.0%** | **{far_list[idx_85]:.2f}%** | **{frr_list[idx_85]:.2f}%** |\n")
    lines.append(sep + "\n")
    lines.append("## Bieu do FAR vs FRR\n")
    lines.append("![FAR/FRR Curve](threshold_chart.png)\n")
    lines.append("![Confusion Matrix](confusion_matrix.png)\n")
    lines.append(sep + "\n")
    lines.append("## Bang so lieu chi tiet FAR / FRR theo tung nguong Cosine\n")
    lines.append("| Cosine Similarity | Similarity% (UI) | FAR (%) | FRR (%) |")
    lines.append("|:---:|:---:|:---:|:---:|")
    for i, t in enumerate(thresholds):
        sp = cosine_to_sim_percent(float(t))
        lines.append(f"| {t:.2f} | {sp:.1f}% | {far_list[i]:.2f}% | {frr_list[i]:.2f}% |")

    with open(report_path, "w", encoding="utf-8") as rpt:
        rpt.write("\n".join(lines) + "\n")

    print(f"[Report] Saved to: {report_path}")
    print("\nDONE. Evaluation complete.")


if __name__ == "__main__":
    run_evaluation()
