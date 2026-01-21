# plot_auc_boxplot.py
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === 改成你的 AUC 文件夹 ===
AUC_DIR = Path(r"D:\CAMR_paper\multimodal_survival_prediction\multimodal_survival_prediction\CAMR_results\LGG\all\AUC")

# 想要的展示顺序（没找到的会自动追加在后面）
method_order = ["En-Cox", "Lasso-Cox", "DeepSurv", "CAMR", "MAF-Surv"]

def normalize_label_from_filename(name: str) -> str:
    m = re.search(r"AUC_formula16_(.+?)(?:\.[^.]+)?$", name, flags=re.IGNORECASE)
    suf_raw = (m.group(1) if m else Path(name).stem)
    suf = suf_raw.lower().strip()
    suf_norm = suf.replace("-", "_")

    if suf_norm.startswith(("en_cox", "encox")):
        return "En-Cox"
    if suf_norm.startswith(("lasso_cox", "lassocox")):
        return "Lasso-Cox"
    if suf_norm.startswith("deepsurv"):
        return "DeepSurv"
    if suf_norm.startswith("camr"):
        return "CAMR"
    # ★ 修正处：同时支持 maf_surv / maf-surv / mafsurv
    if suf_norm.startswith(("maf_surv", "maf-surv", "mafsurv")):
        return "MAF-Surv"

    # 兜底：简单美化
    return suf.replace("_", "-").title()


def read_auc_series(p: Path) -> np.ndarray:
    """读取第二列AUC → 数值向量；支持 csv/xlsx/xls。"""
    ext = p.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(p)
    else:
        # 对于 xlsx/xls，让 pandas 自选引擎；若失败再给出提示
        try:
            df = pd.read_excel(p)
        except Exception as e:
            raise RuntimeError(f"read_excel 失败（{p.name}）：{e}")

    if df.shape[1] < 2:
        raise ValueError(f"{p.name} 至少需要两列（第2列为AUC）")

    auc_col = df.columns[1]
    s = pd.to_numeric(df[auc_col], errors="coerce").dropna()
    return s.to_numpy(dtype=float)

# 收集文件
files = []
for ext in (".csv", ".xlsx", ".xls"):
    files.extend(AUC_DIR.glob(f"AUC_formula16_*{ext}"))

if not files:
    raise FileNotFoundError(f"未在 {AUC_DIR} 找到 AUC_formula16_* 文件")

print("[INFO] 发现文件：")
for f in files:
    print("  -", f.name)

# 读取与汇总
data = {}
for f in files:
    label = normalize_label_from_filename(f.name)
    try:
        arr = read_auc_series(f)
        if arr.size == 0:
            print(f"[WARN] {f.name} 的第2列没有有效数值，跳过")
            continue
        data[label] = arr
        print(f"[OK]   {f.name}  ->  '{label}', n={arr.size}, mean={arr.mean():.4f}")
    except Exception as e:
        print(f"[SKIP] 读取失败 {f.name}: {e}")

if not data:
    raise RuntimeError("没有任何 AUC 数据被成功读取，请检查文件格式/第2列内容。")

# 按顺序准备数据（缺的自动排后）
labels = [m for m in method_order if m in data] + [m for m in data if m not in method_order]
values = [data[m] for m in labels]

# 打印汇总
print("\n=== 汇总（均值±标准差） ===")
for m, v in zip(labels, values):
    print(f"{m:>10s}: {np.mean(v):.4f} ± {np.std(v, ddof=0):.4f} (n={len(v)})")

# 画箱线图（仿论文风格）
plt.rcParams.update({'font.size': 16})  # 全局字体大小
plt.figure(figsize=(10, 6))
bp = plt.boxplot(values, labels=labels, patch_artist=True, showfliers=False, widths=0.6)
for box in bp["boxes"]:
    box.set_facecolor("#A7C7E7")
    box.set_alpha(0.6)
for median in bp["medians"]:
    median.set_color("black")
    median.set_linewidth(2)

plt.ylabel("AUC")
plt.title("the AUC value of MAF-Surv and other methods on LGG")
plt.grid(axis="y", alpha=0.3)
plt.ylim(0.75, 1.00)

out_png = AUC_DIR / "AUC_boxplot_LUSC.png"
plt.tight_layout()
plt.savefig(out_png, dpi=300)
plt.close()
print(f"\n[OK] Boxplot saved to: {out_png}")
