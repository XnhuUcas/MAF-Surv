import pickle
from pathlib import Path
from typing import Dict, Iterator, Tuple, Literal, Union

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler


def load_fused_splits_from_pkl(
    pkl_path: Union[str, Path],
    n_splits: int = 5,
    val_size: float = 0.15,
    random_state: int = 111,
    scaler: Literal["standard", "robust"] = "standard",
    print_info: bool = True,
) -> Iterator[Dict[str, np.ndarray]]:
    """
    从 .pkl 读取三模态数据，做“先融合再标准化”，并生成 K 折划分的数据字典。
    每次 yield 一个 fold 的数据：
        {
          "X_tr","X_va","X_te",   # 已标准化后的特征 (float32)
          "T_tr","T_va","T_te",   # 时间 (float64)
          "E_tr","E_va","E_te",   # 事件(1=事件,0=删失) (int64)
          "fold": fold_index,     # 从 1 开始
          "scaler": fitted_scaler # 已拟合的 scaler（如需复用可取出）
        }
    说明：
      - 只用训练集拟合 scaler；验证/测试仅 transform，避免数据泄漏
      - 早期融合顺序： [X_gene, X_cna, X_path]（保持一致即可）
      - 进行分层：以 (1 - E) 作为 stratify，使事件/删失比例在各折尽量一致
    """
    pkl_path = Path(pkl_path)

    # ---------- 读取与结构兼容 ----------
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    data = obj["datasets"] if isinstance(obj, dict) and "datasets" in obj else obj

    required = ["x_gene", "x_path", "x_cna", "censored", "survival"]
    for k in required:
        if k not in data:
            raise KeyError(f"pkl 文件中缺少键 '{k}'，实际包含：{list(data.keys())}")

    X_gene = np.asarray(data["x_gene"])
    X_path = np.asarray(data["x_path"])
    X_cna  = np.asarray(data["x_cna"])
    E_all  = np.asarray(data["censored"]).astype(int)   # 1=事件, 0=删失
    T_all  = np.asarray(data["survival"]).astype(float)

    if not (len(X_gene) == len(X_path) == len(X_cna) == len(E_all) == len(T_all)):
        raise ValueError("各数组样本数不一致，请检查 pkl。")

    n = len(T_all)
    fused_dim = X_gene.shape[1] + X_cna.shape[1] + X_path.shape[1]

    if print_info:
        print(f"[INFO] pkl 加载成功：样本数={n} | 维度 gene={X_gene.shape[1]} "
              f"cna={X_cna.shape[1]} path={X_path.shape[1]} | 融合后维度={fused_dim}")

    # ---------- 选择 scaler ----------
    if scaler == "standard":
        Scaler = StandardScaler
    elif scaler == "robust":
        Scaler = RobustScaler
    else:
        raise ValueError("scaler 仅支持 'standard' 或 'robust'")

    # ---------- K 折 + 内部验证 ----------
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (trainval_idx, test_idx) in enumerate(skf.split(np.arange(n), E_all), 1):
        tr_idx, va_idx = train_test_split(
            trainval_idx,
            test_size=val_size,
            random_state=random_state,
            stratify=E_all[trainval_idx],
        )

        # 早期融合（先拼接再统一标准化）
        X_tr_raw = np.hstack([X_gene[tr_idx], X_cna[tr_idx], X_path[tr_idx]]).astype(np.float32)
        X_va_raw = np.hstack([X_gene[va_idx], X_cna[va_idx], X_path[va_idx]]).astype(np.float32)
        X_te_raw = np.hstack([X_gene[test_idx], X_cna[test_idx], X_path[test_idx]]).astype(np.float32)

        sc = Scaler().fit(X_tr_raw)                 # 只用训练集拟合
        X_tr = sc.transform(X_tr_raw).astype(np.float32)
        X_va = sc.transform(X_va_raw).astype(np.float32)
        X_te = sc.transform(X_te_raw).astype(np.float32)

        T_tr, T_va, T_te = T_all[tr_idx], T_all[va_idx], T_all[test_idx]
        E_tr, E_va, E_te = E_all[tr_idx], E_all[va_idx], E_all[test_idx]

        if print_info and fold == 1:
            m, s = float(X_tr.mean()), float(X_tr.std())
            print(f"[INFO] 统一标准化后(训练集)：mean≈{m:.4f}, std≈{s:.4f}；"
                  f"X 形状：train {X_tr.shape} / val {X_va.shape} / test {X_te.shape}")

        yield dict(
            X_tr=X_tr, X_va=X_va, X_te=X_te,
            T_tr=T_tr, T_va=T_va, T_te=T_te,
            E_tr=E_tr, E_va=E_va, E_te=E_te,
            fold=fold, scaler=sc,
        )