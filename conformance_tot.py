#!/usr/bin/env python3
# ⚠️ 生成AI使用・要検証
"""conformance_tot — 本リポの 全域化が vendored cuda_total(監査済み核)と 一致するかの 照合。

  総合の 第一歩(2026-07-21): 順伝播の 全域化を 手書きから 監査済み核への 委譲に 切替。
  切替の 動機 = この照合が 見つけた 本物の 穴: **旧手書き sat は NaN を 素通しした**
  (over = (a > MAX) が NaN 比較で False → out = raw = NaN のまま)。
  cuda_total._sat は NaN→(0, 境界なし+SUNK)。0×inf 経路などで NaN が 湧いた瞬間、
  旧版は「NaN を 出さない」の 看板に 反していた。

  手書きのまま 残す部分(意図的): SatDiv の **逆伝播全域化**(勾配フラグ) — cuda_total は
  順方向のみで autograd を 持たない。ここが 本リポの 発明であり、核の 責務外。
  ただし その 値意味論(_safe_div_val)は cuda_total.tot_div と 照合する。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from physics_tot_train import sat, _safe_div_val
from cuda_total import Tot, tot_div, _sat, GE, LE, SUNK

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def battery():
    vals = [0.0, -0.0, 1.5, -2.0, 1e38, -1e38, 1e39, -1e39, 1e-38, 1e-45, -1e-45,
            float("inf"), float("-inf"), float("nan")]
    return torch.tensor(vals, dtype=torch.float32, device=DEV)


def main():
    x = battery()
    print("① sat ≡ cuda_total._sat (委譲後は 恒真だが、意味論の 回帰として 常設)")
    f0 = torch.zeros_like(x, dtype=torch.uint8)
    v1, f1 = sat(x.clone(), f0.clone())
    v2, f2 = _sat(x.clone(), DEV)
    assert torch.equal(v1, v2) and torch.equal(f1, f2)
    nan_i = torch.isnan(x)
    assert not torch.isnan(v1).any(), "NaN が 素通しされている(旧バグの 再発)"
    assert (f1[nan_i] == (GE | LE | SUNK)).all(), "NaN 入力は (0, 境界なし+SUNK) のはず"
    print(f"   battery {len(x)} 値: 値・フラグ 完全一致・NaN は 名指しされる ✓")

    print("② _safe_div_val(SatDiv の 値意味論) ≡ cuda_total.tot_div(クリーン入力の 値)")
    torch.manual_seed(0)
    num = torch.randn(4096, device=DEV) * torch.logspace(-30, 30, 4096, device=DEV)
    den = torch.randn(4096, device=DEV) * torch.logspace(-30, 30, 4096, device=DEV)
    den[::7] = 0.0                                        # a/0 = 0 の 規約も 照合
    got = _safe_div_val(num, den)
    ref = tot_div(Tot(num.double()), Tot(den.double()))
    assert torch.equal(got, ref.val), "除算の 値意味論が 監査済み核と 不一致"
    print(f"   4096 ケース(0除算 585 込み): 値 完全一致 ✓")

    print("③ 勾配意味論の 回帰: 飽和枝の 勾配は 0・非飽和枝は 素通し")
    y = torch.tensor([1.0, 1e39, float("nan")], device=DEV, requires_grad=True)
    v, _ = sat(y, torch.zeros(3, dtype=torch.uint8, device=DEV))
    v.sum().backward()
    assert y.grad[0] == 1.0 and y.grad[1] == 0.0 and y.grad[2] == 0.0
    print("   非飽和→1・飽和→0・NaN→0 ✓ (学習は 生きた行だけから 学ぶ)")
    print("★ conformance: 順伝播の 全域化は 監査済み核と 同一・発明部分(勾配全域化)は 健在")


if __name__ == "__main__":
    main()
