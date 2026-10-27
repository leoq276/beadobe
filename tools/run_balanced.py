"""临时脚本：跑折中版量化"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.palette_loader import load_palette
from core.quantize import quantize
from core.bead_chart import render_grid_only, save_bead_chart

pal = load_palette()
img = ROOT / "outputs" / "cat_seedream.png"

# 从 cat_analysis.json 读 key_features
analysis = json.loads((ROOT / "outputs" / "cat_analysis.json").read_text(encoding="utf-8"))
FEATURE_DEFAULT_SIZE = {
    "eye": (0.075, 0.045), "ear": (0.075, 0.085), "mouth": (0.060, 0.022),
    "nose": (0.030, 0.025), "brow": (0.060, 0.020), "whisker": (0.080, 0.015),
    "lid": (0.060, 0.020), "head": (0.55, 0.50), "face": (0.50, 0.45),
}
key_features = {}
for f in analysis.get("key_features", []):
    name = f.get("name") or ""
    if not name:
        continue
    pos = f.get("position")
    if not pos:
        continue
    cx, cy = pos
    sw, sh = (0.05, 0.04)
    for kw, sz in FEATURE_DEFAULT_SIZE.items():
        if kw in name.lower():
            sw, sh = sz
            break
    key_features[name] = [max(0, cx - sw / 2), max(0, cy - sh / 2),
                          min(1, cx + sw / 2), min(1, cy + sh / 2)]

dark_codes = {c.code for c in pal if c.lab[0] < 30}
print("Protected dark codes:", sorted(dark_codes))

# 折中参数
result = quantize(
    img, grid_w=52, grid_h=52, palette=pal,
    min_island_size=2, cleanup=True, detail_mode=True,
    min_sub_pct=0.06, min_contrast=18.0,
    key_features=key_features, outline_preserve=True,
    protected_codes=dark_codes,
)
print(f"\n折中版: {result.color_count} 色, 清理 {result.removed_islands} 格")
print("TOP 12:")
for code, cnt in sorted(result.stats.items(), key=lambda x: -x[1])[:12]:
    c = pal.by_code(code)
    hex_ = c.hex if c else "?"
    print(f"  {code:<6} {cnt:>4} 格  hex={hex_}")

save_bead_chart(
    result, ROOT / "outputs" / "cat_bead_chart_52x52_balanced.png",
    cell_px=14, label_every=10,
    title="拼豆图纸 52x52 (折中)", palette_name="MARD",
)
render_grid_only(result, cell_px=14, with_grid=True).save(
    ROOT / "outputs" / "cat_bead_grid_only_52x52_balanced.png"
)
print("\n[OK] 折中版已保存")
