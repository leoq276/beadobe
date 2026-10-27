"""verify_outputs.py - 检查 outputs 目录里的产物"""
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image

od = Path("outputs")
print("=" * 60)
print("outputs/ 目录内容")
print("=" * 60)
for f in sorted(od.iterdir()):
    print(f"  {f.name:<35} {f.stat().st_size:>8} bytes")

print()
print("=" * 60)
print("SVG 解析验证")
print("=" * 60)
svg = od / "cat_cartoon.svg"
tree = ET.parse(svg)
root = tree.getroot()
children = list(root)
print(f"root tag        : {root.tag}")
print(f"viewBox         : {root.get('viewBox')}")
print(f"size            : {root.get('width')} x {root.get('height')}")
# 数 shape 元素（排除网格线）
shape_kinds = {}
for c in children:
    tag = c.tag.split("}")[-1]
    shape_kinds[tag] = shape_kinds.get(tag, 0) + 1
print(f"子元素分类统计 : {shape_kinds}")

print()
print("=" * 60)
print("PNG 验证")
print("=" * 60)
for png in ["cat_cartoon_preview.png", "cat_cartoon_clean.png"]:
    p = od / png
    im = Image.open(p)
    print(f"  {png:<35} size={im.size}  mode={im.mode}")
