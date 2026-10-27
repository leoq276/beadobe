# -*- mode: python ; coding: utf-8 -*-
# 拼豆图纸生成器打包 spec（最小环境 beadbuild 用）
# 用法: conda run -n beadbuild pyinstaller bead_studio.spec
from PyInstaller.utils.hooks import collect_submodules

datas = [
    ("palettes/mard.csv", "palettes"),
    ("config.yaml", "."),
    (".env.example", "."),
    ("icon_transparent.png", "."),
]

hiddenimports = []

a = Analysis(
    ["bead_studio.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # 排除项目用不到的重型 Qt 组件
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineQuick",
        "PySide6.QtPdf",
        "PySide6.QtPdfWidgets",
        "PySide6.QtQuick",
        "PySide6.QtQml",
        "PySide6.Qt3DCore",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
        "PySide6.QtMultimedia",
        "PySide6.QtBluetooth",
        "PySide6.QtNetworkAuth",
        "PySide6.QtPositioning",
        "PySide6.QtSensors",
        "PySide6.QtSerialPort",
        "PySide6.QtTest",
        "PySide6.QtSql",
        # 排除非项目依赖（保险）
        "matplotlib",
        "torch",
        "tensorflow",
        "keras",
        "jax",
        "pandas",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# 过滤掉用不到的二进制：FFmpeg 视频 IO 后端（拼豆只处理静态图片，不需要视频读写）
# 二进制元组第一项是带目录前缀的路径（如 cv2\opencv_videoio_ffmpeg500_64.dll）
a.binaries = [
    b for b in a.binaries
    if "opencv_videoio_ffmpeg" not in str(b[0]).lower()
]

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BeadStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # GUI 程序，不弹控制台
    icon="icon_bead.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="BeadStudio",
)
