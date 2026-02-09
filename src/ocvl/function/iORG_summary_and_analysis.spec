# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['iORG_summary_and_analysis.py'],
    pathex=[],
    binaries=[],
    datas=[('../../.venv/Lib/site-packages/ssqueezepy/configs.ini','ssqueezepy')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={
		"matplotlib": {
			"backends":["QtAgg","SVG"],
		},
	},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='iORG_summary_and_analysis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='iORG_summary_and_analysis',
)
