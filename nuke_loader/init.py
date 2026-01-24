"""
MikeMagic - Nuke Loader
This file is executed by ~/.nuke/init.py
"""
import nuke
import sys
import os

# MikeMagic root - hardcoded for reliability with exec()
MIKEMAGIC_ROOT = "D:/MikeMagic"

# Add MikeMagic to Python path
if MIKEMAGIC_ROOT not in sys.path:
    sys.path.append(MIKEMAGIC_ROOT)

# Register Nuke plugin paths
nuke.pluginAddPath(f"{MIKEMAGIC_ROOT}/mike_magic")

# External tools (pyd_playground)
# nuke.pluginAddPath("D:/pyd_playground/cotracker_roto/MMTracker_for_nuke/gizmos")

# Initialize mm_roto_toolkit
if nuke.GUI:
    try:
        import mm_roto_toolkit
        mm_roto_toolkit.startup()
    except Exception as e:
        print(f"MM-Roto toolkit failed to start: {e}")

print(f"[MikeMagic] Loaded from: {MIKEMAGIC_ROOT}")
