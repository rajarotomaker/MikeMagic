# Mike Magic Nuke Plugin
# Place this folder in your .nuke directory

import nuke
import os

# Get the directory where this plugin is installed
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))

# Add plugin directory to Nuke's plugin path
nuke.pluginAddPath(PLUGIN_DIR)

print("[Mike Magic] Plugin loaded from: {}".format(PLUGIN_DIR))
