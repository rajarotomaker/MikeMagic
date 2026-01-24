# Mike Magic - Nuke Node Registration
import nuke
import MikeMagic

# Register for Tab search with icon
nuke.menu("Nodes").addCommand("MikeMagic", "MikeMagic.create_mike_magic_node()", icon="splash.png")
