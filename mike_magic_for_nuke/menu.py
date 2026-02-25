# Mike Magic - Nuke Node Registration
import nuke
import MikeMagic

# Register for Tab search with icon
nuke.menu("Nodes").addCommand("MikeMagic", "MikeMagic.create_mike_magic_node()", icon="splash.png")

# Dev tools - reload all MikeMagic modules
def _reload_mikemagic():
    import importlib
    modules = [
        'MikeMagic',
        'mm_roto_toolkit',
        'mm_roto_toolkit.listener',
        'mm_roto_toolkit.matte_toolkit_merged',
        'mm_roto_toolkit.copy_matte_toolkit_merged',
        'mm_roto_toolkit.less_points_matte',
    ]
    import sys
    for m in modules:
        if m in sys.modules:
            importlib.reload(sys.modules[m])
    print("[MikeMagic] All modules reloaded!")

nuke.menu("Nuke").addCommand("MikeMagic/Reload All", "_reload_mikemagic()")
