"""
MikeMagic Toolkit for Nuke
Auto-import and roto generation from MM-Roto exports
Version: 1.0.0
"""

def startup():
    """Initialize the MM-Roto toolkit - called from init.py"""
    from . import listener
    listener.start_listener()
    print("MikeMagic - Listener started")

