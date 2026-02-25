# MikeMagic Node for Nuke
# Launches MikeMagic with footage, converts DPX/EXR to PNG if needed

import nuke
import subprocess
import os
import tempfile
import shutil
import re
import hashlib

# ============================================================================
# CONFIGURATION - Update this path to your MikeMagic installation
# ============================================================================
MIKE_MAGIC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run_mm.bat")
# ============================================================================

# Formats MikeMagic already supports - pass directly, no conversion
SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp',
                     '.mp4', '.mov', '.avi', '.mkv']

# Formats that need conversion to PNG before loading
CONVERT_FORMATS = ['.exr', '.dpx', '.cin', '.hdr']


def get_temp_base_dir():
    """Get or create base temp directory for PNG conversion"""
    temp_base = os.path.join(tempfile.gettempdir(), "mike_magic_temp")
    if not os.path.exists(temp_base):
        os.makedirs(temp_base)
    return temp_base


def get_cache_folder_name(file_path, first_frame, last_frame):
    """Generate a unique cache folder name based on source file path and frame range"""
    # Create a unique identifier from the file path and frame range
    cache_key = "{}_{}_{}".format(file_path, first_frame, last_frame)
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
    # Include a readable prefix from the filename
    base_name = os.path.splitext(os.path.basename(file_path.replace('%04d', '').replace('####', '')))[0]
    base_name = re.sub(r'[^\w\-]', '_', base_name)[:20]  # Sanitize and truncate
    return "{}_{}".format(base_name, cache_hash)


def get_cache_dir(file_path, first_frame, last_frame):
    """Get cache directory for a specific source sequence"""
    folder_name = get_cache_folder_name(file_path, first_frame, last_frame)
    cache_dir = os.path.join(get_temp_base_dir(), folder_name)
    return cache_dir


def is_cache_valid(cache_dir, first_frame, last_frame):
    """Check if cached PNGs exist for the given frame range"""
    if not os.path.exists(cache_dir):
        return False

    # Check if first and last frame PNGs exist
    first_png = os.path.join(cache_dir, "frame.{:04d}.png".format(int(first_frame)))
    last_png = os.path.join(cache_dir, "frame.{:04d}.png".format(int(last_frame)))

    if os.path.exists(first_png) and os.path.exists(last_png):
        return True

    return False


def clear_cache_dir(cache_dir):
    """Clear a specific cache directory before conversion"""
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir)


def get_file_extension(file_path):
    """
    Get the actual file extension from Nuke's file path.
    Handles sequence patterns like .%04d.exr or .####.dpx
    """
    # Remove frame pattern to get actual extension
    clean_path = file_path
    clean_path = re.sub(r'%\d*d', '0000', clean_path)  # %04d -> 0000
    clean_path = re.sub(r'#+', '0000', clean_path)      # #### -> 0000

    ext = os.path.splitext(clean_path)[1].lower()
    return ext


def resolve_first_frame_path(file_path, first_frame):
    """
    Convert Nuke's sequence notation to actual first frame path.
    e.g., image.%04d.png -> image.0001.png
    """
    result = file_path

    # Handle %0Xd notation
    match = re.search(r'%(\d*)d', result)
    if match:
        padding = int(match.group(1)) if match.group(1) else 1
        frame_str = str(int(first_frame)).zfill(padding)
        result = re.sub(r'%\d*d', frame_str, result)

    # Handle #### notation
    match = re.search(r'(#+)', result)
    if match:
        num_hashes = len(match.group(1))
        frame_str = str(int(first_frame)).zfill(num_hashes)
        result = re.sub(r'#+', frame_str, result)

    return os.path.normpath(result)


def get_input_info(read_node):
    """
    Extract file path and frame range from a Read node.
    Returns (file_path, first_frame, last_frame)
    """
    file_path = read_node['file'].value()
    first_frame = int(read_node['first'].value())
    last_frame = int(read_node['last'].value())

    return file_path, first_frame, last_frame


def convert_to_png(read_node, first_frame, last_frame, file_path, status_node=None):
    """
    Convert sequence to PNG using Nuke's Write node.
    Uses caching to avoid re-conversion if PNGs already exist.
    Returns (success, png_first_frame_path)
    """
    cache_dir = get_cache_dir(file_path, first_frame, last_frame)
    first_frame_path = os.path.join(cache_dir, "frame.{:04d}.png".format(int(first_frame)))

    # Check if cache is valid - skip conversion if PNGs already exist
    if is_cache_valid(cache_dir, first_frame, last_frame):
        print("[MikeMagic] Using cached PNGs from: {}".format(cache_dir))
        if status_node and status_node.knob('status_text'):
            status_node.knob('status_text').setValue("Using cached conversion")
        return True, first_frame_path

    # Cache not valid, need to convert
    clear_cache_dir(cache_dir)
    output_pattern = os.path.join(cache_dir, "frame.%04d.png").replace('\\', '/')

    if status_node and status_node.knob('status_text'):
        status_node.knob('status_text').setValue("Converting frames {}-{}...".format(first_frame, last_frame))

    try:
        # Create Write node connected to Read node
        write_node = nuke.createNode("Write", inpanel=False)
        write_node.setInput(0, read_node)
        write_node['file'].setValue(output_pattern)
        write_node['file_type'].setValue('png')
        write_node['channels'].setValue('rgba')

        # Render
        nuke.execute(write_node, int(first_frame), int(last_frame))

        # Cleanup Write node
        nuke.delete(write_node)

        if os.path.exists(first_frame_path):
            return True, first_frame_path
        else:
            nuke.message("Conversion failed - output file not created")
            return False, None

    except Exception as e:
        try:
            nuke.delete(write_node)
        except:
            pass
        nuke.message("Conversion error: {}".format(str(e)))
        return False, None


def launch_mike_magic(file_path, status_node=None):
    """Launch MikeMagic with the given file path"""
    if not os.path.exists(MIKE_MAGIC_PATH):
        nuke.message("MikeMagic not found at:\n{}\n\nUpdate MIKE_MAGIC_PATH in MikeMagic.py".format(MIKE_MAGIC_PATH))
        return False

    if not os.path.exists(file_path):
        nuke.message("File not found:\n{}".format(file_path))
        return False

    try:
        mike_magic_dir = os.path.dirname(MIKE_MAGIC_PATH)

        subprocess.Popen(
            [MIKE_MAGIC_PATH, file_path],
            cwd=mike_magic_dir,
            shell=True
        )

        if status_node and status_node.knob('status_text'):
            status_node.knob('status_text').setValue("Launched MikeMagic!")

        print("[MikeMagic] Launched: {}".format(file_path))
        return True

    except Exception as e:
        nuke.message("Failed to launch MikeMagic:\n{}".format(str(e)))
        return False


def convert_and_launch(node):
    """Main function: Check format, convert if needed, launch MikeMagic"""
    input_node = node.input(0)

    if not input_node:
        nuke.message("Connect a Read node to MikeMagic first")
        return

    if input_node.Class() != "Read":
        nuke.message("Connect a Read node (not {})".format(input_node.Class()))
        return

    # Get input info
    file_path, first_frame, last_frame = get_input_info(input_node)

    if not file_path:
        nuke.message("Could not get file path from Read node")
        return

    # Get extension
    ext = get_file_extension(file_path)

    if node.knob('status_text'):
        node.knob('status_text').setValue("Processing: {} ({})".format(os.path.basename(file_path), ext))

    # Check if conversion is needed
    if ext in SUPPORTED_FORMATS:
        # No conversion needed - pass directly
        launch_path = resolve_first_frame_path(file_path, first_frame)
        print("[MikeMagic] Format supported, passing directly: {}".format(ext))

    elif ext in CONVERT_FORMATS:
        # Conversion needed
        print("[MikeMagic] Converting {} to PNG...".format(ext))
        success, png_path = convert_to_png(input_node, first_frame, last_frame, file_path, node)
        if not success:
            return
        launch_path = png_path

    else:
        # Unknown format - try passing directly
        launch_path = resolve_first_frame_path(file_path, first_frame)
        print("[MikeMagic] Unknown format {}, trying direct load".format(ext))

    # Launch MikeMagic
    launch_mike_magic(launch_path, node)


def on_launch_button():
    """Called when Launch button is pressed"""
    node = nuke.thisNode()
    convert_and_launch(node)


def create_mike_magic_node():
    """Create the MikeMagic node"""
    node = nuke.createNode("NoOp", inpanel=False)
    node.setName("MikeMagic")
    node.knob('label').setValue("")

    # Custom tab
    tab = nuke.Tab_Knob("mike_magic_tab", "MikeMagic")
    node.addKnob(tab)

    # Info
    info = nuke.Text_Knob("info", "",
        "<b>MikeMagic Launcher</b>\n\n"
        "1. Connect a Read node\n"
        "2. Click 'Launch MikeMagic'\n\n"
        "Supported: PNG, JPG, TIFF, MP4, MOV, etc.\n"
        "Auto-converts: EXR, DPX, CIN, HDR")
    node.addKnob(info)

    # Divider
    node.addKnob(nuke.Text_Knob("div1", "", ""))

    # Status
    status = nuke.String_Knob("status_text", "Status")
    status.setValue("Ready - Connect footage")
    status.setEnabled(False)
    node.addKnob(status)

    # Launch button
    launch_btn = nuke.PyScript_Knob("launch_btn", "Launch MikeMagic", "import MikeMagic; MikeMagic.on_launch_button()")
    node.addKnob(launch_btn)

    # Tile color (orange)
    node.knob('tile_color').setValue(0xFFAA00FF)

    print("[MikeMagic] Node created")
    return node
