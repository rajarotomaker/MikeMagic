"""
MM-Roto Nuke Listener
Receives export notifications from MM-Roto and create nodes
"""

import socket
import json
import threading
import nuke

HOST = 'localhost'
PORT = 9876
_server_running = False
_server_thread = None

def _create_read_node(file_path, frame_range):
    """
    Create Read Node and run the merged matte toolkit workflow.
    """
    try:
        #Create Read Node
        read_node = nuke.nodes.Read(file=file_path)
        read_node['first'].setValue(int(frame_range[0]))
        read_node['last'].setValue(int(frame_range[1]))

        read_node['reload'].execute() # force reload

        # Force format detection by evaluating the node
        # This ensures read_node.format() returns the correct image dimensions
        read_node.width()
        read_node.height()

        # Select the Read Node
        for n in nuke.allNodes():
            n.setSelected(False)
        read_node.setSelected(True)

        # Force Read node initialization - ensure files are actually loaded
        nuke.updateUI()

        print(f"[MM-Roto] Read created: {file_path} [{frame_range[0]} - {frame_range[1]}]")

        # Run the complete merged workflow
        try:
            from . import matte_toolkit_merged
            print("[MM-Roto] Running complete matte workflow...")
            matte_toolkit_merged.run_complete_workflow()
            print("[MM-Roto] Workflow Complete.")
        except Exception as e:
            print(f"[MM-Roto] Workflow failed: {e}")
            import traceback
            traceback.print_exc()

        return read_node
    except Exception as e:
        print(f"[MM-Roto] Import error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def _handle_client(client_socket):
    """ Parse incoming JSON and run on main thread """
    try:
        data   = client_socket.recv(4096).decode()
        message = json.loads(data)

        if message.get('action') == 'import_matte':
            file_path = message.get('file')
            frame_range = message.get('frame_range',[1,100])
            nuke.executeInMainThread(_create_read_node,(file_path, frame_range))
    except Exception as e:
        print(f"[MM-Roto] Listener error: {e}")
    finally:
        client_socket.close()

def _server_loop():
    """Socket server loop running in a daemon thread."""
    global _server_running

    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
        server_socket.bind((HOST, PORT))
        server_socket.listen(5)
        server_socket.settimeout(1.0)

        print(f"[MikeMagic] Listener active on {HOST}:{PORT}")
        
        while _server_running:
            try:
                client_socket , _ = server_socket.accept()
                _handle_client(client_socket)
            except socket.timeout:
                continue
            except Exception as e:
                if _server_running:
                    print(f"[MM-Roto] Server error:{e}")
        
        server_socket.close()
        print("[MM-Roto] Listener Stopped")

    except Exception as e:
        print(f"[MM-Roto] Listener failed to start:{e}")

def start_listener():
    """Public API: start the listener once"""
    global _server_running, _server_thread
    if _server_running:
        return
    _server_running =True
    _server_thread = threading.Thread(target=_server_loop, daemon=True)
    _server_thread.start()

def stop_listener():
    """Public API: stop the listener."""
    global _server_running
    _server_running = False
        