"""
Nuke Integration - Export to Nuke feature
"""
import socket
import json
import os

# Nuke socket configuration
NUKE_HOST = 'localhost'
NUKE_PORT = 9876

def send_to_nuke(file_path, frame_range, colorspace='linear'):
    '''Send export info to nuke via socket'''
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect((NUKE_HOST, NUKE_PORT))

        data = {
            "action": "import_matte",
            "file": file_path,
            "frame_range": frame_range,
            "colorspace": colorspace
        }
        print(file_path)
        sock.sendall(json.dumps(data).encode())
        sock.close()
        return True
    except Exception as e:
        print(f"Send_to_nuke failed: {e}")
        return False

def patch_export_dialog():
    """Add 'Send to Nuke' checkbox to export dialog"""
    from sammie.export_dialog import ExportDialog
    from PySide6.QtWidgets import QCheckBox, QMessageBox, QGroupBox

    original_init = ExportDialog.__init__
    original_start_export = ExportDialog._start_export
    original_export_finished = ExportDialog._export_finished
    
    def new_init(self, parent_window=None):
        original_init(self, parent_window)
        
        self.send_to_nuke_checkbox = QCheckBox("Send to Nuke after export")
        self.send_to_nuke_checkbox.setChecked(False)
        self._nuke_export_info = None
        
        for widget in self.findChildren(QGroupBox):
            if widget.title() == "Format & Settings":
                layout = widget.layout()
                row_count = layout.rowCount()
                layout.insertRow(row_count - 1, "", self.send_to_nuke_checkbox)
                break
        
        print("Nuke checkbox added to Export Dialog")
    
    def new_start_export(self):
        """Capture export info before starting export"""
        # Call original to create the worker
        original_start_export(self)

        # Now capture the info from the newly created worker
        if hasattr(self, 'export_worker') and self.export_worker:
            try:
                from sammie.export_workers import SequenceExportWorker, VideoExportWorker

                first_frame = self.export_worker.start_frame
                last_frame = self.export_worker.end_frame

                # Handle different worker types
                if isinstance(self.export_worker, SequenceExportWorker):
                    # For sequence exports (EXR, PNG)
                    output_dir = self.export_worker.settings.output_dir
                    base_filename = self.export_worker.base_filename
                    format_id = self.export_worker.settings.format_id

                    if format_id == 'exr':
                        output_path = os.path.join(output_dir, f"{base_filename}.%04d.exr")
                    elif format_id == 'png':
                        output_path = os.path.join(output_dir, f"{base_filename}.%04d.png")
                    else:
                        output_path = os.path.join(output_dir, base_filename)

                    # Adjust frame range to match actual file numbering
                    # Files are named with (first_frame + frame_num), so we need to offset
                    from sammie import sammie
                    first_frame = first_frame + sammie.VideoInfo.first_frame
                    last_frame = last_frame + sammie.VideoInfo.first_frame

                elif isinstance(self.export_worker, VideoExportWorker):
                    # For video exports - use first output path
                    output_path = self.export_worker.output_paths[0]
                else:
                    print(f"Unknown worker type: {type(self.export_worker)}")
                    return

                output_path = output_path.replace('\\', '/')

                self._nuke_export_info = {
                    'path': output_path,
                    'frame_range': [first_frame, last_frame]
                }
                print(f"Captured export info: {output_path}, frames {first_frame}-{last_frame}")
            except Exception as e:
                print(f"Could not capture export info: {e}")
                import traceback
                traceback.print_exc()
    
    def new_export_finished(self, success, message):
        print(f"_export_finished called: success={success}")
        
        # Call original first
        original_export_finished(self, success, message)
        
        if success and hasattr(self, 'send_to_nuke_checkbox') and self.send_to_nuke_checkbox.isChecked():
            print("Attempting to send to Nuke...")
            
            if hasattr(self, '_nuke_export_info') and self._nuke_export_info:
                output_path = self._nuke_export_info['path']
                frame_range = self._nuke_export_info['frame_range']
                
                print(f"Sending: {output_path}")
                print(f"Frames: {frame_range[0]}-{frame_range[1]}")
                
                if send_to_nuke(output_path, frame_range):
                    print("Successfully sent to Nuke")
                    QMessageBox.information(self, "Sent to Nuke", 
                        f"Exported to Nuke!\n Nuke will freeze untill RotoScoping \nFile: {output_path}\nFrames: {frame_range[0]}-{frame_range[1]}")
                else:
                    print("Failed to send to Nuke")
                    QMessageBox.warning(self, "Nuke Not Running", 
                        "Could not connect to Nuke.\n\nMake sure Nuke is running with the listener script active.")
            else:
                print("No export info found")
        else:
            print("Skipping Nuke send")
    
    ExportDialog.__init__ = new_init
    ExportDialog._start_export = new_start_export
    ExportDialog._export_finished = new_export_finished
    print("Nuke integration patch applied")


# Auto-apply patch
patch_export_dialog()
