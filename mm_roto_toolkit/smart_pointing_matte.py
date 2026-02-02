import nuke
import cv2
import numpy as np
import os

try:
    import _curveknob
except ImportError:
    pass

# --- Helper Functions (Same as before) ---

def _clean_matte_crisp(img):
    float_img = img.astype(float)
    contrast = 2.0 
    float_img = (float_img - 128) * contrast + 128
    float_img = np.clip(float_img, 0, 255).astype(np.uint8)
    
    denoised = cv2.GaussianBlur(float_img, (9, 9), 0)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((7, 7), np.uint8) 
    solid = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return solid

def _calculate_smooth_tangents(points):
    result = []
    pts = points.reshape(-1, 2)
    count = len(pts)
    
    for i in range(count):
        prev = pts[i-1]
        curr = pts[i]
        nxt = pts[(i+1) % count]
        
        chord = nxt - prev
        
        v1 = prev - curr
        v2 = nxt - curr
        angle = 180.0
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))

        if angle < 110.0: # Sharp Corner
            data = {"anchor": curr.tolist(), "in": curr.tolist(), "out": curr.tolist()}
        else:
            smoothing = 0.25 
            t_in = curr - (chord * smoothing)
            t_out = curr + (chord * smoothing)
            data = {"anchor": curr.tolist(), "in": t_in.tolist(), "out": t_out.tolist()}
            
        result.append(data)
    return result

def _align_contour_indices(prev_pts, curr_pts):
    if prev_pts is None: return curr_pts
    p0 = prev_pts[0, 0]
    curr_flat = curr_pts.reshape(-1, 2)
    dists = np.sum((curr_flat - p0)**2, axis=1)
    best_idx = np.argmin(dists)
    aligned = np.roll(curr_pts, -best_idx, axis=0)
    return aligned

def _resample_curve_to_n(contour, n_points):
    pts = contour.reshape(-1, 2).astype(float)
    closed_pts = np.vstack((pts, pts[0]))
    diffs = closed_pts[1:] - closed_pts[:-1]
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    total_len = cum_dists[-1]
    target_dists = np.linspace(0, total_len, n_points, endpoint=False)
    new_x = np.interp(target_dists, cum_dists, closed_pts[:, 0])
    new_y = np.interp(target_dists, cum_dists, closed_pts[:, 1])
    resampled = np.column_stack((new_x, new_y)).astype(np.float32)
    return resampled.reshape(-1, 1, 2)

# --- Main Workflow ---

def run_complete_workflow():
    # --------------------------------------------
    # STEP 0: Setup
    # --------------------------------------------
    try:
        read = nuke.selectedNode()
        if read.Class() != "Read": raise ValueError
    except:
        nuke.message("Select the Read node.")
        return

    raw_path = read['file'].value()
    base_dir = os.path.dirname(raw_path)
    file_name = os.path.basename(raw_path)
    
    cleaned_dir = os.path.join(base_dir, "mm_optimized_mattes").replace("\\", "/")
    if not os.path.exists(cleaned_dir): os.makedirs(cleaned_dir)

    first = int(read['first'].value())
    last = int(read['last'].value())
    
    task = nuke.ProgressTask("MM-Roto: Adaptive Shaping")

    # --------------------------------------------
    # STEP 1: CLEAN & ANALYZE
    # --------------------------------------------
    processed_paths = []
    max_perimeter = 0.0
    
    for f in range(first, last + 1):
        if task.isCancelled(): return
        task.setMessage(f"Analyzing Frame {f}")
        task.setProgress(int((f-first)/(last-first+1) * 30))

        if "####" in file_name: actual = file_name.replace("####", str(f).zfill(4))
        elif "%" in file_name: actual = file_name % f
        else: actual = file_name

        in_path = os.path.join(base_dir, actual).replace("\\", "/")
        out_path = os.path.join(cleaned_dir, actual).replace("\\", "/")
        
        if not os.path.exists(in_path): 
            processed_paths.append(None)
            continue

        img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        if img is None: 
            processed_paths.append(None)
            continue

        clean = _clean_matte_crisp(img)
        cv2.imwrite(out_path, clean)
        processed_paths.append(out_path)
        
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(cnt, True)
            if peri > max_perimeter:
                max_perimeter = peri

    if max_perimeter == 0:
        nuke.message("No matte content found.")
        return

    TARGET_POINT_COUNT = int(max(10, max_perimeter / 40.0))
    print(f"Global Target Points calculated: {TARGET_POINT_COUNT}")

    # --------------------------------------------
    # STEP 2: GENERATE SHAPES
    # --------------------------------------------
    roto_animation = {} 
    prev_points_array = None 
    
    for i, f_path in enumerate(processed_paths):
        if task.isCancelled(): return
        current_frame = first + i
        task.setMessage(f"Solving Frame {current_frame}")
        task.setProgress(30 + int((i/len(processed_paths)) * 60))
        
        if f_path is None: continue
        
        img = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            prev_points_array = None 
            continue
            
        raw_cnt = max(contours, key=cv2.contourArea)
        resampled = _resample_curve_to_n(raw_cnt, TARGET_POINT_COUNT)
        
        if prev_points_array is not None:
            aligned = _align_contour_indices(prev_points_array, resampled)
        else:
            aligned = resampled
            
        prev_points_array = aligned
        
        roto_animation[current_frame] = _calculate_smooth_tangents(aligned)

    # --------------------------------------------
    # STEP 3: BUILD ROTO (Corrected for Zero Feather)
    # --------------------------------------------
    task.setMessage("Building Roto Node...")
    task.setProgress(95)
    
    roto = nuke.createNode("Roto")
    roto['cliptype'].setValue("no clip")
    curves = roto['curves']
    root = curves.rootLayer
    
    shape = _curveknob.Shape(curves)
    shape.name = "AutoMatte_Hard"
    root.append(shape)
    
    for _ in range(TARGET_POINT_COUNT):
        shape.append(_curveknob.ShapeControlPoint())
        
    H = float(nuke.root().height())
    frames_sorted = sorted(roto_animation.keys())
    
    for frame in frames_sorted:
        points_data = roto_animation[frame]
        
        for i, pt_data in enumerate(points_data):
            cp = shape[i]
            
            ax, ay = pt_data['anchor']
            ix, iy = pt_data['in']
            ox, oy = pt_data['out']
            
            # --- TANGENT CALCULATION ---
            # Calculate the relative vectors for the main curve
            left_vec = (ix - ax, (H - iy) - (H - ay), 0)
            right_vec = (ox - ax, (H - oy) - (H - ay), 0)
            
            # 1. Set Main Curve Tangents
            cp.center.addPositionKey(frame, (ax, H - ay, 0))
            cp.leftTangent.addPositionKey(frame, left_vec)
            cp.rightTangent.addPositionKey(frame, right_vec)
            
            # 2. FORCE FEATHER MATCH
            # To remove the feather visibility, the feather curve must match the main curve.
            # Feather Center = (0,0,0) offset (sits on main point)
            # Feather Tangents = SAME vectors as Main Tangents (curves with main shape)
            cp.featherCenter.addPositionKey(frame, (0, 0, 0))
            cp.featherLeftTangent.addPositionKey(frame, left_vec)
            cp.featherRightTangent.addPositionKey(frame, right_vec)

    curves.changed()
    task.setProgress(100)
    print("MM-Roto: Complete (Exact Hard Edge).")

if __name__ == "__main__":
    run_complete_workflow()
    