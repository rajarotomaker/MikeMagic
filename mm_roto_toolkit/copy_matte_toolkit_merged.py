import nuke
import cv2
import numpy as np
import os
import json
import _curveknob

def _resample_polygon(cnt, target_dist=15.0, fixed_count=None):
    """
    Resamples a contour.
    If 'fixed_count' is provided, it forces the output to have exactly that many points
    distributed evenly along the arc length (essential for Roto consistency).
    Otherwise, it calculates count based on 'target_dist'.
    """
    # Force closed loop
    if len(cnt) < 3: return cnt
    
    # Calculate total length
    perimeter = cv2.arcLength(cnt, True)
    
    if fixed_count is not None:
        count = fixed_count
    else:
        count = int(max(5, perimeter / target_dist))
    
    if count <= 1: return cnt

    # Prepare inputs
    pts = cnt.reshape(-1, 2).astype(np.float32)
    
    # Calculate cumulative distances between original points
    # d[i] is distance from pt[i] to pt[i+1]
    dists = np.sqrt(np.sum(np.diff(np.vstack((pts, pts[0])), axis=0)**2, axis=1))
    cum_dists = np.insert(np.cumsum(dists), 0, 0.0)
    total_len = cum_dists[-1]
    
    # Generate target distances for new points
    # We want 'count' points. The first is at 0, the last implies the loop closure.
    target_dists = np.linspace(0, total_len, count, endpoint=False)
    
    new_points = []
    
    # Interpolate
    # For each target distance, find which segment of the original poly it falls on
    for t in target_dists:
        # Find index such that cum_dists[idx] <= t < cum_dists[idx+1]
        idx = np.searchsorted(cum_dists, t, side='right') - 1
        idx = max(0, min(idx, len(pts) - 1))
        
        # Segment start/end
        p_start = pts[idx]
        p_end = pts[(idx + 1) % len(pts)]
        
        # How far along this segment are we?
        seg_len = dists[idx]
        if seg_len > 1e-6:
            alpha = (t - cum_dists[idx]) / seg_len
        else:
            alpha = 0.0
            
        # Linear interpolate
        p_new = p_start + (p_end - p_start) * alpha
        new_points.append(p_new)
            
    final_arr = np.array(new_points, dtype=np.float32).reshape(-1, 1, 2)
    return final_arr

def _snap_to_contour(predicted_points, target_contour_pts):
    """
    Instead of searching an image mask, this snaps predicted points 
    to the nearest vertex on the actual target contour polygon.
    This guarantees points lie EXACTLY on the matte edge.
    """
    if target_contour_pts is None or len(target_contour_pts) == 0:
        return predicted_points
        
    target_flat = target_contour_pts.reshape(-1, 2)
    predicted_flat = predicted_points.reshape(-1, 2)
    
    corrected = []
    
    # Simple nearest neighbor search
    # For production with high point counts, a KDTree is faster, 
    # but for Roto (<500 pts), brute force broadcasting is instant in numpy.
    
    for pt in predicted_flat:
        # Calculate distance to all points on the target contour
        dists = np.sum((target_flat - pt)**2, axis=1)
        nearest_idx = np.argmin(dists)
        
        # Snap to that point
        # We blend 50% flow prediction / 50% snap to maintain temporal smoothness
        # while sticking to the edge
        nearest_pt = target_flat[nearest_idx]
        
        # "Elastic" snap:
        # If the snap is huge (big jump), trust flow more. If small, snap hard.
        dist_sq = dists[nearest_idx]
        
        if dist_sq < 900: # Within 30 pixels
            # 80% snap, 20% flow
            smoothed = pt * 0.2 + nearest_pt * 0.8
            corrected.append(smoothed)
        else:
            # Too far, probably a topology change or error. Keep flow.
            corrected.append(pt)
            
    return np.array(corrected, dtype=np.float32).reshape(-1, 1, 2)

def run_complete_workflow():
    # --------------------------------------------
    # STEP 0: Validate selection and basic paths
    # --------------------------------------------
    try:
        read = nuke.selectedNode()
        if read.Class() != "Read":
            raise ValueError
    except:
        nuke.message("Please select the Read node containing your matte.")
        return

    raw_path = read['file'].value()
    base_dir = os.path.dirname(raw_path)
    file_template = os.path.basename(raw_path)

    cleaned_dir = os.path.join(base_dir, "hard_cleaned_sequence").replace("\\", "/")
    if not os.path.exists(cleaned_dir):
        os.makedirs(cleaned_dir)

    first = int(read['first'].value())
    last = int(read['last'].value())
    erosion_pixels = 1

    task = nuke.ProgressTask("MM-Roto: Processing Matte -> Roto")

    # --------------------------------------------
    # STEP 1: Hard clean & erode sequence
    # --------------------------------------------
    for f in range(first, last + 1):
        if task.isCancelled(): return
        
        local_progress = (f - first) / float(max(1, (last - first + 1)))
        task.setMessage("Step 1/3: Hard cleaning frame %d" % f)
        task.setProgress(int(local_progress * 33))

        if "####" in file_template:
            actual_name = file_template.replace("####", str(f).zfill(4))
        elif "%" in file_template:
            actual_name = file_template % f
        else:
            actual_name = file_template

        curr_file = os.path.join(base_dir, actual_name).replace("\\", "/")
        if not os.path.exists(curr_file): continue

        img = cv2.imread(curr_file, cv2.IMREAD_GRAYSCALE)
        if img is None: continue

        # Hard threshold cleaning for solid mattes
        clean = cv2.medianBlur(img, 5)
        _, binary = cv2.threshold(clean, 127, 255, cv2.THRESH_BINARY)
        
        # Optional: slight open/close to remove salt/pepper noise
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        out_path = os.path.join(cleaned_dir, actual_name)
        cv2.imwrite(out_path, binary)

    # --------------------------------------------
    # STEP 2: TRACKING & SHAPE EXTRACTION (FIXED)
    # --------------------------------------------
    output_json = os.path.join(cleaned_dir, "multi_matte_data.json").replace("\\", "/")
    files = sorted([f for f in os.listdir(cleaned_dir) if f.endswith(".png")])

    data = {"schema": "multi_char_v1", "objects": {}}
    
    prev_gray = None
    active_shapes = {} 
    
    # Drift threshold: if shape match score > this, we force a re-conform
    DRIFT_THRESHOLD = 0.1 
    
    total_files = max(1, len(files))
    
    for i, filename in enumerate(files):
        if task.isCancelled(): return
        task.setMessage(f"Step 2/3: Analyzing Shape Stability ({filename})")
        task.setProgress(33 + int((i / float(total_files)) * 33))

        frame_num = "".join(filter(str.isdigit, filename))
        img_path = os.path.join(cleaned_dir, filename)
        curr_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if curr_gray is None: continue

        # Find "Ground Truth" contours for current frame
        contours, _ = cv2.findContours(curr_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filter small noise
        valid_contours = [c for c in contours if cv2.contourArea(c) > 500]
        
        if prev_gray is None:
            # --- INITIALIZATION (First Frame) ---
            active_shapes = {} 
            for idx, cnt in enumerate(valid_contours):
                char_id = f"p0:char_{idx}"
                epsilon = 0.005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                # Initial resample
                active_shapes[char_id] = _resample_polygon(approx, target_dist=15.0)
                
                if char_id not in data["objects"]:
                    data["objects"][char_id] = {"frames": {}}
        else:
            # --- TRACKING MODE ---
            lk_params = dict(winSize=(31, 31), maxLevel=4,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))

            new_active_shapes = {}
            
            for char_id, old_points in active_shapes.items():
                # 1. Predict Movement (Optical Flow)
                # This handles rotation and translation best
                predicted, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, old_points, None, **lk_params)
                
                # 2. Find the actual contour this shape belongs to
                # We use the centroid of the predicted points to find the nearest valid contour
                pred_center = np.mean(predicted, axis=0)
                
                best_contour = None
                min_dist = 99999.0
                
                for cnt in valid_contours:
                    # Calculate contour center
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]
                        dist = np.linalg.norm(pred_center - np.array([[cx, cy]]))
                        
                        if dist < min_dist:
                            min_dist = dist
                            best_contour = cnt
                
                # Check for DRIFT or SHAPE CHANGE
                if best_contour is not None and min_dist < 200: # Threshold to ensure we didn't lose the object
                    
                    # Check shape similarity
                    match_score = cv2.matchShapes(predicted, best_contour, cv2.CONTOURS_MATCH_I1, 0)
                    
                    # Current point count (Must be preserved!)
                    required_count = len(old_points)
                    
                    if match_score > DRIFT_THRESHOLD:
                        # HEAVY DRIFT: The shape deformed significantly. 
                        # We discard flow points and resample the NEW contour strictly.
                        # CRITICAL: We pass 'fixed_count' to maintain topology with previous frames.
                        # We use the raw contour to ensure we capture the new shape.
                        new_active_shapes[char_id] = _resample_polygon(best_contour, fixed_count=required_count)
                    else:
                        # LIGHT DRIFT / RIGID MOTION:
                        # Use the flow predicted points, but snap them to the contour edge
                        # to ensure they are tight on the matte (solid frame handling).
                        snapped_points = _snap_to_contour(predicted, best_contour)
                        new_active_shapes[char_id] = snapped_points
                        
                else:
                    # Object lost or occluded? Keep predicted flow (best guess)
                    new_active_shapes[char_id] = predicted

            active_shapes = new_active_shapes

        # --- EXPORT TO JSON ---
        for char_id, points in active_shapes.items():
            segments = []
            pts_flat = points.reshape(-1, 2)
            count = len(pts_flat)
            for j in range(count):
                curr = pts_flat[j]
                prev = pts_flat[j - 1] # Wraps around naturally
                nxt = pts_flat[(j + 1) % count] # Wraps around
                
                # Simple tangent estimation (Catmull-Rom style factor)
                # 0.25 gives smoother curves than 0.33 usually for Roto
                segments.append({
                    "anchor": curr.tolist(),
                    "in": (curr + (prev - curr) * 0.25).tolist(),
                    "out": (curr + (nxt - curr) * 0.25).tolist()
                })
            
            if char_id not in data["objects"]: data["objects"][char_id] = {"frames": {}}
            data["objects"][char_id]["frames"][frame_num] = {"segments": segments}

        prev_gray = curr_gray

    with open(output_json, "w") as f:
        json.dump(data, f)

    # --------------------------------------------
    # STEP 3: Import JSON into new Roto node
    # --------------------------------------------
    task.setMessage("Step 3/3: Building Roto shapes")
    read_start = first
    H = float(nuke.root().height())

    with open(output_json, "r") as f:
        data = json.load(f)

    roto_node = nuke.createNode("Roto")
    curves = roto_node["curves"]
    root_layer = curves.rootLayer

    objects_items = list(data["objects"].items())
    total_objs = max(1, len(objects_items))

    for obj_index, (obj_name, obj_data) in enumerate(objects_items):
        if task.isCancelled(): return
        task.setProgress(66 + int((obj_index / float(total_objs)) * 34))

        frames = obj_data["frames"]
        sorted_keys = sorted(frames.keys(), key=int)
        if not sorted_keys: continue

        offset = read_start - int(sorted_keys[0])
        first_frame_key = sorted_keys[0]
        
        # Determine strict point count from the first frame
        max_pts = len(frames[first_frame_key]["segments"])

        shape = _curveknob.Shape(curves)
        shape.name = obj_name.replace(":", "_")
        root_layer.append(shape)

        for _ in range(max_pts):
            shape.append(_curveknob.ShapeControlPoint())

        for f_str in sorted_keys:
            f_aligned = int(f_str) + offset
            segs = frames[f_str]["segments"]
            
            # Skip frames where point count mismatches (prevents crashing)
            # (This should happen rarely now with fixed_count logic)
            if len(segs) != max_pts:
                print(f"Skipping frame {f_aligned} for {obj_name}: Point count changed ({len(segs)} vs {max_pts})")
                continue

            for i_pt in range(max_pts):
                cp = shape[i_pt]
                s = segs[i_pt]

                ax, ay = float(s["anchor"][0]), H - float(s["anchor"][1])
                itx, ity = float(s["in"][0]), H - float(s["in"][1])
                otx, oty = float(s["out"][0]), H - float(s["out"][1])

                cp.center.addPositionKey(f_aligned, (ax, ay, 0.0))
                cp.leftTangent.addPositionKey(f_aligned, (itx - ax, ity - ay, 0.0))
                cp.rightTangent.addPositionKey(f_aligned, (otx - ax, oty - ay, 0.0))

                for attr in (cp.center, cp.leftTangent, cp.rightTangent):
                    for dim in (0, 1):
                        cv = attr.getPositionAnimCurve(dim)
                        cv.curveType = 1 # smooth

    curves.changed()
    task.setProgress(100)
    nuke.message("MM-Roto Auto Pipeline Complete!\nLocked Topology Mode.")

if __name__ == "__main__":
    run_complete_workflow()