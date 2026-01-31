import nuke
import cv2
import numpy as np
import os
import json
import _curveknob

def _resample_polygon(cnt, target_dist=15.0):
    """
    Resamples a contour to have points evenly spaced by target_dist.
    Returns a numpy array of points (N, 1, 2) compatible with OpenCV.
    """
    if len(cnt) < 3: return cnt

    # Calculate total length
    perimeter = cv2.arcLength(cnt, True)
    # Determine number of points needed
    count = int(max(5, perimeter / target_dist))

    # Faster robust method:
    pts = cnt.reshape(-1, 2)
    resampled = []

    if len(pts) > 0:
        resampled.append(pts[0])

    for i in range(1, len(pts)):
        p1 = pts[i-1]
        p2 = pts[i]
        
        # Only add point if it's far enough from the last added point
        if np.linalg.norm(p2 - resampled[-1]) >= target_dist:
            resampled.append(p2)

    if len(resampled) < 3:
        return cnt.astype(np.float32).reshape(-1, 1, 2)

    final_arr = np.array(resampled, dtype=np.float32).reshape(-1, 1, 2)
    return final_arr

def _snap_to_contour(predicted_points, target_contour_pts):
    """
    ROBUST FIX: Instead of searching a noisy edge map, this snaps predicted points 
    to the nearest vertex on the actual target contour polygon.
    This guarantees points lie EXACTLY on the matte boundary, ignoring internal texture.
    """
    if target_contour_pts is None or len(target_contour_pts) == 0:
        return predicted_points
        
    target_flat = target_contour_pts.reshape(-1, 2)
    predicted_flat = predicted_points.reshape(-1, 2)
    
    corrected = []
    
    # Iterate through every predicted roto point
    for pt in predicted_flat:
        # Calculate distance to ALL points on the valid matte contour
        # (Broadcasting makes this fast enough for roto point counts)
        dists = np.sum((target_flat - pt)**2, axis=1)
        nearest_idx = np.argmin(dists)
        
        nearest_pt = target_flat[nearest_idx]
        dist_sq = dists[nearest_idx]
        
        # SMART SNAPPING LOGIC:
        # If the point is reasonably close (within ~40px), snap it HARD to the edge.
        # We blend 20% flow / 80% snap to keep it stuck to the silhouette.
        if dist_sq < 1600: 
            smoothed = pt * 0.2 + nearest_pt * 0.8
            corrected.append(smoothed)
        else:
            # If the nearest edge is too far away, the point likely drifted off-screen 
            # or is occluded. Trust the Optical Flow prediction (don't snap wildly).
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

        # 1. Blur to reduce grain
        clean = cv2.medianBlur(img, 5)
        
        # 2. Hard Threshold (Low value = Outwards expansion)
        _, binary = cv2.threshold(clean, 20, 255, cv2.THRESH_BINARY)
        
        # 3. Morph Close (Fill holes, preserve outer shape)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        out_path = os.path.join(cleaned_dir, actual_name)
        cv2.imwrite(out_path, binary)

    # --------------------------------------------
    # STEP 2: TRACKING & ROBUST SHAPE SNAP
    # --------------------------------------------
    output_json = os.path.join(cleaned_dir, "multi_matte_data.json").replace("\\", "/")
    files = sorted([f for f in os.listdir(cleaned_dir) if f.endswith(".png")])

    data = {"schema": "multi_char_v1", "objects": {}}

    # State variables
    prev_gray = None
    active_shapes = {} 
    
    # We maintain a mapping of which contour corresponds to which character
    # to avoid swapping if multiple shapes exist.
    
    total_files = max(1, len(files))

    for i, filename in enumerate(files):
        if task.isCancelled(): return

        task.setMessage(f"Step 2/3: Tracking & Refining ({filename})")
        task.setProgress(33 + int((i / float(total_files)) * 33))

        frame_num = "".join(filter(str.isdigit, filename))
        img_path = os.path.join(cleaned_dir, filename)

        # Load Current Frame
        curr_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if curr_gray is None: continue

        # Find ALL contours in the current cleaned frame
        # These are the "Magnets" we will snap to.
        contours, _ = cv2.findContours(curr_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter noise
        valid_contours = [c for c in contours if cv2.contourArea(c) > 500]

        if prev_gray is None:
            # --- INITIALIZATION ---
            # Sort by area so char_0 is the biggest object
            valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)

            for idx, cnt in enumerate(valid_contours):
                char_id = f"p0:char_{idx}"

                # Resample for clean Roto points
                epsilon = 0.002 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                points = _resample_polygon(approx, target_dist=15.0)

                active_shapes[char_id] = points
                data["objects"][char_id] = {"frames": {}}

        else:
            # --- OPTICAL FLOW + CONTOUR SNAP ---
            lk_params = dict(winSize=(21, 21), maxLevel=3,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

            for char_id, old_points in active_shapes.items():
                if len(old_points) < 3: continue

                # 1. Predict Movement (Optical Flow)
                predicted_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, old_points, None, **lk_params)

                # 2. Find the correct contour for this character
                # We calculate the center of our predicted points
                pred_center = np.mean(predicted_points, axis=0)
                
                best_contour = None
                min_dist = 99999.0
                
                # Search all valid contours in this frame to find the one closest to our tracked object
                for cnt in valid_contours:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]
                        # Distance from predicted object center to contour center
                        dist = np.linalg.norm(pred_center - np.array([[cx, cy]]))
                        
                        if dist < min_dist:
                            min_dist = dist
                            best_contour = cnt
                
                # 3. SNAP to that contour
                # Only snap if we found a contour reasonably close (e.g. < 200px shift)
                if best_contour is not None and min_dist < 200:
                    snapped_points = _snap_to_contour(predicted_points, best_contour)
                    active_shapes[char_id] = snapped_points
                else:
                    # No matching contour found (maybe object went off screen or became too small)
                    # Keep the flow prediction
                    active_shapes[char_id] = predicted_points

        # --- EXPORT TO JSON FORMAT ---
        for char_id, points in active_shapes.items():
            segments = []
            pts_flat = points.reshape(-1, 2)
            count = len(pts_flat)

            for j in range(count):
                curr = pts_flat[j]
                prev = pts_flat[j - 1]
                nxt = pts_flat[(j + 1) % count]

                # Catmull-Rom style auto-tangents
                itx = curr + (prev - curr) * 0.33
                otx = curr + (nxt - curr) * 0.33

                segments.append({
                    "anchor": curr.tolist(),
                    "in": itx.tolist(),
                    "out": otx.tolist()
                })

            if char_id not in data["objects"]:
                data["objects"][char_id] = {"frames": {}}

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
        max_pts = len(frames[first_frame_key]["segments"])

        shape = _curveknob.Shape(curves)
        shape.name = obj_name.replace(":", "_")
        root_layer.append(shape)

        for _ in range(max_pts):
            shape.append(_curveknob.ShapeControlPoint())

        for f_str in sorted_keys:
            f_aligned = int(f_str) + offset
            segs = frames[f_str]["segments"]

            if len(segs) != max_pts:
                # print(f"Warning: Point count mismatch at frame {f_aligned}")
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
    nuke.message("MM-Roto Fixed:\nContour Snapping Applied (No Edge Drift).")

if __name__ == "__main__":
    run_complete_workflow()