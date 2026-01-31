import nuke
import cv2
import numpy as np
import os
import json

try:
    import _curveknob
except ImportError:
    pass

def _clean_matte_crisp(img):
    """
    Enhance whites/blacks to create a solid, crisp matte.
    """
    # Contrast Stretch
    float_img = img.astype(float)
    contrast = 2.0 
    float_img = (float_img - 128) * contrast + 128
    float_img = np.clip(float_img, 0, 255).astype(np.uint8)
    
    # Denoise
    denoised = cv2.GaussianBlur(float_img, (9, 9), 0)
    
    # Otsu Threshold + Morph Close
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morph kernel to fill small gaps
    kernel = np.ones((7, 7), np.uint8) 
    solid = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return solid

def _format_linear_points(points):
    """
    Returns points with ZERO tangents. 
    This prevents any curvature ('feathers') and results in a hard poly-line.
    """
    result = []
    pts = points.reshape(-1, 2)
    
    for curr in pts:
        # By setting 'in' and 'out' equal to 'anchor', the tangent length is 0.
        # This forces Nuke to treat the segment as a straight line.
        data = {
            "anchor": curr.tolist(), 
            "in": curr.tolist(), 
            "out": curr.tolist()
        }
        result.append(data)
    return result

def _align_contour_indices(prev_pts, curr_pts):
    """
    Rotates the array of current points so that index 0 
    is closest to index 0 of the previous frame.
    Prevents the shape from 'spinning'.
    """
    if prev_pts is None: return curr_pts
    
    p0 = prev_pts[0, 0] # (x, y)
    
    # Find point in curr_pts closest to p0
    curr_flat = curr_pts.reshape(-1, 2)
    dists = np.sum((curr_flat - p0)**2, axis=1)
    best_idx = np.argmin(dists)
    
    # Rotate array
    aligned = np.roll(curr_pts, -best_idx, axis=0)
    return aligned

def _resample_curve_to_n(contour, n_points):
    """
    Resamples a contour to exactly n_points spaced equally by arc length.
    Returns array of shape (n_points, 1, 2) to match CV2 format.
    """
    # Ensure standard numpy shape (N, 2)
    pts = contour.reshape(-1, 2).astype(float)
    
    # Calculate distance between each consecutive point
    # np.vstack puts the first point at the end to close the loop for measurement
    closed_pts = np.vstack((pts, pts[0]))
    
    # Calculate Euclidean distance between points
    diffs = closed_pts[1:] - closed_pts[:-1]
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    
    # Calculate cumulative distance (0.0, dist_0, dist_0+dist_1, ...)
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    total_len = cum_dists[-1]
    
    # Create the target distances (equally spaced)
    target_dists = np.linspace(0, total_len, n_points, endpoint=False)
    
    # Interpolate X and Y coordinates separately based on target distances
    new_x = np.interp(target_dists, cum_dists, closed_pts[:, 0])
    new_y = np.interp(target_dists, cum_dists, closed_pts[:, 1])
    
    # Combine and reshape to (N, 1, 2)
    resampled = np.column_stack((new_x, new_y)).astype(np.float32)
    return resampled.reshape(-1, 1, 2)


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
    
    task = nuke.ProgressTask("MM-Roto: Linear Shaping")

    # --------------------------------------------
    # STEP 1: CLEAN & ANALYZE (Determine Point Count)
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

        # Clean
        clean = _clean_matte_crisp(img)
        cv2.imwrite(out_path, clean)
        processed_paths.append(out_path)
        
        # Measure Perimeter
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(cnt, True)
            if peri > max_perimeter:
                max_perimeter = peri

    if max_perimeter == 0:
        nuke.message("No matte content found.")
        return

    # Calculate Target Points
    TARGET_POINT_COUNT = int(max(10, max_perimeter / 20.0))
    print(f"Global Target Points calculated: {TARGET_POINT_COUNT}")

    # --------------------------------------------
    # STEP 2: GENERATE SHAPES (Resample per frame)
    # --------------------------------------------
    
    roto_animation = {} # { frame: [point_data] }
    prev_points_array = None # For index alignment
    
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
        
        # RESAMPLE
        resampled = _resample_curve_to_n(raw_cnt, TARGET_POINT_COUNT)
        
        # ALIGN
        if prev_points_array is not None:
            aligned = _align_contour_indices(prev_points_array, resampled)
        else:
            aligned = resampled
            
        prev_points_array = aligned
        
        # CHANGED: Use linear formatting instead of smooth tangents
        roto_animation[current_frame] = _format_linear_points(aligned)

    # --------------------------------------------
    # STEP 3: BUILD ROTO
    # --------------------------------------------
    task.setMessage("Building Roto Node...")
    task.setProgress(95)
    
    roto = nuke.createNode("Roto")
    roto['cliptype'].setValue("no clip")
    curves = roto['curves']
    root = curves.rootLayer
    
    shape = _curveknob.Shape(curves)
    shape.name = "AutoMatte_Linear"
    root.append(shape)
    
    # Create Points
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
            
            # Keyframe
            cp.center.addPositionKey(frame, (ax, H - ay, 0))
            
            # These will now be (0,0,0), creating a hard corner
            cp.leftTangent.addPositionKey(frame, (ix - ax, (H - iy) - (H - ay), 0))
            cp.rightTangent.addPositionKey(frame, (ox - ax, (H - oy) - (H - ay), 0))

    curves.changed()
    task.setProgress(100)
    print("MM-Roto: Complete (Linear).")

if __name__ == "__main__":
    run_complete_workflow()