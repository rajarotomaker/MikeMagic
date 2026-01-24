import nuke
import cv2
import numpy as np
import os
import json
import _curveknob


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

    # Single processing dialog for all steps
    task = nuke.ProgressTask("MM-Roto: Processing Matte → Roto")

    # --------------------------------------------
    # STEP 1: Hard clean & erode sequence
    # --------------------------------------------
    for f in range(first, last + 1):
        if task.isCancelled():
            return

        local_progress = (f - first) / float(max(1, (last - first + 1)))
        task.setMessage("Step 1/3: Hard cleaning frame %d" % f)
        task.setProgress(int(local_progress * 33))

        # Resolve filename
        if "####" in file_template:
            actual_name = file_template.replace("####", str(f).zfill(4))
        elif "%" in file_template:
            actual_name = file_template % f
        else:
            actual_name = file_template

        curr_file = os.path.join(base_dir, actual_name).replace("\\", "/")
        if not os.path.exists(curr_file):
            continue

        img = cv2.imread(curr_file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        clean = cv2.medianBlur(img, 5)
        _, binary = cv2.threshold(clean, 240, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=erosion_pixels)
        final = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)

        out_path = os.path.join(cleaned_dir, actual_name)
        cv2.imwrite(out_path, final)

    # --------------------------------------------
    # STEP 2: Extract shapes and write JSON
    # --------------------------------------------
    output_json = os.path.join(cleaned_dir, "multi_matte_data.json").replace("\\", "/")
    files = sorted([f for f in os.listdir(cleaned_dir) if f.endswith(".png")])

    data = {"schema": "multi_char_v1", "objects": {}}

    total_files = max(1, len(files))
    for i, filename in enumerate(files):
        if task.isCancelled():
            return

        local_progress = i / float(total_files)
        task.setMessage("Step 2/3: Extracting contours (%s)" % filename)
        task.setProgress(33 + int(local_progress * 33))

        frame_num = "".join(filter(str.isdigit, filename))
        img_path = os.path.join(cleaned_dir, filename)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for idx, cnt in enumerate(contours):
            if cv2.contourArea(cnt) < 100:
                continue

            char_id = "p0:char_%d" % idx
            if char_id not in data["objects"]:
                data["objects"][char_id] = {"frames": {}}

            epsilon = 0.0005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)

            segments = []
            for j in range(len(approx)):
                curr = approx[j]
                prev = approx[j - 1]
                nxt = approx[(j + 1) % len(approx)]

                itx = curr + (prev - curr) * 0.33
                otx = curr + (nxt - curr) * 0.33

                segments.append({
                    "anchor": curr.tolist(),
                    "in": itx.tolist(),
                    "out": otx.tolist()
                })

            data["objects"][char_id]["frames"][frame_num] = {"segments": segments}

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
        if task.isCancelled():
            return

        task.setProgress(66 + int((obj_index / float(total_objs)) * 34))

        frames = obj_data["frames"]
        sorted_keys = sorted(frames.keys(), key=int)
        if not sorted_keys:
            continue

        offset = read_start - int(sorted_keys[0])
        max_pts = max(len(frames[f]["segments"]) for f in frames)

        shape = _curveknob.Shape(curves)
        shape.name = obj_name.replace(":", "_")
        root_layer.append(shape)

        for _ in range(max_pts):
            shape.append(_curveknob.ShapeControlPoint())

        for f_str in sorted_keys:
            f_aligned = int(f_str) + offset
            segs = frames[f_str]["segments"]
            num_segs = len(segs)

            for i_pt in range(max_pts):
                cp = shape[i_pt]
                s = segs[min(i_pt, num_segs - 1)]

                ax, ay = float(s["anchor"][0]), H - float(s["anchor"][1])
                itx, ity = float(s["in"][0]), H - float(s["in"][1])
                otx, oty = float(s["out"][0]), H - float(s["out"][1])

                cp.center.addPositionKey(f_aligned, (ax, ay, 0.0))
                cp.leftTangent.addPositionKey(f_aligned, (itx - ax, ity - ay, 0.0))
                cp.rightTangent.addPositionKey(f_aligned, (otx - ax, oty - ay, 0.0))

                for attr in (cp.center, cp.leftTangent, cp.rightTangent):
                    for dim in (0, 1):
                        cv = attr.getPositionAnimCurve(dim)
                        cv.curveType = 1

    curves.changed()
    task.setProgress(100)
    nuke.message("MM-Roto Auto Pipeline Complete!\nCleaned matte, extracted shapes and built Roto node.")

# Run the complete workflow automatically when this script is executed
if __name__ == "__main__":
    run_complete_workflow()
