CWSB microscope detection model: cwsb_micro_det_v1

Classes (id: name): 0: egg, 1: frass, 2: hole, 3: larva, 4: ridge
Input size: 512 x 512.  Suggested confidence: 0.25 (raise it for fewer false alarms, lower it to miss less).

Run on a Raspberry Pi (Python):
    pip install ultralytics
    from ultralytics import YOLO
    model = YOLO('cwsb_micro_det_v1_ncnn_model')        # fastest on a Pi; or 'cwsb_micro_det_v1.pt' / 'cwsb_micro_det_v1.onnx'
    r = model.predict('frame.jpg', imgsz=512, conf=0.25)[0]
    for c, s, box in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy):
        print(model.names[int(c)], float(s), box.tolist())

Use the class NAME for the alarm rule, not the number.
Test results are in model_info.json. They come from a very small test set.
