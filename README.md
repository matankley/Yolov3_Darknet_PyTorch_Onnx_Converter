# Yolov3_Darknet_PyTorch_Onnx_Converter
This Repository allows to convert *.weights file of darknet format to *.pt (pytorch format) and *.onnx (ONNX format).
Based on ultralytics repository (archive branch).This module converts *.weights files to *.pt and to *.onnx
    
Please keep in mind to delete the next fields from cfg file because the parser doesnt support them and because they dont impact inference:

    1. jitter
    2. nms_threshold
    3. threshold

Install requirements:

    python -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt

The command for conversion is:
    
    python converter.py yolov3.cfg yolov3.weights 1024 1024
