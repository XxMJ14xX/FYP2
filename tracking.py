import os
import time
import pathlib
import cv2
import numpy as np
import traceback
from periphery import PWM
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.adapters import detect
from PID import PID

controller = PID(0.00005, 0, 0)

# Open PWM pin
pwm = PWM(1, 0)

pwm.frequency = 50
pwm.duty_cycle = 0.95

pwm.enable()

time.sleep(2)

# Set camera resolution and FPS accordingly
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Specify the TensorFlow model
script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, 'ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite')

# Initialize the TF interpreter
interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()

input_size = common.input_size(interpreter)
in_w, in_h = input_size

output_details = interpreter.get_output_details()

try:
    while True:
        # Measure inference time
        st = time.perf_counter_ns()
        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame
        image = cv2.resize(frame, input_size)

        # Run an inference
        common.set_input(interpreter, image)
        interpreter.invoke()
        
        classes = classify.get_classes(interpreter, top_k=1)

        objs = detect.get_objects(interpreter, 0.4, [1, 1])

        output_tensor = interpreter.get_tensor(output_details[0]['index'])
        
        num_detections = len(output_tensor[0])

        img_h, img_w, c = image.shape

        x_scale = img_w/in_w
        y_scale = img_h/in_h
        print(x_scale)
        print(y_scale)

        for obj in objs:
            print("obj")
            print('  id:    ', obj.id)
            print('  score: ', obj.score)
            print('  bbox:  ', obj.bbox)

            x1 = int(obj.bbox.xmin * x_scale)
            y1 = int(obj.bbox.ymin * y_scale)

            x2 = int(obj.bbox.xmax * x_scale)
            y2 = int(obj.bbox.ymax * y_scale)

            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (obj.bbox.xmin*2, int(obj.bbox.ymin*1.5)), (obj.bbox.xmax*2, int(obj.bbox.ymax*1.5)), (0, 255, 0), 2)

        print("Inference Time: ", (time.perf_counter_ns() - st) * 1e-6)

        if len(objs) >= 1:
            # Get center of bbox and center of frame
            center_frame = frame.shape[1] / 2
            
            #center_obj = (objs[0].bbox.xmin + objs[0].bbox.xmax) / 2
            center_obj = (x1 + x2)/2

            # Get the offset and correction from controller
            error = center_obj - center_frame
            corr = controller(error)

            # Update PWM
            pwm.duty_cycle = np.clip(pwm.duty_cycle + corr, 0.9, 0.95)

            print(corr, error, pwm.duty_cycle)

        # Display the output frame on the screen
        cv2.imshow('Object Detection', frame)
        #pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #pil_image.show()
        if cv2.waitKey(1) == ord('q'):
            break
except KeyboardInterrupt:
    pass
# except Exception as e:
#     print("Error", e)
#     traceback.print_exc()

pwm.close()

# Release the VideoCapture object and close the window
cap.release()
cv2.destroyAllWindows()

print('ciao ciao')
