#!pip install mysql-connector-python
import tkinter as tk
from tkinter import messagebox
import mysql.connector

# MySQL Connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Mextscholar@2007",
    database="sambhashana"
)

cursor = db.cursor()

root = tk.Tk()
root.title("Login / Sign Up")
root.geometry("400x350")

# Switch between Login & Signup
def show_login():
    signup_frame.pack_forget()
    login_frame.pack()

def show_signup():
    login_frame.pack_forget()
    signup_frame.pack()

# Login function
def login():
    user = login_username.get()
    pwd = login_password.get()

    cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (user, pwd))
    result = cursor.fetchone()

    if result:
        messagebox.showinfo("Success", "Login Successful!")
        root.destroy()
        open_sambhashana()
    else:
        messagebox.showerror("Error", "Invalid Username or Password")

# Signup function
def signup():
    user = signup_username.get()
    email = signup_email.get()
    pwd = signup_password.get()

    try:
        cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                       (user, email, pwd))
        db.commit()
        messagebox.showinfo("Success", "Signup Successful! Please login.")
        show_login()
    except:
        messagebox.showerror("Error", "Username already exists!")
"""
# Sambhashana GUI (next window)
#def open_sambhashana():
window = tk.Tk()
window.title("Sambhashana")
window.geometry("600x400")
tk.Label(window, text="Welcome to Sambhashana!", font=("Arial", 20)).pack(pady=50)
window.mainloop()"""

# GUI Layouts
login_frame = tk.Frame(root)
signup_frame = tk.Frame(root)

# LOGIN UI
tk.Label(login_frame, text="Login", font=("Arial", 18)).pack(pady=10)

tk.Label(login_frame, text="Username").pack()
login_username = tk.Entry(login_frame)
login_username.pack()

tk.Label(login_frame, text="Password").pack()
login_password = tk.Entry(login_frame, show="*")
login_password.pack()

tk.Button(login_frame, text="Login", command=login).pack(pady=10)
tk.Button(login_frame, text="Create account", command=show_signup).pack()

# SIGNUP UI
tk.Label(signup_frame, text="Sign Up", font=("Arial", 18)).pack(pady=10)

tk.Label(signup_frame, text="Username").pack()
signup_username = tk.Entry(signup_frame)
signup_username.pack()

tk.Label(signup_frame, text="Email").pack()
signup_email = tk.Entry(signup_frame)
signup_email.pack()

tk.Label(signup_frame, text="Password").pack()
signup_password = tk.Entry(signup_frame, show="*")
signup_password.pack()

tk.Button(signup_frame, text="Submit", command=signup).pack(pady=10)
tk.Button(signup_frame, text="Already have an account? Login", command=show_login).pack()

login_frame.pack()
#root.mainloop()

def open_sambhashana():
    import os
    from PyQt5 import QtWidgets, uic
    import object_detection
    import tensorflow as tf
    from object_detection.utils import config_util
    from object_detection.protos import pipeline_pb2
    from google.protobuf import text_format
    import cv2
    import tensorflow as tf
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as viz_utils
    from object_detection.builders import model_builder
    from object_detection.utils import config_util
    import numpy as np
    from matplotlib import pyplot as plt
    
    
    CUSTOM_MODEL_NAME = 'my_sdd_mobnet'
    PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
    PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
    TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
    LABEL_MAP_NAME = 'label_map.pbtxt'
    
    
    paths = {
        'PROJECT_PATH': os.path.join('project'),
        'SCRIPT_PATH': os.path.join('scripts'),
        'APIMODEL_PATH': os.path.join('modelsNEW'),
        'ANNOTATION_PATH': os.path.join('project','annotations'),
        'IMAGE_PATH': os.path.join('project','images'),
        'MODEL_PATH': os.path.join('project','modelsNEW'),
        'PRETRAINED_MODEL_PATH': os.path.join('project','pre-trained-models'),
        'CHECKPOINT_PATH': os.path.join('project','modelsNEW',CUSTOM_MODEL_NAME),
        'OUTPUT_PATH': os.path.join('project','modelsNEW',CUSTOM_MODEL_NAME,'export'),
        'TFJS_PATH': os.path.join('project','modelsNEW',CUSTOM_MODEL_NAME,'tfjsexport'),
        'TFLITE_PATH': os.path.join('project','modelsNEW',CUSTOM_MODEL_NAME,'tfliteexport'),
        'PROTOC_PATH': os.path.join('protoc')
    }
    
    files = {
        'PIPELINE_CONFIG': os.path.join('project','modelsNEW',CUSTOM_MODEL_NAME,'pipeline.config'),
        'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPT_PATH'],TF_RECORD_SCRIPT_NAME),
        'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME),
        'ASL_LABELMAP': os.path.join(paths['ANNOTATION_PATH'],'label_map.pbtxt')
    }
    
    
    configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)
    
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-41')).expect_partial()
    
    @tf.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections
    category_index = label_map_util.create_category_index_from_labelmap(files['ASL_LABELMAP'])
    
    
    def predict_image(file_path):
        img = cv2.imread(file_path)
        image_np = np.array(img)
        
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
        
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections
        
        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        
        label_id_offset = 1
        image_np_with_detections = image_np.copy()
        
        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.8,
                    agnostic_mode=False)
        
        #plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
        #plt.show()
        # Convert BGR image (cv2) to RGB for display later in GUI
        detected_img = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)
        
        return detected_img
    
    def start_camera():
        #thread = threading.Thread(target=run_detection)
        #thread.daemon = True
        #thread.start()
        cap = cv2.VideoCapture(0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        while cap.isOpened(): 
            ret, frame = cap.read()
            
            image_np = np.array(frame)
            
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype = tf.float32)
            detections = detect_fn(input_tensor)
            
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections
        
            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        
            label_id_offset = 1
            image_np_with_detections = image_np.copy()
        
            viz_utils.visualize_boxes_and_labels_on_image_array(
                        image_np_with_detections,
                        detections['detection_boxes'],
                        detections['detection_classes']+label_id_offset,
                        detections['detection_scores'],
                        category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=5,
                        min_score_thresh=.8,
                        agnostic_mode=False)
        
            cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
    
    
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import threading
    #from detection import run_detection   # import your camera function
    from PIL import Image, ImageTk


    def show_output_image(result):
        viewer = tk.Toplevel()
        viewer.title("Detection Result")
    
        # Convert model result to displayable image if necessary
        if isinstance(result, np.ndarray):
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            result = Image.fromarray(result)
    
        img_tk = ImageTk.PhotoImage(result)
    
        lbl = tk.Label(viewer, image=img_tk)
        lbl.image = img_tk  # prevent GC issue
        lbl.pack(padx=20, pady=20)
    
        tk.Button(viewer, text="Close", command=viewer.destroy).pack(pady=10)
    #img_label = tk.Label(main)
    #img_label.pack(pady=20)
    def upload_image():
            file_path = filedialog.askopenfilename(
                filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
            )
            if file_path:
                result = predict_image(file_path)  # ⬅ call your detection function here
                
                if result is None:
                    messagebox.showerror("Error", "Detection failed")
                    return
                
                # Convert numpy image → Tkinter image
                show_output_image(result)
                 
    root = tk.Tk()
    root.title("Sambhashana")
    
    # Full screen or near full-screen feel
    root.geometry("900x600")
    root.configure(bg="#ffffff")  # clean white background
    
    # Title Label
    title_label = tk.Label(
        root,
        text="Sambhashana",
        font=("Montserrat", 40, "bold"),
        bg="#ffffff"
    )
    title_label.pack(pady=40)
    
    # Buttons Frame
    btn_frame = tk.Frame(root, bg="#ffffff")
    btn_frame.pack(pady=50)
    
    # Upload Image button
    upload_btn = tk.Button(
        btn_frame,
        text="Upload Image",
        font=("Arial", 20),
        width=15,
        command=upload_image
    )
    upload_btn.grid(row=0, column=0, padx=30, pady=20)
    
    # Start Camera button
    camera_btn = tk.Button(
        btn_frame,
        text="Start Camera",
        font=("Arial", 20),
        width=15,
        command=start_camera
    )
    camera_btn.grid(row=0, column=1, padx=30, pady=20)

root.mainloop()
