from flask import Flask, render_template, Response, jsonify, make_response
import cv2
import time
import numpy as np
import os
import sqlite3
from datetime import datetime
import csv
from io import StringIO

# YOLO imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: Ultralytics YOLO not found.")
    print("Install with: pip install ultralytics")
    YOLO_AVAILABLE = False

# Thermal camera imports
try:
    import board
    import busio
    import adafruit_mlx90640
    THERMAL_AVAILABLE = True
except ImportError:
    print("Warning: Thermal camera libraries not found.")
    print("Install with: pip install adafruit-circuitpython-mlx90640")
    THERMAL_AVAILABLE = False

app = Flask(__name__)

# Database setup
DATABASE = 'pig_monitoring.db'

def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database with tables"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Create activity logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            track_id INTEGER,
            growth_label TEXT,
            time_since_move_sec REAL,
            status TEXT,
            temperature REAL,
            temp_warning TEXT,
            alert_level TEXT
        )
    ''')
    
    # Create detection summary table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            total_pigs INTEGER,
            healthy_count INTEGER,
            inactive_count INTEGER,
            heat_stress_count INTEGER,
            cold_stress_count INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✓ Database initialized successfully")

# Initialize USB camera
camera = None
CAMERA_AVAILABLE = False

# Initialize YOLO model
yolo_model = None
yolo_labels = None
yolo_enabled = False
min_confidence = 0.5

# Tracking settings
tracking_enabled = False
pig_state = {}  # Store state for each tracked pig
pig_temperatures = {}  # Store temperature for each tracked pig
last_log_time = 0
LOG_INTERVAL = 2.0  # Log every 2 seconds
MOVE_THRESHOLD = 30  # Pixels to consider as movement
logs = []  # Store activity logs

# Temperature thresholds
TEMP_HIGH_THRESHOLD = 45.0  # Heat stress above 40°C
TEMP_LOW_THRESHOLD = 25.0   # Cold stress below 25°C

# YOLO bounding box colors (Tableau 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

def get_center(x1, y1, x2, y2):
    """Calculate center point of bounding box"""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy

def distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_temperature_at_point(thermal_data, x, y, frame_width, frame_height):
    """Get temperature at a specific point from thermal data"""
    if thermal_data is None:
        return None
    
    # Thermal camera is 24x32, scale coordinates
    thermal_x = int((x / frame_width) * 32)
    thermal_y = int((y / frame_height) * 24)
    
    # Clamp to valid range
    thermal_x = max(0, min(31, thermal_x))
    thermal_y = max(0, min(23, thermal_y))
    
    return thermal_data[thermal_y, thermal_x]

def check_temperature_warning(temp):
    """Check temperature and return warning status"""
    if temp is None:
        return None, "UNKNOWN"
    
    if temp >= TEMP_HIGH_THRESHOLD:
        return "⚠️ HEAT STRESS", "CRITICAL"
    elif temp <= TEMP_LOW_THRESHOLD:
        return "❄️ COLD STRESS", "WARNING"
    else:
        return "✓ Normal", "NORMAL"

def init_yolo(model_path='best.pt'):
    """Initialize YOLO model"""
    global yolo_model, yolo_labels, yolo_enabled, YOLO_AVAILABLE
    
    if not YOLO_AVAILABLE:
        print("✗ YOLO not available (ultralytics not installed)")
        return False
    
    if not os.path.exists(model_path):
        print(f"✗ YOLO model not found at: {model_path}")
        print("  Please train a model or provide path to existing model")
        return False
    
    try:
        yolo_model = YOLO(model_path, task='detect')
        yolo_labels = yolo_model.names
        yolo_enabled = True
        print(f"✓ YOLO model loaded successfully from {model_path}")
        print(f"  Classes: {yolo_labels}")
        return True
    except Exception as e:
        print(f"✗ Error loading YOLO model: {e}")
        return False

# Initialize thermal camera
mlx = None
thermal_frame = None
if THERMAL_AVAILABLE:
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        mlx = adafruit_mlx90640.MLX90640(i2c)
        mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
        thermal_frame = np.zeros((24 * 32,))
        print("✓ Thermal camera (MLX90640) initialized successfully!")
    except Exception as e:
        print(f"✗ Error initializing thermal camera: {e}")
        THERMAL_AVAILABLE = False
        mlx = None

def init_camera():
    """Initialize USB camera"""
    global camera, CAMERA_AVAILABLE
    
    # Try multiple video device indices (USB cameras can be on different indices)
    video_indices = [0, 1, 2, 3]
    
    for idx in video_indices:
        try:
            print(f"Trying /dev/video{idx}...")
            test_camera = cv2.VideoCapture(idx)
            
            if test_camera.isOpened():
                # Try to read a frame to verify it's a real capture device
                ret, frame = test_camera.read()
                if ret and frame is not None:
                    # Found a working camera!
                    camera = test_camera
                    
                    # Set camera properties
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    camera.set(cv2.CAP_PROP_FPS, 30)
                    
                    CAMERA_AVAILABLE = True
                    print(f"✓ USB Camera initialized successfully on /dev/video{idx}!")
                    print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
                    return True
                else:
                    test_camera.release()
            else:
                test_camera.release()
        except Exception as e:
            print(f"  Error on /dev/video{idx}: {e}")
            continue
    
    print("✗ No working USB camera found on /dev/video0-3")
    CAMERA_AVAILABLE = False
    return False

# Initialize camera on startup
init_camera()

# Store latest thermal data
latest_thermal_data = None

def get_thermal_frame():
    """Get a frame from the thermal camera"""
    global mlx, thermal_frame, latest_thermal_data
    
    if not THERMAL_AVAILABLE or mlx is None:
        return None
    
    try:
        mlx.getFrame(thermal_frame)
        # Convert to 2D
        thermal = np.reshape(thermal_frame, (24, 32))
        
        # Store raw thermal data for temperature lookups
        latest_thermal_data = thermal.copy()
        
        # Normalize temperature to 0–255
        min_temp = np.min(thermal)
        max_temp = np.max(thermal)
        normalized = np.uint8(
            255 * (thermal - min_temp) / (max_temp - min_temp + 1e-6)
        )
        # Resize for display
        resized = cv2.resize(normalized, (640, 480), interpolation=cv2.INTER_CUBIC)
        # Apply color map
        heatmap = cv2.applyColorMap(resized, cv2.COLORMAP_JET)
        # Add temperature text
        cv2.putText(
            heatmap,
            f"Min: {min_temp:.1f}C  Max: {max_temp:.1f}C",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        return heatmap
    except Exception as e:
        print(f"Error getting thermal frame: {e}")
        return None

def generate_frames():
    """Generator function to continuously capture and yield frames with YOLO detection and tracking"""
    global camera, CAMERA_AVAILABLE, yolo_model, yolo_enabled, min_confidence
    global tracking_enabled, pig_state, last_log_time, logs
    
    if not CAMERA_AVAILABLE or camera is None:
        # Return a placeholder image if camera is not available
        while True:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "USB Camera Not Available", (120, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Please connect USB camera", (120, 260), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.5)
    else:
        while True:
            try:
                # Capture frame from USB camera
                ret, frame = camera.read()
                
                if not ret:
                    print("Failed to grab frame")
                    time.sleep(0.1)
                    continue
                
                current_time = time.time()
                object_count = 0
                
                # Run YOLO detection or tracking if enabled
                if yolo_enabled and yolo_model is not None:
                    try:
                        # Use tracking if enabled, otherwise regular detection
                        if tracking_enabled:
                            # YOLO tracking mode
                            results = yolo_model.track(
                                frame,
                                conf=min_confidence,
                                iou=0.5,
                                persist=True,
                                tracker="bytetrack.yaml",
                                verbose=False
                            )
                        else:
                            # Regular detection mode
                            results = yolo_model(frame, verbose=False)
                        
                        if results[0].boxes is not None:
                            detections = results[0].boxes
                            do_log = (current_time - last_log_time) >= LOG_INTERVAL
                            
                            # Process each detection
                            for i in range(len(detections)):
                                # Get bounding box coordinates
                                xyxy_tensor = detections[i].xyxy.cpu()
                                xyxy = xyxy_tensor.numpy().squeeze()
                                x1, y1, x2, y2 = xyxy.astype(int)
                                
                                # Get class and confidence
                                classidx = int(detections[i].cls.item())
                                classname = yolo_labels[classidx]
                                conf = detections[i].conf.item()
                                
                                # Draw bounding box if confidence is high enough
                                if conf > min_confidence:
                                    object_count += 1
                                    color = bbox_colors[classidx % 10]
                                    
                                    # Tracking mode - show more info
                                    if tracking_enabled and detections[i].id is not None:
                                        track_id = int(detections[i].id.item())
                                        cx, cy = get_center(x1, y1, x2, y2)
                                        
                                        # Get temperature at pig location
                                        frame_height, frame_width = frame.shape[:2]
                                        pig_temp = get_temperature_at_point(latest_thermal_data, cx, cy, frame_width, frame_height)
                                        pig_temperatures[track_id] = pig_temp
                                        
                                        # Check temperature warnings
                                        temp_warning, alert_level = check_temperature_warning(pig_temp)
                                        
                                        # Initialize pig state if new
                                        if track_id not in pig_state:
                                            pig_state[track_id] = {
                                                "last_pos": (cx, cy),
                                                "last_move_time": current_time
                                            }
                                        
                                        # Check movement
                                        last_pos = pig_state[track_id]["last_pos"]
                                        moved_distance = distance((cx, cy), last_pos)
                                        
                                        if moved_distance > MOVE_THRESHOLD:
                                            pig_state[track_id]["last_move_time"] = current_time
                                        
                                        # Update position
                                        pig_state[track_id]["last_pos"] = (cx, cy)
                                        
                                        # Calculate inactive time
                                        time_since_move = current_time - pig_state[track_id]["last_move_time"]
                                        
                                        # Determine status
                                        status = "HEALTHY"
                                        if time_since_move > 60:  # Inactive for 1+ minute
                                            status = "INACTIVE"
                                            color = (0, 165, 255)  # Orange
                                        
                                        # Override color for temperature alerts
                                        if alert_level == "CRITICAL":
                                            color = (0, 0, 255)  # Red for heat stress
                                        elif alert_level == "WARNING":
                                            color = (255, 165, 0)  # Blue for cold stress
                                        
                                        # Log data periodically (and save to database)
                                        if do_log:
                                            log_entry = {
                                                "time": current_time,
                                                "track_id": track_id,
                                                "growth_label": classname,
                                                "time_since_move_sec": round(time_since_move, 1),
                                                "status": status,
                                                "temperature": round(pig_temp, 1) if pig_temp else None,
                                                "temp_warning": temp_warning,
                                                "alert_level": alert_level
                                            }
                                            logs.append(log_entry)
                                            
                                            # Save to database
                                            try:
                                                conn = get_db()
                                                cursor = conn.cursor()
                                                cursor.execute('''
                                                    INSERT INTO activity_logs 
                                                    (track_id, growth_label, time_since_move_sec, status, 
                                                     temperature, temp_warning, alert_level)
                                                    VALUES (?, ?, ?, ?, ?, ?, ?)
                                                ''', (track_id, classname, round(time_since_move, 1), status,
                                                      round(pig_temp, 1) if pig_temp else None, temp_warning, alert_level))
                                                conn.commit()
                                                conn.close()
                                            except Exception as db_error:
                                                print(f"Database error: {db_error}")
                                        
                                        # Draw bounding box
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                        
                                        # Draw labels with tracking info
                                        label1 = f"ID:{track_id} | {classname}"
                                        label2 = f"Inactive: {time_since_move:.0f}s"
                                        label3 = status
                                        
                                        # Add temperature info if available
                                        if pig_temp:
                                            label4 = f"Temp: {pig_temp:.1f}C {temp_warning}"
                                            
                                            cv2.putText(frame, label1, (x1, y1 - 10),
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                            cv2.putText(frame, label2, (x1, y1 + 15),
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                                            cv2.putText(frame, label3, (x1, y1 + 30),
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                                            cv2.putText(frame, label4, (x1, y1 + 45),
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                                        else:
                                            cv2.putText(frame, label1, (x1, y1 - 10),
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                            cv2.putText(frame, label2, (x1, y1 + 15),
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                                            cv2.putText(frame, label3, (x1, y1 + 35),
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                                    
                                    else:
                                        # Regular detection mode - simple box and label
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                        
                                        label = f'{classname}: {int(conf*100)}%'
                                        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                        label_ymin = max(y1, labelSize[1] + 10)
                                        cv2.rectangle(frame, (x1, label_ymin-labelSize[1]-10), 
                                                    (x1+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
                                        cv2.putText(frame, label, (x1, label_ymin-7), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                            
                            # Update log time
                            if do_log:
                                last_log_time = current_time
                        
                        # Display detection/tracking count
                        mode_text = "Tracked" if tracking_enabled else "Detected"
                        cv2.putText(frame, f'{mode_text}: {object_count}', (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                    except Exception as e:
                        print(f"YOLO detection/tracking error: {e}")
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame = buffer.tobytes()
                
                # Yield frame in byte format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Error capturing frame: {e}")
                time.sleep(0.1)

def generate_thermal():
    """Generator function for thermal camera stream"""
    global THERMAL_AVAILABLE
    
    if not THERMAL_AVAILABLE or mlx is None:
        # Return placeholder if thermal camera not available
        while True:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Thermal Camera Not Available", (100, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "MLX90640 not detected", (150, 260), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.5)
    else:
        while True:
            try:
                frame = get_thermal_frame()
                if frame is not None:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    time.sleep(0.25)
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error generating thermal frame: {e}")
                time.sleep(0.1)

@app.route('/')
def index():
    """Home page with video feed"""
    return render_template('index.html', 
                         camera_available=CAMERA_AVAILABLE,
                         thermal_available=THERMAL_AVAILABLE,
                         yolo_enabled=yolo_enabled)

@app.route('/video_feed')
def video_feed():
    """Route for video streaming"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/thermal_feed')
def thermal_feed():
    """Route for thermal camera streaming"""
    return Response(generate_thermal(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """Route to check camera status"""
    return {
        'camera_available': CAMERA_AVAILABLE,
        'camera_initialized': camera is not None and camera.isOpened(),
        'thermal_available': THERMAL_AVAILABLE,
        'thermal_initialized': mlx is not None,
        'yolo_enabled': yolo_enabled,
        'yolo_available': YOLO_AVAILABLE,
        'tracking_enabled': tracking_enabled,
        'tracked_pigs': len(pig_state)
    }

@app.route('/toggle_yolo')
def toggle_yolo():
    """Toggle YOLO detection on/off"""
    global yolo_enabled
    if YOLO_AVAILABLE and yolo_model is not None:
        yolo_enabled = not yolo_enabled
        return {'success': True, 'yolo_enabled': yolo_enabled}
    else:
        return {'success': False, 'message': 'YOLO not available'}

@app.route('/toggle_tracking')
def toggle_tracking():
    """Toggle tracking mode on/off"""
    global tracking_enabled, pig_state
    if YOLO_AVAILABLE and yolo_model is not None and yolo_enabled:
        tracking_enabled = not tracking_enabled
        if tracking_enabled:
            pig_state = {}  # Reset tracking state when enabling
        return {'success': True, 'tracking_enabled': tracking_enabled}
    else:
        return {'success': False, 'message': 'YOLO must be enabled first'}

@app.route('/get_logs')
def get_logs():
    """Get activity logs from database"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM activity_logs 
            ORDER BY timestamp DESC 
            LIMIT 100
        ''')
        rows = cursor.fetchall()
        conn.close()
        
        logs_list = []
        for row in rows:
            logs_list.append({
                'id': row['id'],
                'timestamp': row['timestamp'],
                'track_id': row['track_id'],
                'growth_label': row['growth_label'],
                'time_since_move_sec': row['time_since_move_sec'],
                'status': row['status'],
                'temperature': row['temperature'],
                'temp_warning': row['temp_warning'],
                'alert_level': row['alert_level']
            })
        
        return jsonify({'logs': logs_list})
    except Exception as e:
        print(f"Error fetching logs: {e}")
        return jsonify({'logs': []})

@app.route('/get_summary')
def get_summary():
    """Get summary statistics"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Get latest summary
        cursor.execute('''
            SELECT COUNT(DISTINCT track_id) as total_pigs,
                   SUM(CASE WHEN status = 'HEALTHY' THEN 1 ELSE 0 END) as healthy,
                   SUM(CASE WHEN status = 'INACTIVE' THEN 1 ELSE 0 END) as inactive,
                   SUM(CASE WHEN alert_level = 'CRITICAL' THEN 1 ELSE 0 END) as heat_stress,
                   SUM(CASE WHEN alert_level = 'WARNING' THEN 1 ELSE 0 END) as cold_stress
            FROM activity_logs 
            WHERE timestamp > datetime('now', '-5 minutes')
        ''')
        
        row = cursor.fetchone()
        conn.close()
        
        return jsonify({
            'total_pigs': row['total_pigs'] or 0,
            'healthy': row['healthy'] or 0,
            'inactive': row['inactive'] or 0,
            'heat_stress': row['heat_stress'] or 0,
            'cold_stress': row['cold_stress'] or 0
        })
    except Exception as e:
        print(f"Error fetching summary: {e}")
        return jsonify({
            'total_pigs': 0,
            'healthy': 0,
            'inactive': 0,
            'heat_stress': 0,
            'cold_stress': 0
        })

@app.route('/clear_logs')
def clear_logs():
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM activity_logs")
        conn.commit()
        conn.close()

        pig_state.clear()
        pig_temperatures.clear()

        return {'success': True}
    except Exception as e:
        print(f"Error clearing logs: {e}")
        return {'success': False}

@app.route('/download_logs')
def download_logs():
    try:
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT timestamp, track_id, growth_label,
                   time_since_move_sec, status,
                   temperature, temp_warning, alert_level
            FROM activity_logs
            ORDER BY timestamp ASC
        """)

        rows = cursor.fetchall()
        conn.close()

        # Create CSV in memory
        output = StringIO()
        writer = csv.writer(output)

        # Header row
        writer.writerow([
            "Timestamp",
            "Pig ID",
            "Growth Label",
            "Inactive Time (sec)",
            "Status",
            "Temperature (°C)",
            "Temperature Warning",
            "Alert Level"
        ])

        # Data rows
        for row in rows:
            writer.writerow([
                row["timestamp"],
                row["track_id"],
                row["growth_label"],
                row["time_since_move_sec"],
                row["status"],
                row["temperature"],
                row["temp_warning"],
                row["alert_level"]
            ])

        # Prepare response
        response = make_response(output.getvalue())
        response.headers["Content-Disposition"] = (
            "attachment; filename=pig_activity_logs.csv"
        )
        response.headers["Content-Type"] = "text/csv"

        return response

    except Exception as e:
        print(f"Error exporting logs: {e}")
        return {"success": False, "error": str(e)}, 500


@app.route('/set_confidence/<float:conf>')
def set_confidence(conf):
    """Set minimum confidence threshold"""
    global min_confidence
    min_confidence = max(0.0, min(1.0, conf))
    return {'success': True, 'min_confidence': min_confidence}

@app.route('/reconnect')
def reconnect():
    """Attempt to reconnect to camera"""
    global camera, CAMERA_AVAILABLE
    if camera is not None:
        camera.release()
    success = init_camera()
    return {'success': success, 'camera_available': CAMERA_AVAILABLE}

if __name__ == '__main__':
    try:
        print("=" * 50)
        print("Pig Growth Monitoring System")
        print("=" * 50)
        
        # Initialize database
        init_db()
        
        # Try to initialize YOLO with default model path
        # You can change 'best.pt' to your actual model path
        model_path = 'best.pt'  # Change this to your YOLO model path
        if os.path.exists(model_path):
            init_yolo(model_path)
        else:
            print(f"Note: YOLO model not found at '{model_path}'")
            print("      YOLO detection will be disabled")
            print("      To enable: Place your trained model as 'best.pt' or update model_path in code")
        
        print(f"USB Camera Status: {'✓ Available' if CAMERA_AVAILABLE else '✗ Not Available'}")
        print(f"Thermal Camera Status: {'✓ Available' if THERMAL_AVAILABLE else '✗ Not Available'}")
        print(f"YOLO Detection Status: {'✓ Enabled' if yolo_enabled else '✗ Disabled'}")
        print(f"Database Status: ✓ Connected ({DATABASE})")
        
        if not CAMERA_AVAILABLE:
            print("\nTo fix USB camera:")
            print("  1. Connect your USB camera to the Raspberry Pi")
            print("  2. Check with: ls /dev/video*")
            print("  3. Restart this application")
        if not THERMAL_AVAILABLE:
            print("\nTo fix Thermal camera:")
            print("  1. Connect MLX90640 to I2C (SCL/SDA)")
            print("  2. Install: pip install adafruit-circuitpython-mlx90640")
            print("  3. Restart this application")
        print("\nTemperature Thresholds:")
        print(f"  ⚠️  Heat Stress: ≥ {TEMP_HIGH_THRESHOLD}°C")
        print(f"  ❄️  Cold Stress: ≤ {TEMP_LOW_THRESHOLD}°C")
        print("\nStarting Flask server...")
        print("Access the application at:")
        print("  - Local: http://localhost:5000")
        print("  - LAN: http://YOUR_PI_IP:5000")
        print("=" * 50)
        
        # Run on all network interfaces so it's accessible on LAN
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Clean up camera on exit
        if camera is not None:
            camera.release()
            print("Camera released.")