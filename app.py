from flask import Flask, render_template, Response
import cv2
import time
import numpy as np

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

# Initialize USB camera
camera = None
CAMERA_AVAILABLE = False

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
    video_indices = [0, 1]
    
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

def get_thermal_frame():
    """Get a frame from the thermal camera"""
    global mlx, thermal_frame
    
    if not THERMAL_AVAILABLE or mlx is None:
        return None
    
    try:
        mlx.getFrame(thermal_frame)
        # Convert to 2D
        thermal = np.reshape(thermal_frame, (24, 32))
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
    """Generator function to continuously capture and yield frames"""
    global camera, CAMERA_AVAILABLE
    
    if not CAMERA_AVAILABLE or camera is None:
        # Return a placeholder image if camera is not available
        while True:
            import numpy as np
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
                         thermal_available=THERMAL_AVAILABLE)

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
        'thermal_initialized': mlx is not None
    }

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
        print(f"USB Camera Status: {'✓ Available' if CAMERA_AVAILABLE else '✗ Not Available'}")
        print(f"Thermal Camera Status: {'✓ Available' if THERMAL_AVAILABLE else '✗ Not Available'}")
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