# realtime_dashboard.py
"""
Run with:
    python realtime_dashboard.py --source 0  # local webcam
    python realtime_dashboard.py --source "rtsp://user:pwd@ip/stream"
Browse to http://localhost:5000
"""
import argparse, time, json
from flask import Flask, Response, render_template_string
import cv2
from realtime_analyzer import RealTimeAnalyzer

HTML = """
<!doctype html>
<title>Fiber‑End‑Face Live Defect Dashboard</title>
<style>
 body      { margin:0; font-family:Arial, sans-serif; background:#111; color:#eee;}
 #wrap     { display:flex; flex-direction:row; height:100vh;}
 #videoBox { flex:3; }
 #statsBox { flex:1; padding:10px; overflow-y:auto; background:#222;}
 img       { width:100%; height:auto;}
 table     { width:100%; border-collapse:collapse; font-size:0.9em;}
 th, td    { border:1px solid #444; padding:3px 6px; text-align:center;}
 th        { background:#333;}
 .high     { color:#ff4444; font-weight:bold; }
 .medium   { color:#ffaa44; }
 .low      { color:#44ff44; }
</style>
<div id="wrap">
  <div id="videoBox"><img src="{{ url_for('video_feed') }}"></div>
  <div id="statsBox">
    <h3>Current‑frame defects</h3>
    <table id="defects"><thead>
      <tr><th>ID</th><th>Region</th><th>XY</th><th>Area</th><th>Severity</th></tr>
    </thead><tbody></tbody></table>
    <div id="summary" style="margin-top:20px; padding:10px; background:#333;">
      <h4>Summary</h4>
      <p>Total defects: <span id="total">0</span></p>
      <p>Critical (core): <span id="critical">0</span></p>
    </div>
  </div>
</div>
<script>
const tbody = document.querySelector('#defects tbody');
const total = document.querySelector('#total');
const critical = document.querySelector('#critical');

function poll(){
  fetch('/defect_json').then(r=>r.json()).then(data=>{
     tbody.innerHTML='';
     let critCount = 0;
     data.forEach((d,i)=>{
        const row=document.createElement('tr');
        row.innerHTML=`<td>${i+1}</td><td>${d.region}</td>
                       <td>(${d.cx},${d.cy})</td><td>${d.area}</td>
                       <td class="${d.severity.toLowerCase()}">${d.severity}</td>`;
        tbody.appendChild(row);
        if(d.region === 'core' && d.severity === 'HIGH') critCount++;
     });
     total.textContent = data.length;
     critical.textContent = critCount;
  }).catch(console.error);
}
setInterval(poll, 500);   // update twice a second
</script>
"""

app     = Flask(__name__)
analyzer= None           # filled in main()
last_defects = []

def gen_frames():
    global last_defects
    cap = cv2.VideoCapture(args.source if isinstance(args.source, str) else int(args.source))
    if not cap.isOpened():
        raise IOError(f"Cannot open video source {args.source}")

    frame_count = 0
    fps_timer = time.time()
    fps = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        vis, defects = analyzer.analyze(frame)
        if vis is not None:
            last_defects = defects        # store for JSON route
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_timer)
                fps_timer = time.time()
            
            # Add FPS to frame
            if fps > 0:
                cv2.putText(vis, f"FPS: {fps:.1f}", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            ret, buf = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret: continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
    
    cap.release()

@app.route('/')
def index():   
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():   
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/defect_json')
def defect_json():  
    return Response(json.dumps(last_defects),
                    mimetype='application/json')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=0,
                        help='camera index or RTSP/HTTP stream URL')
    parser.add_argument('--config', default="config.json",
                        help='path to pipeline config for analyzer')
    args = parser.parse_args()

    analyzer = RealTimeAnalyzer(config_path=args.config,
                                fast_segmentation_method="ai",
                                min_frame_interval=0.05)   # ≈20 FPS target

    app.run(host='0.0.0.0', port=5000, threaded=True)