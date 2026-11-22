import subprocess
import time
import json
import base64
import urllib.request
import cv2
import numpy as np
from PIL import Image
import websocket

CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 9222
APP_URL = "http://localhost:8501"

print("🚀 Launching Chrome in headless CDP mode...")
chrome_proc = subprocess.Popen([
    CHROME_PATH,
    "--headless",
    f"--remote-debugging-port={PORT}",
    "--disable-gpu",
    "--remote-allow-origins=*",
    "--window-size=1440,960",
    "--no-sandbox",
    "about:blank"
])

time.sleep(2.5)

try:
    # 1. Connect to Chrome Debugger
    with urllib.request.urlopen(f"http://localhost:{PORT}/json") as r:
        tabs = json.loads(r.read().decode())
    
    ws_url = tabs[0]["webSocketDebuggerUrl"]
    print(f"✓ Connected to DevTools WebSocket: {ws_url}")
    ws = websocket.create_connection(ws_url, timeout=30)
    
    msg_id = 1
    def send_cdp(method, params=None):
        global msg_id
        payload = {"id": msg_id, "method": method, "params": params or {}}
        ws.send(json.dumps(payload))
        msg_id += 1
        while True:
            raw = ws.recv()
            data = json.loads(raw)
            if data.get("id") == payload["id"]:
                return data.get("result", {})

    send_cdp("Page.enable")
    send_cdp("Runtime.enable")

    frames_bgr = []
    frames_pil = []

    def capture_frame(tag=""):
        res = send_cdp("Page.captureScreenshot", {"format": "png"})
        b64 = res.get("data", "")
        if b64:
            img_bytes = base64.b64decode(b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frames_bgr.append(bgr)
            
            # Downscale slightly for animated GIF to keep size reasonable
            pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            pil_small = pil_img.resize((1024, int(1024 * bgr.shape[0] / bgr.shape[1])), Image.Resampling.LANCZOS)
            frames_pil.append(pil_small)
            return bgr
        return None

    def evaluate_js(expr):
        res = send_cdp("Runtime.evaluate", {"expression": expr, "returnByValue": True})
        return res.get("result", {}).get("value")

    print(f"🌐 Navigating to {APP_URL}...")
    send_cdp("Page.navigate", {"url": APP_URL})
    time.sleep(4)

    # Step 1: Initial Dashboard State
    print("📸 Step 1: Capturing Initial Dashboard...")
    for _ in range(6):
        bgr = capture_frame()
        time.sleep(0.3)
    cv2.imwrite("screenshot_1_dashboard.png", bgr)
    print("✓ Saved screenshot_1_dashboard.png")

    # Step 2: Load Elon Musk Case Study
    print("🚀 Step 2: Triggering 'Elon Musk Case' button...")
    js_click_elon = """
    (() => {
        const btns = Array.from(document.querySelectorAll('button'));
        const elonBtn = btns.find(b => b.innerText.includes('Elon Musk Case'));
        if (elonBtn) {
            elonBtn.click();
            return 'clicked';
        }
        return 'not_found';
    })()
    """
    click_res = evaluate_js(js_click_elon)
    print(f"Click result: {click_res}")
    
    # Wait for ingestion and record frames
    for i in range(15):
        bgr = capture_frame()
        time.sleep(0.3)
    cv2.imwrite("screenshot_2_elon_musk_loaded.png", bgr)
    print("✓ Saved screenshot_2_elon_musk_loaded.png")

    # Step 3: Trigger SpaceX Analysis Chip
    print("🔬 Step 3: Triggering 'SpaceX Falcon & Starship' Analysis Chip...")
    js_click_spacex = """
    (() => {
        const btns = Array.from(document.querySelectorAll('button'));
        const spacexBtn = btns.find(b => b.innerText.includes('SpaceX'));
        if (spacexBtn) {
            spacexBtn.click();
            return 'clicked';
        }
        return 'not_found';
    })()
    """
    click_res = evaluate_js(js_click_spacex)
    print(f"Click SpaceX Chip result: {click_res}")

    # Wait for RAG retrieval + LLM synthesis + Graphviz diagram generation
    print("⏳ Waiting for RAG Retrieval & Diagram Generation...")
    for i in range(35):
        bgr = capture_frame()
        time.sleep(0.4)
    cv2.imwrite("screenshot_3_spacex_diagram_generated.png", bgr)
    print("✓ Saved screenshot_3_spacex_diagram_generated.png")

    # Step 4: Click Academic Synthesis Tab
    print("📝 Step 4: Inspecting 'Academic Synthesis' Tab...")
    js_click_synthesis = """
    (() => {
        const tabs = Array.from(document.querySelectorAll('div[data-baseweb="tab"]'));
        const tab = tabs.find(t => t.innerText.includes('Academic Synthesis'));
        if (tab) {
            tab.click();
            return 'clicked';
        }
        return 'not_found';
    })()
    """
    evaluate_js(js_click_synthesis)
    for _ in range(8):
        bgr = capture_frame()
        time.sleep(0.3)
    cv2.imwrite("screenshot_4_academic_synthesis.png", bgr)
    print("✓ Saved screenshot_4_academic_synthesis.png")

    # Step 5: Click Grounded Sources Tab
    print("📚 Step 5: Inspecting 'Grounded Source Evidence' Tab...")
    js_click_sources = """
    (() => {
        const tabs = Array.from(document.querySelectorAll('div[data-baseweb="tab"]'));
        const tab = tabs.find(t => t.innerText.includes('Grounded Source Evidence'));
        if (tab) {
            tab.click();
            return 'clicked';
        }
        return 'not_found';
    })()
    """
    evaluate_js(js_click_sources)
    for _ in range(8):
        bgr = capture_frame()
        time.sleep(0.3)
    cv2.imwrite("screenshot_5_grounded_sources.png", bgr)
    print("✓ Saved screenshot_5_grounded_sources.png")

    # 4. Compile Video Recording
    print(f"🎬 Compiling {len(frames_bgr)} frames into screenrecord_demo.mp4...")
    if frames_bgr:
        h, w, _ = frames_bgr[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter("screenrecord_demo.mp4", fourcc, 4.0, (w, h))
        for f in frames_bgr:
            out_video.write(f)
        out_video.release()
        print("✓ Successfully created screenrecord_demo.mp4")

    # 5. Compile Animated GIF Preview
    print(f"🎞️ Compiling animated GIF preview screenrecord_demo.gif...")
    if frames_pil:
        # Sample every 2nd frame for lightweight GIF
        gif_frames = frames_pil[::2]
        gif_frames[0].save(
            "screenrecord_demo.gif",
            save_all=True,
            append_images=gif_frames[1:],
            duration=250,
            loop=0,
            optimize=True
        )
        print("✓ Successfully created screenrecord_demo.gif")

finally:
    try:
        ws.close()
    except Exception:
        pass
    chrome_proc.terminate()
    print("🏁 Chrome headless session terminated cleanly.")
