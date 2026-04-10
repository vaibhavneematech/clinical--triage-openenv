import os
import subprocess
import time
import sys
import webbrowser

def run_demo():
    print("🚀 INITIALIZING CLINICAL TRIAGE V2 DEMO")
    print("========================================")

    # 1. Start Backend
    print("📡 Starting FastAPI Backend (ClinicalTriageEnv)...")
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "clinical_triage_env.app"],
        env={**os.environ, "PORT": "7860"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # 2. Wait for backend to be ready
    print("⏳ Waiting for backend to warm up...")
    max_retries = 10
    while max_retries > 0:
        try:
            import requests
            resp = requests.get("http://localhost:7860/")
            if resp.status_code == 200:
                print("✅ Backend ONLINE.")
                break
        except:
            pass
        time.sleep(2)
        max_retries -= 1

    # 3. Open Dashboard (if possible)
    print("💻 Opening Dashboard in browser (defaulting to localhost:3000)...")
    print("   Note: Ensure 'npm run dev' is running in the 'clinical-triage-dashboard' directory.")
    webbrowser.open("http://localhost:3000")

    # 4. Start Agent
    print("🧠 Starting LLM Agent (inference.py)...")
    print("   Mode: " + ("LLM (ReAct)" if os.environ.get("USE_LLM") == "true" else "Deterministic (Optimal)"))
    
    agent_env = {**os.environ, "USE_LLM": os.environ.get("USE_LLM", "false")}
    
    # Run agent in current process to see output
    try:
        subprocess.run([sys.executable, "inference.py"], env=agent_env, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted.")
    finally:
        print("🧹 Cleaning up...")
        backend_proc.terminate()
        backend_proc.wait()

if __name__ == "__main__":
    run_demo()
