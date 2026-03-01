"""
Test Chat GUI — Simple web interface to test the voice assistant.
Open http://localhost:5000 in your browser.
Type a message → LLM responds (Streaming) → TTS speaks.
"""
import sys
import os
import io
import time
import json
import base64
import wave
import numpy as np
import requests
import threading
from flask import Flask, render_template_string, request, jsonify, Response

sys.path.insert(0, os.path.dirname(__file__))
import config

app = Flask(__name__)

# --- State ---
tts_engine = None
tts_lock = threading.Lock()
SERVER_URL = config.LOCAL_SERVER_URL  # http://localhost:8401

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice Assistant — Test Chat</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: #0a0a1a;
            color: #e0e0e0;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: linear-gradient(135deg, #1a1a3e 0%, #0d0d2b 100%);
            padding: 16px 24px;
            border-bottom: 1px solid rgba(100, 100, 255, 0.15);
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .header h1 {
            font-size: 18px;
            font-weight: 600;
            background: linear-gradient(135deg, #7c7cff, #4ecdc4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .status {
            font-size: 12px;
            color: #888;
            margin-left: auto;
        }
        .status .dot { 
            display: inline-block;
            width: 8px; height: 8px;
            border-radius: 50%;
            margin-right: 4px;
            vertical-align: middle;
        }
        .dot.green { background: #4ecdc4; box-shadow: 0 0 6px #4ecdc4; }
        .dot.red { background: #ff4444; }
        .dot.yellow { background: #ffaa00; animation: pulse 1s infinite; }
        @keyframes pulse { 50% { opacity: 0.5; } }

        .chat {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .msg {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 16px;
            line-height: 1.5;
            font-size: 14px;
            animation: fadeIn 0.3s ease;
            white-space: pre-wrap;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } }
        .msg.user {
            align-self: flex-end;
            background: linear-gradient(135deg, #4a4aff, #3a3ad0);
            border-bottom-right-radius: 4px;
        }
        .msg.assistant {
            align-self: flex-start;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            border-bottom-left-radius: 4px;
        }
        .msg.system {
            align-self: center;
            font-size: 12px;
            color: #666;
            background: none;
            padding: 4px;
        }
        .msg .meta {
            font-size: 11px;
            color: rgba(255,255,255,0.4);
            margin-top: 6px;
        }
        .thinking {
            align-self: flex-start;
            padding: 12px 20px;
            font-size: 14px;
            color: #888;
        }
        .thinking::after {
            content: '...';
            animation: dots 1.5s infinite;
        }
        @keyframes dots {
            0%, 20% { content: '.'; }
            40% { content: '..'; }
            60%, 100% { content: '...'; }
        }

        .input-area {
            padding: 16px 20px;
            background: #0f0f2a;
            border-top: 1px solid rgba(100,100,255,0.1);
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .input-area input {
            flex: 1;
            padding: 12px 16px;
            border-radius: 12px;
            border: 1px solid rgba(100,100,255,0.2);
            background: rgba(255,255,255,0.04);
            color: #e0e0e0;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s;
        }
        .input-area input:focus {
            border-color: rgba(100,100,255,0.5);
        }
        .input-area button {
            padding: 12px 20px;
            border-radius: 12px;
            border: none;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-send {
            background: linear-gradient(135deg, #4a4aff, #3a3ad0);
            color: white;
        }
        .btn-send:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(74,74,255,0.3); }
        .btn-send:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .btn-record {
            background: rgba(255,60,60,0.15);
            color: #ff6b6b;
            border: 1px solid rgba(255,60,60,0.3);
            min-width: 44px;
        }
        .btn-record:hover { background: rgba(255,60,60,0.25); }
        .btn-record.recording {
            background: #ff4444;
            color: white;
            animation: pulse 0.8s infinite;
        }
        .tts-toggle {
            font-size: 11px;
            color: #666;
            display: flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
            user-select: none;
        }
        .tts-toggle input { accent-color: #4ecdc4; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Voice Assistant — Test Chat</h1>
        <div class="status" id="status">
            <span class="dot yellow"></span> Checking...
        </div>
        <label class="tts-toggle">
            <input type="checkbox" id="ttsEnabled" checked> TTS
        </label>
    </div>

    <div class="chat" id="chat"></div>

    <div class="input-area">
        <button class="btn-record" id="btnRecord" title="Hold to record audio">🎤</button>
        <input type="text" id="msgInput" placeholder="Type a message..." autofocus>
        <button class="btn-send" id="btnSend">Send</button>
    </div>

    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('msgInput');
        const btnSend = document.getElementById('btnSend');
        const btnRecord = document.getElementById('btnRecord');
        const statusEl = document.getElementById('status');
        const ttsEnabled = document.getElementById('ttsEnabled');
        let busy = false;

        // Check server health
        async function checkHealth() {
            try {
                const r = await fetch('/health');
                const d = await r.json();
                statusEl.innerHTML = `<span class="dot green"></span> ${d.backend} · ${d.gpu} · ${d.vram_gb} GB`;
            } catch {
                statusEl.innerHTML = '<span class="dot red"></span> Server offline';
            }
        }
        checkHealth();
        setInterval(checkHealth, 30000);

        function addMsg(role, text, meta) {
            const div = document.createElement('div');
            div.className = `msg ${role}`;
            div.textContent = text;
            if (meta) {
                const m = document.createElement('div');
                m.className = 'meta';
                m.textContent = meta;
                div.appendChild(m);
            }
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            return div;
        }

        function createStreamMsg() {
            const div = document.createElement('div');
            div.className = `msg assistant`;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            return div;
        }

        function playBase64Audio(b64) {
            if (!b64) return;
            const audio = new Audio("data:audio/wav;base64," + b64);
            audio.play().catch(e => console.error("Audio play failed", e));
        }

        async function streamChat(bodyParams) {
            busy = true;
            btnSend.disabled = true;
            input.value = '';
            
            const msgDiv = createStreamMsg();
            let fullText = "";
            let startTime = performance.now();

            try {
                const r = await fetch('/chat_stream', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(bodyParams)
                });
                
                const reader = r.body.getReader();
                const decoder = new TextDecoder("utf-8");
                
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value, {stream: true});
                    const lines = chunk.split('\\n');
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const dataStr = line.replace('data: ', '');
                            try {
                                const data = JSON.parse(dataStr);
                                if (data.chunk) {
                                    fullText += data.chunk;
                                    msgDiv.textContent = fullText;
                                    chat.scrollTop = chat.scrollHeight;
                                }
                                if (data.done) {
                                    const timeTaken = ((performance.now() - startTime)/1000).toFixed(1);
                                    const m = document.createElement('div');
                                    m.className = 'meta';
                                    m.textContent = `${timeTaken}s LLM`;
                                    msgDiv.appendChild(m);
                                    
                                    if (ttsEnabled.checked && fullText) {
                                        fetchTTSAndPlay(fullText);
                                    }
                                }
                                if (data.error) {
                                    msgDiv.textContent = `Error: ${data.error}`;
                                }
                            } catch (e) {}
                        }
                    }
                }
            } catch (e) {
                addMsg('system', `Stream Error: ${e.message}`);
            }
            
            busy = false;
            btnSend.disabled = false;
            input.focus();
        }

        async function fetchTTSAndPlay(text) {
            try {
                const r = await fetch('/tts_only', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: text})
                });
                const d = await r.json();
                if (d.audio_b64) {
                    playBase64Audio(d.audio_b64);
                }
            } catch (e) {
                console.error("TTS fetch failed", e);
            }
        }

        btnSend.onclick = () => { if(input.value.trim()) { addMsg('user', input.value); streamChat({text: input.value}); }};
        input.onkeydown = (e) => { if (e.key === 'Enter' && input.value.trim()) { addMsg('user', input.value); streamChat({text: input.value}); }};

        // --- Mic recording to WAV Conversion ---
        function bufferToWave(abuffer, len) {
            let numOfChan = abuffer.numberOfChannels,
                length = len * numOfChan * 2 + 44,
                buffer = new ArrayBuffer(length),
                view = new DataView(buffer),
                channels = [], i, sample,
                offset = 0,
                pos = 0;

            function setUint16(data) { view.setUint16(pos, data, true); pos += 2; }
            function setUint32(data) { view.setUint32(pos, data, true); pos += 4; }

            setUint32(0x46464952); setUint32(length - 8); setUint32(0x45564157); 
            setUint32(0x20746d66); setUint32(16); setUint16(1); setUint16(numOfChan);
            setUint32(abuffer.sampleRate); setUint32(abuffer.sampleRate * 2 * numOfChan);
            setUint16(numOfChan * 2); setUint16(16); setUint32(0x61746164); setUint32(length - pos - 4);

            for(i = 0; i < abuffer.numberOfChannels; i++) channels.push(abuffer.getChannelData(i));

            while(pos < length) {
                for(i = 0; i < numOfChan; i++) {
                    sample = Math.max(-1, Math.min(1, channels[i][offset]));
                    sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767) | 0;
                    view.setInt16(pos, sample, true); pos += 2;
                }
                offset++;
            }
            return new Blob([buffer], {type: "audio/wav"});
        }

        let mediaRecorder, audioChunks;
        btnRecord.onmousedown = btnRecord.ontouchstart = async (e) => {
            if(busy) return;
            e.preventDefault();
            try {
                const stream = await navigator.mediaDevices.getUserMedia({audio: true});
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                mediaRecorder.start();
                btnRecord.classList.add('recording');
                btnRecord.textContent = '⏹';
            } catch (err) {
                addMsg('system', 'Mic access denied: ' + err.message);
            }
        };
        btnRecord.onmouseup = btnRecord.ontouchend = async () => {
            if (!mediaRecorder || mediaRecorder.state !== 'recording') return;
            btnRecord.classList.remove('recording');
            btnRecord.textContent = '🎤';
            mediaRecorder.stop();
            mediaRecorder.onstop = async () => {
                const blob = new Blob(audioChunks, {type: 'audio/webm'});
                
                // Convert WebM to WAV
                const arrayBuffer = await blob.arrayBuffer();
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                if(arrayBuffer.byteLength === 0) return;
                const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                const wavBlob = bufferToWave(audioBuffer, audioBuffer.length);
                
                const reader = new FileReader();
                reader.onloadend = async () => {
                    const b64 = reader.result.split(',')[1];
                    addMsg('user', '🎤 [Audio message]');
                    streamChat({audio_b64: b64});
                };
                reader.readAsDataURL(wavBlob);
            };
        };
    </script>
</body>
</html>
"""

def np_to_wav_base64(audio_np, sample_rate):
    """Convert numpy array (float32 or int16) to a base64 encoded WAV string."""
    if audio_np is None:
        return ""
    if audio_np.dtype in (np.float32, np.float64):
        audio_np = np.nan_to_num(audio_np, nan=0.0)
        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_int16 = (audio_np * 32767).astype(np.int16)
    else:
        audio_int16 = audio_np.astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/health")
def health():
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=3)
        return jsonify(r.json())
    except:
        return jsonify({"status": "offline", "backend": "none", "gpu": "none", "vram_gb": 0})


@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    """Stream LLM response and handle audio/text seamlessly."""
    data = request.json
    text = data.get("text", "")
    audio_b64 = data.get("audio_b64", "")

    json_payload = {
        "system_prompt": config.SYSTEM_PROMPT,
        "max_new_tokens": 256,
        "stream": True
    }
    if text:
        json_payload["text"] = text
    if audio_b64:
        json_payload["audio_b64"] = audio_b64

    def generate():
        try:
            with requests.post(f"{SERVER_URL}/infer", json=json_payload, stream=True, timeout=120) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith("data: "):
                            # Forward the SSE data straight to frontend
                            yield f"{line_str}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(generate(), mimetype="text/event-stream")


@app.route("/tts_only", methods=["POST"])
def tts_only():
    """Endpoint specifically for the browser to request audio for text."""
    data = request.json
    text = data.get("text", "")
    
    audio_b64 = ""
    try:
        audio_np, sr = _speak(text, play_audio=False)
        if audio_np is not None:
            audio_b64 = np_to_wav_base64(audio_np, sr)
    except Exception as e:
        print(f"[TTS Error] {e}")
        
    return jsonify({"audio_b64": audio_b64})


def _speak(text, play_audio=False):
    """Lazy-load TTS and speak. Returns audio context."""
    global tts_engine
    with tts_lock:
        if tts_engine is None:
            print("[GUI] Loading TTS engine...")
            from tts_engine import TTSEngine
            tts_engine = TTSEngine()
            print("[GUI] TTS ready!")
        return tts_engine.speak(text, play_audio=play_audio)


if __name__ == "__main__":
    print("=" * 50)
    print("Test Chat GUI — http://localhost:5000")
    print("=" * 50)
    print(f"LLM Server: {SERVER_URL}")
    print(f"TTS Backend: {config.TTS_BACKEND}")
    print()
    app.run(host="0.0.0.0", port=5000, debug=False)

