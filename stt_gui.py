import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel
from scipy.signal import resample

import webrtcvad

# Global state
audio_queue = queue.Queue()
is_listening = False
document_lines = []

# Load Whisper model once
print("[INFO] Loading Whisper model...")
model = WhisperModel("medium", compute_type="int8")
print("[INFO] Model loaded.")

def update_text_output():
    text_box.delete(1.0, tk.END)
    text_box.insert(tk.END, "\n".join(document_lines))

def handle_command(text):
    global document_lines
    if "highlight paragraph" in text:
        print("[COMMAND] Detected: highlight paragraph")
        document_lines.append("[[HIGHLIGHT PARAGRAPH]]")
    elif "up on line" in text:
        print("[COMMAND] Detected: up on line")
        document_lines.append("[[MOVE UP LINE]]")
    else:
        print("[TEXT] Appending:", text)
        document_lines.append(text)
    update_text_output()

def transcribe_audio(input_device, sample_rate):
    global is_listening

    vad = webrtcvad.Vad()
    vad.set_mode(0)  # 0 = most sensitive, 3 = least; 2 is a good balance

    buffer = []

    def callback(indata, frames, time, status):
        if status:
            print(f"[AUDIO STATUS] {status}")
        audio_queue.put(indata.copy())

    try:
        with sd.InputStream(callback=callback,
                            channels=1,
                            samplerate=sample_rate,
                            device=input_device,
                            dtype='float32'):
            print(f"[INFO] Listening started with device #{input_device} at {sample_rate} Hz")

            while is_listening:
                chunk = audio_queue.get()
                chunk = np.squeeze(chunk)

                if chunk.ndim > 1:
                    chunk = np.mean(chunk, axis=1)

                # Convert to 16-bit PCM (webrtcvad expects mono 16-bit)
                pcm_chunk = (chunk * 32767).astype(np.int16).tobytes()

                # 30 ms frames for VAD
                frame_duration_ms = 30
                frame_size = int(sample_rate * frame_duration_ms / 1000)
                is_speech = False

                for i in range(0, len(chunk) - frame_size, frame_size):
                    frame = pcm_chunk[i*2:(i+frame_size)*2]  # *2 for bytes
                    if vad.is_speech(frame, sample_rate):
                        is_speech = True
                        break

                if is_speech:
                    print("[VAD] Speech detected.")
                    buffer.extend(chunk)
                else:
                    print("[VAD] Silence skipped.")

                # Transcribe when buffer hits ~2 seconds
                if len(buffer) >= 2 * sample_rate:
                    audio_np = np.array(buffer[:2 * sample_rate])
                    buffer = buffer[2 * sample_rate:]

                    if sample_rate != 16000:
                        target_samples = int(len(audio_np) * 16000 / sample_rate)
                        audio_np = resample(audio_np, target_samples)

                    print(f"[INFO] Transcribing {len(audio_np)} samples")
                    segments_gen, _ = model.transcribe(audio_np, language="en", beam_size=5, best_of=5)
                    segments = list(segments_gen)
                    print(f"[DEBUG] Segments returned: {segments}")

                    for segment in segments:
                        text = segment.text.strip().lower()
                        print(f"[TRANSCRIBED] '{text}'")
                        if text:
                            handle_command(text)

    except Exception as e:
        print(f"[ERROR] in audio stream: {e}")

def start_transcription():
    global is_listening
    is_listening = True
    selected_device = device_combo.get()
    print(f"[INFO] Starting transcription with device: {selected_device}")

    device_index = next((d['index'] for d in sd.query_devices() if d['name'] == selected_device), None)

    if device_index is not None:
        device_info = sd.query_devices(device_index)
        supported_samplerate = int(device_info['default_samplerate'])
        print(f"[INFO] Using sample rate: {supported_samplerate}")

        threading.Thread(target=transcribe_audio,
                         args=(device_index, supported_samplerate),
                         daemon=True).start()

        status_label.config(text="Listening...", foreground="green")
    else:
        status_label.config(text="Device not found.", foreground="red")
        print("[ERROR] Device not found.")

def stop_transcription():
    global is_listening
    is_listening = False
    status_label.config(text="Stopped.", foreground="red")
    print("[INFO] Transcription stopped.")

def save_text():
    with open("output.txt", "w") as f:
        f.write("\n".join(document_lines))
    status_label.config(text="Saved to output.txt", foreground="blue")
    print("[INFO] Transcription saved to output.txt")

# Tkinter GUI setup
root = tk.Tk()
root.title("Voice Transcriber with Commands")

frame = ttk.Frame(root, padding=10)
frame.grid(row=0, column=0, sticky="nsew")

# Input device dropdown
ttk.Label(frame, text="Input Device:").grid(row=0, column=0, sticky="w")
device_names = [d['name'] for d in sd.query_devices() if d['max_input_channels'] > 0]
device_combo = ttk.Combobox(frame, values=device_names, width=50)
device_combo.grid(row=0, column=1, sticky="ew")
if device_names:
    device_combo.current(0)

# Buttons
ttk.Button(frame, text="Start", command=start_transcription).grid(row=1, column=0, pady=5)
ttk.Button(frame, text="Stop", command=stop_transcription).grid(row=1, column=1, pady=5, sticky="w")
ttk.Button(frame, text="Save", command=save_text).grid(row=1, column=1, pady=5, sticky="e")

# Status label
status_label = ttk.Label(frame, text="Idle.", foreground="gray")
status_label.grid(row=2, column=0, columnspan=2, pady=5)

# Output text box
text_box = tk.Text(frame, height=20, width=80)
text_box.grid(row=3, column=0, columnspan=2, pady=10)

root.mainloop()
