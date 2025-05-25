import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel

# Global variables
audio_queue = queue.Queue()
is_listening = False
stream = None
document_lines = []

# Whisper model (load at startup)
model = WhisperModel("base", compute_type="int8")

# Transcription thread
def transcribe_audio(input_device, sample_rate=16000):
    global is_listening, stream

    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_queue.put(indata.copy())

    try:
        with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate,
                            device=input_device, dtype='float32'):
            while is_listening:
                audio_data = audio_queue.get()
                audio_np = np.squeeze(audio_data)
                audio_pcm = (audio_np * 32767).astype(np.int16).tobytes()

                segments, _ = model.transcribe(audio_pcm, language="en", beam_size=1)
                for segment in segments:
                    text = segment.text.strip().lower()
                    if text:
                        handle_command(text)
    except Exception as e:
        print(f"Error in audio stream: {e}")

# Handle commands
def handle_command(text):
    global document_lines
    if "highlight paragraph" in text:
        document_lines.append("[[HIGHLIGHT PARAGRAPH]]")
    elif "up on line" in text:
        document_lines.append("[[MOVE UP LINE]]")
    else:
        document_lines.append(text)
    update_text_output()

def update_text_output():
    text_box.delete(1.0, tk.END)
    text_box.insert(tk.END, "\n".join(document_lines))

def start_transcription():
    global is_listening
    is_listening = True
    selected_device = device_combo.get()
    device_index = next((d['index'] for d in sd.query_devices() if d['name'] == selected_device), None)

    if device_index is not None:
        threading.Thread(target=transcribe_audio, args=(device_index,), daemon=True).start()
        status_label.config(text="Listening...", foreground="green")
    else:
        status_label.config(text="Device not found.", foreground="red")

def stop_transcription():
    global is_listening
    is_listening = False
    status_label.config(text="Stopped.", foreground="red")

def save_text():
    with open("output.txt", "w") as f:
        f.write("\n".join(document_lines))
    status_label.config(text="Saved to output.txt", foreground="blue")

# Tkinter GUI
root = tk.Tk()
root.title("Voice Transcriber with Commands")

frame = ttk.Frame(root, padding=10)
frame.grid(row=0, column=0, sticky="nsew")

# Input device dropdown
ttk.Label(frame, text="Input Device:").grid(row=0, column=0, sticky="w")
device_names = [d['name'] for d in sd.query_devices() if d['max_input_channels'] > 0]
device_combo = ttk.Combobox(frame, values=device_names, width=50)
device_combo.grid(row=0, column=1, sticky="ew")
device_combo.current(0)

# Buttons
ttk.Button(frame, text="Start", command=start_transcription).grid(row=1, column=0, pady=5)
ttk.Button(frame, text="Stop", command=stop_transcription).grid(row=1, column=1, pady=5, sticky="w")
ttk.Button(frame, text="Save", command=save_text).grid(row=1, column=1, pady=5, sticky="e")

# Status
status_label = ttk.Label(frame, text="Idle.", foreground="gray")
status_label.grid(row=2, column=0, columnspan=2, pady=5)

# Text output box
text_box = tk.Text(frame, height=20, width=80)
text_box.grid(row=3, column=0, columnspan=2, pady=10)

root.mainloop()
