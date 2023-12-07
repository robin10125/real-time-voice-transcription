import sounddevice as sd
import numpy as np
import wave
import queue
import time
import threading
from faster_whisper import WhisperModel


###--- Model Setup ---###
model_size = "tiny.en"
# Run on GPU with FP16
#model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
model = WhisperModel(model_size, device="cpu", compute_type="int8")

###--- End Model Setup ---###

###---Constants---###
CHANNELS = 1
RATE = 48000
CHUNK = 1024
RECORD_SECONDS = 5  # Interval for saving audio
OVERLAP_SECONDS = 1  # Interval for overlapping audio
device_id = 6  # Device ID (check device ID using sounddevice.query_devices())

###---End Constants---###

def transcribe_audio_chunk(filename, model):
    
    #"/home/robin/Documents/Tabletop-Assistant/Speech-To-Text/gettysburg10.wav"
    segments, info = model.transcribe(filename, beam_size=5, vad_filter=True)

    #print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        #print(segment.text, end="/")

# Function to save recorded audio chunks
def save_audio_chunk(frames, file_name):
    
    audio_data = np.concatenate(frames)
    int_data = np.int16(audio_data * 32767)  # Convert to int16 format

    wf = wave.open(file_name, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)  # 2 bytes per sample for int16
    wf.setframerate(RATE)
    wf.writeframes(int_data.tobytes())
    wf.close()

    processing_thread = threading.Thread(target=transcribe_audio_chunk, args=(file_name, model))
    processing_thread.start()

# Callback function to capture audio
def callback(indata, frames, time, status):
    # This function will be called for each audio block
    #if status:
        #print(status)
    q.put(indata.copy())

# Queue to hold audio chunks
q = queue.Queue()

# Open an audio stream
with sd.InputStream(device=device_id, channels=CHANNELS, samplerate=RATE, callback=callback, blocksize=CHUNK):
    print("Recording started. Press Ctrl+C to stop.")
    current_frames = []
    overlap_frames = []

    try:
        while True:

            ###----------------------------------------------###
            data = q.get()
            current_frames.append(data)

            # Check if it's time to save the chunk
            if len(current_frames) * CHUNK >= (RECORD_SECONDS + 2*OVERLAP_SECONDS) * RATE:
                # Prepare complete chunk with overlap
                complete_chunk = np.concatenate(overlap_frames + current_frames[:-int(OVERLAP_SECONDS * RATE / CHUNK)])

                # Save the chunk
                save_audio_chunk([complete_chunk], f"output_{int(time.time())}.wav")
                
                # Update overlap for the next chunk
                overlap_frames = current_frames[-int(OVERLAP_SECONDS * RATE / CHUNK):]
                current_frames = overlap_frames.copy()
            ###----------------------------------------------###

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
        if current_frames:
            complete_chunk = overlap_frames + current_frames
            save_audio_chunk(complete_chunk, f"output_{int(time.time())}.wav")