import pyaudio
import wave
import time
import threading
import os
from faster_whisper import WhisperModel


###--- Audio recording parameters ---###
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 1024
RECORD_SECONDS = 4  # Interval for saving audio
OVERLAP_SECONDS = 2  # Interval for overlapping audio
device_id = 6
###--- End Audio recording parameters ---###

def transcribe_audio_chunk(filename, model, callback=None):
    
    #"/home/robin/Documents/Tabletop-Assistant/Speech-To-Text/gettysburg10.wav"
    segments, info = model.transcribe(filename, beam_size=5, vad_filter=True)

    #print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    for segment in segments:
        
        #print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        print(segment.text, end=" ", flush=True)


# Function to save recorded audio chunks
def process_audio_chunk(frames, model, repository_name):
    file_name = f"{repository_name}/output_{int(time.time())}.wav"
    wf = wave.open(file_name, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pa.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    #print(f"Saved: {file_name}")

    processing_thread = threading.Thread(target=transcribe_audio_chunk, args=(file_name, model))
    processing_thread.start()

def delete_file(file_path):
    # Logic to delete the file
    print(f"Deleting {file_path}")
    os.remove(file_path)

def start_recording_overlap(stream, model):
    
    #frames holds the current audio chunk, overlap_frames holds a small portion of the end the previous chunk
    current_frames = []
    overlap_frames = []
    session_chunk_frames = []

    next_chunk_start_time = int(time.time())

    session_name = f"session_file_at_{next_chunk_start_time}.wav"

    repository_name = f"session_file_chunks{next_chunk_start_time}"
    os.makedirs(repository_name, exist_ok=True)
    loop_count = 0
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            current_frames.append(data)

            # Calculate elapsed time
            elapsed_time = len(current_frames) * CHUNK / RATE

            if elapsed_time >= RECORD_SECONDS + 2*OVERLAP_SECONDS:
                # Prepare the chunk for processing
                processing_frames = current_frames
                processing_start_time = next_chunk_start_time

                # Isolate non-overlapping frames by cutting out OVERLAP_SECONDS of frames from the start and end
                session_chunk_frames = current_frames[int(OVERLAP_SECONDS * RATE / CHUNK):-int(OVERLAP_SECONDS * RATE / CHUNK)]

                # Update overlap for the next chunk
                overlap_frames = current_frames[-int(OVERLAP_SECONDS * RATE / CHUNK):]
                current_frames = overlap_frames.copy()

                # Update the start time for the next chunk
                next_chunk_start_time = int(time.time())
                process_audio_chunk(processing_frames, model, repository_name)

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")

###-------------------------------------------###
def start_recording(stream, model):
    start_time = time.time()
    current_frames = []

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            current_frames.append(data)

            # Check if 10 seconds have passed and save the audio chunk
            if time.time() - start_time >= RECORD_SECONDS:
                process_audio_chunk(current_frames, model)
                current_frames = []  # Reset frames
                start_time = time.time()  # Reset timer

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
        
    # Save any remaining audio
    if current_frames:
        process_audio_chunk(current_frames, model)
###-------------------------------------------###

###--- Model Setup ---###
model_size = "tiny.en"
# Run on GPU with FP16
#model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
model = WhisperModel(model_size, device="cpu", compute_type="int8")
###--- End Model Setup ---###

# Initialize PyAudio
pa = pyaudio.PyAudio()

# Open stream
stream = pa.open(format=FORMAT,
                 channels=CHANNELS,
                 rate=RATE,
                 input=True,
                 input_device_index=device_id,
                 frames_per_buffer=CHUNK)

print("Recording started. Press Ctrl+C to stop.")
start_recording_overlap(stream, model)

# Cleanup
stream.stop_stream()
stream.close()
pa.terminate()