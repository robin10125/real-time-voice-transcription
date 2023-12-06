import pyaudio
import wave
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


###--- Audio recording parameters ---###
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 1024
RECORD_SECONDS = 5  # Interval for saving audio
OVERLAP_SECONDS = 1  # Interval for overlapping audio
device_id = 6
###--- End Audio recording parameters ---###

def transcribe_audio_chunk(filename, model):
    
    #"/home/robin/Documents/Tabletop-Assistant/Speech-To-Text/gettysburg10.wav"
    segments, info = model.transcribe(filename, beam_size=5, vad_filter=True)

    #print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

# Function to save recorded audio chunks
def save_audio_chunk(frames, model):
    file_name = f"output_{int(time.time())}.wav"
    wf = wave.open(file_name, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pa.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Saved: {file_name}")

    processing_thread = threading.Thread(target=transcribe_audio_chunk, args=(file_name, model))
    processing_thread.start()
###-------------------------------------------###
def start_recording_overlap(stream, model):
    
    current_frames = []
    overlap_frames = []
    next_chunk_start_time = int(time.time())

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            current_frames.append(data)

            # Calculate elapsed time
            elapsed_time = len(current_frames) * CHUNK / RATE

            if elapsed_time >= RECORD_SECONDS + OVERLAP_SECONDS:
                # Prepare the chunk for processing
                processing_frames = overlap_frames + current_frames
                processing_start_time = next_chunk_start_time

                # Update overlap for the next chunk
                overlap_frames = current_frames[-int(OVERLAP_SECONDS * RATE / CHUNK):]
                current_frames = overlap_frames.copy()

                # Update the start time for the next chunk
                next_chunk_start_time = int(time.time())

                save_audio_chunk(processing_frames, model)

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
                save_audio_chunk(current_frames, model)
                current_frames = []  # Reset frames
                start_time = time.time()  # Reset timer

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
        
    # Save any remaining audio
    if current_frames:
        save_audio_chunk(current_frames, model)

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