import pyaudio
import wave
import time
import threading
import os
import io


from faster_whisper import WhisperModel

# TODO: Switch from threading to multiprocessing module to utulize CPU more effectively for larger models
#       Set up algorithm to chunk audio and text based off of periods of silence
#       Feed text chunks to LLM
#       Shortcut disk read write cycle for audio chunks by using io.BytesIO() instead of wave.open()


###--- Audio recording parameters ---###
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 1024
RECORD_SECONDS = 4  # Interval for saving audio
OVERLAP_SECONDS = 1  # Interval for overlapping audio
device_id = 6
###--- End Audio recording parameters ---###

###############################################################################################


###--- Functions ---###

def transcribe_audio_chunk(filename, model, callback=None):
    
    #start= time.time()

    trans_chunk = False
    segments, info = model.transcribe(filename, beam_size=5, vad_filter=True, word_timestamps=True)

    #print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    for segment in segments:
        for word in segment.words:
            
            #check words that exist across chunks.  These words would be duplicated in the output
            if word.start <= RECORD_SECONDS + OVERLAP_SECONDS and word.end >= RECORD_SECONDS + OVERLAP_SECONDS and not trans_chunk:
                trans_chunk = True
                print("(trans chunk word) [%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
                #print(word.word, end="", flush=True)
            if (word.end <= RECORD_SECONDS+OVERLAP_SECONDS and word.start >= OVERLAP_SECONDS):
                if not trans_chunk:
                    print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
                    #print(word.word, end="", flush=True)
                trans_chunk = False
    #end = time.time()
    #print(end-start)
    if callback:
        callback(filename)

# Function to save recorded audio chunks
def process_audio_chunk(frames, model, repository_name):
    
    file_name = f"{repository_name}/output_{int(time.time())}.wav"
    wf = wave.open(file_name, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pa.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    '''print("start pre-processing audio chunk")
    audio_data = b''.join(frames)
    
    audio_stream = io.BytesIO(audio_data)
    audio_stream.seek(0)

    print("end pre-processing audio chunk")'''
    

    processing_thread = threading.Thread(target=transcribe_audio_chunk, args=(file_name, model))
    #print("start thread")
    #processing_thread = threading.Thread(target=transcribe_audio_chunk, args=(audio_stream, model))
    processing_thread.start()
    

def start_recording_overlap(stream, model):
    
    #frames holds the current audio chunk, overlap_frames holds a small portion of the end the previous chunk
    current_frames = []
    overlap_frames = []

    next_chunk_start_time = int(time.time())

    repository_name = f"session_file_chunks{next_chunk_start_time}"
    os.makedirs(repository_name, exist_ok=True)
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            current_frames.append(data)

            # Calculate elapsed time
            elapsed_time = len(current_frames) * CHUNK / RATE

            if elapsed_time >= RECORD_SECONDS + 2*OVERLAP_SECONDS:
                
                # Prepare the chunk for processing
                processing_frames = current_frames

                # Update overlap for the next chunk
                overlap_frames = current_frames[-int(2*OVERLAP_SECONDS * RATE / CHUNK):]
                current_frames = overlap_frames.copy()

                # Update the start time for the next chunk
                next_chunk_start_time = int(time.time())
                process_audio_chunk(processing_frames, model, repository_name)

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")

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

###--- End Functions ---###

###############################################################################################

###--- Model Setup ---###
model_size = "tiny.en"
#model_size = "small.en"
# Run on GPU with FP16
#model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
model = WhisperModel(model_size, device="cpu", compute_type="int8")
###--- End Model Setup ---###

###############################################################################################


###----------Setup Audio Stream----------###
pa = pyaudio.PyAudio()

# Open stream
stream = pa.open(format=FORMAT,
                 channels=CHANNELS,
                 rate=RATE,
                 input=True,
                 input_device_index=device_id,
                 frames_per_buffer=CHUNK)

input("Program started. Press any key to start recording.  Press Ctrl+C to stop.")
start_recording_overlap(stream, model)
###----------End Setup Audio Stream----------###

# Cleanup
stream.stop_stream()
stream.close()
pa.terminate()