import pyaudio
import time
import threading
import os
import io
import wave


from faster_whisper import WhisperModel

# TODO: Switch from threading to multiprocessing module to utulize CPU more effectively for larger models
#       Set up algorithm to chunk audio and text based off of periods of silence
#       Feed text chunks to LLM


###--- Audio recording parameters ---###
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 1024
RECORD_SECONDS = 2  # Interval for saving audio
OVERLAP_SECONDS = 1  # Interval for overlapping audio
device_id = 6
###--- End Audio recording parameters ---###

###############################################################################################

###--- Functions ---###

def transcribe_audio_chunk(filename, model):

    trans_chunk = False
    segments, info = model.transcribe(filename, beam_size=5, vad_filter=True, word_timestamps=True)

    #print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    for segment in segments:
        for word in segment.words:
            
            #check words that exist across chunks.  These words would be duplicated in the output
            if word.start <= RECORD_SECONDS + OVERLAP_SECONDS and word.end >= RECORD_SECONDS + OVERLAP_SECONDS and not trans_chunk:
                trans_chunk = True
                #print("(trans chunk word) [%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
                print(word.word, end="", flush=True)
            if (word.end <= RECORD_SECONDS+OVERLAP_SECONDS and word.start >= OVERLAP_SECONDS):
                if not trans_chunk:
                    #print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
                    print(word.word, end="", flush=True)
                trans_chunk = False

# process_audio_chunks handles the audio processing, taking the frames and reference to the data model as inputs
def process_audio_chunk(frames, model, sample_width):
    
    #convert frames into wav file like BytesIO object
    wav_stream = io.BytesIO()
    with wave.open(wav_stream, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(sample_width)
        wf.setframerate(RATE)
        for frame in frames:
            wf.writeframes(frame)
    wav_stream.seek(0)
    return wav_stream
    
    
# start recording_overlap handles reading the data from audio stream and feeding it to the transcription pipeline
def start_recording_overlap(stream, model, sample_width):
    
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

                # Process the chunk
                wav_stream = process_audio_chunk(processing_frames, model, sample_width)
                #Transcribe the chunk in a separate thread
                processing_thread = threading.Thread(target=transcribe_audio_chunk, args=(wav_stream, model))
                processing_thread.start()  
    
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")

#main function
def main():
    
    ###--------- Model Setup ---------###
    model_size = "tiny.en"
    #model_size = "base.en"

    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    ###--------- End Model Setup ---------###

    ###---------Setup Audio Stream---------###
    pa = pyaudio.PyAudio()
    sample_width = pa.get_sample_size(FORMAT)
    stream = pa.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_id,
                    frames_per_buffer=CHUNK)

    input("Program started. Press any key to start recording.  Press Ctrl+C to stop.")
    start_recording_overlap(stream, model, sample_width)
    ###---------End Setup Audio Stream---------###

    #Cleanup 
    stream.stop_stream()
    stream.close()
    pa.terminate()

###--- End Functions ---### 
#####################################################################################################

if __name__ == "__main__":
    main()