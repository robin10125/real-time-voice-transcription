import pyaudio
import time
import threading
import os
import io
import wave

import multiprocessing

from faster_whisper import WhisperModel

# TODO: Switch from threading to multiprocessing module to utulize CPU more effectively for larger models
#       Set up algorithm to chunk audio and text based off of periods of silence
#       Feed text chunks to LLM
#       Deal with multiprocessing termination
#       Set up queue structure

###--- Audio recording parameters ---###
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 1024
RECORD_SECONDS = 4  # Interval for saving audio
OVERLAP_SECONDS = 2  # Interval for overlapping audio
device_id = 6
###--- End Audio recording parameters ---###

###############################################################################################
######------ Test code for multiprocessing ------######

def model_server(input_queue, output_queue):
    
    model = initialize_model()
    while True:
        #Receive audio data from the recording program
        #TODO check sync of input_queue.get
        audio_data = input_queue.get()


        # NOTE: Temporary shutdown signal
        if audio_data is None:  
            output_queue.put(None)  # Signal display process to shut down
            break

        # Transcribe the audio data into segments of text
        segments = model.transcribe(audio_data, beam_size=5, vad_filter=True, word_timestamps=True) 

        # Send the result to the display program
        output_queue.put(process_transcript(segments))

def process_transcript(segments):

    trans_chunk = False
    output_string= ""
    #print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    for segment in segments:
        for word in segment.words:
            
            #check words that exist across chunks.  These words would be duplicated in the output
            if word.start <= RECORD_SECONDS + OVERLAP_SECONDS and word.end >= RECORD_SECONDS + OVERLAP_SECONDS and not trans_chunk:
                trans_chunk = True
                #output_string += "(trans chunk word) [%.2fs -> %.2fs] %s" % (word.start, word.end, word.word)
                output_string += word.word
                
            if (word.end <= RECORD_SECONDS+OVERLAP_SECONDS and word.start >= OVERLAP_SECONDS):
                if not trans_chunk:
                    #output_string += "[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word)
                    output_string += word.word
                trans_chunk = False

    return output_string

def initialize_model(size="tiny.en"):

    model_size = size
    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return model


def output_transcript(output_queue):
    while True:
        text_data = output_queue.get()
        if text_data is None:
            print("\nDisplay process terminated.")
            break
        print(text_data, end="", flush=True)


######------ End Test code for multiprocessing ------######
###############################################################################################

###--- Functions ---###

#transcribe_audio_chunk handles the transcription of the audio chunk
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

# process_audio_chunks handles the audio processing turning frames into a file like object
def process_audio_chunk(frames, sample_width):
    
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
def start_recording_overlap(stream, sample_width, input_queue):
    
    #frames holds the current audio chunk, overlap_frames holds a small portion of the end the previous chunk
    current_frames = []
    overlap_frames = []

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            current_frames.append(data)

            # Calculate elapsed time
            elapsed_time = len(current_frames) * CHUNK / RATE

            if elapsed_time >= RECORD_SECONDS + 2*OVERLAP_SECONDS:
                
                # Prepare the chunk for processing
                processing_frames = current_frames

                # Process the chunk
                wav_stream = process_audio_chunk(processing_frames, sample_width)
                input_queue.put(wav_stream)

                # Update overlap for the next chunk
                overlap_frames = current_frames[-int(2*OVERLAP_SECONDS * RATE / CHUNK):]
                current_frames = overlap_frames.copy()
    
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")

def main():

    ###---------Setup Audio Stream---------###
    pa = pyaudio.PyAudio()
    sample_width = pa.get_sample_size(FORMAT)
    stream = pa.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_id,
                    frames_per_buffer=CHUNK)
    ###---------End Setup Audio Stream---------###

    ###--------- Multiprocessing Setup ---------###
    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()
    
    server_process = multiprocessing.Process(target=model_server, args=(input_queue, output_queue))
    display_proccess = multiprocessing.Process(target=output_transcript, args=(output_queue,))

    server_process.start()
    display_proccess.start()
    ###--------- End Multiprocessing Setup ---------###

    #Wait for user input to start recording
    input("Program started. Press any key to start recording.  Press Ctrl+C to stop.")
    start_recording_overlap(stream, sample_width, input_queue)

    #Cleanup 
    server_process.terminate()
    display_proccess.terminate()

    stream.stop_stream()
    stream.close()
    pa.terminate()

###--- End Functions ---### 

if __name__ == "__main__":
    main()