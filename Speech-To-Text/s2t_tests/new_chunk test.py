import pyaudio
import tkinter as tk
import time
import io
import wave
import multiprocessing
from faster_whisper import WhisperModel

# TODO: Switch from threading to multiprocessing module to utulize CPU more effectively for larger models
#       Set up algorithm to chunk audio and text based off of periods of silence
#       Feed text chunks to LLM
#       Deal with multiprocessing termination -temporary stopgap is to send None to the queue and check for it
#       Set up queue structure
#       Make sure if multiple data chunks are added to the input or output queue, that they are all processed,
#           and not just the most recently added chunk

###--- Audio recording parameters ---###
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 1024
RECORD_SECONDS = 3  # Interval for saving audio

PRIOR_OVERLAP_SECONDS = 10  
POSTERIOR_OVERLAP_SECONDS = 2  

OVERLAP_SECONDS = 2  # Interval for overlapping audio
device_id = 6
###--- End Audio recording parameters ---###

###############################################################################################

######------ Functions ------######

#- Multiprocessing functions -#

def model_server(input_queue, output_queue):
    
    model = initialize_model()
    transcript_list=[]
    prev_words=[]
    while True:
        #Receive audio data from the recording program
        #TODO check sync of input_queue.get
        audio_data = input_queue.get()
        time_at = time.time()
        # NOTE: Temporary shutdown signal
        if audio_data is None: 
            print("\nTranscription process terminated.") 
            output_queue.put(None)  # Signal display process to shut down
            break

        # Transcribe the audio data into segments of text
        segments, info = model.transcribe(audio_data, beam_size=5, vad_filter=True, word_timestamps=True)
        
        word_list = []
        for segment in segments:
            for word in segment.words:
                word_list.append(word.word)
        merge_transcripts(word_list, transcript_list)
        words = process_transcript(segments)

        prev_words=word_list.copy()

        output = str(time_at)+": "+words+"\n"
        output_queue.put(output)

def merge_transcripts(word_list, transcript_list):
    #finds the optimal point of overlap offset and rewrites the end of transcript list based off of word_list
    pass

def longest_shared_substring(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    longest = 0
    end_index = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > longest:
                    longest = dp[i][j]
                    end_index = i
            else:
                dp[i][j] = 0

    return text1[end_index - longest: end_index]

def merge_texts(text1, text2):
    overlap = longest_shared_substring(text1, text2)
    overlap_len = len(overlap)
    merged_text = text1[:-overlap_len] + text2 if overlap_len > 0 else text1 + text2
    return merged_text   


def process_transcript(segments):

    output_string= ""
    #print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    for segment in segments:
        for word in segment.words: 
            output_string += word.word
               

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

#TODO: Check effects of blocking behavior of queue.get on the tkinter window
def output_to_window(output_queue):
    root = tk.Tk()
    root.title("Tabletop Assistant")
    root.geometry("800x600")
    root.configure(bg="grey")
    text_box = tk.Text(root, bg="grey", fg="white")
    text_box.pack()
    while True:
        text_data = output_queue.get()
        if text_data is None:
            print("\nDisplay process terminated.")
            break
        text_box.insert(tk.END, text_data)
        text_box.see(tk.END)
        root.update_idletasks()
        root.update()

#- End Multiprocessing functions -#

###############################################################################################

#- Audio processing functions -#

# process_audio_chunks handles the audio processing, turning frames into a file like object
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
def process_stream(stream, sample_width, input_queue):
    
    #frames holds the current audio chunk, overlap_frames holds a small portion of the end the previous chunk
    current_frames = []
    overlap_frames = []
    
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            current_frames.append(data)

            # Calculate elapsed time
            elapsed_time = len(current_frames) * CHUNK / RATE

            if (elapsed_time >= RECORD_SECONDS + PRIOR_OVERLAP_SECONDS + POSTERIOR_OVERLAP_SECONDS):
                
                # Prepare the chunk for processing
                processing_frames = current_frames

                # Process the chunk
                wav_stream = process_audio_chunk(processing_frames, sample_width)
                input_queue.put(wav_stream)

                #remove RECORDING_SECONDS + POSTERIOR_OVERLAP_SECONDS of frames from the start of current_frames
                #TODO: Make sure garbage collection is working efficiently for this code, as it will be constantly making new lists
                current_frames = current_frames[int((RECORD_SECONDS + POSTERIOR_OVERLAP_SECONDS) * RATE / CHUNK):]
    
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
        input_queue.put(None)  # Signal model process to shut down
#- End Audio processing functions -#

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
    #display_process = multiprocessing.Process(target=output_transcript, args=(output_queue,))
    display_process = multiprocessing.Process(target=output_to_window, args=(output_queue,))

    server_process.start()
    display_process.start()

    ###--------- End Multiprocessing Setup ---------###

    #Wait for user input to start recording
    input("Program started. Press any key to start recording.  Press Ctrl+C to stop.")
    process_stream(stream, sample_width, input_queue)
    server_process.join()
    display_process.join()
    #Cleanup 
    server_process.terminate()
    display_process.terminate()

    stream.stop_stream()
    stream.close()
    pa.terminate()

###--- End Functions ---### 

if __name__ == "__main__":
    main()