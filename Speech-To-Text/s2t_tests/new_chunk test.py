import pyaudio
import tkinter as tk
import time
import io
import wave
import multiprocessing
import difflib
from faster_whisper import WhisperModel
import datetime
import uuid
import os 

# TODO: Switch from threading to multiprocessing module to utulize CPU more effectively for larger models
#       Set up algorithm to chunk audio and text based off of periods of silence
#       Feed text chunks to LLM
#       Deal with multiprocessing termination -temporary stopgap is to send None to the queue and check for it
#       Set up queue structure
#       Make sure if multiple data chunks are added to the input or output queue, that they are all processed,
#           and not just the most recently added chunk
#       Implement silence detection
#       Finish implementing sentence aware audio buffer pruning

###--- Audio recording parameters ---###
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 1024

AUDIO_CHUNK_LENGTH = 10 # Length of audio chunks in seconds
device_id = 6
###--- End Audio recording parameters ---###

###############################################################################################

######------ Functions ------######

#- Multiprocessing functions -#

#TODO: Implement verified/immutable transcript vs unverified transcript, so that verified transcript is not changed or taking up compuation resources
def model_server(input_queue, output_queue, buffer_prune_queue, log_queue):
    
    model = initialize_model(log_queue)
    
    confirmed_transcript=''
    while True:

        #Receive audio data from the recording program
        audio_data, chunk_id = input_queue.get()

        # NOTE: Temporary shutdown signal
        if audio_data is None: 
            log_queue.put(f'{datetime.datetime.now().strftime("%H:%M:%S")}|TERMINATED_TRANSCRIPTION_PROCESS|Termination process signal recieved by model server')
            print("\nTranscription process terminated.") 
            output_queue.put(None)  # Signal display process to shut down
            break
        
        time_at = datetime.datetime.now().strftime("%H:%M:%S")
        unconfirmed_transcript=''
        
        # Transcribe the audio data into segments of text
        segments, info = model.transcribe(audio_data, beam_size=5, vad_filter=True, word_timestamps=True)

        num_sentences = 0
        words_list = []

        #segments is a generator object that will use the whisper model autoregressively to generate text trascripts from the audio data provided in model.transcribe
        for segment in segments:
            for word in segment.words:
                if "." in word.word or "?" in word.word or "!" in word.word or "..." in word.word:
                    num_sentences += 1
                    words_list.append((word.start, word.end, word.word, True))
                else: words_list.append((word.start, word.end, word.word, False))
        #confirmed_signal tells the program where to prune the audio buffer
        confirmed_signal = 0
        
        print(f'num_sentences: {num_sentences}')
        
        #prune all but the last full sentence and all words afterwards, and calculate the timestamp in the audio buffer of this pruning location
        if num_sentences == 2 and words_list[-1][3] == False or num_sentences > 2:
            prune_prime= False
            #Backward loop through words_list to find the index of the end of the second last full sentence
            for i in range(len(words_list)-1, -1, -1):
                word_tuple = words_list[i]
                if word_tuple[3] == True: 
                    if prune_prime == True:
                        confirmed_signal = i
                        buffer_prune_queue.put(word_tuple[1])
                        break
                    prune_prime = True

            #Forward loop through words_list to create the confirmed transcript        
            for j in range(confirmed_signal+1):
                confirmed_transcript += words_list[j][2]

        #Forward loop through words_list to create the unconfirmed transcript (executes regardless of num sentences)
        for k in range(confirmed_signal, len(words_list)):
            unconfirmed_transcript += words_list[k][2]
        log_queue.put(f'{datetime.datetime.now().strftime("%H:%M:%S")}|{chunk_id}|TRANSCRIPT|{confirmed_transcript}{unconfirmed_transcript}')
        output = (str(time_at), confirmed_transcript, unconfirmed_transcript)
        output_queue.put(output)

#TODO: merge_transcripts is currently unused, delete later if evidently not needed after testing
def merge_transcripts(t_prime, words_chunk):
    s = difflib.SequenceMatcher(None, t_prime, words_chunk)
    match = s.find_longest_match(0, len(t_prime), 0, len(words_chunk))
    
    #TODO: Use content agnostic time based function to add words together in this case
    if match.size < 3:
        return t_prime

    overlap_start = match.b
    return t_prime + words_chunk[overlap_start + match.size:]

def initialize_model(log_queue, size="tiny.en"):

    model_size = size
    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    log_queue.put(f'{datetime.datetime.now().strftime("%H:%M:%S")}|INITIALIZED_MODEL|{model_size}')

    return model

#TODO: Rewrite with new transcript structure
def output_transcript(output_queue):
    while True:
        text_data = output_queue.get()
        if text_data is None:
            print("\nDisplay process terminated.")
            break
        print(f'Confirmed transcript: \n {text_data[1]} \n' 
            f'Unconfirmed transcript: \n {text_data[2]} \n Last Updated: {text_data[0]} \n'
            f'############################################################################################################################# \n')

#TODO: Check effects of blocking behavior of queue.get on the tkinter window
#TODO: Rewrite with new transcript structure
#TODO: Add scroll bar to window
#TODO: Add resizeable window
def output_to_window(output_queue):
    root = tk.Tk()
    root.title("Tabletop Assistant")
    root.geometry("800x600")
    root.configure(bg="grey")
    
    # Create text boxes for confirmed and unconfirmed transcript data
    confirmed_text_box = tk.Text(root, bg="grey", fg="white")
    confirmed_text_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    unconfirmed_text_box = tk.Text(root, bg="grey", fg="white")
    unconfirmed_text_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    while True:
        text_data = output_queue.get()
        if text_data is None:
            print("\nDisplay process terminated.")
            break
        
        timestamp, confirmed_transcript, unconfirmed_transcript = text_data[0], text_data[1], text_data[2]
        
        # Clear the unconfirmed text box and insert new unconfirmed data
        unconfirmed_text_box.delete("1.0", tk.END)
        unconfirmed_text_box.insert(tk.END, unconfirmed_transcript)
        
        # Insert new confirmed data at the end of the confirmed text box
        confirmed_text_box.insert(tk.END, confirmed_transcript)
        
        # Scroll to the end of both text boxes
        confirmed_text_box.see(tk.END)
        unconfirmed_text_box.see(tk.END)
        
        root.update_idletasks()
        root.update()

def logger(log_queue, repository_path):
    session_file = f'{repository_path}/session_log_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_{uuid.uuid4()}.txt'
    while True:
        log_data = log_queue.get()
        if log_data is None:
            print("\nLog process terminated.")
            break
        with open(session_file, "a") as log_file:
            log_file.write(log_data + "\n")  # Write the log data to the file
        log_file.close()

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
def save_audio_chunk(frames, sample_width, logger_queue, chunk_id, repository_path):
    
    file_name = f"{repository_path}/{chunk_id}.wav"
    wf = wave.open(file_name, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()   
    logger_queue.put(f'{datetime.datetime.now().strftime("%H:%M:%S")}|{chunk_id}|SAVED_AUDIO_CHUNK|{file_name}')

# start recording_overlap handles reading the data from audio stream and feeding it to the transcription pipeline
# TODO Change chunking behaviour to be based off of sentence content.  This requires getting transcriptions back from the model server
def process_stream(stream, sample_width, input_queue, buffer_prune_queue, log_queue, repository_path):
    
    log_queue.put(f'{datetime.datetime.now().strftime("%H:%M:%S")}|STARTED_RECORDING|')

    #frames holds the current audio chunk
    current_frames = []
    silence = False

    try:
        reference_time = 0
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            current_frames.append(data)
            # Calculate elapsed time
            elapsed_time = len(current_frames) * CHUNK / RATE

            # TODO Implement silence detection
            if (elapsed_time - reference_time >= AUDIO_CHUNK_LENGTH) or silence == True:
                chunk_id = f'audio_chunk_{datetime.datetime.now().strftime("%H:%M:%S")}_{uuid.uuid4()}'
                # buffer_prune queue will get pushed to it the time where the audio buffer should be pruned when its appropriate
                #TODO: Implement sentinel value for checking queue emptyness instead of .empty()
                if not buffer_prune_queue.empty():
                    prune_signal = buffer_prune_queue.get()
                    current_frames = current_frames[int(prune_signal * RATE / CHUNK):]
                
                reference_time = elapsed_time
                # Prepare the chunk for processing
                processing_frames = current_frames

                # Process the chunk
                wav_stream = process_audio_chunk(processing_frames, sample_width)
                save_audio_chunk(processing_frames, sample_width, log_queue, chunk_id, repository_path)

                input_queue.put((wav_stream, chunk_id))
                
                
    
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
        #send custom chunk id
        input_queue.put((None,f'termination_chunk_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_{uuid.uuid4()}'))  # Signal model process to shut down
        log_queue.put(None)
#- End Audio processing functions -#

def main():

    ###---------Setup Logging---------###
    session_id = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_{uuid.uuid4()}'
    repository_path = f"log_files/{session_id}"
    os.makedirs(repository_path, exist_ok=True)
    ###---------End Setup Logging---------###

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
    buffer_prune_queue = multiprocessing.Queue()
    log_queue = multiprocessing.Queue()

    server_process = multiprocessing.Process(target=model_server, args=(input_queue, output_queue, buffer_prune_queue, log_queue))
    display_process = multiprocessing.Process(target=output_transcript, args=(output_queue,))
    #display_process = multiprocessing.Process(target=output_to_window, args=(output_queue,))

    log_process = multiprocessing.Process(target=logger, args=(log_queue,repository_path,))

    server_process.start()
    display_process.start()
    log_process.start()
    ###--------- End Multiprocessing Setup ---------###

    #Wait for user input to start recording
    input("Program started. Press any key to start recording.  Press Ctrl+C to stop.")
    output_queue.put((str(0),"Begin!",""))
    process_stream(stream, sample_width, input_queue, buffer_prune_queue, log_queue, repository_path)


    server_process.join()
    display_process.join()
    log_process.join()
    #Cleanup 
    server_process.terminate()
    display_process.terminate()
    log_process.terminate()

    stream.stop_stream()
    stream.close()
    pa.terminate()

###--- End Functions ---### 

if __name__ == "__main__":
    main()