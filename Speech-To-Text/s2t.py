import pyaudio
import tkinter as tk
import io
import wave
import multiprocessing
from faster_whisper import WhisperModel
import datetime
import uuid
import os 

# TODO: Measure compute and memory of multiprocessing modules
#       Set up algorithm to chunk audio and text based off of periods of silence
#       Feed text chunks to LLM
#       Deal with multiprocessing termination -temporary stopgap is to send None to the queue and check for it
#       Refine queue structure
#       Implement silence detection
#       Integrate unconfirmed and confirmed transcript pieces together for LLM inference
#       Test microphone input to make sure it is working properly, and to prompt user toroblems if it is not
#       Clean up logging
#       Implement program uptime clock

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
    """
    model_server transcribes audio data into text segments.

    Args:
        input_queue (Queue): Queue for receiving audio data from the recording program.
        output_queue (Queue): Queue for sending transcribed text segments to the display process.
        buffer_prune_queue (Queue): Queue for sending pruning location in the audio buffer.
        log_queue (Queue): Queue for logging information.

    Returns:
        None
    """

    model = initialize_model(log_queue)
    
    confirmed_transcript=''
    while True:
        # Receive audio data from the recording program
        audio_data, chunk_id = input_queue.get()

        # NOTE: Temporary shutdown signal
        if audio_data is None: 
            print("\nTranscription process terminated.") 
            break
        
        time_at = datetime.datetime.now().strftime("%H:%M:%S")
        unconfirmed_transcript=''
        
        # Transcribe the audio data into segments of text
        segments, info = model.transcribe(audio_data, beam_size=5, vad_filter=True, word_timestamps=True)

        num_sentences = 0
        words_list = []

        # segments is a generator object that will use the whisper model autoregressively to generate text transcripts from the audio data provided in model.transcribe
        for segment in segments:
            for word in segment.words:
                if "." in word.word or "?" in word.word or "!" in word.word or "..." in word.word:
                    num_sentences += 1
                    words_list.append((word.start, word.end, word.word, True))
                else:
                    words_list.append((word.start, word.end, word.word, False))
        
        # confirmed_signal tells the program where to prune the audio buffer
        confirmed_signal = 0
        
        print(f'num_sentences: {num_sentences}')
        
        # prune all but the last full sentence and all words afterwards, and calculate the timestamp in the audio buffer of this pruning location
        if num_sentences == 2 and words_list[-1][3] == False or num_sentences > 2:
            prune_prime = False
            # Backward loop through words_list to find the index of the end of the second last full sentence
            for i in range(len(words_list)-1, -1, -1):
                word_tuple = words_list[i]
                if word_tuple[3] == True: 
                    if prune_prime == True:
                        confirmed_signal = i
                        buffer_prune_queue.put(word_tuple[1])
                        break
                    prune_prime = True

            # Forward loop through words_list to create the confirmed transcript        
            for j in range(confirmed_signal+1):
                confirmed_transcript += words_list[j][2]

        # Forward loop through words_list to create the unconfirmed transcript (executes regardless of num sentences)
        for k in range(confirmed_signal, len(words_list)):
            unconfirmed_transcript += words_list[k][2]
        
        log_queue.put(f'{datetime.datetime.now().strftime("%H:%M:%S")}|{chunk_id}|TRANSCRIPT|{confirmed_transcript}{unconfirmed_transcript}')
        output = (str(time_at), confirmed_transcript, unconfirmed_transcript)
        output_queue.put(output)

def initialize_model(log_queue, size="tiny.en"):
    """
    Initializes a WhisperModel object with the specified size.

    Args:
        log_queue (Queue): A queue to store log messages.
        size (str, optional): The size of the model. Defaults to "tiny.en".

    Returns:
        WhisperModel: The initialized WhisperModel object.
    """

    model_size = size
    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    log_queue.put(f'{datetime.datetime.now().strftime("%H:%M:%S")}|INITIALIZED_MODEL|{model_size}')

    return model

def output_transcript(output_queue, session_id):
    """
    Writes the transcripts to a file.

    Args:
        output_queue (Queue): The queue containing the text data.
        session_id (str): The ID of the session.

    Returns:
        None
    """
    session_repo = f'{session_id}/transcripts/'
    while True:
        text_data = output_queue.get()
        
        if text_data is None:
            with open(session_file, "a") as transcript_file:
                transcript_file.write(unconfirmed_transcript)  
                transcript_file.close()
            print("\nDisplay process terminated.")
            break

        confirmed_transcript = text_data[1]
        unconfirmed_transcript = text_data[2]

        session_file = f'{session_repo}/transcript.txt'
        with open(session_file, "a") as transcript_file:
            transcript_file.write(confirmed_transcript)  # Write the log data to the file
        transcript_file.close()

        print(f'Confirmed transcript: \n {text_data[1]} \n' 
            f'Unconfirmed transcript: \n {text_data[2]} \n Last Updated: {text_data[0]} \n'
            f'############################################################################################################################# \n')

#TODO: Check effects of blocking behavior of queue.get on the tkinter window
#TODO: Rewrite with new transcript structure
#TODO: Add scroll bar to window
#TODO: Add resizeable window
#TODO: Rewrite with new transcript structure
def output_to_window(output_queue):
    """
    Display the transcript data in a GUI window.
    
    Args:
        output_queue (Queue): A queue containing the transcript data to be displayed.
    """
    
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
    """
    Writes log data from a queue to a file.

    Args:
        log_queue (Queue): The queue containing log data.
        repository_path (str): The path to the repository.

    Returns:
        None
    """
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

def process_audio_chunk(frames, sample_width):
    """
    Process audio frames and convert them into a wav file-like BytesIO object.

    Args:
        frames (list): List of audio frames.
        sample_width (int): Width of each audio sample in bytes.

    Returns:
        BytesIO: Wav file-like BytesIO object containing the processed audio.

    """
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
    """
    Save audio chunk as a WAV file.

    Args:
        frames (list): List of audio frames.
        sample_width (int): Sample width in bytes.
        logger_queue (Queue): Queue for logging messages.
        chunk_id (str): ID of the audio chunk.
        repository_path (str): Path to the repository.

    Returns:
        None
    """
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
    """
    Process the audio stream in chunks and perform necessary operations on each chunk.

    Args:
        stream (audio stream): The audio stream to process.
        sample_width (int): The sample width of the audio stream.
        input_queue (queue): The queue to put processed audio chunks into.
        buffer_prune_queue (queue): The queue to receive prune signals for audio buffer.
        log_queue (queue): The queue to log events and messages.
        repository_path (str): The path to the repository.

    Returns:
        None
    """

    start_time=datetime.datetime.now().strftime("%H:%M:%S")   
    log_queue.put(f'{start_time}|STARTED_RECORDING|')

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
                chunk_id = f'AudioChunk_{datetime.datetime.now().strftime("%H:%M:%S")}_{uuid.uuid4()}'
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
        return
        
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
    display_process = multiprocessing.Process(target=output_transcript, args=(output_queue,session_id,))
    #display_process = multiprocessing.Process(target=output_to_window, args=(output_queue,))

    log_process = multiprocessing.Process(target=logger, args=(log_queue,repository_path,))

    server_process.start()
    display_process.start()
    log_process.start()
    ###--------- End Multiprocessing Setup ---------###

    #Wait for user input to start recording
    input("Program started. Press any key to start recording.  Press Ctrl+C to stop.")
    output_queue.put((str(0),"Beginning transcription! \n",""))
    process_stream(stream, sample_width, input_queue, buffer_prune_queue, log_queue, repository_path)

    #send custom chunk id
    input_queue.put((None,f'termination_chunk_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_{uuid.uuid4()}'))  # Signal model process to shut down
    log_queue.put(None)
    output_queue.put(None)  # Signal display process to shut down


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

######------ End Functions ------###### 

if __name__ == "__main__":
    main()