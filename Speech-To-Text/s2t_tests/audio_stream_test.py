import pyaudio
import wave
import datetime

print("!!!!!!!---------------Start Program---------------!!!!!!!")
# Create an instance of PyAudio
pa = pyaudio.PyAudio()

# List all available audio devices
info = pa.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
print("!!!!!!!---------------Open PyAudio---------------!!!!!!!")
# Scan through devices and print them
for i in range(0, numdevices):
    if pa.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
        print(f"Input Device id {i} - {pa.get_device_info_by_host_api_device_index(0, i).get('name')}")


device_id = 6

if device_id == 6: framerate = 48000
else: framerate = 44100

stream = pa.open(format=pyaudio.paInt16,
                 channels=1,
                 rate=framerate,
                 input=True,
                 input_device_index=device_id,
                 frames_per_buffer=1024)

frames = []  # A list to store the audio chunks
print("!!!!!!!--------------Opening PyAudio stream--------------!!!!!!!")
try:
    while True:
        data = stream.read(1024)
        frames.append(data)
except KeyboardInterrupt:
    # Stop and close the stream when Ctrl+C is pressed
    stream.stop_stream()
    stream.close()
    pa.terminate()
    # Close PyAudio
    print("!!!!!!!---------------Terminated PyAudio---------------!!!!!!!")

print("!!!!!!!--------------Recording Finished--------------!!!!!!!")
# Save the recorded data as a WAV file
#get date/time timestamp for file name
now = datetime.datetime.now()
filename = "output_test_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".wav"

wf = wave.open(filename, 'wb')
wf.setnchannels(1)
wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
wf.setframerate(framerate)
wf.writeframes(b''.join(frames))
wf.close()
print("!!!!!!!-------Saved to " + filename+" -------!!!!!!!")