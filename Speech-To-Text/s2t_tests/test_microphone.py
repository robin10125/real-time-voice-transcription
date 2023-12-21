import pyaudio
pa = pyaudio.PyAudio()
print("!!!!!!!---------------Open PyAudio---------------!!!!!!!")

info = pa.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

# Scan through devices and print them
for i in range(0, numdevices):
    if pa.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
        print(f"Input Device id {i} - {pa.get_device_info_by_host_api_device_index(0, i).get('name')}")

device_id = 6
device_info = pa.get_device_info_by_index(device_id)

print(f"Device Name: {device_info['name']}")
print(f"Input Channels: {device_info['maxInputChannels']}")
print(f"Output Channels: {device_info['maxOutputChannels']}")
print(f"Default Sample Rate: {device_info['defaultSampleRate']}")
pa.terminate()
