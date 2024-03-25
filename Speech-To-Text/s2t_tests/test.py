import numpy as np
import time

# Create a large list of floating-point numbers
large_list = [np.float64(i) for i in range(1000000)]

# Measure conversion time
start_time = time.time()
array = np.frombuffer(b''.join(large_list), dtype=np.float64)
end_time = time.time()

print(f"Conversion took {end_time - start_time} seconds.")