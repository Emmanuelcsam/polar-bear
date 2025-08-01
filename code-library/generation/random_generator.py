import random
import json
import time

def generate_random():
    value = random.randint(0, 255)
    print(f"[RANDOM_GEN] Generated: {value}")
    
    with open('random_value.json', 'w') as f:
        json.dump({'value': value, 'timestamp': time.time()}, f)
    
    return value

if __name__ == "__main__":
    while True:
        generate_random()
        time.sleep(0.1)  # Adjust speed as needed