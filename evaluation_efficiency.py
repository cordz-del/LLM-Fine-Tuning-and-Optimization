# evaluation_efficiency.py
import time

def simulate_response(prompt):
    # Simulate an API call (replace with real API call)
    time.sleep(0.5)  # simulate latency
    response = "This is a simulated response for prompt: " + prompt
    tokens_used = len(response.split())
    return response, tokens_used

def evaluate_response(prompt):
    start_time = time.time()
    response, tokens = simulate_response(prompt)
    latency = time.time() - start_time
    print(f"Response: {response}")
    print(f"Tokens used: {tokens}")
    print(f"Latency: {latency:.2f} seconds")
    return tokens, latency

# Example usage:
evaluate_response("Explain prompt engineering.")
