import numpy as np

def polynomial_to_trellis(constraint_length : np.int32, generator : np.ndarray):
    memory_size = constraint_length - 1
    num_states = 2**(memory_size)
    
    next_state = np.array([[0] * 2 for _ in range(num_states)])
    output = np.array([[[0] * len(generator)] * 2 for _ in range(num_states)])

    for s in range(num_states):
        for u in (0, 1):
            state = (u << memory_size) | s
            next_state[s][u] = state >> 1
            output[s][u] = [np.bitwise_count(state & g) & 1 for g in generator]

    return next_state, output

def convolutional_encode(input : np.ndarray, 
                        constraint_length : np.int32,
                        generator : np.ndarray):
    output = np.array([[0] * len(generator) for _ in range(len(input))])
    memory_size = constraint_length - 1

    state = 0

    for i in range(len(input)):
        state = ((input[i] << (memory_size)) | (state >> 1))
        output[i] = [np.bitwise_count(state & g) & 1 for g in generator]
    
    return output

def convolutional_decode(input : np.ndarray,
                         constraint_length : np.int32,
                         generator : np.ndarray):
    num_states = 2**(constraint_length - 1)
    next_state, output = polynomial_to_trellis(constraint_length, generator)

    metrics = np.array([np.nan] * num_states)
    metrics[0] = 0

    prev_state = [[np.nan] * num_states for _ in range(len(input))]
    prev_input = [[np.nan] * num_states for _ in range(len(input))]

    for t in range(len(input)):
        new_metrics = np.array([np.nan] * num_states)

        for s in range(num_states):
            m = metrics[s]

            if np.isnan(m):
                continue

            for u in (0, 1):
                ns = next_state[s][u]

                hamming_distance = np.sum(input[t] ^ output[s][u])

                new_metric = m + hamming_distance

                if np.isnan(new_metrics[ns]):
                    new_metrics[ns] = new_metric
                    prev_state[t][ns] = s
                    prev_input[t][ns] = u

                if new_metric < new_metrics[ns]:
                    new_metrics[ns] = new_metric
                    prev_state[t][ns] = s
                    prev_input[t][ns] = u

        metrics = new_metrics

    best_end = np.argmin(metrics)

    decoded = np.array([0] * len(input))
    state = best_end

    for t in reversed(range(len(input))):
        decoded[t] = prev_input[t][state]
        state = prev_state[t][state]

    return decoded

def main():
    bits = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1])
    constraint_length = np.int32(7)
    generator = np.array([0b1_110111, 0b1_101001, 0b1_001011])
    encoded = convolutional_encode(bits, constraint_length, generator)
    decoded = convolutional_decode(encoded, constraint_length, generator)
    print(np.array_equal(bits, decoded))

if __name__ == "__main__":
    main()