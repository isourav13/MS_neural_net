import numpy as np
import tensorflow as tf


# Define the reported errors and tests
reported_errors = ['timeout', 'data_mismatch', 'connection_error', 'server_down', 'invalid_response', 'authentication_error']
tests = ['curl', 'wget', 'ping', 'traceroute', 'nslookup', 'telnet', 'netcat', 'tcpdump', 'dns_query', 'ssl_check']

# Generate one-hot encodings for reported errors and tests
error_to_index = {error: i for i, error in enumerate(reported_errors)}
test_to_index = {test: i for i, test in enumerate(tests)}

# Define training data
training_data = [
    (error_to_index['timeout'], [test_to_index['curl'], test_to_index['ping'], test_to_index['telnet']]), # Example 1
    (error_to_index['data_mismatch'], [test_to_index['curl'], test_to_index['wget'], test_to_index['ping']]), # Example 2
    # Add more training data here as needed
]

print(training_data)

# Convert training data to numpy arrays
X_train_error = np.array([tf.one_hot(error, len(reported_errors)) for error, _ in training_data])
X_train_test = np.array([np.sum([tf.one_hot(test, len(tests)) for test in test_indices], axis=0) for _, test_indices in training_data])
X_train = np.concatenate((X_train_error, X_train_test), axis=1)

print("Train_Error", X_train_error)
print("Train_Test", X_train_test)
print("X_Train Input: ", X_train)



# Define the target labels (y_train)
y_train = np.zeros((len(training_data), len(tests))) # Initialize target tensor
for i, (_, tests_indices) in enumerate(training_data):
    y_train[i, tests_indices] = 1  # Set relevant test indices to 1

print("Y_Train Target: ", y_train)

# Define the neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(reported_errors) + len(tests),)), # Input layer
    tf.keras.layers.Dense(64, activation='relu'), # Hidden layer with ReLU activation
    tf.keras.layers.Dense(32, activation='relu'), # Another hidden layer with ReLU activation
    tf.keras.layers.Dense(len(tests), activation='sigmoid') # Output layer with sigmoid activation for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# After training, you can use the model to predict the relevance of tests for any reported error
# For example, to predict for 'timeout' error
error_input = tf.one_hot(error_to_index['timeout'], len(reported_errors)).numpy().reshape(1, -1)
test_input = np.sum([tf.one_hot(test_index, len(tests)) for test_index in training_data[0][1]], axis=0).reshape(1, -1)
combined_input = np.concatenate((error_input, test_input), axis=1)
predicted_probs = model.predict(combined_input)
test_relevance = {tests[i]: predicted_probs[0][i] for i in range(len(tests))}
print("Relevance of tests for 'timeout' error:")
print(test_relevance)
