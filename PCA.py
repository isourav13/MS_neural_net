import numpy as np
import tensorflow as tf


reported_errors = ['timeout', 'data_mismatch', 'connection_error', 'server_down', 'invalid_response', 'authentication_error']
tests = ['curl', 'wget', 'ping', 'traceroute', 'nslookup', 'telnet', 'netcat', 'tcpdump', 'dns_query', 'ssl_check']

error_to_index = {error: i for i, error in enumerate(reported_errors)}
test_to_index = {test: i for i, test in enumerate(tests)}

training_data = [
    (error_to_index['timeout'], [test_to_index['curl'], test_to_index['ping'], test_to_index['telnet']]), 
    (error_to_index['data_mismatch'], [test_to_index['curl'], test_to_index['wget'], test_to_index['ping']]), 
]

print(training_data)

X_train_error = np.array([tf.one_hot(error, len(reported_errors)) for error, _ in training_data])
X_train_test = np.array([np.sum([tf.one_hot(test, len(tests)) for test in test_indices], axis=0) for _, test_indices in training_data])
X_train = np.concatenate((X_train_error, X_train_test), axis=1)

print("Train_Error", X_train_error)
print("Train_Test", X_train_test)
print("X_Train Input: ", X_train)



y_train = np.zeros((len(training_data), len(tests))) # Initialize target tensor
for i, (_, tests_indices) in enumerate(training_data):
    y_train[i, tests_indices] = 1  

print("Y_Train Target: ", y_train)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(reported_errors) + len(tests),)), 
    tf.keras.layers.Dense(64, activation='relu'), 
    tf.keras.layers.Dense(32, activation='relu'),  # Feature extraction layer
    tf.keras.layers.Dense(len(tests), activation='sigmoid') 
])

# Access the output of the second-last layer using `get_layer`
extracted_features = model.get_layer(index=-2).output

# Define a new model to map features to semantics (optional)
feature_to_semantics_model = tf.keras.Sequential([
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dense(len(tests), activation='softmax')  # Softmax for probability distribution
])

# Compile the feature_to_semantics_model (if used)
feature_to_semantics_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("feature_to_semantics_model", feature_to_semantics_model)

# Train the main model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32)

# After training, you can use the model to predict

error_input = tf.one_hot(error_to_index['timeout'], len(reported_errors)).numpy().reshape(1, -1)
test_input = np.sum([tf.one_hot(test_index, len(tests)) for test_index in training_data[0][1]], axis=0).reshape(1, -1)
combined_input = np.concatenate((error_input, test_input), axis=1)

# Get extracted features
extracted_features_output = model.predict(combined_input)[0].reshape(1, -1)
print("extracted_features_output", extracted_features_output)

# Use the separate model for semantic interpretation (optional)
predicted_semantics = feature_to_semantics_model.predict(extracted_features_output)[0]
semantics_dict = {i: tests[i] for i in range(len(tests))}
print(semantics_dict)
print("Semantic relevance of tests for 'timeout' error (using separate model):")
print({semantics_dict[i]: predicted_semantics[i] for i in range(len(predicted_semantics))})

# Or interpret features directly (if not using separate model)
predicted_probs = model.predict(combined_input)
