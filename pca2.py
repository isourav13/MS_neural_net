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

# One-hot encoding of input data
X_train_error = np.array([tf.one_hot(error, len(reported_errors)) for error, _ in training_data])
X_train_test = np.array([np.sum([tf.one_hot(test, len(tests)) for test in test_indices], axis=0) for _, test_indices in training_data])
X_train = np.concatenate((X_train_error, X_train_test), axis=1)

# Target data
y_train = np.zeros((len(training_data), len(tests))) # Initialize target tensor
for i, (_, tests_indices) in enumerate(training_data):
    y_train[i, tests_indices] = 1  

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(reported_errors) + len(tests),)), 
    tf.keras.layers.Dense(64, activation='relu'), 
    tf.keras.layers.Dense(32, activation='relu'), 
    tf.keras.layers.Dense(len(tests), activation='sigmoid') 
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Extract features from the last layer
feature_extractor_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
features = feature_extractor_model.predict(X_train)

# Map features to semantics
# Define custom formulas for each semantic
# Specificity of error: Sum of absolute feature values for error-specific neurons
# Impact of error on tests: Sum of absolute feature values for test-specific neurons
# Coverage of tests: Mean of feature values for all test-specific neurons

# Calculate weights for each test/test type based on semantics
specificity_weights = np.abs(features[:, :len(reported_errors)]).sum(axis=1)
impact_weights = np.abs(features[:, len(reported_errors):]).sum(axis=1)
coverage_weights = np.abs(features[:, len(reported_errors):]).mean(axis=1)

# Normalize weights
specificity_weights /= specificity_weights.sum()
impact_weights /= impact_weights.sum()
coverage_weights /= coverage_weights.sum()

# Calculate final weights for each test/test type
final_weights = (specificity_weights + impact_weights + coverage_weights) / 3

# Map weights to test names
test_weights = {tests[i]: final_weights[i] for i in range(len(tests))}
print("Weights of each test/test type based on semantics:")
print(test_weights)
