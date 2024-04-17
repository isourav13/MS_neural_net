import numpy as np
import tensorflow as tf

# Define reported errors and tests
reported_errors = ['timeout', 'data_mismatch', 'connection_error', 'server_down', 'invalid_response', 'authentication_error']
tests = ['curl', 'wget', 'ping', 'traceroute', 'nslookup', 'telnet', 'netcat', 'tcpdump', 'dns_query', 'ssl_check']

# Define semantic features and their levels
semantic_features = ['specificity', 'coverage', 'fault_location']
semantic_levels = {'specificity': ['low', 'medium', 'high'],
                   'coverage': ['low', 'medium', 'high'],
                   'fault_location': ['internal', 'same_organization', 'same_subnet']}

# Create dictionaries for mapping errors and tests to indices
error_to_index = {error: i for i, error in enumerate(reported_errors)}
test_to_index = {test: i for i, test in enumerate(tests)}

# Define training data
training_data = [
    (error_to_index['timeout'], [test_to_index['curl'], test_to_index['ping'], test_to_index['telnet']]), 
    (error_to_index['data_mismatch'], [test_to_index['curl'], test_to_index['wget'], test_to_index['ping']]), 
]

# Create target tensors for each semantic feature
y_train_specificity = np.zeros((len(training_data), len(tests), len(semantic_levels['specificity'])))
y_train_coverage = np.zeros((len(training_data), len(tests), len(semantic_levels['coverage'])))
y_train_fault_location = np.zeros((len(training_data), len(tests), len(semantic_levels['fault_location'])))

# Populate target tensors
for i, (_, tests_indices) in enumerate(training_data):
    for test_index in tests_indices:
        # Assigning equal weights initially, you can modify this according to your specific criteria
        y_train_specificity[i, test_index] = [1/3, 1/3, 1/3]
        y_train_coverage[i, test_index] = [1/3, 1/3, 1/3]
        y_train_fault_location[i, test_index] = [1/3, 1/3, 1/3]

# Concatenate target tensors for training
y_train = np.concatenate((y_train_specificity, y_train_coverage, y_train_fault_location), axis=-1)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(reported_errors) + len(tests),)), 
    tf.keras.layers.Dense(64, activation='relu'), 
    tf.keras.layers.Dense(32, activation='relu'), 
    tf.keras.layers.Dense(len(tests) * len(semantic_features) * len(semantic_levels['specificity']) * len(semantic_levels['coverage']) * len(semantic_levels['fault_location']), activation='sigmoid') 
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# After training, you can use the model to predict the relevance of tests for any reported error
error_input = tf.one_hot(error_to_index['timeout'], len(reported_errors)).numpy().reshape(1, -1)
test_input = np.sum([tf.one_hot(test_index, len(tests)) for test_index in training_data[0][1]], axis=0).reshape(1, -1)
combined_input = np.concatenate((error_input, test_input), axis=1)
predicted_probs = model.predict(combined_input)

# Reshape and extract predictions for each semantic feature
predicted_probs = predicted_probs.reshape(len(tests), len(semantic_features), len(semantic_levels['specificity']), len(semantic_levels['coverage']), len(semantic_levels['fault_location']))

# Print the relevance of tests for 'timeout' error for each semantic feature
for feature_index, feature in enumerate(semantic_features):
    print(f"Relevance of tests for 'timeout' error - {feature}:")
    for test_index, test in enumerate(tests):
        print(f"Test: {test}")
        for level_index, level in enumerate(semantic_levels[feature]):
            relevance = predicted_probs[test_index, feature_index, :, :, level_index]
            print(f"{level}: {relevance}")
