import numpy as np
import tensorflow as tf

# Define semantic features and levels
specificity_levels = ['low', 'medium', 'high']
coverage_levels = ['low', 'medium', 'high']
fault_location_levels = ['internal', 'same_org', 'same_subnet']

# Define reported errors and tests
reported_errors = ['timeout', 'data_mismatch', 'connection_error', 'server_down', 'invalid_response', 'authentication_error']
tests = ['curl', 'wget', 'ping', 'traceroute', 'nslookup', 'telnet', 'netcat', 'tcpdump', 'dns_query', 'ssl_check']

# Mapping reported errors and tests to indices
error_to_index = {error: i for i, error in enumerate(reported_errors)}
test_to_index = {test: i for i, test in enumerate(tests)}

# Training data
training_data = [
    (error_to_index['timeout'], [test_to_index['curl'], test_to_index['ping'], test_to_index['telnet']], 'high', 'low', 'internal'), 
    (error_to_index['data_mismatch'], [test_to_index['curl'], test_to_index['wget'], test_to_index['ping']], 'medium', 'medium', 'same_org'), 
]

# One-hot encoding of error and test data
X_train_error = np.array([tf.one_hot(error, len(reported_errors)) for error, _, _, _, _ in training_data])
X_train_test = np.array([np.sum([tf.one_hot(test, len(tests)) for test in test_indices], axis=0) for _, test_indices, _, _, _ in training_data])
X_train_specificity = np.array([tf.one_hot(specificity_levels.index(spec), len(specificity_levels)) for _, _, spec, _, _ in training_data])
X_train_coverage = np.array([tf.one_hot(coverage_levels.index(cov), len(coverage_levels)) for _, _, _, cov, _ in training_data])
X_train_fault_location = np.array([tf.one_hot(fault_location_levels.index(loc), len(fault_location_levels)) for _, _, _, _, loc in training_data])

# Concatenate all inputs
X_train = np.concatenate((X_train_error, X_train_test, X_train_specificity, X_train_coverage, X_train_fault_location), axis=1)

# Define target tensor
y_train = {
    'specificity_output': X_train_specificity,
    'coverage_output': X_train_coverage,
    'fault_location_output': X_train_fault_location
}

# Define model architecture
error_input = tf.keras.layers.Input(shape=(len(reported_errors),), name='error_input')
test_input = tf.keras.layers.Input(shape=(len(tests),), name='test_input')
specificity_input = tf.keras.layers.Input(shape=(len(specificity_levels),), name='specificity_input')
coverage_input = tf.keras.layers.Input(shape=(len(coverage_levels),), name='coverage_input')
fault_location_input = tf.keras.layers.Input(shape=(len(fault_location_levels),), name='fault_location_input')

concatenated_inputs = tf.keras.layers.concatenate([error_input, test_input, specificity_input, coverage_input, fault_location_input])

dense_1 = tf.keras.layers.Dense(64, activation='relu')(concatenated_inputs)
dense_2 = tf.keras.layers.Dense(32, activation='relu')(dense_1)

# Specificity classifier
specificity_output = tf.keras.layers.Dense(len(specificity_levels), activation='softmax', name='specificity_output')(dense_2)

# Coverage classifier
coverage_output = tf.keras.layers.Dense(len(coverage_levels), activation='softmax', name='coverage_output')(dense_2)

# Fault location classifier
fault_location_output = tf.keras.layers.Dense(len(fault_location_levels), activation='softmax', name='fault_location_output')(dense_2)

# Compile the model with three outputs
model = tf.keras.Model(inputs=[error_input, test_input, specificity_input, coverage_input, fault_location_input], outputs=[specificity_output, coverage_output, fault_location_output])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics={'specificity_output': 'accuracy', 'coverage_output': 'accuracy', 'fault_location_output': 'accuracy'})

# Train the model
model.fit({'error_input': X_train_error, 'test_input': X_train_test, 'specificity_input': X_train_specificity, 'coverage_input': X_train_coverage, 'fault_location_input': X_train_fault_location},
          y_train,
          epochs=50, batch_size=32)

# After training, you can use the model to predict the relevance of tests for any reported error
error_index = error_to_index['timeout']
test_indices = [test_to_index['curl'], test_to_index['ping'], test_to_index['telnet']]
specificity_level = 'high'
coverage_level = 'low'
fault_location_level = 'internal'

# Prepare input for prediction
error_input = tf.one_hot(error_index, len(reported_errors)).numpy().reshape(1, -1)
test_input = np.sum([tf.one_hot(test_index, len(tests)) for test_index in test_indices], axis=0).reshape(1, -1)
specificity_input = tf.one_hot(specificity_levels.index(specificity_level), len(specificity_levels)).numpy().reshape(1, -1)
coverage_input = tf.one_hot(coverage_levels.index(coverage_level), len(coverage_levels)).numpy().reshape(1, -1)
fault_location_input = tf.one_hot(fault_location_levels.index(fault_location_level), len(fault_location_levels)).numpy().reshape(1, -1)

# Predict
predicted_specificity, predicted_coverage, predicted_fault_location = model.predict([error_input, test_input, specificity_input, coverage_input, fault_location_input])

print("Predicted Specificity:", predicted_specificity)
print("Predicted Coverage:", predicted_coverage)
print("Predicted Fault Location:", predicted_fault_location)
