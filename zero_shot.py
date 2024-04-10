import numpy as np
import tensorflow as tf

# Reported errors and tests
reported_errors = ['timeout', 'data_mismatch', 'connection_error', 'server_down', 'invalid_response', 'authentication_error']
tests = ['curl', 'wget', 'ping', 'traceroute', 'nslookup', 'telnet', 'netcat', 'tcpdump', 'dns_query', 'ssl_check']

# Semantic hierarchy mapping tests to their specificity levels
# I defined a semantic hierarchy (hierarchical_semantics) that assigns each test a specificity level.
# I implemented a function predict_relevance_hierarchical_semantics() that predicts the relevance of tests based on their hierarchical semantics. This function uses the specificity level of the error to determine which tests are relevant, and then predicts their relevance using the trained model.

#Hierarchical levels

hierarchical_semantics = {
    'curl': 0, 'wget': 0, 'ping': 1, 'traceroute': 1, 'nslookup': 1, 'telnet': 2, 'netcat': 2,
    'tcpdump': 2, 'dns_query': 3, 'ssl_check': 3
}

# Association semantics mapping errors to associated tests
# Association semantics capture the relationships between errors and tests, allowing the model to infer relevance based on these associations.
# A new function predict_relevance_association_semantics() is implemented to predict the relevance of tests based on the associated tests for a given error.

association_semantics = {
    'timeout': ['ping', 'traceroute'],  
    'data_mismatch': ['curl', 'wget', 'ping'], 
    # Define associations for other errors...
}

# Mapping errors and tests to indices
error_to_index = {error: i for i, error in enumerate(reported_errors)}
test_to_index = {test: i for i, test in enumerate(tests)}

# Training data
training_data = [
    (error_to_index['timeout'], [test_to_index['curl'], test_to_index['ping'], test_to_index['telnet']]), 
    (error_to_index['data_mismatch'], [test_to_index['curl'], test_to_index['wget'], test_to_index['ping']]), 
]

# Prepare input features for training
X_train_error = np.array([tf.one_hot(error, len(reported_errors)) for error, _ in training_data])
X_train_test = np.array([np.sum([tf.one_hot(test, len(tests)) for test in test_indices], axis=0) for _, test_indices in training_data])
X_train = np.concatenate((X_train_error, X_train_test), axis=1)

# Prepare target labels for training
y_train = np.zeros((len(training_data), len(tests)))
for i, (_, tests_indices) in enumerate(training_data):
    y_train[i, tests_indices] = 1  

# Model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(reported_errors) + len(tests),)), 
    tf.keras.layers.Dense(64, activation='relu'), 
    tf.keras.layers.Dense(32, activation='relu'), 
    tf.keras.layers.Dense(len(tests), activation='sigmoid') 
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model training
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Function to predict relevance using association semantics
def predict_relevance_association_semantics(error):
    associated_tests = association_semantics.get(error, [])  # Get associated tests for the error
    test_indices = [test_to_index[test] for test in associated_tests]  # Convert tests to indices
    error_index = error_to_index[error]  # Get error index
    error_input = tf.one_hot(error_index, len(reported_errors)).numpy().reshape(1, -1)  # One-hot encode error
    test_input = np.sum([tf.one_hot(test_index, len(tests)) for test_index in test_indices], axis=0).reshape(1, -1)  # One-hot encode associated tests and sum
    combined_input = np.concatenate((error_input, test_input), axis=1)  # Concatenate error and test inputs
    predicted_probs = model.predict(combined_input)  # Predict probabilities
    test_relevance = {tests[i]: predicted_probs[0][i] for i in range(len(tests))}  # Map probabilities to test names
    return test_relevance

# Example usage
error = 'timeout'
print("Relevance of tests for '{}' error based on association semantics:".format(error))
print(predict_relevance_association_semantics(error))




