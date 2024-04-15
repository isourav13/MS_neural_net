import numpy as np
import tensorflow as tf

# Define semantic features and their levels
semantic_features = {
    'specificity': ['low', 'medium', 'high'],
    'coverage': ['low', 'medium', 'high'],
    'fault_location': ['local', 'organization', 'subnet']
}

# Define the probabilities of semantic features based on their levels (just placeholders)
semantic_probs = {
    'specificity': [0.3, 0.5, 0.2],
    'coverage': [0.2, 0.4, 0.4],
    'fault_location': [0.5, 0.3, 0.2]
}

reported_errors = ['timeout', 'data_mismatch', 'connection_error', 'server_down', 'invalid_response', 'authentication_error']
tests = ['curl', 'wget', 'ping', 'traceroute', 'nslookup', 'telnet', 'netcat', 'tcpdump', 'dns_query', 'ssl_check']

error_to_index = {error: i for i, error in enumerate(reported_errors)}
test_to_index = {test: i for i, test in enumerate(tests)}

# Define training data
training_data = [
    (error_to_index['timeout'], [test_to_index['curl'], test_to_index['ping'], test_to_index['telnet']]), 
    (error_to_index['data_mismatch'], [test_to_index['curl'], test_to_index['wget'], test_to_index['ping']]), 
]

# Convert training data to one-hot encodings
X_train_error = np.array([tf.one_hot(error, len(reported_errors)) for error, _ in training_data])
X_train_test = np.array([np.sum([tf.one_hot(test, len(tests)) for test in test_indices], axis=0) for _, test_indices in training_data])
X_train = np.concatenate((X_train_error, X_train_test), axis=1)

# Define target tensor
y_train = np.zeros((len(training_data), len(tests)))
for i, (_, tests_indices) in enumerate(training_data):
    y_train[i, tests_indices] = 1  

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(reported_errors) + len(tests),)), 
    tf.keras.layers.Dense(64, activation='relu'), 
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(tests) * len(semantic_features), activation='sigmoid') # Add a layer for semantic features
])

# Custom loss function for zero-shot learning
def custom_loss(y_true, y_pred):
    # Reshape y_pred to have the same shape as y_true
    y_pred_reshaped = tf.reshape(y_pred, (-1, len(tests), len(semantic_features)))
    # Initialize loss
    loss = 0
    # Calculate loss for each semantic feature
    for feature, levels in semantic_features.items():
        # Get probabilities of the semantic feature
        feature_probs = semantic_probs[feature]
        # Calculate loss for each level of the feature
        for i, level in enumerate(levels):
            # Weighted sum of probabilities
            weighted_sum = tf.reduce_sum(tf.multiply(y_pred_reshaped[:, :, i], feature_probs[i]))
            # Add to the total loss
            loss += weighted_sum * tf.cast(y_true[:, :], tf.float32)
    return loss

# Compile the model with custom loss
model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# After training, you can use the model to predict the relevance of tests for any reported error
error_input = tf.one_hot(error_to_index['timeout'], len(reported_errors)).numpy().reshape(1, -1)
test_input = np.sum([tf.one_hot(test_index, len(tests)) for test_index in training_data[0][1]], axis=0).reshape(1, -1)
combined_input = np.concatenate((error_input, test_input), axis=1)
predicted_probs = model.predict(combined_input)
test_relevance = {tests[i]: predicted_probs[0][i] for i in range(len(tests))}
print("Relevance of tests for 'timeout' error:")
print(test_relevance)
