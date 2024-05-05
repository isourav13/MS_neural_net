import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Semantic features for reported errors
reported_errors = ['timeout', 'data_mismatch', 'connection_error', 'server_down', 'invalid_response', 'authentication_error']
error_features = {
    'timeout': [0.8, 0.2, 0.4, 0.5],
    'data_mismatch': [0.5, 0.6, 0.3, 0.4],
    'connection_error': [0.7, 0.3, 0.6, 0.7],
    'server_down': [0.9, 0.1, 0.8, 0.9],
    'invalid_response': [0.6, 0.4, 0.5, 0.6],
    'authentication_error': [0.3, 0.5, 0.2, 0.3]
}

# Semantic features for tests
tests = ['curl', 'wget', 'ping', 'traceroute', 'nslookup', 'telnet', 'netcat', 'tcpdump', 'dns_query', 'ssl_check']
test_features = {
    'curl': [0.1, 0.8, 0.4, 0.2],
    'wget': [0.2, 0.7, 0.3, 0.3],
    'ping': [0.3, 0.6, 0.1, 0.1],
    'traceroute': [0.7, 0.9, 0.5, 0.5],
    'nslookup': [0.4, 0.5, 0.2, 0.3],
    'telnet': [0.6, 0.7, 0.3, 0.4],
    'netcat': [0.5, 0.6, 0.4, 0.4],
    'tcpdump': [0.8, 0.9, 0.7, 0.6],
    'dns_query': [0.4, 0.5, 0.3, 0.3],
    'ssl_check': [0.9, 0.8, 0.6, 0.7]
}

# Create training data
X = []
y = []
for error in reported_errors:
    for test in tests:
        X.append(error_features[error] + test_features[test])
        y.append(test in error)

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(len(error_features) + len(test_features),)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Predict probabilities for each test for a particular error
error_to_predict = 'timeout'
X_predict = np.array([error_features[error_to_predict] + test_features[test] for test in tests])
probabilities = model.predict(X_predict)

# Print probabilities
for test, prob in zip(tests, probabilities):
    print(f"Probability of test '{test}' for error '{error_to_predict}': {prob[0]}")
