import numpy as np
import tensorflow as tf

# Define semantic features
semantic_features = {
    'specificity': {'level1': [0.1, 0.2, 0.3],
                    'level2': [0.2, 0.3, 0.4],
                    'level3': [0.3, 0.4, 0.5]},
    'coverage': {'level1': [0.3, 0.4, 0.5],
                 'level2': [0.4, 0.5, 0.6],
                 'level3': [0.5, 0.6, 0.7]},
    'fault_location': {'level1': [0.2, 0.3, 0.4],
                       'level2': [0.3, 0.4, 0.5],
                       'level3': [0.4, 0.5, 0.6]}
}

# Define reported errors and tests
reported_errors = ['timeout', 'data_mismatch', 'connection_error', 'server_down', 'invalid_response', 'authentication_error']
tests = ['curl', 'wget', 'ping', 'traceroute', 'nslookup', 'telnet', 'netcat', 'tcpdump', 'dns_query', 'ssl_check']

# Mapping reported errors and tests to indices
error_to_index = {error: i for i, error in enumerate(reported_errors)}
test_to_index = {test: i for i, test in enumerate(tests)}

# Training data
training_data = [
    (error_to_index['timeout'], [test_to_index['curl'], test_to_index['ping'], test_to_index['telnet']]), 
    (error_to_index['data_mismatch'], [test_to_index['curl'], test_to_index['wget'], test_to_index['ping']]), 
]

# Preparing training inputs
X_train_error = np.array([tf.one_hot(error, len(reported_errors)) for error, _ in training_data])
X_train_test = np.array([np.sum([tf.one_hot(test, len(tests)) for test in test_indices], axis=0) for _, test_indices in training_data])
X_train = np.concatenate((X_train_error, X_train_test), axis=1)

# Preparing training targets
y_train = np.zeros((len(training_data), len(tests))) # Initialize target tensor
for i, (_, tests_indices) in enumerate(training_data):
    y_train[i, tests_indices] = 1  

# Define custom layer for discovering abstract features and mapping them to semantic features

""" class SemanticMappingLayer(tf.keras.layers.Layer):
    def __init__(self, num_features, num_classes, **kwargs):
        super(SemanticMappingLayer, self).__init__(**kwargs)
        self.num_features = num_features
        self.num_classes = num_classes
        self.semantic_weights = self.add_weight(shape=(self.num_features, self.num_classes),
                                                initializer='random_normal',
                                                trainable=True)

    def call(self, inputs):
        abstract_features = tf.matmul(inputs, self.semantic_weights)
        return abstract_features """

class SemanticMappingLayer(tf.keras.layers.Layer):
    def __init__(self, num_features, num_classes, **kwargs):
        super(SemanticMappingLayer, self).__init__(**kwargs)
        self.num_features = num_features
        self.num_classes = num_classes
        self.semantic_weights = self.add_weight(shape=(self.num_features // 2, self.num_classes),
                                                initializer='random_normal',
                                                trainable=True)

    def call(self, inputs):
        abstract_features = tf.matmul(inputs[:, :self.num_features // 2], self.semantic_weights)
        return abstract_features


# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(reported_errors) + len(tests),)), 
    tf.keras.layers.Dense(64, activation='relu'), 
    tf.keras.layers.Dense(32, activation='relu'), 
    SemanticMappingLayer(num_features=len(reported_errors) + len(tests), num_classes=len(tests)),  # Custom layer added
    tf.keras.layers.Dense(len(tests), activation='sigmoid') 
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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
