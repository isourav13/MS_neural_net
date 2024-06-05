# Static weight table 
static_weights = {
    "Verify service responds within timeout limit": [0.87, 0.93, 0.22, 0.78, 0.11, 0.02, 0.21, 0.25],
    "Test for correct data formatting": [0.72, 0.68, 0.13, 0.27, 0.99, 0.96, 0.12, 0.19],
    "Ensure service handles invalid input gracefully": [0.85, 0.74, 0.92, 0.11, 0.19, 0.36, 0.04, 0.00],
    "Verify service handles large payloads properly": [0.49, 0.53, 0.82, 0.97, 0.08, 0.21, 0.15, 0.00],
    "Check for proper authentication": [0.68, 0.79, 0.34, 0.15, 0.01, 0.02, 0.87, 0.92],
    "Verify service returns expected status codes": [0.01, 0.00, 0.11, 0.00, 0.71, 0.89, 0.00, 0.06],
    "Ensure service health endpoint is accessible": [1.00, 0.90, 0.02, 0.00, 0.91, 0.82, 0.00, 0.00],
    "Test for proper caching": [0.20, 0.09, 0.47, 0.33, 0.82, 0.77, 0.18, 0.04],
    "Verify service dependencies are reachable": [0.97, 0.92, 0.81, 0.90, 0.00, 0.12, 0.76, 0.48],
    "Test for DNS resolution": [0.99, 0.99, 0.96, 0.94, 0.02, 0.12, 0.00, 0.00],
    "Check for network latency": [0.99, 0.97, 0.85, 0.92, 0.00, 0.00, 0.01, 0.00],
    "Ensure proper network routing": [0.76, 0.81, 0.92, 0.60, 0.37, 0.18, 0.00, 0.00],
    "Test for service discovery": [0.37, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.49],
    "Istio destination rule": [0.58, 0.67, 0.00, 0.00, 0.13, 0.00, 0.00, 0.54],
    "Istio request routing": [0.92, 0.88, 0.04, 0.00, 0.00, 0.00, 0.91, 0.98],
    "Check for Istio mTLS encryption": [0.75, 0.69, 0.01, 0.00, 0.86, 0.99, 0.00, 0.82],
    "Test for database connectivity": [0.53, 0.49, 0.61, 0.00, 0.79, 0.88, 0.00, 0.00],
    "Verify database query execution": [0.61, 0.85, 0.98, 0.99, 0.00, 0.00, 0.00, 0.00],
    "Verify the environment type configuration": [0.68, 0.72, 0.44, 0.00, 0.99, 0.92, 0.00, 0.00],
    "Check maximum file size": [0.00, 0.77, 0.61, 0.43, 0.66, 0.55, 0.00, 0.00],
    "Verify cron jobs are scheduled": [0.56, 0.84, 0.95, 0.98, 0.00, 0.00, 0.00, 0.00],
    "Role-Based Access Control checks": [0.45, 0.57, 0.39, 0.00, 0.92, 0.85, 0.00, 0.00],
    "Resource utilization checks": [0.32, 0.41, 0.28, 0.00, 0.87, 0.94, 0.00, 0.00],
    "Database Grants": [0.48, 0.53, 0.58, 0.00, 0.79, 0.88, 0.00, 0.00],
    "API/JWT keys configured": [0.96, 0.92, 0.07, 0.00, 0.00, 0.00, 0.89, 0.97],
    "Port accessibility": [0.88, 0.91, 0.12, 0.00, 0.00, 0.00, 0.93, 0.86],
}

# Box-specific tests mapping 
box_tests = {
    "box1": ["Verify service responds within timeout limit", "Test for correct data formatting"],
    "box2": ["Ensure service handles invalid input gracefully", "Verify service handles large payloads properly"],
    "box3": ["Check for proper authentication", "Verify service returns expected status codes"],
    "box4": ["Ensure service health endpoint is accessible", "Test for proper caching"],
    "box5": ["Verify service dependencies are reachable", "Test for DNS resolution"],
}

# Conditions for dynamic weight adjustment
specific_conditions = ["multiple_errors_pointing_to_same_box", "network_issue_prevalent"]

# Dynamic adjustment rules
def adjust_weights(static_weights, errors):
    dynamic_weights = static_weights.copy()
    
    for error in errors:
        if error in specific_conditions:
            if error == "multiple_errors_pointing_to_same_box":
                adjustment_factor = 0.2  # Example adjustment factor
                for test in static_weights:
                    dynamic_weights[test] = [w + adjustment_factor for w in dynamic_weights[test]]
            elif error == "network_issue_prevalent":
                for test in static_weights:
                    if "network" in test.lower():
                        dynamic_weights[test] = [w * 1.1 for w in dynamic_weights[test]]  # Increase by 10%
    
    return dynamic_weights

# Priority algorithm
def prioritize_tests(dynamic_weights, flowchart_path):
    test_priorities = {}
    for box in flowchart_path:
        for test in box_tests.get(box, []):
            test_priorities[test] = max(dynamic_weights[test])  # Using max weight as priority
    
    # Sort tests based on priority (highest weight first)
    sorted_tests = sorted(test_priorities.items(), key=lambda x: x[1], reverse=True)
    return [test[0] for test in sorted_tests]

# usage
reported_errors = ["network_issue", "service_slow"]
flowchart_path = ["box1", "box2", "box3"]  # Determined by following the flowchart based on errors

# Adjust weights based on reported errors
dynamic_weights = adjust_weights(static_weights, reported_errors)

# Prioritize tests based on adjusted weights and flowchart path
priority_tests = prioritize_tests(dynamic_weights, flowchart_path)

print("Tests to perform in order of priority:", priority_tests)
