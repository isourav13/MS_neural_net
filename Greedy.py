from collections import defaultdict

def scoring_function(call_graph, available_microservices, microservice, problem_class, historical_error_rates):
  """
  Calculates a score for a microservice based on call frequency and historical error rate.

  Args:
    call_graph: A dictionary representing the call graph, where keys are microservices and values are lists of called microservices.
    available_microservices: A set of microservices already replicated in the DT.
    microservice: The microservice to be scored.
    problem_class: The class of the reported problem.
    historical_error_rates: A dictionary storing historical error rates for each microservice and problem class combination.

  Returns:
    A score for the microservice.
  """
  call_count = 0
  for caller in call_graph:
    if microservice in call_graph[caller] and caller in available_microservices:
      call_count += 1

  error_rate = historical_error_rates.get((microservice, problem_class), 0.0)

  # Weight call frequency and error rate (adjust weights as needed)
  score = 0.6 * call_count + 0.4 * error_rate
  return score

def greedy_microservice_replication(call_graph, problem_class, available_microservices, historical_error_rates):
  """
  Implements the greedy algorithm for microservice replication.

  Args:
    call_graph: A dictionary representing the call graph.
    problem_class: The class of the reported problem.
    available_microservices: A set of microservices already replicated in the DT.
    historical_error_rates: A dictionary storing historical error rates.

  Returns:
    A prioritized list of microservices to be added to the DT.
  """
  microservices_to_add = []
  while True:
    # Identify candidate microservices
    candidate_microservices = [m for m in call_graph
                              if any(caller in available_microservices for caller in call_graph[m])
                              and m not in available_microservices]
    print(candidate_microservices)

    if not candidate_microservices:
      break

    # Score each candidate microservice
    scores = {m: scoring_function(call_graph, available_microservices, m, problem_class, historical_error_rates) for m in candidate_microservices}
    # Select the microservice with the highest score
    highest_scoring_microservice = max(scores, key=scores.get)
    available_microservices.add(highest_scoring_microservice)
    microservices_to_add.append(highest_scoring_microservice)

    # Will implement  root cause identification logic here (potentially with a lightweight test)
    # If the root cause is identified, break the loop

  return microservices_to_add

call_graph = {
  "A": ["B", "C"],
  "B": ["D"],
  "C": ["E", "B"],
  "D": [],
  "E": ["A"],
}

available_microservices = {"B"}
problem_class = "AuthenticationError"
historical_error_rates = {("A", "AuthenticationError"): 0.6, ("B", "AuthenticationError"): 0.3, ("C", "AuthenticationError"): 0.5, ("D", "AuthenticationError"): 0.9, ("E", "AuthenticationError"): 0.8}

microservices_to_add = greedy_microservice_replication(call_graph, problem_class, available_microservices, historical_error_rates)

print("Microservices to add:", microservices_to_add)


