from evaluation.trust_flag import generate_trust_flag

def main():
# Simulate all checks passing.
 passing_checks = {
"perturbation_budget": {"passed": True},
"pixel_range": {"passed": True},
"prediction_outputs": {"passed": True},
"metric_consistency": {"passed": True},
}


 result = generate_trust_flag(passing_checks)

 print("========== TRUST FLAG TEST ==========")
 print("Status:", result["status"])
 print("Summary:", result["summary"])
 print(f"Checks passed: {result['passed']}/{result['total']}")
 print("=====================================")

 assert result["status"] == "PASS"

# Verify that a failed check produces FAIL.
failing_checks = {
    "perturbation_budget": {"passed": False},
    "pixel_range": {"passed": True},
}

failed_result = generate_trust_flag(failing_checks)

print("\nTesting failed validation...")
print("Status:", failed_result["status"])

assert failed_result["status"] == "FAIL"

print("\nAll Trust Flag tests passed!")

if __name__ == "__main__":
 main()
