def generate_trust_flag(checks):


 if not checks:
    return {
        "status": "WARNING",
        "summary": "No evaluation checks were provided.",
        "passed": 0,
        "failed": 0,
        "total": 0,
    }

 passed = 0
 failed = 0
 missing = 0

 for result in checks.values():
    if not isinstance(result, dict) or "passed" not in result:
        missing += 1
    elif result["passed"] is True:
        passed += 1
    else:
        failed += 1

 total = len(checks)

 if failed > 0:
    status = "FAIL"
    summary = f"{failed} evaluation check(s) failed."
 elif missing > 0:
    status = "WARNING"
    summary = f"{missing} evaluation check(s) have invalid results."
 else:
    status = "PASS"
    summary = "All configured evaluation checks passed."

 return {
    "status": status,
    "summary": summary,
    "passed": passed,
    "failed": failed,
    "total": total,
}

