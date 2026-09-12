"""Simple API test harness for assignment activity 3.3.

Run:
    python -m api.test_api
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from api.app import app


def main() -> None:
    client = TestClient(app)

    checks = [
        ("/health", 200),
        ("/api/v1/application/details", 200),
        ("/api/v1/flow/details", 200),
        ("/api/v1/dataset/details", 200),
        ("/api/v1/preprocessing/details", 200),
        ("/api/v1/model/details", 200),
        ("/api/v1/output/artifacts", 200),
        ("/api/v1/does-not-exist", 404),
    ]

    for path, expected_status in checks:
        response = client.get(path)
        assert response.status_code == expected_status, (
            f"Expected {expected_status} for {path}, got {response.status_code}. "
            f"Response body: {response.text}"
        )
        print(f"{path} -> HTTP {response.status_code}")
        if response.headers.get("content-type", "").startswith("application/json"):
            payload = response.json()
            print(json.dumps(payload, indent=2)[:1000])
        print("-" * 80)

    print("All API checks passed.")


if __name__ == "__main__":
    main()
