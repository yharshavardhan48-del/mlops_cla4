import sys

import requests


API_URL = "http://127.0.0.1:8000/predict"
PASSENGER = {
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S",
}


def main() -> None:
    try:
        response = requests.post(API_URL, json=PASSENGER, timeout=10)
    except requests.exceptions.ConnectionError as error:
        print(
            f"Could not connect to {API_URL}. Start the API with "
            "'python -m uvicorn app:app --reload' and try again. "
            f"Details: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    except requests.exceptions.RequestException as error:
        print(f"Request to {API_URL} failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print(f"Status code: {response.status_code}")
    try:
        print(response.json())
    except requests.exceptions.JSONDecodeError:
        print(response.text)
        if not response.ok:
            raise SystemExit(1)

    if not response.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
