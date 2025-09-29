import requests
import json

url = "http://localhost:8501/api/v1/broadcast/recommendations"

payload = {
    "broadcastTime": "2024-01-15T20:00:00",
    "recommendationCount": 3
}

headers = {
    'Content-Type': 'application/json'
}

print("--- Starting API test from within the container ---")
try:
    # 컨테이너 내부에서는 localhost로 접근
    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120) # 타임아웃을 넉넉하게 120초로 설정
    response.raise_for_status()  # 200번대 상태 코드가 아니면 예외 발생
    
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    # JSON 출력을 위해 pretty-print
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

except requests.exceptions.HTTPError as errh:
    print(f"Http Error: {errh}")
    print(f"Response content: {response.text}")
except requests.exceptions.ConnectionError as errc:
    print(f"Error Connecting: {errc}")
except requests.exceptions.Timeout as errt:
    print(f"Timeout Error: {errt}")
except requests.exceptions.RequestException as err:
    print(f"Oops: Something Else: {err}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
