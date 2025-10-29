import requests

class HttpPostClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def post_json(self, endpoint, data, headers=None):
        url = self.base_url + endpoint
        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print("HTTP request failed: " + e)
            return None