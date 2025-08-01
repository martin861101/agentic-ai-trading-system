import requests

class TravilyClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.travily.io/v1"

    def get_latest_events(self, market="forex"):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.get(f"{self.base_url}/events?market={market}", headers=headers)
        if resp.status_code == 200:
            return resp.json()
        else:
            return {}
# MacroForecaster utils
