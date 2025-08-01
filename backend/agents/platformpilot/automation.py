from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

class WebAutomation:
    def __init__(self, headless=True):
        options = Options()
        options.headless = headless
        self.driver = webdriver.Chrome(options=options)

    def search_and_click(self, search_url: str, link_text: str):
        self.driver.get(search_url)
        time.sleep(3)  # wait for page load (replace with smarter wait)

        try:
            link = self.driver.find_element(By.LINK_TEXT, link_text)
            link.click()
            return True
        except Exception as e:
            print(f"Link not found or error: {e}")
            return False

    def close(self):
        self.driver.quit()
# PlatformPilot automation logic
