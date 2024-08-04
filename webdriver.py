from selenium import webdriver
import os

# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from webdriver_manager.chrome import ChromeDriverManager

from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import SessionNotCreatedException

URL = "https://ckonline.tbgtom.com/DatabaseEdit.aspx?id=0"

xpaths = {
    "block": '//*[@id="ctl00_ContentPlaceHolder1_txtSection"]',
    "lot": '//*[@id="ctl00_ContentPlaceHolder1_txtLot"]',
    "grave": '//*[@id="ctl00_ContentPlaceHolder1_txtGrave"]',
    "inter_date": '//*[@id="ctl00_ContentPlaceHolder1_txtInterDate"]',
    "cremains": '//*[@id="ctl00_ContentPlaceHolder1_IsCremains"]',
    "first_name": '//*[@id="ctl00_ContentPlaceHolder1_txtInterFirstName"]',
    "last_name": '//*[@id="ctl00_ContentPlaceHolder1_txtInterLastName"]',
    "gender": '//*[@id="ctl00_ContentPlaceHolder1_cboGender"]',
    "undertaker": '//*[@id="ctl00_ContentPlaceHolder1_txtUndertaker"]',
    "death_cause": '//*[@id="ctl00_ContentPlaceHolder1_txtDeathCause"]',
    "res_address": '//*[@id="ctl00_ContentPlaceHolder1_cfLate_Residence_Address"]',
    "res_city": '//*[@id="ctl00_ContentPlaceHolder1_cfLate_Residence_City"]',
    "res_state": '//*[@id="ctl00_ContentPlaceHolder1_cfLate_Residence_State"]',
    "initials": '//*[@id="ctl00_ContentPlaceHolder1_cfInitials"]',
    "age_at_death": '//*[@id="ctl00_ContentPlaceHolder1_cfAge_at_Death"]',
    "plot_card_num": '//*[@id="ctl00_ContentPlaceHolder1_cfPlot_Card_Num"]',
}


# def chrome_driver():
#     options = Options()
#     options.set_capability("detach", True)

#     driver = webdriver.Firefox()
#     url = driver.command_executor._url  # "http://127.0.0.1:60622/hub"
#     session_id = driver.session_id

#     print(f"URL: {url}")

#     try:

#         driver = webdriver.Remote(command_executor=url, options=options)
#         driver.close()  # this prevents the dummy browser
#         driver.session_id = session_id

#     except SessionNotCreatedException as e:
#         print(e)

profile_path = os.getenv("PROFILE_PATH", "")

ff_options = webdriver.FirefoxOptions()
profile = FirefoxProfile(profile_path)
ff_options.profile = profile
service = Service(GeckoDriverManager().install(), options=ff_options)
driver = webdriver.Firefox(service=service)

driver.get(URL)
assert "Crypt Keeper" in driver.title

element = driver.find_element(
    By.XPATH, '//*[@id="ctl00_ContentPlaceHolder1_txtSection"]'
)
element.send_keys("sale")
