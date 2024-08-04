from selenium import webdriver

from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import SessionNotCreatedException

URL = "https://ckonline.tbgtom.com/DatabaseEdit.aspx?id=0"


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


options = Options()
options.add_experimental_option("detach", True)


driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()), options=options
)

driver.get(URL)
driver.maximize_window()

element = driver.find_element(
    By.XPATH, '//*[@id="ctl00_ContentPlaceHolder1_txtSection"]'
)
element.send_keys("sale")
