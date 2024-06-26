# TODO set up flask callback to redirect to auth url

import json
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow, Flow
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/drive"]


def handle_auth():
    """Shows basic usage of the Docs API.
    Prints the title of a sample document.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=8000)  # 8000
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    fetch_lot_from_gdrive("CO-DAR", credentials=creds)


def fetch_lot_from_gdrive(lot_str: str, credentials):
    with build("drive", "v3", credentials=credentials) as service:
        collections = service.files()
        request = collections.list(
            q=f"name contains '{lot_str}' and mimeType = 'application/pdf'"
        )

        try:
            response = request.execute()
            # print(response, end="\n***************************\n")
            data = json.dumps(response, sort_keys=True, indent=4)
            print(data)

        except HttpError as e:
            print(f"Error response : {e}")


if __name__ == "__main__":
    handle_auth()
