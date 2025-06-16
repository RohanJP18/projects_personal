import os
import base64
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from notion_client import Client as NotionClient

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
creds = None

if os.path.exists('token.json'):
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
else:
    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
    creds = flow.run_local_server(port=0)
    with open('token.json', 'w') as token:
        token.write(creds.to_json())

service = build('gmail', 'v1', credentials=creds)

notion = NotionClient(auth="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")
DATABASE_ID = "214ee7089d9e80c68bf3e874455d12b1"

def get_task_emails():
    results = service.users().messages().list(userId='me', q="is:unread").execute()
    messages = results.get('messages', [])

    print(f" Found {len(messages)} unread emails")

    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
        headers = msg_data['payload']['headers']
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')

        print(" Adding to Notion:", subject)

        response = notion.pages.create(
            parent={"database_id": DATABASE_ID},
            properties={
                "Name": {
                    "title": [{"text": {"content": subject}}]
                },
                "Status Update": {
                    "status": {"name": "Not started"}
                }
            }
        )


        print(" Notion page created!")

        service.users().messages().modify(
            userId='me',
            id=msg['id'],
            body={'removeLabelIds': ['UNREAD']}
        ).execute()

get_task_emails()
