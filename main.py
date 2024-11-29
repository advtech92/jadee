import os
import sqlite3
import time
import torch
import discord
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from datetime import datetime, timedelta, timezone
from model import JadeModel
from dotenv import load_dotenv
from collections import deque
import uuid as uuid_lib
import json

# Constants
SCOPES = ['https://www.googleapis.com/auth/youtube.readonly']
DATABASE_FILE = 'global_user_data.db'  # Updated database file name
CHANNEL_HANDLE = 'UCsVJcf4KbO8Vz308EKpSYxw'
STREAM_KEYWORD = "Live"

# Load environment variables
load_dotenv()

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
client = discord.Client(intents=intents)

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = JadeModel().to(device)

# Context management for conversation continuity
conversation_history = deque(maxlen=5)  # Store the last 5 messages for context
training_data = []  # Store live messages for training purposes

# Profile Manager
class ProfileManager:
    def __init__(self):
        self._create_profiles_table()

    def _create_profiles_table(self):
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS global_profiles (
                uuid TEXT PRIMARY KEY,
                discord_user_id TEXT UNIQUE,
                youtube_channel_id TEXT UNIQUE,
                points INTEGER DEFAULT 0,
                last_interaction TIMESTAMP,
                subscription_status TEXT,
                first_seen_as_member TIMESTAMP,
                has_opted_in INTEGER DEFAULT 0
            )
        ''')
        conn.commit()
        conn.close()

    def get_or_create_uuid(self, discord_id=None, youtube_id=None):
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        uuid = None

        if discord_id:
            cursor.execute("SELECT uuid FROM global_profiles WHERE discord_user_id = ?", (discord_id,))
            result = cursor.fetchone()
            if result:
                uuid = result[0]

        if not uuid and youtube_id:
            cursor.execute("SELECT uuid FROM global_profiles WHERE youtube_channel_id = ?", (youtube_id,))
            result = cursor.fetchone()
            if result:
                uuid = result[0]

        if not uuid:
            uuid = str(uuid_lib.uuid4())
            cursor.execute('''
                INSERT INTO global_profiles (uuid, discord_user_id, youtube_channel_id)
                VALUES (?, ?, ?)
            ''', (uuid, discord_id, youtube_id))
            conn.commit()

        conn.close()
        return uuid

    def update_subscription_status(self, youtube_id, status):
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE global_profiles
            SET subscription_status = ?, last_interaction = ?
            WHERE youtube_channel_id = ?
        ''', (status, datetime.utcnow(), youtube_id))
        conn.commit()
        conn.close()

    def delete_user_data(self, uuid):
        # Delete user data to comply with GDPR
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM global_profiles WHERE uuid = ?', (uuid,))
        user_data = cursor.fetchone()
        if user_data:
            with open(f'deleted_user_data_{uuid}.json', 'w') as f:
                json.dump({
                    'uuid': user_data[0],
                    'discord_user_id': user_data[1],
                    'youtube_channel_id': user_data[2],
                    'points': user_data[3],
                    'last_interaction': user_data[4],
                    'subscription_status': user_data[5],
                    'first_seen_as_member': user_data[6],
                    'has_opted_in': user_data[7]
                }, f)
        cursor.execute('DELETE FROM global_profiles WHERE uuid = ?', (uuid,))
        conn.commit()
        conn.close()

    def has_opted_in(self, uuid):
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute('SELECT has_opted_in FROM global_profiles WHERE uuid = ?', (uuid,))
        result = cursor.fetchone()
        conn.close()
        return result and result[0] == 1

    def set_opt_in(self, uuid, opted_in=True):
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE global_profiles
            SET has_opted_in = ?
            WHERE uuid = ?
        ''', (1 if opted_in else 0, uuid))
        conn.commit()
        conn.close()

profile_manager = ProfileManager()

# YouTube API Functions
def get_authenticated_service():
    flow = InstalledAppFlow.from_client_secrets_file(
        'client_secret.json', SCOPES)
    creds = flow.run_local_server(port=63355)
    with open('token.json', 'w') as token:
        token.write(creds.to_json())
    return build('youtube', 'v3', credentials=creds)

def find_correct_live_video(youtube, channel_id, keyword):
    request = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        eventType="live",
        type="video"
    )
    response = request.execute()
    items = response.get('items', [])
    for item in items:
        title = item['snippet']['title']
        if keyword.lower() in title.lower():
            return item['id']['videoId']
    return None

def get_live_chat_id(youtube, video_id):
    request = youtube.videos().list(
        part="liveStreamingDetails",
        id=video_id
    )
    response = request.execute()
    items = response.get('items', [])
    if items:
        return items[0]['liveStreamingDetails'].get('activeLiveChatId')
    return None

def monitor_youtube_chat(youtube, live_chat_id):
    if not live_chat_id:
        print("No valid live chat ID found.")
        return False

    next_page_token = None
    while True:
        try:
            request = youtube.liveChatMessages().list(
                liveChatId=live_chat_id,
                part="snippet,authorDetails",
                maxResults=200,
                pageToken=next_page_token
            )
            response = request.execute()

            if 'items' in response and response['items']:
                for item in response['items']:
                    user_id = item['authorDetails']['channelId']
                    display_name = item['authorDetails']['displayName']
                    is_moderator = item['authorDetails']['isChatModerator']
                    is_member = item['authorDetails']['isChatSponsor']
                    message = item['snippet']['displayMessage']

                    uuid = profile_manager.get_or_create_uuid(youtube_id=user_id)
                    if is_member:
                        profile_manager.update_subscription_status(user_id, "subscribed")
                    else:
                        profile_manager.update_subscription_status(user_id, "none")

                    print(f"[{datetime.utcnow()}] {display_name}: {message} (UUID: {uuid})")

                    # Add live chat message to training data if the user has opted in
                    if profile_manager.has_opted_in(uuid):
                        training_data.append((display_name, message))

                next_page_token = response.get('nextPageToken')

            else:
                print("No new messages detected; continuing to poll...")

        except Exception as e:
            print(f"Error while monitoring chat: {e}")
            time.sleep(30)  # Wait before retrying in case of an error

        time.sleep(10)  # Adjust this delay as needed

# Discord Event Handlers
@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    # Link the Discord user to the correct global profile UUID
    uuid = profile_manager.get_or_create_uuid(discord_id=str(message.author.id))

    # Ensure user has opted in before interacting
    if not profile_manager.has_opted_in(uuid):
        await message.channel.send("Please type '!optin' to confirm that you agree to data usage and interaction with this bot.")
        return

    if message.content.lower() == '!optin':
        profile_manager.set_opt_in(uuid, True)
        await message.channel.send("You have successfully opted in to data usage.")
        return

    # Add the message to conversation history for context
    conversation_history.append(message.content)

    # Generate a response using Jade with context
    context = "\n".join(conversation_history)
    response = model.generate_response(context)
    if response:
        await message.channel.send(response)

    print(f"Discord Interaction: User {message.author} (UUID: {uuid})")

# Main Function to Start Both Services
def main():
    youtube = get_authenticated_service()
    channel_id = profile_manager.get_or_create_uuid(youtube_id=CHANNEL_HANDLE)
    video_id = find_correct_live_video(youtube, channel_id, STREAM_KEYWORD)
    if video_id:
        live_chat_id = get_live_chat_id(youtube, video_id)
        if live_chat_id:
            print("Monitoring YouTube live chat...")
            monitor_youtube_chat(youtube, live_chat_id)
        else:
            print("No live chat ID available.")
    else:
        print("Could not find the correct live stream or it is not live.")

    client.run(os.getenv('DISCORD_TOKEN'))

if __name__ == "__main__":
    main()
