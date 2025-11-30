"""Модуль для сбора новостей из Telegram каналов."""
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytz

import requests
from telethon import TelegramClient
from telethon.tl.types import PeerChannel

from src.models import Post


class TelegramPostsParser:
    """Парсер для сбора постов из Telegram каналов."""
    
    MAX_CHANNELS_PER_REQUEST = 100

    def __init__(
        self,
        tgstat_api_token: str,
        tgstat_base_url: str,
        telethon_session_name: str,
        telethon_api_id: int,
        telethon_api_hash: str,
        whitelist_categories_path: Optional[str] = None,
    ):
        self.tgstat_api_token = tgstat_api_token
        self.tgstat_base_url = tgstat_base_url
        self.telethon_session_name = telethon_session_name
        self.telethon_api_id = telethon_api_id
        self.telethon_api_hash = telethon_api_hash
        self.telethon_client = TelegramClient(
            telethon_session_name, telethon_api_id, telethon_api_hash
        )
        self.whitelist_categories = self._load_whitelist_categories(whitelist_categories_path)
        self.request_delay = 0.1
        self.msk_tz = pytz.timezone("Europe/Moscow")

    def _load_whitelist_categories(self, whitelist_categories_path: str) -> List[str]:
        """Load whitelist categories from categories.json file."""
        if not whitelist_categories_path:
            return []

        categories_file = Path(whitelist_categories_path)
        if not categories_file.exists():
            raise FileNotFoundError(f"{whitelist_categories_path} file not found")

        with open(categories_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [category["code"] for category in data["categories"]]

    def get_categories(self) -> List[Dict[str, str]]:
        """Get list of available channel categories from TGstat API."""
        url = f"{self.tgstat_base_url}/database/categories"
        params = {"token": self.tgstat_api_token}

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        if data["status"] != "ok":
            raise Exception(f"API error: {data.get('error', 'Unknown error')}")
        return [cat for cat in data["response"] if cat["code"] in self.whitelist_categories]

    def search_channels_by_category(
        self, category_code: str, n_channels: int
    ) -> List[Dict[str, Any]]:
        """Search channels in a specific category."""
        url = f"{self.tgstat_base_url}/channels/search"
        limit = min(n_channels, self.MAX_CHANNELS_PER_REQUEST)
        params = {
            "token": self.tgstat_api_token,
            "category": category_code,
            "limit": limit,
            "peer_type": "channel",
            "country": "ru",
            "language": "russian",
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        if data["status"] != "ok":
            raise Exception(f"API error: {data.get('error', 'Unknown error')}")

        return data["response"]["items"]

    def collect_channels_metadata(self, output_file: str, n_channels: int) -> List[Dict[str, Any]]:
        """Collect metadata for channels from all categories and save to file."""
        categories = self.get_categories()
        all_channels = []

        for category in categories:
            try:
                channels = self.search_channels_by_category(category["code"], n_channels)
                for channel in channels:
                    channel_data = {
                        "uri": channel["link"],
                        "category": category["name"],
                    }
                    all_channels.append(channel_data)
                time.sleep(self.request_delay)

            except Exception as e:
                print(f"Error processing category {category['name']}: {str(e)}")
                continue

        output_path = Path(output_file)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump({"channels": all_channels}, f, ensure_ascii=False, indent=2)
        return all_channels

    def _normalize_datetime(self, dt: datetime) -> datetime:
        """Нормализует datetime к MSK таймзоне."""
        if dt.tzinfo is None:
            dt = pytz.utc.localize(dt)
        return dt.astimezone(self.msk_tz)

    def filter_posts_by_window(
        self, posts: List[Post], window_start: datetime, window_end: datetime
    ) -> List[Post]:
        """Фильтрует посты по временному окну W."""
        filtered = []
        for post in posts:
            try:
                post_dt = datetime.fromisoformat(post.date.replace("Z", "+00:00"))
                post_dt = self._normalize_datetime(post_dt)
                
                if window_start <= post_dt <= window_end:
                    filtered.append(post)
            except Exception as e:
                print(f"Error parsing date for post {post.id}: {e}")
                continue
        return filtered

    async def get_channel_posts(
        self, channel_id: str, limit: int, min_date: Optional[str], channel_category: str
    ) -> List[Post]:
        """
        Get posts from a specific channel.

        Args:
            channel_id (str): Channel identifier (@username, t.me/username)
            limit (int, optional): Number of posts to return.
            min_date (Any, optional): Start timestamp for post filtering.
            channel_category (str): Category of telegram channel

        Returns:
            List[Post]: Response containing posts
        """
        await self.telethon_client.start()

        channel_posts = []
        async for message in self.telethon_client.iter_messages(
            channel_id, offset_date=datetime.strptime(min_date, "%Y-%m-%d").date(), limit=limit
        ):
            if message.text:
                channel_posts.append(
                    Post(
                        id=message.id,
                        date=str(message.date),
                        text=message.text or "",
                        views=message.views or 0,
                        forwards=message.forwards or 0,
                        channel_category=[{"code": channel_category}],
                        reactions=message.reactions.to_dict() if message.reactions else None,
                        channel_id=message.peer_id.channel_id
                        if isinstance(message.peer_id, PeerChannel)
                        else message.peer_id.chat_id,
                    )
                )
            time.sleep(self.request_delay)
        return channel_posts

    async def collect_channels_posts(
        self,
        channels: List[Dict[str, str]],
        output_file: str,
        min_date: Optional[str],
        limit: int,
    ) -> List[Post]:
        """
        Collect posts from multiple channels and save them to a file.

        Args:
            channels (List[Dict[str, str]]): List of channel metadata dictionaries
            output_file (str, optional): Path to save the posts. Defaults to "posts.json".
            min_date (str, optional): Start timestamp for post filtering.
            limit (int): Number of posts to collect per channel.

        Returns:
            List[Post]: List of all channel's parsed posts.
        """
        all_posts: List[Post] = []
        total_channels = len(channels)
        successful_channels = 0
        failed_channels = 0
        total_posts_collected = 0

        print(f"\nStarting to collect posts from {total_channels} channels...")
        print(f"Posts per channel: {limit}")

        for idx, channel in enumerate(channels, 1):
            channel_uri = channel["uri"]
            print(f"\nProcessing channel {idx}/{total_channels}: {channel_uri}")

            try:
                channel_posts = []
                posts = await self.get_channel_posts(
                    channel_id=channel_uri,
                    limit=limit,
                    min_date=min_date,
                    channel_category=channel["category"],
                )
                channel_posts.extend(posts)

                total_posts_collected += len(channel_posts)
                all_posts.extend(channel_posts)
                successful_channels += 1

                print(f"Successfully collected {len(channel_posts)} posts from {channel_uri}")

            except Exception as e:
                failed_channels += 1
                print(f"Error while processing channel {channel_uri}: {str(e)}")
                continue

        print(
            f"\nCollection completed: {successful_channels} channels processed successfully, "
            f"{failed_channels} failed, {total_posts_collected} total posts collected"
        )

        output_path = Path(output_file)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"posts": [item.model_dump(mode="python") for item in all_posts]},
                f,
                ensure_ascii=False,
                indent=2,
            )
        return all_posts

