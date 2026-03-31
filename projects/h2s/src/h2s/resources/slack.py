"""Slack resource for H2S alert notifications."""

from dagster import ConfigurableResource
from slack_sdk import WebClient


class SlackAlertResource(ConfigurableResource):
    """Slack resource that wraps a bot token and target channel."""

    token: str
    channel: str

    def get_client(self) -> WebClient:
        return WebClient(token=self.token)