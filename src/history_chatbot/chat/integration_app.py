"""Explicit fictional-fixture ASGI factory for local integration smoke tests."""

import os
from pathlib import Path

from history_chatbot.chat.api import create_app
from history_chatbot.chat.demo_journey import InMemoryDemoJourneyProvider
from history_chatbot.chat.service import create_development_integration_service


def create_integration_app():
    runtime_dir = Path(
        os.environ.get(
            "INTEGRATION_RUNTIME_DIR", ".runtime/development-integration"
        )
    )
    return create_app(
        service=create_development_integration_service(runtime_dir=runtime_dir),
        journey_provider=InMemoryDemoJourneyProvider(),
    )
