from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class EventBus:
    def __init__(self):
        self._subscribers = defaultdict(list)

    def subscribe(self, event_type: str, handler):
        """Subscribe a handler to an event type."""
        logger.info(f"Subscribing {handler.__name__} to event '{event_type}'")
        self._subscribers[event_type].append(handler)

    def publish(self, event_type: str, data: dict):
        """Publish an event to all subscribed handlers."""
        logger.info(f"Publishing event '{event_type}' with data: {data}")
        if event_type in self._subscribers:
            for handler in self._subscribers[event_type]:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"Error in event handler {handler.__name__} for event '{event_type}': {e}")
        else:
            logger.warning(f"No subscribers for event type '{event_type}'")

# Create a singleton instance of the EventBus
event_bus = EventBus()
