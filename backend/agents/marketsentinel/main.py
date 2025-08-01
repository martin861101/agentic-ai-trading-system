import logging
import time
from backend.orchestrator.event_bus import event_bus

logger = logging.getLogger(__name__)

class MarketSentinel:
    """
    The MarketSentinel agent is responsible for monitoring market data,
    such as price updates, and publishing relevant events.
    """
    AGENT_NAME = "MarketSentinel"

    def __init__(self):
        logger.info(f"{self.AGENT_NAME} agent initialized.")
        self.running = False

    def start(self):
        """
        Starts the agent's main loop.
        """
        self.running = True
        logger.info(f"{self.AGENT_NAME} agent started.")
        event_bus.publish(
            "agent_lifecycle",
            {"agent_name": self.AGENT_NAME, "status": "started"}
        )
        self.run()

    def stop(self):
        """
        Stops the agent's main loop.
        """
        self.running = False
        logger.info(f"{self.AGENT_NAME} agent stopped.")
        event_bus.publish(
            "agent_lifecycle",
            {"agent_name": self.AGENT_NAME, "status": "stopped"}
        )

    def run(self):
        """
        The main loop of the agent.
        In a real implementation, this would connect to a data source
        and monitor for market updates.
        """
        while self.running:
            logger.info(f"{self.AGENT_NAME} is monitoring the market...")

            # In a real scenario, you would get data from an API
            # and publish it to the event bus.
            # Example:
            # market_data = self.fetch_market_data("AAPL")
            # event_bus.publish("market_data_update", {"symbol": "AAPL", "data": market_data})

            # For this placeholder, we'll just sleep.
            time.sleep(15)

def main():
    """
    Main function to run the MarketSentinel agent.
    """
    try:
        sentinel = MarketSentinel()
        sentinel.start()
    except KeyboardInterrupt:
        logger.info("MarketSentinel agent shutting down.")
        sentinel.stop()

if __name__ == "__main__":
    # This allows running the agent as a standalone script
    logging.basicConfig(level=logging.INFO)
    main()
