#!/usr/bin/env python3
"""
Fixed Agentic AI Trading System - Complete Setup Script
Creates the full project structure with corrected indentation
"""

import os
import subprocess
import sys
from pathlib import Path
import json

def create_file(path: str, content: str):
    """Create a file with the given content, creating directories as needed."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ Created: {path}")

def create_project_structure():
    """Create the complete agentic trading system project structure."""
    
    project_name = "agentic-trading"
    
    # Create main project directory
    if os.path.exists(project_name):
        print(f"Directory {project_name} already exists. Please remove it first or choose a different name.")
        return
    
    os.makedirs(project_name)
    os.chdir(project_name)
    
    print(f"🚀 Creating Agentic AI Trading System in {os.getcwd()}")
    
    # Root level files
    create_root_files()
    
    # Backend structure
    create_backend_structure()
    
    # Frontend structure  
    create_frontend_structure()
    
    # Docker and deployment
    create_deployment_files()
    
    print("\n🎉 Agentic Trading System created successfully!")
    print("\n📋 Next steps:")
    print("1. cd agentic-trading")
    print("2. Update .env with your API keys (copy from .env.example)")
    print("3. docker-compose up -d")
    print("4. Access frontend at http://localhost:3000")
    print("5. API docs at http://localhost:8000/docs")

def create_root_files():
    """Create root level configuration files."""
    
    # README.md
    readme_content = """# Agentic AI Trading System

A modular, microservices-based AI trading system with specialized agents for different aspects of trading analysis and execution.

## Architecture

- **ChartAnalyst**: Pattern recognition and technical analysis
- **RiskManager**: Position sizing and risk assessment  
- **MarketSentinel**: Volatility monitoring and scalping opportunities
- **MacroForecaster**: News analysis and macro event impact
- **TacticBot**: Trade execution logic and timing
- **PlatformPilot**: Platform automation and logging

## Quick Start

1. Copy environment file: `cp .env.example .env`
2. Update API keys in `.env`
3. Run: `docker-compose up -d`
4. Access dashboard: http://localhost:3000
5. API documentation: http://localhost:8000/docs

## Development

### Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn orchestrator.api:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm start
```

## Features

- Real-time WebSocket communication
- Event-driven agent coordination
- Comprehensive logging and analytics
- Modern React dashboard with TradingView integration
- Docker containerization
- PostgreSQL database with TimescaleDB extensions
"""
    create_file("README.md", readme_content)
    
    # .gitignore
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Node
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.next/
out/
dist/

# Environment files
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# Database
*.db
*.sqlite

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
logs/
*.log

# Docker
.dockerignore
"""
    create_file(".gitignore", gitignore_content)

def create_backend_structure():
    """Create the complete backend structure with all agents and orchestrator."""
    
    # Backend requirements
    requirements_content = """fastapi==0.104.1
uvicorn[standard]==0.24.0
websockets==12.0
aioredis==2.0.1
sqlalchemy==2.0.23
alembic==1.12.1
psycopg2-binary==2.9.8
pydantic==2.5.0
pydantic-settings==2.0.3
python-multipart==0.0.6
httpx==0.25.2
pandas==2.1.3
numpy==1.25.2
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
selenium==4.15.2
python-dotenv==1.0.0
asyncio-mqtt==0.13.0
schedule==1.2.0
loguru==0.7.2
"""
    create_file("backend/requirements.txt", requirements_content)
    
    # Configuration
    config_content = """from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://postgres:password@localhost:5432/agentic_trading"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # API Keys - UPDATE THESE!
    travily_api_key: Optional[str] =         "tacticbot": """
async def process_signal(input_data: AgentInput) -> AgentOutput:
    \"\"\"Process tactical trading analysis\"\"\"
    from .model import model
    from .utils import TacticbotUtils
    
    # Get aggregated signals from other agents
    agent_signals = input_data.data.get("agent_signals", [])
    market_conditions = input_data.data.get("market_conditions", {})
    
    # Run tactical analysis
    analysis = await model.predict({
        "agent_signals": agent_signals,
        "market_conditions": market_conditions,
        "symbol": input_data.symbol,
        "timeframe": input_data.timeframe
    })
    
    return AgentOutput(
        timestamp=datetime.now(),
        symbol=input_data.symbol,
        confidence=analysis.get("confidence", 0.0),
        signal_type=analysis.get("signal_type"),
        reasoning=analysis.get("reasoning", "Tactical execution analysis"),
        data=analysis,
        metadata={"agent_type": "tactical_execution"}
    )
""",
        "platformpilot": """
async def process_signal(input_data: AgentInput) -> AgentOutput:
    \"\"\"Process platform automation\"\"\"
    from .model import model
    from .utils import PlatformpilotUtils
    
    # Get trade decision data
    trade_decision = input_data.data.get("trade_decision", {})
    
    # Process automation request
    analysis = await model.predict({
        "trade_decision": trade_decision,
        "symbol": input_data.symbol,
        "automation_type": input_data.data.get("automation_type", "logging")
    })
    
    return AgentOutput(
        timestamp=datetime.now(),
        symbol=input_data.symbol,
        confidence=1.0,  # Platform actions are deterministic
        reasoning=analysis.get("reasoning", "Platform automation execution"),
        data=analysis,
        metadata={"agent_type": "platform_automation"}
    )
"""
    }
    
    return logic_map.get(agent_name, """
async def process_signal(input_data: AgentInput) -> AgentOutput:
    \"\"\"Default signal processing\"\"\"
    return AgentOutput(
        timestamp=datetime.now(),
        symbol=input_data.symbol,
        confidence=0.5,
        reasoning="Default agent processing",
        data={"status": "processed"},
        metadata={"agent_type": "default"}
    )
""")

def get_agent_utils(agent_name: str) -> str:
    """Generate agent-specific utility functions."""
    
    utils_map = {
        "chartanalyst": """
def detect_patterns(candles: List[Dict]) -> Dict:
    \"\"\"Detect chart patterns in candle data\"\"\"
    if len(candles) < 20:
        return {"pattern": "insufficient_data", "confidence": 0.0}
    
    # Placeholder pattern detection
    patterns = ["bullish_engulfing", "bearish_engulfing", "doji", "hammer", "shooting_star"]
    import random
    detected_pattern = random.choice(patterns)
    
    return {
        "pattern": detected_pattern,
        "confidence": random.uniform(0.3, 0.9),
        "support_levels": [random.uniform(100, 200) for _ in range(3)],
        "resistance_levels": [random.uniform(200, 300) for _ in range(3)]
    }

def calculate_indicators(candles: List[Dict]) -> Dict:
    \"\"\"Calculate technical indicators\"\"\"
    if not candles:
        return {}
    
    # Simple moving averages (placeholder)
    closes = [float(c.get("close", 0)) for c in candles[-20:]]
    if len(closes) >= 10:
        sma_10 = sum(closes[-10:]) / 10
        sma_20 = sum(closes) / len(closes)
        return {"sma_10": sma_10, "sma_20": sma_20}
    
    return {}
""",
        "riskmanager": """
def calculate_position_size(portfolio_value: float, risk_percent: float, stop_loss_distance: float) -> Dict:
    \"\"\"Calculate optimal position size based on risk management\"\"\"
    if stop_loss_distance <= 0 or portfolio_value <= 0:
        return {"position_size": 0, "reason": "invalid_parameters"}
    
    risk_amount = portfolio_value * (risk_percent / 100)
    position_size = risk_amount / stop_loss_distance
    
    return {
        "position_size": round(position_size, 2),
        "risk_amount": risk_amount,
        "risk_percent": risk_percent,
        "stop_loss_distance": stop_loss_distance
    }

def assess_risk_reward(entry_price: float, stop_loss: float, take_profit: float) -> Dict:
    \"\"\"Assess risk-reward ratio\"\"\"
    if entry_price <= 0:
        return {"risk_reward": 0, "assessment": "invalid"}
    
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    
    if risk <= 0:
        return {"risk_reward": float('inf'), "assessment": "no_risk"}
    
    ratio = reward / risk
    
    return {
        "risk_reward": round(ratio, 2),
        "risk": risk,
        "reward": reward,
        "assessment": "good" if ratio >= 2.0 else "acceptable" if ratio >= 1.5 else "poor"
    }
""",
        "marketsentinel": """
def analyze_volatility(price_data: List[float], window: int = 20) -> Dict:
    \"\"\"Analyze market volatility\"\"\"
    if len(price_data) < window:
        return {"volatility": 0, "regime": "unknown"}
    
    import numpy as np
    returns = np.diff(np.log(price_data))
    volatility = np.std(returns) * np.sqrt(252)  # Annualized
    
    # Classify volatility regime
    if volatility > 0.3:
        regime = "high"
    elif volatility > 0.15:
        regime = "medium"
    else:
        regime = "low"
    
    return {
        "volatility": round(volatility, 4),
        "regime": regime,
        "returns_std": np.std(returns),
        "avg_return": np.mean(returns)
    }

def detect_scalping_opportunities(tick_data: List[Dict]) -> List[Dict]:
    \"\"\"Detect short-term scalping opportunities\"\"\"
    opportunities = []
    
    if len(tick_data) < 100:
        return opportunities
    
    # Simple momentum-based opportunities (placeholder)
    for i in range(10, len(tick_data) - 10):
        current_price = float(tick_data[i].get("price", 0))
        prev_prices = [float(tick_data[j].get("price", 0)) for j in range(i-10, i)]
        
        if current_price > max(prev_prices) * 1.001:  # 0.1% breakout
            opportunities.append({
                "type": "bullish_momentum",
                "price": current_price,
                "timestamp": tick_data[i].get("timestamp"),
                "strength": min((current_price / max(prev_prices) - 1) * 1000, 10)
            })
    
    return opportunities[:5]  # Return top 5
"""
    }
    
    return utils_map.get(agent_name, "# Agent-specific utilities go here")

def get_model_prediction(agent_name: str) -> str:
    """Generate model prediction logic for each agent."""
    
    prediction_map = {
        "chartanalyst": """
prediction = {
    "signal_type": "BUY" if input_data.get("candles", [{}])[-1].get("close", 0) > input_data.get("candles", [{}])[-2].get("close", 0) else "SELL",
    "confidence": 0.75,
    "pattern": "bullish_engulfing",
    "price_zones": {
        "support": 1970.0,
        "resistance": 1980.0
    },
    "reasoning": "Strong bullish pattern detected with high volume confirmation"
}
""",
        "riskmanager": """
portfolio_value = input_data.get("portfolio", {}).get("total_value", 10000)
risk_percent = 2.0  # 2% risk per trade

prediction = {
    "position_size": portfolio_value * 0.02,
    "stop_loss_percent": 1.0,
    "take_profit_percent": 2.0,
    "risk_reward_ratio": 2.0,
    "confidence": 0.85,
    "reasoning": f"Calculated 2% risk on portfolio of ${portfolio_value}"
}
""",
        "marketsentinel": """
prediction = {
    "volatility_regime": "medium",
    "scalping_opportunities": 3,
    "market_sentiment": "bullish",
    "volatility_score": 0.65,
    "confidence": 0.72,
    "reasoning": "Medium volatility detected with several scalping opportunities"
}
""",
        "macroforecaster": """
prediction = {
    "news_impact": "positive",
    "economic_bias": "bullish",
    "impact_score": 0.7,
    "key_events": ["Fed meeting", "Employment data"],
    "confidence": 0.68,
    "reasoning": "Positive economic indicators suggest bullish bias"
}
""",
        "tacticbot": """
prediction = {
    "signal_type": "BUY",
    "entry_timing": "immediate",
    "exit_strategy": "trailing_stop",
    "position_allocation": 0.25,
    "confidence": 0.8,
    "reasoning": "Multiple agents confirm bullish signal with good risk/reward"
}
""",
        "platformpilot": """
prediction = {
    "automation_status": "executed",
    "platform_actions": ["log_signal", "send_alert", "update_dashboard"],
    "execution_time": datetime.now().isoformat(),
    "confidence": 1.0,
    "reasoning": "Platform automation completed successfully"
}
"""
    }
    
    return prediction_map.get(agent_name, """
prediction = {
    "status": "processed",
    "confidence": 0.5,
    "reasoning": "Default prediction logic"
}
""")

def create_orchestrator_files():
    """Create the orchestrator service files."""
    
    # Create orchestrator directory
    create_file("backend/orchestrator/__init__.py", "")
    
    # Event bus (Redis pub/sub wrapper)
    event_bus_content = """import asyncio
import aioredis
import json
import logging
from typing import Dict, Any, Callable, List
from datetime import datetime

logger = logging.getLogger(__name__)

class EventBus:
    \"\"\"Redis-based event bus for agent communication\"\"\"
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis = None
        self.pubsub = None
        self.subscribers = {}
        self.running = False
        
    async def connect(self):
        \"\"\"Connect to Redis\"\"\"
        try:
            self.redis = await aioredis.from_url(self.redis_url)
            self.pubsub = self.redis.pubsub()
            logger.info("Connected to Redis event bus")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self):
        \"\"\"Disconnect from Redis\"\"\"
        self.running = False
        if self.pubsub:
            await self.pubsub.close()
        if self.redis:
            await self.redis.close()
        logger.info("Disconnected from Redis")
    
    async def publish(self, channel: str, message: Dict[str, Any]):
        \"\"\"Publish message to channel\"\"\"
        try:
            if not self.redis:
                await self.connect()
            
            # Add metadata
            enriched_message = {
                **message,
                "timestamp": datetime.now().isoformat(),
                "channel": channel
            }
            
            await self.redis.publish(channel, json.dumps(enriched_message))
            logger.debug(f"Published to {channel}: {message}")
            
        except Exception as e:
            logger.error(f"Failed to publish to {channel}: {e}")
            raise
    
    async def subscribe(self, channel: str, callback: Callable):
        \"\"\"Subscribe to channel with callback\"\"\"
        try:
            if channel not in self.subscribers:
                self.subscribers[channel] = []
            
            self.subscribers[channel].append(callback)
            await self.pubsub.subscribe(channel)
            logger.info(f"Subscribed to channel: {channel}")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to {channel}: {e}")
            raise
    
    async def start_listening(self):
        \"\"\"Start listening for messages\"\"\"
        if not self.pubsub:
            await self.connect()
        
        self.running = True
        logger.info("Started listening for events")
        
        try:
            while self.running:
                message = await self.pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    await self._handle_message(message)
                    
        except Exception as e:
            logger.error(f"Error in event loop: {e}")
        finally:
            logger.info("Stopped listening for events")
    
    async def _handle_message(self, message):
        \"\"\"Handle incoming message\"\"\"
        try:
            channel = message['channel'].decode('utf-8')
            data = json.loads(message['data'].decode('utf-8'))
            
            if channel in self.subscribers:
                for callback in self.subscribers[channel]:
                    await callback(channel, data)
                    
        except Exception as e:
            logger.error(f"Error handling message: {e}")

# Global event bus instance
event_bus = EventBus()
"""
    create_file("backend/orchestrator/event_bus.py", event_bus_content)
    
    # Main orchestrator API
    orchestrator_api_content = """from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Dict
import asyncio
import logging
import json
from datetime import datetime
import sys
import os

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.db_session import get_db, init_db
from db.models import TradeSignal, Agent, TradeOutcome
from .event_bus import event_bus
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agentic Trading Orchestrator",
    description="Central orchestrator for AI trading agents",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections.copy():
            try:
                await connection.send_json(message)
            except:
                if connection in self.active_connections:
                    self.active_connections.remove(connection)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    \"\"\"Initialize services on startup\"\"\"
    try:
        # Initialize database
        init_db()
        
        # Connect to event bus
        await event_bus.connect()
        
        # Start event listener
        asyncio.create_task(event_bus.start_listening())
        
        logger.info("Orchestrator started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start orchestrator: {e}")
        # Don't raise in startup to allow for development without Redis

@app.on_event("shutdown")
async def shutdown_event():
    \"\"\"Cleanup on shutdown\"\"\"
    try:
        await event_bus.disconnect()
    except:
        pass
    logger.info("Orchestrator shut down")

# REST API Endpoints

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/agents")
async def get_agents():
    \"\"\"Get all registered agents\"\"\"
    # Mock data for demo
    agents = [
        {"name": "chartanalyst", "status": "active", "port": 8001},
        {"name": "riskmanager", "status": "active", "port": 8002},
        {"name": "marketsentinel", "status": "active", "port": 8003},
        {"name": "macroforecaster", "status": "active", "port": 8004},
        {"name": "tacticbot", "status": "active", "port": 8005},
        {"name": "platformpilot", "status": "active", "port": 8006},
    ]
    return agents

@app.get("/signals")
async def get_recent_signals(limit: int = 50):
    \"\"\"Get recent trading signals\"\"\"
    # Mock data for demo
    signals = [
        {
            "symbol": "EURUSD",
            "signal_type": "BUY",
            "confidence": 0.85,
            "agent_name": "chartanalyst",
            "timestamp": datetime.now().isoformat(),
            "reasoning": "Strong bullish pattern detected"
        }
    ]
    return signals

@app.post("/manual_signal")
async def create_manual_signal(signal_data: Dict):
    \"\"\"Create manual trading signal\"\"\"
    try:
        # Broadcast manual signal
        await manager.broadcast({
            "type": "manual_signal",
            "data": signal_data
        })
        
        return {"status": "success", "message": "Manual signal created"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back or handle client messages
            await websocket.send_text(f"Message received: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.orchestrator_port)
"""
    create_file("backend/orchestrator/api.py", orchestrator_api_content)

def create_frontend_structure():
    """Create the React frontend structure."""
    
    # Package.json
    package_json = """{
  "name": "agentic-trading-frontend",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "axios": "^1.6.0",
    "recharts": "^2.8.0",
    "lightweight-charts": "^4.1.0",
    "@mui/material": "^5.14.0",
    "@emotion/react": "^11.11.0",
    "@emotion/styled": "^11.11.0",
    "@mui/icons-material": "^5.14.0",
    "moment": "^2.29.4",
    "lodash": "^4.17.21"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  },
  "proxy": "http://localhost:8000"
}"""
    create_file("frontend/package.json", package_json)
    
    # Main App component
    app_js_content = """import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  AppBar,
  Toolbar,
  Chip,
  Alert
} from '@mui/material';
import LiveSignalFeed from './components/LiveSignalFeed';
import ChartOverlay from './components/ChartOverlay';
import AgentLogsPanel from './components/AgentLogsPanel';
import MacroEventFeed from './components/MacroEventFeed';
import TradeBook from './components/TradeBook';
import { SignalWebSocket } from './services/websocket';
import './App.css';

function App() {
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [signals, setSignals] = useState([]);
  const [decisions, setDecisions] = useState([]);
  const [agentLogs, setAgentLogs] = useState([]);

  useEffect(() => {
    // Initialize WebSocket connection
    const ws = new SignalWebSocket('ws://localhost:8000/ws');
    
    ws.onMessage((data) => {
      handleWebSocketMessage(data);
    });

    ws.ws.onopen = () => setConnectionStatus('connected');
    ws.ws.onclose = () => setConnectionStatus('disconnected');
    ws.ws.onerror = () => setConnectionStatus('error');

    // Add sample data for demo
    setTimeout(() => {
      setSignals([
        {
          agent_name: 'chartanalyst',
          symbol: 'EURUSD',
          signal_type: 'BUY',
          confidence: 0.85,
          reasoning: 'Bullish engulfing pattern detected',
          timestamp: new Date().toISOString()
        }
      ]);
      
      setAgentLogs([
        {
          id: 1,
          agent_name: 'chartanalyst',
          symbol: 'EURUSD',
          confidence: 0.85,
          reasoning: 'Strong bullish pattern with volume confirmation',
          timestamp: new Date().toISOString(),
          data: { pattern: 'bullish_engulfing', strength: 8.5 }
        }
      ]);
    }, 2000);

    return () => {
      ws.ws.close();
    };
  }, []);

  const handleWebSocketMessage = (data) => {
    console.log('WebSocket message:', data);
  };

  const getStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'success';
      case 'disconnected': return 'error';
      case 'error': return 'error';
      default: return 'warning';
    }
  };

  return (
    <div className="App">
      <AppBar position="static" sx={{ backgroundColor: '#1a1a1a' }}>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            🤖 Agentic AI Trading System
          </Typography>
          <Chip 
            label={`Connection: ${connectionStatus}`}
            color={getStatusColor()}
            variant="outlined"
          />
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ mt: 2 }}>
        <Grid container spacing={2}>
          {/* Top Row - Charts and Live Feed */}
          <Grid item xs={12} md={8}>
            <Paper sx={{ p: 2, height: '400px' }}>
              <Typography variant="h6" gutterBottom>
                📈 Chart Analysis
              </Typography>
              <ChartOverlay decisions={decisions} />
            </Paper>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 2, height: '400px', overflow: 'auto' }}>
              <Typography variant="h6" gutterBottom>
                📡 Live Signals
              </Typography>
              <LiveSignalFeed signals={signals} />
            </Paper>
          </Grid>

          {/* Middle Row - Agent Logs and Macro Events */}
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2, height: '350px', overflow: 'auto' }}>
              <Typography variant="h6" gutterBottom>
                🧠 Agent Intelligence
              </Typography>
              <AgentLogsPanel logs={agentLogs} />
            </Paper>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2, height: '350px', overflow: 'auto' }}>
              <Typography variant="h6" gutterBottom>
                🌍 Macro Events
              </Typography>
              <MacroEventFeed events={[]} />
            </Paper>
          </Grid>

          {/* Bottom Row - Trade Book */}
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                📊 Trade Book
              </Typography>
              <TradeBook trades={[]} decisions={decisions} />
            </Paper>
          </Grid>
        </Grid>
      </Container>
    </div>
  );
}

export default App;
"""
    create_file("frontend/src/App.js", app_js_content)
    
    # Create React components
    create_react_components()

def create_react_components():
    """Create React UI components."""
    
    # Component files
    components = [
        ("LiveSignalFeed", get_live_signal_component()),
        ("ChartOverlay", get_chart_component()),
        ("AgentLogsPanel", get_agent_logs_component()),
        ("MacroEventFeed", get_macro_events_component()),
        ("TradeBook", get_trade_book_component())
    ]
    
    for name, content in components:
        create_file(f"frontend/src/components/{name}.js", content)
    
    # Services
    create_file("frontend/src/services/websocket.js", get_websocket_service())
    create_file("frontend/src/services/api.js", get_api_service())
    
    # Styles and other files
    create_file("frontend/src/App.css", get_app_css())
    create_file("frontend/src/index.js", get_index_js())
    create_file("frontend/src/index.css", get_index_css())
    create_file("frontend/public/index.html", get_index_html())

def get_live_signal_component():
    return """import React from 'react';
import {
  List,
  ListItem,
  ListItemText,
  Chip,
  Box,
  Typography,
  Avatar
} from '@mui/material';
import moment from 'moment';

const LiveSignalFeed = ({ signals }) => {
  const getSignalColor = (signalType) => {
    switch (signalType) {
      case 'BUY': return 'success';
      case 'SELL': return 'error';
      case 'HOLD': return 'warning';
      default: return 'default';
    }
  };

  const getAgentAvatar = (agentName) => {
    const avatars = {
      chartanalyst: '📈',
      riskmanager: '🛡️',
      marketsentinel: '👁️',
      macroforecaster: '🌍',
      tacticbot: '🎯',
      platformpilot: '🤖'
    };
    return avatars[agentName] || '🔍';
  };

  if (!signals.length) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="100%">
        <Typography variant="body2" color="textSecondary">
          Waiting for signals...
        </Typography>
      </Box>
    );
  }

  return (
    <List sx={{ padding: 0 }}>
      {signals.map((signal, index) => (
        <ListItem key={index} divider sx={{ py: 1 }}>
          <Avatar sx={{ mr: 2, bgcolor: 'transparent', fontSize: '1.2em' }}>
            {getAgentAvatar(signal.agent_name)}
          </Avatar>
          <ListItemText
            primary={
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Typography variant="subtitle2">
                  {signal.symbol}
                </Typography>
                <Chip
                  label={signal.signal_type || 'ANALYSIS'}
                  color={getSignalColor(signal.signal_type)}
                  size="small"
                />
              </Box>
            }
            secondary={
              <Box>
                <Typography variant="body2" color="textSecondary">
                  {signal.agent_name}: {signal.reasoning}
                </Typography>
                <Typography variant="caption" color="textSecondary">
                  Confidence: {(signal.confidence * 100).toFixed(1)}% • 
                  {moment(signal.timestamp).fromNow()}
                </Typography>
              </Box>
            }
          />
        </ListItem>
      ))}
    </List>
  );
};

export default LiveSignalFeed;"""

def get_chart_component():
    return """import React from 'react';
import { Box, Typography } from '@mui/material';

const ChartOverlay = ({ decisions }) => {
  return (
    <Box sx={{ height: '100%', position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <Typography variant="h6" color="textSecondary">
        📊 Trading Chart (TradingView integration placeholder)
      </Typography>
    </Box>
  );
};

export default ChartOverlay;"""

def get_agent_logs_component():
    return """import React, { useState } from 'react';
import {
  List,
  ListItem,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Typography,
  Chip,
  Box,
  Tab,
  Tabs
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import moment from 'moment';

const AgentLogsPanel = ({ logs }) => {
  const [selectedAgent, setSelectedAgent] = useState('all');
  const [expanded, setExpanded] = useState(false);

  const agents = ['all', 'chartanalyst', 'riskmanager', 'marketsentinel', 'macroforecaster', 'tacticbot', 'platformpilot'];
  
  const filteredLogs = selectedAgent === 'all' 
    ? logs 
    : logs.filter(log => log.agent_name === selectedAgent);

  const handleChange = (event, newValue) => {
    setSelectedAgent(newValue);
  };

  const handleAccordionChange = (panel) => (event, isExpanded) => {
    setExpanded(isExpanded ? panel : false);
  };

  const getAgentColor = (agentName) => {
    const colors = {
      chartanalyst: 'primary',
      riskmanager: 'secondary',
      marketsentinel: 'info',
      macroforecaster: 'warning',
      tacticbot: 'success',
      platformpilot: 'default'
    };
    return colors[agentName] || 'default';
  };

  if (!logs.length) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="100%">
        <Typography variant="body2" color="textSecondary">
          No agent logs available
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Tabs
        value={selectedAgent}
        onChange={handleChange}
        variant="scrollable"
        scrollButtons="auto"
        sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}
      >
        {agents.map(agent => (
          <Tab
            key={agent}
            label={agent === 'all' ? 'All' : agent}
            value={agent}
            sx={{ minWidth: 'auto', textTransform: 'capitalize' }}
          />
        ))}
      </Tabs>

      <List sx={{ padding: 0, maxHeight: '250px', overflow: 'auto' }}>
        {filteredLogs.map((log, index) => (
          <Accordion 
            key={log.id || index}
            expanded={expanded === `panel${index}`}
            onChange={handleAccordionChange(`panel${index}`)}
            sx={{ mb: 1 }}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Box display="flex" alignItems="center" width="100%">
                <Chip
                  label={log.agent_name}
                  color={getAgentColor(log.agent_name)}
                  size="small"
                  sx={{ mr: 2 }}
                />
                <Typography variant="body2" sx={{ flexGrow: 1 }}>
                  {log.symbol} - {log.reasoning?.substring(0, 50)}...
                </Typography>
                <Typography variant="caption" color="textSecondary">
                  {moment(log.timestamp).format('HH:mm:ss')}
                </Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Box>
                <Typography variant="body2" gutterBottom>
                  <strong>Reasoning:</strong> {log.reasoning}
                </Typography>
                <Typography variant="body2" gutterBottom>
                  <strong>Confidence:</strong> {(log.confidence * 100).toFixed(1)}%
                </Typography>
                {log.data && (
                  <Box mt={1}>
                    <Typography variant="caption" display="block" gutterBottom>
                      <strong>Data:</strong>
                    </Typography>
                    <pre style={{ fontSize: '11px', overflow: 'auto', maxHeight: '100px' }}>
                      {JSON.stringify(log.data, null, 2)}
                    </pre>
                  </Box>
                )}
              </Box>
            </AccordionDetails>
          </Accordion>
        ))}
      </List>
    </Box>
  );
};

export default AgentLogsPanel;"""

def get_macro_events_component():
    return """import React from 'react';
import {
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Typography,
  Chip,
  Box
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  TrendingFlat,
  Announcement,
  AttachMoney,
  Public
} from '@mui/icons-material';
import moment from 'moment';

const MacroEventFeed = ({ events }) => {
  const getEventIcon = (eventType) => {
    switch (eventType) {
      case 'NEWS': return <Announcement />;
      case 'ECONOMIC': return <AttachMoney />;
      case 'EARNINGS': return <TrendingUp />;
      default: return <Public />;
    }
  };

  const getBiasIcon = (bias) => {
    switch (bias) {
      case 'BULLISH': return <TrendingUp color="success" />;
      case 'BEARISH': return <TrendingDown color="error" />;
      case 'NEUTRAL': return <TrendingFlat color="action" />;
      default: return <TrendingFlat />;
    }
  };

  const getBiasColor = (bias) => {
    switch (bias) {
      case 'BULLISH': return 'success';
      case 'BEARISH': return 'error';
      case 'NEUTRAL': return 'default';
      default: return 'default';
    }
  };

  if (!events.length) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="100%">
        <Typography variant="body2" color="textSecondary">
          No macro events available
        </Typography>
      </Box>
    );
  }

  return (
    <List sx={{ padding: 0 }}>
      {events.map((event, index) => (
        <ListItem key={index} divider sx={{ py: 1 }}>
          <ListItemIcon>
            {getEventIcon(event.event_type)}
          </ListItemIcon>
          <ListItemText
            primary={
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Typography variant="subtitle2">
                  {event.event_name}
                </Typography>
                <Box display="flex" alignItems="center" gap={1}>
                  {getBiasIcon(event.forecast_bias)}
                  <Chip
                    label={event.forecast_bias}
                    color={getBiasColor(event.forecast_bias)}
                    size="small"
                  />
                </Box>
              </Box>
            }
            secondary={
              <Box>
                <Typography variant="body2" color="textSecondary">
                  Impact Score: {event.impact_score}/10 • Source: {event.source}
                </Typography>
                <Typography variant="caption" color="textSecondary">
                  {moment(event.event_time).format('MMM DD, HH:mm')}
                </Typography>
              </Box>
            }
          />
        </ListItem>
      ))}
    </List>
  );
};

export default MacroEventFeed;"""

def get_trade_book_component():
    return """import React, { useState } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Typography,
  Box,
  Tab,
  Tabs,
  Tooltip
} from '@mui/material';
import moment from 'moment';

const TradeBook = ({ trades, decisions }) => {
  const [activeTab, setActiveTab] = useState(0);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const getSignalColor = (signal) => {
    switch (signal) {
      case 'BUY': return 'success';
      case 'SELL': return 'error';
      case 'HOLD': return 'warning';
      default: return 'default';
    }
  };

  const getPnLColor = (pnl) => {
    if (pnl > 0) return 'success';
    if (pnl < 0) return 'error';
    return 'default';
  };

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(value);
  };

  const DecisionsTable = () => (
    <TableContainer component={Paper} sx={{ maxHeight: 300 }}>
      <Table stickyHeader>
        <TableHead>
          <TableRow>
            <TableCell>Time</TableCell>
            <TableCell>Symbol</TableCell>
            <TableCell>Signal</TableCell>
            <TableCell>Confidence</TableCell>
            <TableCell>Agents</TableCell>
            <TableCell>Reasoning</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {decisions.length === 0 ? (
            <TableRow>
              <TableCell colSpan={6} align="center">
                <Typography variant="body2" color="textSecondary">
                  No trading decisions yet
                </Typography>
              </TableCell>
            </TableRow>
          ) : (
            decisions.map((decision, index) => (
              <TableRow key={index}>
                <TableCell>
                  <Typography variant="caption">
                    {moment(decision.timestamp).format('HH:mm:ss')}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Typography variant="body2" fontWeight="bold">
                    {decision.symbol}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Chip
                    label={decision.signal}
                    color={getSignalColor(decision.signal)}
                    size="small"
                  />
                </TableCell>
                <TableCell>
                  <Typography variant="body2">
                    {(decision.confidence * 100).toFixed(1)}%
                  </Typography>
                </TableCell>
                <TableCell>
                  <Typography variant="caption">
                    {decision.contributing_agents?.length || 0} agents
                  </Typography>
                </TableCell>
                <TableCell>
                  <Tooltip title={decision.reasoning}>
                    <Typography variant="body2" noWrap sx={{ maxWidth: 200 }}>
                      {decision.reasoning}
                    </Typography>
                  </Tooltip>
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </TableContainer>
  );

  const TradesTable = () => (
    <TableContainer component={Paper} sx={{ maxHeight: 300 }}>
      <Table stickyHeader>
        <TableHead>
          <TableRow>
            <TableCell>Entry Time</TableCell>
            <TableCell>Exit Time</TableCell>
            <TableCell>Symbol</TableCell>
            <TableCell>Entry Price</TableCell>
            <TableCell>Exit Price</TableCell>
            <TableCell>P&L</TableCell>
            <TableCell>P&L %</TableCell>
            <TableCell>Status</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {trades.length === 0 ? (
            <TableRow>
              <TableCell colSpan={8} align="center">
                <Typography variant="body2" color="textSecondary">
                  No completed trades yet
                </Typography>
              </TableCell>
            </TableRow>
          ) : (
            trades.map((trade, index) => (
              <TableRow key={index}>
                <TableCell>
                  <Typography variant="caption">
                    {moment(trade.entry_time).format('MM/DD HH:mm')}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Typography variant="caption">
                    {trade.exit_time ? moment(trade.exit_time).format('MM/DD HH:mm') : '-'}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Typography variant="body2" fontWeight="bold">
                    {trade.symbol}
                  </Typography>
                </TableCell>
                <TableCell>
                  {formatCurrency(trade.entry_price)}
                </TableCell>
                <TableCell>
                  {trade.exit_price ? formatCurrency(trade.exit_price) : '-'}
                </TableCell>
                <TableCell>
                  <Typography color={getPnLColor(trade.pnl)}>
                    {trade.pnl ? formatCurrency(trade.pnl) : '-'}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Typography color={getPnLColor(trade.pnl_percentage)}>
                    {trade.pnl_percentage ? `${trade.pnl_percentage.toFixed(2)}%` : '-'}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Chip
                    label={trade.success_flag ? 'Profit' : 'Loss'}
                    color={trade.success_flag ? 'success' : 'error'}
                    size="small"
                  />
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </TableContainer>
  );

  return (
    <Box>
      <Tabs value={activeTab} onChange={handleTabChange} sx={{ mb: 2 }}>
        <Tab label={`Decisions (${decisions.length})`} />
        <Tab label={`Completed Trades (${trades.length})`} />
      </Tabs>
      
      {activeTab === 0 && <DecisionsTable />}
      {activeTab === 1 && <TradesTable />}
    </Box>
  );
};

export default TradeBook;"""

def get_websocket_service():
    return """export class SignalWebSocket {
  constructor(url) {
    this.ws = new WebSocket(url);
    this.reconnectInterval = 5000;
    this.maxReconnectAttempts = 5;
    this.reconnectAttempts = 0;
    
    this.setupEventListeners();
  }

  setupEventListeners() {
    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
    };

    this.ws.onclose = (event) => {
      console.log('WebSocket disconnected');
      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        setTimeout(() => {
          this.reconnect();
        }, this.reconnectInterval);
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  reconnect() {
    console.log(`Attempting to reconnect... (${this.reconnectAttempts + 1}/${this.maxReconnectAttempts})`);
    this.reconnectAttempts++;
    this.ws = new WebSocket(this.ws.url);
    this.setupEventListeners();
  }

  onMessage(callback) {
    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        callback(data);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };
  }

  send(data) {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.warn('WebSocket is not open. Ready state:', this.ws.readyState);
    }
  }

  close() {
    this.ws.close();
  }
}
"""

def get_api_service():
    return """import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

class ApiService {
  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('API Request Error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => {
        console.log(`API Response: ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        console.error('API Response Error:', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );
  }

  // Health check
  async checkHealth() {
    const response = await this.client.get('/health');
    return response.data;
  }

  // Get agents
  async getAgents() {
    const response = await this.client.get('/agents');
    return response.data;
  }

  // Get recent signals
  async getRecentSignals(limit = 50) {
    const response = await this.client.get(`/signals?limit=${limit}`);
    return response.data;
  }

  // Create manual signal
  async createManualSignal(signalData) {
    const response = await this.client.post('/manual_signal', signalData);
    return response.data;
  }
}

export default new ApiService();"""

def get_app_css():
    return """.App {
  text-align: left;
  background-color: #f5f5f5;
  min-height: 100vh;
}

.App-header {
  background-color: #1a1a1a;
  padding: 20px;
  color: white;
}

/* Custom scrollbar */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}

/* Material-UI overrides */
.MuiPaper-root {
  border-radius: 8px !important;
}

.MuiChip-root {
  font-weight: 600 !important;
}

/* Animation classes */
.fade-in {
  animation: fadeIn 0.3s ease-in;
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

.slide-in {
  animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
  from { transform: translateY(-10px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}"""

def get_index_js():
    return """import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: '"Roboto","Helvetica","Arial",sans-serif',
    h6: {
      fontWeight: 600,
    },
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 8,
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 600,
        },
      },
    },
  },
});

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <App />
    </ThemeProvider>
  </React.StrictMode>
);"""

def get_index_css():
    return """body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  background-color: #f5f5f5;
}

code {
  font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',
    monospace;
}

* {
  box-sizing: border-box;
}

html, body, #root {
  height: 100%;
  margin: 0;
  padding: 0;
}"""

def get_index_html():
    return """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <link rel="icon" href="%PUBLIC_URL%/favicon.ico" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta
      name="description"
      content="Agentic AI Trading System - Multi-agent trading platform"
    />
    <title>🤖 Agentic AI Trading System</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>"""

def create_deployment_files():
    """Create Docker and deployment configuration files."""
    
    # Docker Compose
    docker_compose = """version: '3.8'

services:
  # Redis for event bus
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - agentic_network

  # PostgreSQL database
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: agentic_trading
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - agentic_network

  # Orchestrator service
  orchestrator:
    build:
      context: ./backend
      dockerfile: Dockerfile.orchestrator
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/agentic_trading
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - postgres
    networks:
      - agentic_network
    volumes:
      - ./backend:/app
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:

networks:
  agentic_network:
    driver: bridge"""
    create_file("docker-compose.yml", docker_compose)
    
    # Orchestrator Dockerfile
    orchestrator_dockerfile = """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["python", "-m", "uvicorn", "orchestrator.api:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    create_file("backend/Dockerfile.orchestrator", orchestrator_dockerfile)
    
    # Environment file template
    env_template = """# Database Configuration
DATABASE_URL=postgresql://postgres:password@postgres:5432/agentic_trading

# Redis Configuration
REDIS_URL=redis://redis:6379

# API Keys - UPDATE THESE!
TRAVILY_API_KEY=your_travily_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Security
SECRET_KEY=your-super-secret-key-change-this-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Agent Configuration
CHARTANALYST_PORT=8001
RISKMANAGER_PORT=8002
MARKETSENTINEL_PORT=8003
MACROFORECASTER_PORT=8004
TACTICBOT_PORT=8005
PLATFORMPILOT_PORT=8006

# Orchestrator
ORCHESTRATOR_PORT=8000

# Frontend
REACT_APP_API_URL=http://localhost:8000
"""
    create_file(".env.example", env_template)
    
    # Makefile for easy commands
    makefile_content = """# Agentic Trading System Makefile

.PHONY: help install start stop restart logs clean test

help:
	@echo "Available commands:"
	@echo "  install    - Install dependencies"
	@echo "  start      - Start all services"
	@echo "  stop       - Stop all services"
	@echo "  restart    - Restart all services"
	@echo "  logs       - View logs"
	@echo "  clean      - Clean up containers and volumes"
	@echo "  test       - Run tests"

install:
	@echo "Installing dependencies..."
	cd backend && pip install -r requirements.txt
	cd frontend && npm install

start:
	@echo "Starting Agentic Trading System..."
	docker-compose up -d

stop:
	@echo "Stopping all services..."
	docker-compose down

restart:
	@echo "Restarting all services..."
	docker-compose restart

logs:
	@echo "Viewing logs..."
	docker-compose logs -f

clean:
	@echo "Cleaning up..."
	docker-compose down -v
	docker system prune -f

test:
	@echo "Running tests..."
	cd backend && python -m pytest tests/ || echo "No tests found"

dev-backend:
	@echo "Starting backend in development mode..."
	cd backend && python -m uvicorn orchestrator.api:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	@echo "Starting frontend in development mode..."
	cd frontend && npm start

setup-db:
	@echo "Setting up database..."
	docker-compose up -d postgres redis
	sleep 5
	cd backend && python -c "from db.db_session import init_db; init_db()"

health-check:
	@echo "Checking service health..."
	curl -f http://localhost:8000/health || echo "Orchestrator: DOWN"
"""
    create_file("Makefile", makefile_content)

if __name__ == "__main__":
    create_project_structure()
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Security
    secret_key: str = "your-secret-key-change-this"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Agent ports
    chartanalyst_port: int = 8001
    riskmanager_port: int = 8002
    marketsentinel_port: int = 8003
    macroforecaster_port: int = 8004
    tacticbot_port: int = 8005
    platformpilot_port: int = 8006
    
    # Orchestrator
    orchestrator_port: int = 8000
    
    class Config:
        env_file = ".env"

settings = Settings()
"""
    create_file("backend/config.py", config_content)
    
    # Database models
    models_content = """from sqlalchemy import Column, Integer, String, DateTime, Float, JSON, ForeignKey, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

Base = declarative_base()

class Agent(Base):
    __tablename__ = "agents"
    
    agent_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    model = Column(String)
    status = Column(String, default="active")
    endpoint = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    signals = relationship("TradeSignal", back_populates="agent")

class TradeSignal(Base):
    __tablename__ = "trade_signals"
    
    signal_id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    symbol = Column(String, index=True)
    timeframe = Column(String)
    agent_id = Column(Integer, ForeignKey("agents.agent_id"))
    signal_type = Column(String)  # BUY, SELL, HOLD
    confidence = Column(Float)
    signal_data = Column(JSON)
    macro_context = Column(JSON)
    processed = Column(Boolean, default=False)
    
    # Relationships
    agent = relationship("Agent", back_populates="signals")
    outcome = relationship("TradeOutcome", back_populates="signal", uselist=False)

class TradeOutcome(Base):
    __tablename__ = "trade_outcomes"
    
    outcome_id = Column(Integer, primary_key=True, index=True)
    signal_id = Column(Integer, ForeignKey("trade_signals.signal_id"), unique=True)
    entry_price = Column(Float)
    exit_price = Column(Float)
    entry_time = Column(DateTime(timezone=True))
    exit_time = Column(DateTime(timezone=True))
    pnl = Column(Float)
    pnl_percentage = Column(Float)
    success_flag = Column(Boolean)
    notes = Column(Text)
    
    # Relationships  
    signal = relationship("TradeSignal", back_populates="outcome")

class MacroEvent(Base):
    __tablename__ = "macro_events"
    
    event_id = Column(Integer, primary_key=True, index=True)
    signal_id = Column(Integer, ForeignKey("trade_signals.signal_id"))
    event_name = Column(String)
    event_type = Column(String)  # NEWS, ECONOMIC, EARNINGS, etc.
    impact_score = Column(Float)
    forecast_bias = Column(String)  # BULLISH, BEARISH, NEUTRAL
    source = Column(String)
    event_time = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class AgentPerformance(Base):
    __tablename__ = "agent_performance"
    
    performance_id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.agent_id"))
    date = Column(DateTime(timezone=True), server_default=func.now())
    total_signals = Column(Integer, default=0)
    successful_signals = Column(Integer, default=0)
    accuracy_rate = Column(Float, default=0.0)
    avg_confidence = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
"""
    create_file("backend/db/models.py", models_content)
    
    # Database session
    db_session_content = """from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from config import settings
import logging

logger = logging.getLogger(__name__)

# Create engine
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=300,
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def init_db():
    \"\"\"Initialize database tables\"\"\"
    from db.models import Base
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
"""
    create_file("backend/db/db_session.py", db_session_content)
    
    create_file("backend/db/__init__.py", "")
    
    # Create all agents
    create_agent_files()
    
    # Create orchestrator
    create_orchestrator_files()

def create_agent_files():
    """Create all agent microservices."""
    
    agents = [
        ("chartanalyst", 8001, "Technical pattern recognition and chart analysis"),
        ("riskmanager", 8002, "Risk assessment and position sizing"),
        ("marketsentinel", 8003, "Market volatility and scalping opportunities"),
        ("macroforecaster", 8004, "Macro economic events and news analysis"), 
        ("tacticbot", 8005, "Trade execution timing and tactics"),
        ("platformpilot", 8006, "Platform automation and trade logging")
    ]
    
    for agent_name, port, description in agents:
        create_agent_service(agent_name, port, description)

def create_agent_service(agent_name: str, port: int, description: str):
    """Create an individual agent service."""
    
    # Main agent service
    main_content = f"""from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
import asyncio
import httpx
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="{agent_name.title()} Agent",
    description="{description}",
    version="1.0.0"
)

class AgentInput(BaseModel):
    symbol: str
    timeframe: str
    data: Dict
    context: Optional[Dict] = None

class AgentOutput(BaseModel):
    agent_name: str = "{agent_name}"
    timestamp: datetime
    symbol: str
    confidence: float
    signal_type: Optional[str] = None
    reasoning: str
    data: Dict
    metadata: Optional[Dict] = None

# Agent-specific logic based on type
{get_agent_logic(agent_name)}

@app.get("/health")
async def health_check():
    return {{"status": "healthy", "agent": "{agent_name}", "timestamp": datetime.now()}}

@app.post("/analyze", response_model=AgentOutput)
async def analyze(input_data: AgentInput):
    try:
        logger.info(f"Received analysis request for {{input_data.symbol}}")
        
        # Process the input data
        result = await process_signal(input_data)
        
        # Publish result to event bus (if available)
        await publish_result(result)
        
        return result
        
    except Exception as e:
        logger.error(f"Analysis failed: {{e}}")
        raise HTTPException(status_code=500, detail=str(e))

async def publish_result(result: AgentOutput):
    \"\"\"Publish result to Redis event bus\"\"\"
    try:
        # This would connect to Redis in production
        logger.info(f"Publishing result for {{result.symbol}}")
    except Exception as e:
        logger.warning(f"Failed to publish result: {{e}}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port={port})
"""
    
    create_file(f"backend/agents/{agent_name}/main.py", main_content)
    
    # Agent utils
    utils_content = f"""import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class {agent_name.title()}Utils:
    \"\"\"Utility functions for {agent_name} agent\"\"\"
    
    @staticmethod
    def validate_input(data: Dict) -> bool:
        \"\"\"Validate input data format\"\"\"
        required_fields = ["symbol", "timeframe"]
        return all(field in data for field in required_fields)
    
    @staticmethod
    def calculate_confidence(factors: Dict) -> float:
        \"\"\"Calculate confidence score based on multiple factors\"\"\"
        if not factors:
            return 0.0
        
        # Simple weighted average - customize per agent
        weights = {{
            "strength": 0.4,
            "volume": 0.3, 
            "trend": 0.2,
            "volatility": 0.1
        }}
        
        score = 0.0
        total_weight = 0.0
        
        for factor, value in factors.items():
            if factor in weights:
                score += weights[factor] * value
                total_weight += weights[factor]
        
        return min(max(score / total_weight if total_weight > 0 else 0.0, 0.0), 1.0)
    
    @staticmethod
    def format_output(analysis: Dict) -> Dict:
        \"\"\"Format analysis output for consistency\"\"\"
        return {{
            "analysis": analysis,
            "timestamp": pd.Timestamp.now().isoformat(),
            "version": "1.0"
        }}

# Agent-specific utility functions
{get_agent_utils(agent_name)}
"""
    
    create_file(f"backend/agents/{agent_name}/utils.py", utils_content)
    
    # Agent model (AI integration placeholder)
    model_content = f"""from abc import ABC, abstractmethod
from typing import Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    \"\"\"Base class for AI model integration\"\"\"
    
    def __init__(self, model_name: str = "{agent_name}_model"):
        self.model_name = model_name
        self.initialized = False
    
    @abstractmethod
    async def predict(self, input_data: Dict) -> Dict:
        \"\"\"Make prediction using the AI model\"\"\"
        pass
    
    @abstractmethod
    def load_model(self):
        \"\"\"Load the AI model\"\"\"
        pass

class {agent_name.title()}Model(BaseModel):
    \"\"\"AI model for {agent_name} agent\"\"\"
    
    def __init__(self):
        super().__init__("{agent_name}_model")
        self.load_model()
    
    def load_model(self):
        \"\"\"Load the specific AI model for this agent\"\"\"
        # TODO: Integrate with actual AI model (Mistral, Kimi, etc.)
        logger.info(f"Loading {{self.model_name}} model...")
        self.initialized = True
        logger.info(f"{{self.model_name}} model loaded successfully")
    
    async def predict(self, input_data: Dict) -> Dict:
        \"\"\"Make prediction using the loaded model\"\"\"
        if not self.initialized:
            raise RuntimeError("Model not initialized")
        
        # TODO: Replace with actual model inference
        # This is a placeholder implementation
        
        {get_model_prediction(agent_name)}
        
        return prediction

# Global model instance
model = {agent_name.title()}Model()
"""
    
    create_file(f"backend/agents/{agent_name}/model.py", model_content)
    create_file(f"backend/agents/{agent_name}/__init__.py", "")

def get_agent_logic(agent_name: str) -> str:
    """Generate agent-specific logic based on agent type."""
    
    logic_map = {
        "chartanalyst": """
async def process_signal(input_data: AgentInput) -> AgentOutput:
    \"\"\"Process chart analysis signal\"\"\"
    from .model import model
    from .utils import ChartanalystUtils
    
    # Extract candle data
    candles = input_data.data.get("candles", [])
    
    if not candles:
        raise ValueError("No candle data provided")
    
    # Run AI model analysis
    analysis = await model.predict({
        "candles": candles,
        "symbol": input_data.symbol,
        "timeframe": input_data.timeframe
    })
    
    return AgentOutput(
        timestamp=datetime.now(),
        symbol=input_data.symbol,
        confidence=analysis.get("confidence", 0.0),
        signal_type=analysis.get("signal_type"),
        reasoning=analysis.get("reasoning", "Chart pattern analysis"),
        data=analysis,
        metadata={"agent_type": "technical_analysis"}
    )
""",
        "riskmanager": """
async def process_signal(input_data: AgentInput) -> AgentOutput:
    \"\"\"Process risk management analysis\"\"\"
    from .model import model
    from .utils import RiskmanagerUtils
    
    # Get portfolio info and signal details
    portfolio = input_data.data.get("portfolio", {})
    signal_data = input_data.data.get("signal", {})
    
    # Run risk analysis
    analysis = await model.predict({
        "portfolio": portfolio,
        "signal": signal_data,
        "symbol": input_data.symbol
    })
    
    return AgentOutput(
        timestamp=datetime.now(),
        symbol=input_data.symbol,
        confidence=analysis.get("confidence", 0.0),
        reasoning=analysis.get("reasoning", "Risk assessment analysis"),
        data=analysis,
        metadata={"agent_type": "risk_management"}
    )
""",
        "marketsentinel": """
async def process_signal(input_data: AgentInput) -> AgentOutput:
    \"\"\"Process market volatility analysis\"\"\"
    from .model import model
    from .utils import MarketsentinelUtils
    
    # Get market data
    market_data = input_data.data.get("market_data", {})
    volatility_data = input_data.data.get("volatility", {})
    
    # Run volatility analysis
    analysis = await model.predict({
        "market_data": market_data,
        "volatility": volatility_data,
        "symbol": input_data.symbol,
        "timeframe": input_data.timeframe
    })
    
    return AgentOutput(
        timestamp=datetime.now(),
        symbol=input_data.symbol,
        confidence=analysis.get("confidence", 0.0),
        reasoning=analysis.get("reasoning", "Market volatility analysis"),
        data=analysis,
        metadata={"agent_type": "volatility_analysis"}
    )
""",
        "macroforecaster": """
async def process_signal(input_data: AgentInput) -> AgentOutput:
    \"\"\"Process macro economic analysis\"\"\"
    from .model import model
    from .utils import MacroforecasterUtils
    
    # Get news and economic data
    news_data = input_data.data.get("news", [])
    economic_data = input_data.data.get("economic_events", [])
    
    # Run macro analysis
    analysis = await model.predict({
        "news": news_data,
        "economic_events": economic_data,
        "symbol": input_data.symbol,
        "context": input_data.context
    })
    
    return AgentOutput(
        timestamp=datetime.now(),
        symbol=input_data.symbol,
        confidence=analysis.get("confidence", 0.0),
        reasoning=analysis.get("reasoning", "Macro economic analysis"),
        data=analysis,
        metadata={"agent_type": "macro_analysis"}
    )
""",
        "tacticbot": """
async def process_signal(input_data: AgentInput) -> AgentOutput:
    \"\"\"Process tactical trading analysis\"\"\"
    from .model import model
    from .utils import TacticbotUtils
    
    # Get aggregated signals from other agents
    agent_signals = input_data.data.get("agent_signals", [])
    market_conditions = input_data.data.get("market_conditions", {})
    
    # Run tactical analysis
    analysis = await model.predict({
        "agent_signals": agent_signals,
        "market_conditions