# 🧠 Agentic AI System (Multi-Model, Role-Based)

A modular agentic AI framework that routes tasks to the best-suited LLMs using OpenRouter. Designed for flexible autonomy, powerful tool use, and role-based delegation — with Gradio UI and Docker support.

---

## 🚀 Features

- 🔀 **Multi-agent roles**: Planner, Executor, Critic
- 🤖 **Model routing** (via OpenRouter):
  - Kimi K2 — planning
  - Mistral 3.2 — tool use
  - TNG Chimera — evaluation
- 🌐 **Gradio UI** to interact with agents
- 🐳 **Docker-ready** deployment
- 🧩 Easily extendable (add more agents/tools)

---

## 📦 Setup Instructions

### 1. Clone and Install

```bash
pip install -r requirements.txt
```

Create a `.env` file in the root:

```
OPENROUTER_API_KEY=your-openrouter-key
```

### 2. Run CLI version

```bash
python run.py
```

### 3. Run Gradio UI

```bash
python gradio_ui.py
```

Then open `http://localhost:7860`.

---

## 🐳 Docker Usage

```bash
docker build -t agentic-app .
docker run -p 7860:7860 agentic-app
```

---

## 🧠 Agent Roles & Models

| Agent Role        | Model                   |
|-------------------|-------------------------|
| Planner           | Kimi K2 (Free)          |
| Executor          | Mistral Small 3.2       |
| Critic/Analyzer   | TNG Chimera             |

Modify the models easily in `agents/*.py` files.

---

## 📁 Project Structure

```
agentic_custom_project/
├── agents/
├── services/
├── run.py
├── gradio_ui.py
├── Dockerfile
├── .env
├── requirements.txt
└── README.md
```

---

## 🛠️ Extend Ideas

- Add new agents (e.g., Coder, Fixer, Researcher)
- Integrate tools (web search, bash, APIs)
- Add Langfuse or logging backend
- Connect to Slack/Email/Webhooks

---

## License

MIT — open and free to modify.
