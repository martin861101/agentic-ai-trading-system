// WebSocket clientexport class SignalWebSocket {
  constructor(url) {
    this.ws = new WebSocket(url);
    this.ws.onopen = () => console.log("WebSocket connected");
    this.ws.onclose = () => console.log("WebSocket disconnected");
  }

  onMessage(callback) {
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      callback(data);
    };
  }

  send(data) {
    this.ws.send(JSON.stringify(data));
  }
}

