import { useState } from "react";
import "./App.css";

function App() {
  const [message, setMessage] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!message) return;

    setLoading(true);
    setResponse("");

    try {
      const res = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          message: message
        })
      });

      const data = await res.json();
      setResponse(data.response);
    } catch (err) {
      setResponse("Error connecting to backend.");
    }

    setLoading(false);
  };

  return (
    <div style={{ maxWidth: "700px", margin: "50px auto", fontFamily: "Arial" }}>
      <h1>RAG AI Assistant</h1>

      <textarea
        rows="4"
        style={{ width: "100%", padding: "10px" }}
        placeholder="Ask something..."
        value={message}
        onChange={(e) => setMessage(e.target.value)}
      />

      <button onClick={sendMessage} style={{ marginTop: "10px" }}>
        {loading ? "Thinking..." : "Send"}
      </button>

      <div style={{ marginTop: "30px" }}>
        <h3>Response:</h3>
        <p>{response}</p>
      </div>
    </div>
  );
}

export default App;