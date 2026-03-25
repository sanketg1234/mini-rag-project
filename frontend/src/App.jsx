import { useState } from 'react';
import axios from 'axios';

function App() {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userMessage = { role: 'user', content: query };
    setMessages((prev) => [...prev, userMessage]);
    setQuery('');
    setIsLoading(true);

    try {
      // Call your running Python FastAPI backend
      const response = await axios.post("https://mini-rag-project-fwkn.onrender.com/ask", {
        query: userMessage.content,
      });

      const aiMessage = {
        role: 'ai',
        content: response.data.answer,
        context: response.data.context, // The retrieved document chunks
      };

      setMessages((prev) => [...prev, aiMessage]);
    } catch (error) {
      console.error("Error fetching response:", error);
      setMessages((prev) => [...prev, { role: 'ai', content: 'Sorry, there was an error connecting to the server.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', fontFamily: 'Arial, sans-serif', padding: '20px' }}>
      <h1 style={{ textAlign: 'center', color: '#333' }}>Construction Assistant AI</h1>
      
      <div style={{ height: '60vh', overflowY: 'auto', border: '1px solid #ccc', padding: '20px', borderRadius: '8px', marginBottom: '20px', backgroundColor: '#f9f9f9' }}>
        {messages.length === 0 ? (
          <p style={{ textAlign: 'center', color: '#888' }}>Ask a question about the construction documents!</p>
        ) : (
          messages.map((msg, index) => (
            <div key={index} style={{ marginBottom: '20px', textAlign: msg.role === 'user' ? 'right' : 'left' }}>
              
              {/* Message Bubble */}
              <div style={{ display: 'inline-block', padding: '10px 15px', borderRadius: '15px', backgroundColor: msg.role === 'user' ? '#007bff' : '#e0e0e0', color: msg.role === 'user' ? '#fff' : '#000', maxWidth: '80%' }}>
                {msg.content}
              </div>

              {/* Context Display (Mandatory Transparency Requirement) */}
              {msg.context && (
                <div style={{ marginTop: '10px', fontSize: '0.85em', color: '#555', backgroundColor: '#fff', padding: '10px', border: '1px solid #ddd', borderRadius: '5px', textAlign: 'left' }}>
                  <strong>Retrieved Context:</strong>
                  {msg.context.map((ctx, i) => (
                    <div key={i} style={{ marginTop: '5px', borderBottom: '1px dashed #ccc', paddingBottom: '5px' }}>
                      <em>Source: {ctx.source}</em><br />
                      {ctx.content}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))
        )}
        {isLoading && <p style={{ color: '#888' }}>AI is thinking...</p>}
      </div>

      <form onSubmit={sendMessage} style={{ display: 'flex', gap: '10px' }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g., What factors affect construction delays?"
          style={{ flex: 1, padding: '10px', borderRadius: '5px', border: '1px solid #ccc', fontSize: '16px' }}
        />
        <button type="submit" disabled={isLoading} style={{ padding: '10px 20px', borderRadius: '5px', border: 'none', backgroundColor: '#28a745', color: '#fff', fontSize: '16px', cursor: isLoading ? 'not-allowed' : 'pointer' }}>
          Send
        </button>
      </form>
    </div>
  );
}

export default App;