# CrewAI SQL Agent

A powerful SQL query generator that uses CrewAI framework with locally hosted Ollama models to translate natural language prompts into executable PostgreSQL queries.

## 🚀 Features

- **Natural Language to SQL**: Convert plain English queries into PostgreSQL SQL statements
- **CrewAI Integration**: Multi-agent system for intelligent query analysis and generation
- **Local Model Support**: Uses Ollama with tinyllama:1.1b-chat model
- **Database Integration**: Dynamic PostgreSQL connection with schema inspection
- **Modern UI**: Beautiful Streamlit interface with real-time feedback
- **Query History**: Track and review previous queries
- **Export Results**: Download query results as CSV files

## 🏗️ Architecture

The system consists of four main components:

1. **Database Manager** (`src/database_manager.py`): Handles PostgreSQL connections and metadata extraction
2. **Ollama LLM** (`src/ollama_llm.py`): Integrates with local Ollama service
3. **SQL Agent** (`src/sql_agent.py`): Main CrewAI agent orchestrator
4. **Streamlit UI** (`app.py`): User interface for interaction

### CrewAI Agents

- **SQL Analyst**: Analyzes natural language queries and identifies requirements
- **Database Expert**: Provides database schema context and insights
- **SQL Developer**: Generates accurate PostgreSQL queries
- **Query Validator**: Validates and optimizes generated SQL

## 📋 Prerequisites

- Python 3.8 or higher
- PostgreSQL database
- Ollama (with tinyllama:1.1b-chat model)
- Git

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd CrewAI-SQLAgent-FineTuned
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Run Setup Script

```bash
python setup.py
```

### 4. Install Ollama

1. Download and install Ollama from [https://ollama.ai/](https://ollama.ai/)
2. Start Ollama service
3. Pull the required model:

```bash
ollama pull tinyllama:1.1b-chat
```

### 5. Configure Database

1. Set up a PostgreSQL database
2. Update `config/database_config.json` with your database credentials:

```json
{
  "database": {
    "host": "localhost",
    "port": 5432,
    "database": "your_database_name",
    "username": "your_username",
    "password": "your_password",
    "schema": "public"
  },
  "ollama": {
    "base_url": "http://localhost:11434",
    "model": "tinyllama:1.1b-chat"
  }
}
```

## 🚀 Usage

### Start the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Interface

1. **Connect to Database**: Click "Connect to Database" in the sidebar
2. **Test Ollama**: Click "Test Ollama Connection" to verify model availability
3. **Enter Query**: Type your natural language query in the text area
4. **Generate SQL**: Click "Generate SQL" to create the PostgreSQL query
5. **Execute Query**: Click "Execute SQL" to run the query and see results
6. **Download Results**: Use the download button to save results as CSV

### Example Queries

- "Show me all users who registered in the last 30 days"
- "Count total orders by status"
- "Find top 10 customers by total spent"
- "Calculate average order value by month"
- "List products that have never been ordered"

## 📁 Project Structure

```
CrewAI-SQLAgent-FineTuned/
├── app.py                      # Main Streamlit application
├── setup.py                    # Setup script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── config/
│   └── database_config.json    # Database and Ollama configuration
├── data/
│   └── sample_queries.json     # Example queries for model training
├── src/
│   ├── database_manager.py     # PostgreSQL connection and metadata
│   ├── ollama_llm.py          # Ollama integration
│   └── sql_agent.py           # Main CrewAI agent system
└── logs/                       # Application logs (created automatically)
```

## 🔧 Configuration

### Database Configuration

Edit `config/database_config.json` to match your PostgreSQL setup:

```json
{
  "database": {
    "host": "localhost",
    "port": 5432,
    "database": "your_database",
    "username": "your_username",
    "password": "your_password",
    "schema": "public"
  }
}
```

### Ollama Configuration

The same file contains Ollama settings:

```json
{
  "ollama": {
    "base_url": "http://localhost:11434",
    "model": "tinyllama:1.1b-chat"
  }
}
```

### Model Parameters

You can adjust model parameters in the Streamlit interface:
- **Temperature**: Controls randomness (0.1-0.9)
- **Max Tokens**: Maximum response length (1024-8192)

## 📊 Sample Data

The `data/sample_queries.json` file contains example queries for model conditioning:

```json
{
  "queries": [
    {
      "natural_language": "Show me all users",
      "sql": "SELECT * FROM users;",
      "description": "Basic select all users query"
    }
  ]
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Verify PostgreSQL is running
   - Check credentials in `config/database_config.json`
   - Ensure database exists and is accessible

2. **Ollama Connection Failed**
   - Verify Ollama service is running: `ollama serve`
   - Check if model is downloaded: `ollama list`
   - Pull model if missing: `ollama pull tinyllama:1.1b-chat`

3. **SQL Generation Errors**
   - Check database schema exists
   - Verify table names in natural language query
   - Review debug information in the UI

4. **Import Errors**
   - Ensure virtual environment is activated
   - Run `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

### Logs

Check the `logs/` directory for detailed error information.

## 🔒 Security Considerations

- Store database credentials securely
- Use environment variables for production deployments
- Implement proper access controls for database
- Consider query execution limits and timeouts

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [CrewAI](https://github.com/joaomdmoura/crewAI) for the multi-agent framework
- [Ollama](https://ollama.ai/) for local model hosting
- [Streamlit](https://streamlit.io/) for the web interface
- [PostgreSQL](https://www.postgresql.org/) for the database system

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the debug information in the UI
3. Create an issue on the repository
4. Check the logs in the `logs/` directory 