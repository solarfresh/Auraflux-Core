# Auraflux-Core
The intelligent Python package and project core. It orchestrates AI agents to collect, integrate, and structure information, turning scattered data into organized knowledge.

---

# Core Components

The package architecture is built on a clear separation of concerns, ensuring modularity and scalability. Key components are:

## **1. Agents**
This directory holds AI agent definitions using **Autogen**. Each agent specializes in a specific task, allowing for collaborative workflows.
* **Coordinator Agent:** The orchestrator that receives commands and delegates tasks.
* **Crawler Agent:** Interacts with web scraping services to gather raw data.
* **Knowledge Agent:** Transforms unstructured data into a structured format for the knowledge graph.

## **2. Schemas**
This directory defines data models, ensuring consistency across the system.
* **Agent Schemas:** Govern communication protocols between agents.
* **Knowledge Schemas:** Define the structure of information stored in the knowledge graph.

## **3. Skills**
`skills` contains low-level abilities for agents to interact with the outside world.
* **Database Connectors:** Modules for connecting to Neo4j and PostgreSQL.
* **API Clients:** Modules for making API calls to external services.

## **4. Workflows**
This section holds predefined scripts that combine agents and skills to accomplish specific tasks, serving as a blueprint for new workflows.