# Real Estate Agency Chatbot

This repository contains a Natural Language Processing (NLP)-based chatbot developed as part of a university coursework project. The chatbot simulates an intelligent estate agency assistant that helps users search for properties, arrange viewings, and contact agents via a conversational interface.

## Project Overview

The chatbot allows users to:

- Search for properties using natural language queries (e.g., location, price, number of bedrooms)
- Request detailed information about specific property listings
- Schedule property viewings
- Obtain contact details for estate agents
- Ask general questions and receive assistance on how to use the system

It employs a rule-based intent recognition system underpinned by TF-IDF vectorisation and cosine similarity scoring.

## Features

- Intent recognition using cosine similarity over TF-IDF vectors
- Context tracking for multi-step conversations
- Property and agent data management using pandas and CSV files
- User-friendly command-line interface (CLI)
- Modular design to support easy addition of new functionality

## Technologies and Libraries

- Python 3.x
- `nltk` – Text preprocessing and stemming
- `pandas` – Data manipulation and dataset handling
- `scikit-learn` – Vectorisation and similarity measurement
- `joblib` – Model and data persistence
- `re` – Regular expressions for information extraction
- `tabulate` – Formatting outputs for the CLI

## Project Structure

```

├── chatbot.py             # Main chatbot application
├── chatbot_models.py     # Preprocessing and model training script
├── intents/               # CSV files containing intents and responses
├── models/                # Saved models and vectorisers
├── data/
│   ├── property.csv       # Property listings data
│   └── agent.csv          # Estate agent information
├── README.md              # Project documentation

```

## How It Works

1. Loads and vectorises a set of pre-written intents and responses.
2. Accepts a user input and transforms it into a TF-IDF vector.
3. Identifies the closest matching intent using cosine similarity.
4. Executes the corresponding function (e.g., property search, viewing booking, FAQ).
5. Maintains simple context to support follow-up questions and smoother interactions.

## Example Queries

- "I'm looking for a flat in Leeds under £900"
- "Can you tell me more about property number 7?"
- "I'd like to book a viewing for Friday afternoon"
- "What can you help me with?"

## Design Considerations

- Designed for clarity and ease of use in conversational interactions
- Modular and extensible codebase for future enhancements
- Artificially generated datasets used for demonstration purposes
- Data preprocessing includes custom stopword removal and stemming

## Future Enhancements

- Integration of more advanced NLP models (e.g., transformer-based architectures)
- Improved memory and context handling for multi-intent conversations
- Support for more flexible search inputs (e.g., ranges and comparative queries)
- Deployment via web interface or messaging platforms

## Licence

This project is for academic purposes and is distributed under the MIT Licence.
