# Part 2 - Chat with LLM

## Overview
The Chat with LLM workshop will guide you through four essential techniques used for interacting with LLMs:
* Simple chat request
* RAG
* Structured outputs
* Tool calling

The application runs in the CLI and expects a user prompt. The user then selects one of the available techniques to interact with the LLM. The model will respond. The messages inside the conversation are stored in memory. The application will keep running until the user types "exit".

## Quick Start

### Prerequisites
The following are already installed on the Raspberry Pi:
* [Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html)
* [Llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#cpu-build)

### Deploying the models
```bash
llama-server --embeddings --hf-repo second-state/All-MiniLM-L6-v2-Embedding-GGUF --hf-file  all-MiniLM-L6-v2-ggml-model-f16.gguf --port 8081 # embeddings model available on localhost:8081
llama-server --jinja --hf-repo MaziyarPanahi/gemma-3-1b-it-GGUF --hf-file gemma-3-1b-it.Q5_K_M.gguf # llm available on localhost:8080
```

## Workshop
You will be working inside the `workshop.rs` file. The full implementation is available in the `full_demo.rs` file, in case you get stuck.
In order to run the workshop, execute:
```bash
RUST_LOG=info cargo run --bin workshop
```
### 1. Simple Chat Request
During the workshop, we will be using Gemma 3 1B as our language model. The models are deployed using llama.cpp, which exposes an OpenAI-compatible API on port 8080.

We have defined the necessary structs to interact with the model API.

A chat request consists of the model name, an array of messages and optionally tools and response format.

A message consists of the role (user, assistant, system) and the content.

Complete the TODO 1 to implement the chat interaction logic.

### 2. Retrieval-Augmented Generation (RAG)

In this section, we will implement a RAG system that combines the language model with a document retrieval system.

The embeddings model is also deployed using llama.cpp and exposes a slightly different API on port 8081.

A RAG system is implemented as follows:
1. Calculate embeddings on documents inside the knowledge base.
2. Calculate the embedding of the user query.
3. Store the document embeddings in a vector database (for simplicity, we will use an in-memory vector store).
4. Get the most similar documents from the knowledge base using the query embedding, with a metric such as cosine similarity.
5. Pass the retrieved documents as context to the language model and generate a response.

Here are some examples that you can add to the database and ask questions about them:

1. The secret code to access the project is 'quantum_leap_42'.
2. Alice is the lead engineer for the new 'Orion' feature.
3. The project deadline has been moved to next Friday.


For this exercise, solve TODO 2 to implement the document retrieval logic.

### 3. Structured Outputs
Structured outputs are a way to format the model's responses, such that they can be parsed by other systems. Information extraction is a common use case for structured outputs, where the model is asked to extract specific information from a given text.

Structured outputs are defined by a JSON Schema that describes the structure of the expected output.

The schema is passed in the API request in the `response_format` field. An example schema for extracting the city from a given text looks like this:

```json
{
    "type": "json_schema",
    "json_schema": {
        "name": "example_schema",
        "schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                }
            }
        }
    }
}
```

In the background, llama.cpp parses this schema and creates a GBNF grammar that guides the model's response generation. More information in the [llama.cpp documentation](https://github.com/ggml-org/llama.cpp/tree/master/grammars).

Keep in mind that using structured outputs can degrade the performance of LLMs, as shown by [Tam et al.](https://arxiv.org/abs/2408.02442)

For this exercise, solve TODO 3 in order to extract the name, city and age of user from a given text.

Here's an example prompt you can use to test your implementation:
```John is a 25 years old software engineer living in New York.```

### 4. Tool Calling
LLMs are very good at generating text, but they are not very good at performing tasks that require letter-perfect accuracy, such as calculations. Try asking the model to calculate the sum of two numbers over 10000, and you will see that it often makes mistakes.
These weaknesses can be mitigated by using tools, which are functions that can be called by the model to perform specific tasks.

Tool calling is a technique that builds on structured outputs. It allows the user to define functions that can be called by the language model and executed during the conversation.

Tool calling also uses structured outputs under the hood, as defining a tool is done using a JSON Schema.

A tool for calculating the sum of two numbers might look like this:

```json
[
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "num1": {
                        "type": "integer",
                        "description": "The first number."
                    },
                    "num2": {
                        "type": "integer",
                        "description": "The second number."
                    },
                },
                "required": [
                    "num1",
                    "num2",
                ]
            }
        }
    }
]
```

In this exercise, solve TODO 4 to implement a tool that calculates mathematical operations (add, subtract, multiply, divide) between two numbers.


### 5. Extra
Congratulations, you implemented a basic agent! If you want to extend it, you can try these other options:
1. Replace the in-memory RAG implementation with a proper vector database (e.g. Qdrant).
2. Add more tools for the agent to use - e.g. a web search tool, a bash file finding tool, etc.
3. Try to extract data from other types of documents (e.g. logs) or use other data types of [JSON Schema](https://json-schema.org/understanding-json-schema/reference/type) (e.g. arrays, enums).