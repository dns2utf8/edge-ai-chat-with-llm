use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::error::Error;
use std::io::{self, Write};
use log::info;

// --- 1. COMMON TYPES (Provided) ---
// These are the data structures for interacting with the LLM API.

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Tool {
    pub r#type: String,
    pub function: Function,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Function {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: FunctionCall,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Deserialize, Debug)]
pub struct AssistantMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Deserialize, Debug)]
pub struct Choice {
    pub message: AssistantMessage,
}

#[derive(Deserialize, Debug)]
pub struct ChatCompletion {
    pub choices: Vec<Choice>,
}

#[derive(Serialize, Debug)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
}

#[derive(Serialize, Debug, Clone)]
pub struct ResponseFormat {
    pub r#type: String,
    pub json_schema: JsonSchemaWrapper,
}

#[derive(Serialize, Debug, Clone)]
pub struct JsonSchemaWrapper {
    pub name: String,
    pub schema: Value,
}

#[derive(Serialize, Deserialize, Debug)]
struct UserInfo {
    name: String,
    city: String,
    age: u32,
}

// --- 2. RETRIEVAL-AUGMENTED GENERATION (RAG) ---
// Giving our LLM knowledge about private documents.

mod rag {
    use super::*;

    #[derive(Deserialize)]
    struct EmbeddingResponse {
        #[serde(rename = "index")]
        _index: i32,
        embedding: Vec<Vec<f32>>,
    }

    pub struct VectorDB {
        client: Client,
        embeddings_url: String,
        documents: Vec<(String, Vec<f32>)>,
    }

    impl VectorDB {
        pub fn new(embeddings_url: &str) -> Self {
            VectorDB {
                client: Client::new(),
                embeddings_url: embeddings_url.to_string(),
                documents: Vec::new(),
            }
        }

        async fn get_embeddings(&self, text: &str) -> Result<Vec<f32>, Box<dyn Error>> {
            let payload = json!({ "content": text });
            let response: Vec<EmbeddingResponse> = self
                .client
                .post(&self.embeddings_url)
                .json(&payload)
                .send()
                .await?
                .json()
                .await?;
            Ok(response.first()
                .ok_or("No embeddings found")?
                .embedding
                .first()
                .ok_or("No embedding vector found")?
                .clone())
        }

        pub async fn add_document(&mut self, doc: &str) -> Result<(), Box<dyn Error>> {
            info!("Embedding document: \"{}\"", doc);
            // TODO 2.0: Embed the document and add it to the toy vector database - self.documents in this case.
            !unimplemented!()
        }

        pub async fn find_most_similar(&self, query: &str) -> Result<Option<(String, f32)>, Box<dyn Error>> {
            const SIMILARITY_THRESHOLD: f32 = 0.1;

            // TODO 2.2: Get the embedding for the user's query, using self.get_embeddings.

            // TODO 2.4: Find the document most similar to the user's query.
            // Iterate through `self.documents`. For each `(doc, doc_embedding)`, calculate the
            // `cosine_similarity` with the `query_embedding`. If the similarity is the best
            // so far, update `best_match`.

            // TODO 2.5: Check if the similarity for the best match is above the SIMILARITY_THRESHOLD.
            // You can play around with the threshold value to see how it affects the results.

            unimplemented!()
        }
    }

    /// Calculates the cosine similarity between two vectors.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        // TODO 2.3: Calculate the cosine similarity of vectors `a` and `b`.
        // The formula is: (A · B) / (||A|| * ||B||)
        // 1.  Dot Product (A · B) - this is the sum of the products of corresponding elements.
        //     Formula: Σ(a_i * b_i)
        // 2.  L2 Norm - this is the square root of the sum of the squares of the elements.
        //     Formula: sqrt(Σ(a_i^2))
        // 3.  If either norm is 0.0, return 0.0 to prevent division by zero.
        unimplemented!()
    }
}

// --- 3. TOOLING ---
// Giving our LLM the ability to use external functions.

mod tools {
    use super::*;

    /// Defines the tools our LLM can use.
    pub fn get_tools_definition() -> Vec<Tool> {
        // TODO 4: Define a tool named `calculate`, that uses the `calculate` function.
        // Hint: Parameters are defined by a JSON schema, you can use its features, such as enums, descriptions and nested objects.
        vec![]
    }
    
    async fn calculate(num1: i64, num2: i64, op: &str) -> Result<i64, Box<dyn Error>> {
        match op {
            "add" => Ok(num1 + num2),
            "subtract" => Ok(num1 - num2),
            "multiply" => Ok(num1 * num2),
            "divide" => {
                if num2 == 0 {
                    Err("Cannot divide by zero".into())
                } else {
                    Ok(num1 / num2)
                }
            },
            _ => Err(format!("Unknown operation: {}", op).into()),
        }
    }

    /// Executes a tool call requested by the LLM.
    pub async fn execute_tool_call(tool_call: &ToolCall) -> Result<String, Box<dyn Error>> {
        if tool_call.function.name == "calculate" {
            info!("> Tool call: calculate");
            let args: Value = serde_json::from_str(&tool_call.function.arguments)?;
            let num1 = args["num1"].as_i64().ok_or("Missing num1 for calculate")?;
            let num2 = args["num2"].as_i64().ok_or("Missing num2 for calculate")?;
            let op = args["op"].as_str().ok_or("Missing op for calculate")?;
            let result = calculate(num1, num2, op).await?;
            Ok(format!("<tool_result>{}</tool_result>", result))
        }
        else {
            Err(format!("Unknown tool: {}", tool_call.function.name).into())
        }
    }
}

// --- 4. LLM INTERACTION ---
async fn call_llm_api(
    client: &Client,
    llm_url: &str,
    model: &str,
    messages: Vec<Message>,
    use_tools: bool,
    response_format: Option<ResponseFormat>,
) -> Result<ChatCompletion, Box<dyn Error>> {
    let tools = if use_tools {
        Some(tools::get_tools_definition())
    } else {
        None
    };
    
    // TODO 1: Implement the API call to the LLM server.
    // This function will send the conversation history and available tools to the LLM.
    //
    // Your task:
    // 1.  Create a `ChatRequest` struct.
    //     - `model`: Use the `model` parameter.
    //     - `messages`: Use the `messages` parameter.
    //     - `tools`: If `use_tools` is true, call `tools::get_tools_definition()`. Otherwise, use an empty vector.
    // 2.  Use the `reqwest::Client` to make a POST request.
    //     - The URL should be `"{llm_url}/v1/chat/completions"`.
    //     - Send the `ChatRequest` as JSON using the `.json()` method.
    //     - `await` the response.
    // 3.  Return the `ChatCompletion`.
    
    unimplemented!()
}

// --- 5. AGENT LOGIC ---

/// Handles a simple response from the LLM.
async fn handle_simple_response(client: &Client, llm_url: &str, model: &str, messages: &mut Vec<Message>, user_input: &str) -> Result<(), Box<dyn Error>> {
    info!("> Simple Mode: Generating a response...");
    messages.push(Message { role: "user".to_string(), content: user_input.to_string() });

    let response = call_llm_api(&client, llm_url, model, messages.clone(), false, None).await?;
    if let Some(content) = &response.choices[0].message.content {
        println!("ASSISTANT: {}", content);
        messages.push(Message { role: "assistant".to_string(), content: content.clone() });
    } else {
        println!("ASSISTANT: <no response>");
    }
    Ok(())
}

/// Handles a query using the RAG pipeline.
async fn handle_rag_response(client: &Client, llm_url: &str, model: &str, db: &rag::VectorDB, messages: &mut Vec<Message>, user_input: &str) -> Result<(), Box<dyn Error>> {
    info!("> RAG Mode: Searching internal documents...");
    
    let context = db.find_most_similar(user_input).await?;
    let augmented_prompt = if let Some((ctx, score)) = context {
        info!("> Found relevant context: \"{}\", with score: {}", ctx, score);
        format!(
            "Answer the user's query based *only* on the following context.\n\nContext: \"{}\"\n\nUser query: {}",
            ctx, user_input
        )
    } else {
        info!("> No relevant context found in internal documents for your query.");
        user_input.to_string()
    };

    let mut rag_messages = messages.clone();
    rag_messages.push(Message { role: "user".to_string(), content: augmented_prompt });

    let response = call_llm_api(client, llm_url, model, rag_messages, false, None).await?;
    if let Some(content) = &response.choices[0].message.content {
        println!("ASSISTANT: {}", content);
        messages.push(Message { role: "user".to_string(), content: user_input.to_string() });
        messages.push(Message { role: "assistant".to_string(), content: content.clone() });
    } else {
        println!("ASSISTANT: <no response>");
    }
    Ok(())
}

/// Handles a query that may require using tools.
async fn handle_tool_response(client: &Client, llm_url: &str, model: &str, messages: &mut Vec<Message>, user_input: &str) -> Result<(), Box<dyn Error>> {
    let mut tool_messages = messages.clone();

    let tool_prompt = format!(
        "Use the available tools to answer the user's query if needed. Respond with the result\n\nUser query: {}",
        user_input
    );

    tool_messages.push(Message { role: "user".to_string(), content: tool_prompt.to_string() });

    // First, call the LLM to see if it wants to use a tool.
    let response = call_llm_api(client, llm_url, model, tool_messages.clone(), true, None).await?;
    let assistant_message = &response.choices[0].message;

    if let Some(tool_calls) = &assistant_message.tool_calls {
        // The LLM wants to use a tool.
        tool_messages.push(Message { role: "assistant".to_string(), content: serde_json::to_string(&tool_calls)? });

        for tool_call in tool_calls {
            let tool_result = tools::execute_tool_call(tool_call).await?;
            info!("> Tool result: {}", tool_result);
            tool_messages.push(Message { role: "user".to_string(), content: tool_result });
        }

        // Now, call the LLM again with the tool results to get a final answer.
        let final_response = call_llm_api(client, llm_url, model, tool_messages, false, None).await?;
        if let Some(content) = &final_response.choices[0].message.content {
            println!("ASSISTANT: {}", content);
            messages.push(Message { role: "user".to_string(), content: user_input.to_string() });
            messages.push(Message { role: "assistant".to_string(), content: content.clone() });
        } else {
            println!("ASSISTANT: <no response after tool use>");
        }
    } else if let Some(content) = &assistant_message.content {
        // The LLM answered directly without using a tool.
        println!("ASSISTANT: {}", content);
        messages.push(Message { role: "user".to_string(), content: user_input.to_string() });
        messages.push(Message { role: "assistant".to_string(), content: content.clone() });
    }
    Ok(())
}

/// Handles a query by extracting structured data.
async fn handle_structured_output_response(client: &Client, llm_url: &str, model: &str, messages: &mut Vec<Message>, user_input: &str) -> Result<(), Box<dyn Error>> {
    info!("> Structured Output Mode: Extracting information...");

    let structured_prompt = format!(
        "From the following text, extract the requested info.\n\nText: \"{}\"",
        user_input
    );

    let structured_messages = vec![Message { role: "user".to_string(), content: structured_prompt }];

    // TODO 3.1: Define the response_format
    let response_format: Option<ResponseFormat> = None;

    let response = call_llm_api(client, llm_url, model, structured_messages, false, response_format).await?;


    // TODO 3.2: Parse the response as JSON, using the UserInfo struct defined above.
    unimplemented!()
}


// --- 6. MAIN LOOP ---
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let llm_url = "http://localhost:8080";
    let embeddings_url = "http://localhost:8081/embedding";
    let model = "gemma-3-1b-it";

    let client = Client::new();

    // --- Setup RAG Database ---
    let mut db = rag::VectorDB::new(embeddings_url);

    // TODO 2.1: Add internal documents that the model could not know. For example:

    // --- Main Loop ---
    println!("\nWelcome! I am a helpful assistant. How can I help you today?");
    println!("(Type 'exit' to quit)\n");

    let mut messages: Vec<Message> = Vec::new();
    loop {
        print!("USER: ");
        io::stdout().flush()?;
        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input)?;
        let user_input = user_input.trim();

        if user_input.eq_ignore_ascii_case("exit") {
            break;
        }

        println!("Choose mode: (1) Simple response, (2) RAG, (3) Structured Outputs, (4) Tool Calling");
        print!("Mode: ");
        io::stdout().flush()?;
        let mut mode_input = String::new();
        io::stdin().read_line(&mut mode_input)?;
        let mode = mode_input.trim();

        let result = match mode {
            "1" => handle_simple_response(&client, llm_url, model, &mut messages, user_input).await,
            "2" => handle_rag_response(&client, llm_url, model, &db, &mut messages, user_input).await,
            "3" => handle_structured_output_response(&client, llm_url, model, &mut messages, user_input).await,
            "4" => handle_tool_response(&client, llm_url, model, &mut messages, user_input).await,
            _ => {
                println!("Invalid mode selected. Please enter '1', '2', '3', or '4'.");
                Ok(())
            }
        };

        if let Err(e) = result {
            eprintln!("An error occurred: {}", e);
        }

        println!();
    }

    Ok(())
}

