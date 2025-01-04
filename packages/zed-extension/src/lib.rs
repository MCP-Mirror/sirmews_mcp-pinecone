use zed::prelude::*;
use anyhow::Result;
// For deserializing JSON configuration into Rust structs
use serde::Deserialize;


// Define the settings for the Pinecone extension. This matches Zed's settings.json
#[derive(Debug, Default, Deserialize)]
struct PineconeSettings {
    api_key: Option<String>,
    index_name: Option<String>,
}

// Define the Pinecone extension
pub struct PineconeExtension;

impl PineconeExtension {
    // Helper function to check if mcp-pinecone is installed
    fn check_mcp_pinecone() -> Result<()> {
        let output = ProcessCommand::new("python3")
            .arg("-c")
            .arg("import mcp_pinecone")
            .output()?;

        if !output.status.success() {
            // Convert stderr to string for the error message
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!(
                "mcp-pinecone is not installed. Please install it with 'pip install mcp-pinecone or 'uv pip install mcp-pinecone'", 
            ));
        }

        Ok(())
    }
}

// Implement the Extension trait for the Pinecone extension
impl Extension for PineconeExtension {
    fn init(self) -> Result<()> {
        Ok(())
    }



    // Tell Zed how to run the Pinecone extension
    fn context_server_command(
        &mut self,
        context_server_id: &ContextServerId,
        project: &Project,
    ) -> Result<Command> {
        // Check if the context server ID is "pinecone"
        if context_server_id.as_ref() == "pinecone" {

            // First check if mcp-pinecone is installed
            Self::check_mcp_pinecone()?;

            // Get settings from Zed's settings.json
            let settings = project
                .settings()
                .get::<PineconeSettings>("context_servers.pinecone.settings")
                .unwrap_or_default();

            // Build environment variables for the MCP server
            // Use a hashmap to store the environment variables because MCP expects a JSON object
            let mut env = std::collections::HashMap::new();
            if let Some(api_key) = settings.api_key {
                env.insert("PINECONE_API_KEY".to_string(), api_key);
            }
            if let Some(index_name) = settings.index_name {
                env.insert("PINECONE_INDEX_NAME".to_string(), index_name);
            }
            

            // Use installed mcp-pinecone package
            Ok(Command {
                command: "python3".into(),
                args: vec!["-m".into(), "mcp_pinecone".into()],
                env: Some(env),
            })
        } else {
            Err(anyhow::anyhow!("Unknown context server ID"))
        }
    }
}

// Register the Pinecone extension with Zed
zed::register_extension!(PineconeExtension);