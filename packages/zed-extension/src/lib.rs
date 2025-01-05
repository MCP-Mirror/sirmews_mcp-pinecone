use serde::Deserialize;
use zed::settings::ContextServerSettings;
use zed_extension_api::{self as zed, serde_json, Command, ContextServerId, Project, Result};

// Define the Pinecone extension
struct PineconeExtension;

// Define the settings for the Pinecone extension. This matches Zed's settings.json
#[derive(Debug, Deserialize)]
struct PineconeSettings {
    api_key: String,
    index_name: String,
}

// Implement the Extension trait for the Pinecone extension
impl zed::Extension for PineconeExtension {
    fn new() -> Self {
        Self
    }

    // Tell Zed how to run the Pinecone extension
    fn context_server_command(
        &mut self,
        _context_server_id: &ContextServerId,
        project: &Project,
    ) -> Result<Command> {

        // First check if mcp-pinecone is installed
        // Self::check_mcp_pinecone()?;

        let settings = ContextServerSettings::for_project("pinecone-context-server", project)?;
        let Some(settings) = settings.settings else {
            return Err("missing `api_key` and `index_name` settings".into());
        };
        let settings: PineconeSettings =
            serde_json::from_value(settings).map_err(|e| e.to_string())?;    

        // Use installed mcp-pinecone package
        Ok(Command {
            command: "python3".into(),
            args: vec![
                "-c".into(),
                "import mcp_pinecone; mcp_pinecone.main()".into()
            ],
            env: vec![
                ("PINECONE_API_KEY".to_string(), settings.api_key),
                ("PINECONE_INDEX_NAME".to_string(), settings.index_name),
            ],
        })
    }
}

// Register the Pinecone extension with Zed
zed::register_extension!(PineconeExtension);