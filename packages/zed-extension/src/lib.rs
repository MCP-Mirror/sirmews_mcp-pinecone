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
    python_path: Option<String>,
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

        let settings = ContextServerSettings::for_project("pinecone-context-server", project)?;
        let Some(settings) = settings.settings else {
            return Err("missing `api_key` and `index_name` settings".into());
        };
        let settings: PineconeSettings =
            serde_json::from_value(settings).map_err(|e| e.to_string())?;    

        // If python_path is not empty, use the default python path
        // I presume you use uv because it's simply the best
        let python_path = settings.python_path.unwrap_or_else(|| "uv".to_string());

        // Use installed mcp-pinecone package
        Ok(Command {
            command: python_path.clone(),
            args: vec![
                "--directory".into(),
                "/Users/nav/Documents/projects/mcp-pinecone/packages/mcp-server".into(),
                "run".into(),
                "mcp-pinecone".into(),
            ],
            env: vec![
                ("PINECONE_API_KEY".to_string(), settings.api_key),
                ("PINECONE_INDEX_NAME".to_string(), settings.index_name),
                ("PYTHON_PATH".to_string(), python_path),
            ],
        })
    }
}

// Register the Pinecone extension with Zed
zed::register_extension!(PineconeExtension);