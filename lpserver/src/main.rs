use clap::Parser;
use lpserver::{config::Config, server::TEServer};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::parse();
    let num_threads = config.num_agents;
    let server = TEServer::new(config);
    server.run(num_threads)?;
    Ok(())
}
