# SNS

Implementation of paper: _SNS: Smart Node Selection for Scalable Traffic Engineering in Segment Routing Networks_, IEEE Transactions on Network and Service Management (IEEE TNSM)


## Structure

- `lpserver`: receives LP tasks, solves them in parallel, return optimized TE objectives & solutions & solving time
- python scripts: handle data, generate LP tasks & send tasks to `lpserver`, train & test DRL agents


## Dependencies

> [!WARNING]
> This project started in 2021, so the versions of the dependencies are now outdated.
> Upgrading to the newer versions of the dependencies may cause runtime errors due to breaking changes in the APIs.

Selected important dependencies:

- Rust

```toml
grb = "1.3.0"
petgraph = { git = "https://github.com/wlh320/petgraph", branch = "multi_predecessors" }
rayon = "1.5.3"
serde = { version = "1.0", features = ["derive"] }
serde-pickle = "1.1.1"
zmq = "0.10.0"
phf = { version = "0.11", features = ["macros"] }
clap = { version = "4", features = ["derive", "env"] }
```

- Python

```
torch==1.8.2

torch-geometric==1.7.2
torch-scatter==2.0.9
torch-sparse==0.6.12

zeromq==4.3.3
networkx==2.6.2
networkit==10.1
```
