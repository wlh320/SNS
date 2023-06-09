use clap::Parser;
use phf::phf_map;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// global config of SREnv
#[derive(Parser, Debug)]
#[clap(author, version, about)]
pub struct Config {
    #[clap(short, long, default_value = "1")]
    pub id: usize,
    #[clap(short, long, default_value = "GEANT")]
    pub toponame: String,
    #[clap(short, long, default_value = "/home/wlh/coding/SNS/data/")]
    pub data_dir: PathBuf,
    #[clap(long, default_value = "24")]
    pub num_agents: usize,
    #[clap(long)]
    pub time_limit: Option<f64>,
}
impl Config {
    pub fn topo_path(&self) -> PathBuf {
        let mut path = self.data_dir.clone();
        path.push(&self.toponame);
        path
    }
    pub fn tm_path(&self) -> PathBuf {
        let mut path = self.data_dir.clone();
        path.push(format!("{}TM.pkl", &self.toponame));
        path
    }
}

pub static TLMAP: phf::Map<&'static str, f64> = phf_map! {
    "Abilene" => 1.0,
    "GEANT" => 5.0,
    "germany50" => 50.0,
    "rf1755" => 300.0,
    "rf6461" => 3000.0
};

/// following are structs used to communicate with python code

#[derive(Debug, Serialize, Deserialize)]
pub enum Cand {
    Network(Vec<usize>),   // the whole network share the same candidates
    Node(Vec<Vec<usize>>), // each node share the same candidates
    Flow(Vec<Vec<usize>>), // each real or fake flow has its own candidates
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Candidates {
    pub num_cd: usize,
    pub cands: Cand,
}

#[derive(Debug, Serialize, Deserialize, Copy, Clone)]
pub enum LPKind {
    LP,
    ILP,
}

#[derive(Debug, Serialize, Deserialize, Copy, Clone)]
pub enum ObjKind {
    MLU,
    MT,
}

#[derive(Serialize, Deserialize)]
pub struct Task {
    pub idx: Option<usize>,
    pub lp_kind: LPKind,
    pub obj_kind: ObjKind,
    #[serde(flatten)]
    pub inodes: Option<Candidates>,
    pub tnodes: Option<Vec<usize>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TEResult {
    pub idx: usize,
    #[serde(flatten)]
    pub sol: Solution,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Solution {
    pub obj: f64,
    pub time: f64,
    pub pol: usize,
}

impl std::fmt::Debug for Task {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cands = self.inodes.as_ref().map(|c| match c.cands {
            Cand::Network(_) => format!("Network({})", c.num_cd),
            Cand::Node(_) => format!("Node({})", c.num_cd),
            Cand::Flow(_) => format!("Flow({})", c.num_cd),
        });
        let fsel = self.tnodes.is_some();
        write!(
            f,
            "Task [ idx: {:?} lp: {:?}, obj: {:?}, inodes: {:?} fsel: {:?} ]",
            self.idx, self.lp_kind, self.obj_kind, cands, fsel
        )
    }
}
