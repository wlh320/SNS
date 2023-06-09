use std::collections::{HashMap, HashSet};
use std::path::Path;

use petgraph::algo::{all_shortest_paths, toposort, FloatMeasure};
use petgraph::graphmap::DiGraphMap;
use petgraph::visit::{EdgeRef, NodeIndexable};

use crate::config::{Config, ObjKind, Solution, Task, TLMAP};
use crate::lp;

#[derive(Debug, Default, Eq, PartialOrd, Ord, Copy, Clone)]
pub struct EdgeAttr {
    weight: u32,
    cap: u32,
}

impl EdgeAttr {
    pub fn new(weight: u32, cap: u32) -> EdgeAttr {
        EdgeAttr { weight, cap }
    }
}

impl PartialEq for EdgeAttr {
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight
    }
}

impl std::ops::Add for EdgeAttr {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            weight: self.weight + other.weight,
            cap: self.cap,
        }
    }
}

impl FloatMeasure for EdgeAttr {
    fn zero() -> Self {
        EdgeAttr { cap: 0, weight: 0 }
    }

    fn infinite() -> Self {
        EdgeAttr {
            cap: 0,
            weight: 0x3f3f3f3f,
        }
    }
}

pub type TM = Vec<Vec<f64>>;
type FMap = HashMap<(usize, usize), HashMap<(usize, usize), f64>>;

pub fn load_tm_list(path: impl AsRef<Path>) -> Vec<TM> {
    let content = std::fs::read(path).unwrap();
    serde_pickle::from_slice(&content, Default::default()).unwrap()
}
pub fn load_topology(path: impl AsRef<Path>) -> DiGraphMap<usize, EdgeAttr> {
    let mut graph = DiGraphMap::new();
    let content: String = std::fs::read_to_string(path).unwrap();
    let node_count = content
        .lines()
        .flat_map(|line| {
            line.trim()
                .splitn(4, ' ')
                .take(2)
                .map(|x| x.parse::<usize>().unwrap())
        })
        .max()
        .unwrap();
    for i in 0..node_count {
        graph.add_node(i);
    }
    for line in content.lines() {
        match line.trim().splitn(5, ' ').collect::<Vec<_>>()[..] {
            [src, dst, weight, cap, ..] => {
                let src: usize = src.parse().unwrap();
                let dst: usize = dst.parse().unwrap();
                let weight: u32 = weight.parse().unwrap();
                let cap: u32 = cap.parse().unwrap();
                let attr = EdgeAttr::new(weight, cap);
                graph.add_edge(src, dst, attr);
            }
            _ => panic!("Invalid input file"),
        }
    }
    // println!("nodes: {} edges: {}", graph.node_count(), graph.edge_count());
    graph
}

/// save data related to network topology
#[derive(Debug)]
pub struct TESolver {
    cfg: Config,
    graph: DiGraphMap<usize, EdgeAttr>,
    pub f_data: FMap,
    edges: Vec<(usize, usize)>,
    caps: Vec<f64>,
}

impl TESolver {
    pub fn new(cfg: Config) -> TESolver {
        let graph = load_topology(cfg.topo_path());
        let mut solver = TESolver {
            cfg,
            graph,
            f_data: HashMap::new(),
            edges: Vec::new(),
            caps: Vec::new(),
        };
        solver.precompute_graph_info(); // compute edge, cap info related to topology
        solver.precompute_f(); // compute f,g function related to 2-SR
        solver
    }

    fn compute_ecmp_link_frac(
        &self,
        i: usize,
        j: usize,
        load: f64,
    ) -> HashMap<(usize, usize), f64> {
        let idx = |i| NodeIndexable::from_index(&self.graph, i);
        let ix = |i| self.graph.to_index(i);
        let paths = all_shortest_paths(&self.graph, idx(i), idx(j)).unwrap();
        // println!("{i} -> {j}: {:?}", paths);
        // build DAG
        let mut dag = DiGraphMap::new();
        let mut node_succ = HashMap::<usize, HashSet<usize>>::new();
        let mut node_load = HashMap::<usize, f64>::new();
        for p in paths {
            for (s, t) in p.iter().zip(p.iter().skip(1)) {
                let set = node_succ.entry(*s).or_default();
                set.insert(*t);
                dag.add_edge(*s, *t, 0.0);
            }
        }
        // compute fractions
        node_load.entry(idx(i)).or_insert(load);
        for node in toposort(&dag, None).unwrap().into_iter() {
            let nexthops = node_succ.get(&node);
            if nexthops.is_none() {
                continue;
            }
            let nexthops = nexthops.unwrap();
            let next_load = node_load[&node] / nexthops.len() as f64;
            for nexthop in nexthops {
                let w = dag.edge_weight_mut(node, *nexthop).unwrap();
                *w += next_load;
                *node_load.entry(*nexthop).or_insert(0.0) += next_load;
            }
        }
        let mut ans = HashMap::new();
        for e in dag.all_edges() {
            let (s, t) = (ix(e.source()), ix(e.target()));
            ans.insert((s, t), *e.weight());
        }
        ans
    }

    fn precompute_f(&mut self) {
        let node_count = self.graph.node_count();
        for i in 0..node_count {
            for j in 0..node_count {
                if i != j {
                    let ans = self.compute_ecmp_link_frac(i, j, 1.0);
                    self.f_data.insert((i, j), ans);
                }
            }
        }
    }

    fn precompute_graph_info(&mut self) {
        for (s, t, attr) in self.graph.all_edges() {
            self.edges.push((s, t));
            self.caps.push(attr.cap as f64);
        }
    }

    #[inline]
    pub fn f(&self, i: usize, j: usize, e: (usize, usize)) -> f64 {
        if !self.f_data.contains_key(&(i, j)) {
            0.0
        } else {
            let fracs = self.f_data.get(&(i, j)).unwrap();
            *fracs.get(&e).unwrap_or(&0.0)
        }
    }
    #[inline]
    pub fn g(&self, i: usize, j: usize, k: usize, e: (usize, usize)) -> f64 {
        self.f(i, k, e) + self.f(k, j, e)
    }

    fn convert_tm(&self, tm: &TM) -> (Vec<(usize, usize)>, Vec<f64>, Vec<f64>) {
        let num_nodes = tm.len();
        let mut flows = Vec::new();
        let mut demands = Vec::new();
        let loads = vec![0f64; self.edges.len()];
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if i == j {
                    continue;
                }
                if tm[i][j].abs() > f64::EPSILON {
                    flows.push((i, j));
                    demands.push(tm[i][j]);
                }
            }
        }
        (flows, demands, loads)
    }

    fn convert_tm_with_tnodes(
        &self,
        tm: &TM,
        tnodes: HashSet<usize>,
    ) -> (Vec<(usize, usize)>, Vec<f64>, Vec<f64>) {
        let num_nodes = tm.len();
        let mut flows = Vec::new();
        let mut demands = Vec::new();
        let mut link_bg: HashMap<(usize, usize), f64> = HashMap::new();
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if i == j {
                    continue;
                }
                if tnodes.contains(&i) {
                    if tm[i][j].abs() > f64::EPSILON {
                        flows.push((i, j));
                        demands.push(tm[i][j]);
                    }
                } else if tm[i][j].abs() > f64::EPSILON {
                    let bg_on_link = self.compute_ecmp_link_frac(i, j, tm[i][j]);
                    for (link, load) in bg_on_link {
                        *link_bg.entry(link).or_default() += load;
                    }
                }
            }
        }
        let mut loads = Vec::new();
        for (src, dst) in &self.edges {
            loads.push(*link_bg.entry((*src, *dst)).or_default());
        }
        (flows, demands, loads)
    }

    fn compute_revised_frac(
        &self,
        i: usize,
        j: usize,
        ratios: &HashMap<(usize, usize), f64>,
        load: f64,
    ) -> f64 {
        let idx = |i| NodeIndexable::from_index(&self.graph, i);
        let paths = all_shortest_paths(&self.graph, idx(i), idx(j)).unwrap();
        // println!("{i} -> {j}: {:?}", paths);
        // build DAG
        let mut dag = DiGraphMap::new();
        let mut node_succ = HashMap::<usize, HashSet<usize>>::new();
        let mut node_load = HashMap::<usize, f64>::new();
        for p in paths {
            for (s, t) in p.iter().zip(p.iter().skip(1)) {
                let set = node_succ.entry(*s).or_default();
                set.insert(*t);
                dag.add_edge(*s, *t, 0.0);
            }
        }
        // compute fractions
        node_load.entry(idx(i)).or_insert(load);
        for node in toposort(&dag, None).unwrap().into_iter() {
            let nexthops = node_succ.get(&node);
            if nexthops.is_none() {
                continue;
            }
            let nexthops = nexthops.unwrap();
            let next_load = node_load[&node] / nexthops.len() as f64;
            for &nexthop in nexthops {
                let w = dag.edge_weight_mut(node, nexthop).unwrap();
                *w += next_load * ratios[&(node, nexthop)];
                *node_load.entry(nexthop).or_insert(0.0) += next_load * ratios[&(node, nexthop)];
            }
        }
        node_load[&j]
    }

    fn revise_true_reward(
        &self,
        tm: &TM,
        lp_satisfied: f64,
        tnodes: &HashSet<usize>,
        demands: &[f64],
        loads: &[f64],
    ) -> f64 {
        // 1. compute real ratios
        let mut ratios: HashMap<(usize, usize), f64> = HashMap::new();
        for (e, l) in self.graph.all_edges().zip(loads) {
            let (s, t, attr) = e;
            let ratio = if l.abs() <= f64::EPSILON {
                // avoid division by zero
                1.0
            } else {
                // ** small load: 1.0; large load: real ratio **
                l.min(attr.cap as f64) / l
            };
            ratios.insert((s, t), ratio);
        }
        // 2. ECMP and compute satisfied background loads
        let mut load_demands = 0.0;
        let mut load_satisfied = 0.0;
        let num_nodes = tm.len();
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if i == j {
                    continue;
                }
                if !tnodes.contains(&i) {
                    load_demands += tm[i][j];
                    load_satisfied += self.compute_revised_frac(i, j, &ratios, tm[i][j]);
                }
            }
        }
        // 3. compute selected satisfied
        let selected_demands = demands.iter().sum::<f64>();
        let selected_satisfied = lp_satisfied * selected_demands;

        let ans = (selected_satisfied + load_satisfied) / (selected_demands + load_demands);
        println!("before revision: {:?} after: {:?}", lp_satisfied, ans);
        ans
    }

    pub fn solve(&self, tm: &TM, task: Task) -> Result<Solution, grb::Error> {
        // prepare data
        let num_nodes = tm.len();
        let tl = self.cfg.time_limit.unwrap_or(TLMAP[&self.cfg.toponame]);
        // flows, demands, [loads]
        let (flows, demands, loads) = match &task.tnodes {
            Some(tnodes) => self.convert_tm_with_tnodes(tm, tnodes.clone().into_iter().collect()),
            None => self.convert_tm(tm),
        };
        // caps, edges
        let caps = &self.caps;
        let edges = &self.edges;
        // task-related parameter
        let (inodes, tnodes) = (task.inodes, task.tnodes);
        let (lp_kind, obj_kind) = (task.lp_kind, task.obj_kind);

        // solve
        let mut solution = lp::solve(
            self, tl, num_nodes, caps, edges, flows, &demands, &loads, inodes, lp_kind, obj_kind,
        )?;

        // revise result if matches tnodes + objkind::MT
        match tnodes {
            Some(tnodes) if matches!(obj_kind, ObjKind::MT) => {
                let tnodes = tnodes.into_iter().collect();
                solution.obj = self.revise_true_reward(tm, solution.obj, &tnodes, &demands, &loads);
            }
            _ => (),
        }
        Ok(solution)
    }
}
