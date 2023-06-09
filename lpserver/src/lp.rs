use grb::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

use crate::config::{Cand, Candidates, LPKind, ObjKind, Solution};
use crate::te::TESolver;

#[inline]
fn init_model(time_limit: f64) -> Result<Model, grb::Error> {
    // 1. create model
    let env = Env::new("")?; // mute gurobi.log
    let mut model = Model::with_env("mlu_lp", &env)?;
    model.set_param(param::Threads, 1)?;
    model.set_param(param::Method, 2)?;
    model.set_param(param::TimeLimit, time_limit)?;
    model.set_param(param::OutputFlag, 0)?;
    Ok(model)
}

fn add_action(
    model: &mut Model,
    num_f: usize,
    num_cd: usize,
    kind: LPKind,
) -> Result<HashMap<usize, Vec<Var>>, grb::Error> {
    let mut action: HashMap<usize, Vec<Var>> = HashMap::new();
    for f in 0..num_f {
        let mut vs = Vec::new();
        for cd in 0..num_cd {
            let name = &format!("action_{}_{}", f, cd);
            let var = match kind {
                LPKind::ILP => add_binvar!(model, name: name)?,
                LPKind::LP => add_ctsvar!(model, name: name)?,
            };
            vs.push(var);
        }
        action.insert(f, vs);
    }
    Ok(action)
}

#[inline]
fn get_candidate(cands: &Option<Candidates>, f: usize, src: usize, cd: usize) -> usize {
    let cands = cands.as_ref().map(|c| &c.cands);
    match cands {
        None => cd,
        Some(Cand::Network(v)) => v[cd],
        Some(Cand::Node(v)) => v[src][cd],
        Some(Cand::Flow(v)) => v[f][cd],
    }
}

pub fn solve(
    solver: &TESolver,
    tl: f64,                    // time limit
    num_n: usize,               // node numbers
    caps: &[f64],               // capacity 1xe
    edges: &[(usize, usize)],   // edges ex2
    flows: Vec<(usize, usize)>, // flows fx2
    demands: &[f64],          // demands 1xf
    loads: &[f64],            // current traffic loads on each edge
    inodes: Option<Candidates>, // candidate intermediate midpoints
    lp_kind: LPKind,
    obj_kind: ObjKind,
) -> Result<Solution, grb::Error> {
    let now = Instant::now();
    // 0. initilize input
    let num_f = flows.len();
    let num_cd = inodes.as_ref().map(|c| c.num_cd).unwrap_or(num_n);

    // 1. create model
    let mut model = init_model(tl)?;

    // 2. create decision variables
    let theta: Var = add_ctsvar!(model, name: "theta", bounds: 0.0..)?;
    let action = add_action(&mut model, num_f, num_cd, lp_kind)?;

    // 3. demand constraints
    for f in action.keys() {
        let con = match obj_kind {
            ObjKind::MLU => c!(action[f].iter().grb_sum() >= 1),
            ObjKind::MT => c!(action[f].iter().grb_sum() <= 1),
        };
        let name = &format!("con_d{}", f);
        model.add_constr(name, con)?;
    }

    // 4. utilization constraints
    for e in 0..edges.len() {
        let mut vars = Vec::new();
        for (f, (src, dst)) in flows.iter().enumerate() {
            for cd in 0..num_cd {
                let cand = get_candidate(&inodes, f, *src, cd);
                let ratio = solver.g(*src, *dst, cand, edges[e]);
                if ratio.abs() > f64::EPSILON {
                    vars.push((demands[f] * ratio, action[&f][cd]));
                }
            }
        }
        let con = match obj_kind {
            ObjKind::MLU => {
                let load = loads[e];
                c!(vars.into_iter().map(|(a, b)| a * b).grb_sum() <= theta * caps[e] - load)
            }
            ObjKind::MT => {
                let load = loads[e].min(caps[e]);
                c!(vars.into_iter().map(|(a, b)| a * b).grb_sum() <= caps[e] - load)
            }
        };
        let name = &format!("con_e{}", e);
        model.add_constr(name, con)?;
    }

    // 5. set objective and optimze model
    match obj_kind {
        ObjKind::MLU => model.set_objective(theta, grb::ModelSense::Minimize)?,
        ObjKind::MT => {
            let bw = action
                .iter()
                .map(|(k, v)| demands[*k] * v.grb_sum())
                .grb_sum();
            model.set_objective(bw, grb::ModelSense::Maximize)?;
        }
    }
    model.optimize()?;

    let elapsed = now.elapsed().as_millis() as f64 / 1000_f64;

    // 6. get solution
    let obj = model.get_attr(attr::ObjVal)?;
    let obj = match obj_kind {
        ObjKind::MLU => obj,
        ObjKind::MT => obj / (demands.iter().sum::<f64>() + f64::EPSILON),
    };
    let policies = action.iter().map(|(&f, rates)| {
        let (src, dst) = flows[f];
        rates.iter().enumerate().filter(|(cd, v)| {
            let cand = get_candidate(&inodes, f, src, *cd);
            let rate = model.get_obj_attr(attr::X, v).unwrap();
            cand != src && cand != dst && rate.abs() > f64::EPSILON
        }).count()
    }).sum();
    let sol = Solution { obj, time: elapsed, pol: policies };
    Ok(sol)
}
