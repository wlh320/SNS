use crate::config::{Config, TEResult, Task};
use crate::te::{self, TESolver, TM};

fn _pickle_solution(sol: TEResult) -> Result<Vec<u8>, serde_pickle::error::Error> {
    let res = serde_pickle::to_vec(&sol, Default::default())?;
    Ok(res)
}

fn _unpickle_task(buf: &[u8]) -> Result<Task, serde_pickle::error::Error> {
    let task: Task = serde_pickle::from_slice(buf, Default::default())?;
    Ok(task)
}

pub struct TEServer {
    id: usize,
    tms: Vec<TM>,
    solver: TESolver,
}

impl TEServer {
    pub fn new(cfg: Config) -> TEServer {
        let id = cfg.id;
        let tms = te::load_tm_list(cfg.tm_path());
        println!("Creating LPServer [{:?}]", &cfg);
        let solver = TESolver::new(cfg);
        TEServer { id, tms, solver }
    }

    pub fn solve(&self, task: Task) -> Result<TEResult, grb::Error> {
        let idx = task.idx.unwrap();
        let tm = &self.tms[idx];
        let sol = self.solver.solve(tm, task)?;
        let (obj, time) = (sol.obj, sol.time);
        let result = TEResult { idx, sol };
        println!("Solve TM idx: {idx} Ans: {obj:.3} Time: {time:.3} s");
        Ok(result)
    }

    pub fn run(self, num_threads: usize) -> Result<(), Box<dyn std::error::Error>> {
        // create zmq socket
        let id = self.id;
        let ctx = zmq::Context::new();
        let pull_sock = ctx.socket(zmq::PULL)?;
        pull_sock.connect(&format!("ipc:///tmp/tasks{id}"))?;
        let push_sock = ctx.socket(zmq::PUSH)?;
        push_sock.connect(&format!("ipc:///tmp/results{id}"))?;

        // create thread pool
        let this = std::sync::Arc::new(self);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()?;
        let (tx, rx) = std::sync::mpsc::channel();
        println!("Prepare done. Start to listen...");

        // spawn thread to send result
        let handle = std::thread::spawn(move || {
            for solution in rx {
                let ans = _pickle_solution(solution).unwrap();
                push_sock.send(ans, 0).ok();
            }
            push_sock.send("quit", 0).ok();
        });

        // handle tasks
        loop {
            let msg = pull_sock.recv_bytes(0)?;
            let task = _unpickle_task(&msg)?;
            let tx = tx.clone();
            let this = this.clone();
            println!("Recv task: {:?}", task);
            if task.idx.is_none() {
                break;
            }
            pool.spawn(move || {
                let idx = task.idx.unwrap();
                match this.solve(task) {
                    Ok(result) => tx.send(result).unwrap(),
                    Err(e) => eprintln!("Solve task {:?} failed, error: {}", idx, e),
                }
            });
        }
        drop(tx); // need to close all senders
        handle.join().ok();
        Ok(())
    }
}
