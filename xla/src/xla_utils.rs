// System
use std::path::PathBuf;
use std::process::{Command, Output};

/// This is used to run local scripts, including the run_xla_cpu_gpu.py script to
/// compile and run XLA .pb for CPU and GPU.
pub fn run_local_script(command: &str, path_arg: Option<PathBuf>, args: &[&str]) -> Output {
    let mut cmd = Command::new(command);
    if let Some(path) = path_arg {
        cmd.arg(path);
    }
    for arg in args {
        cmd.arg(arg);
    }
    let cmd_string = format!("{:?}", cmd);
    let cmd_string = str::replace(&cmd_string, "\"", "");
    println!("Running command '{}'", cmd_string);
    //panic!("You should run this locally");
    let msg = format!("Failed to execute script '{}'", cmd_string);
    let output = cmd.output().expect(&msg);
    assert!(
        output.status.success(),
        "Script '{:?}' failed: \n{}, stdout: \n{}",
        cmd,
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );

    if !output.stderr.is_empty() {
        eprintln!(
            "Python script stderr: \n{}",
            String::from_utf8_lossy(&output.stderr),
        );
    }
    println!("Successfully executed script {:?}", cmd);
    output
}
