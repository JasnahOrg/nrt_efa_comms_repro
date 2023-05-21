// System
use std::collections::hash_map::DefaultHasher;
use std::env;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::path::Path;
use std::rc::Rc;
use std::time::Duration;

// Third Party
use prost::Message;
use serde::Serialize;

// Local
use crate::bindings::nrt;
use crate::trn::{allocate_tensors, handler_save_outputs, iterate_tensors, load_tensor_values};
use crate::xla::HloModuleProto;
use crate::xla_utils::run_local_script;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum XLAHardware {
    TRN,
}

#[derive(Serialize, Default)]
// TODO: This should eventually be replaced with a Tensor type that holds these things.
// NOTE: The only data required by Trainium is `inputs`.
pub struct XLARunInput {
    pub input_names: Vec<String>,
    pub inputs: Vec<Rc<Vec<f32>>>,
    pub input_shapes: Vec<Vec<u64>>,
}

impl XLARunInput {
    pub fn new(input_names: &[&str], inputs: Vec<Vec<f32>>, input_shapes: Vec<Vec<u64>>) -> Self {
        Self {
            input_names: input_names.iter().map(|x| x.to_string()).collect(),
            inputs: inputs.iter().map(|x| Rc::new(x.clone())).collect(),
            input_shapes,
        }
    }

    pub fn inputs(inputs: Vec<Rc<Vec<f32>>>) -> Self {
        Self {
            inputs,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CommInitInfo {
    pub ip_address_of_rank0: Option<String>,
    pub port_of_rank0: Option<u32>,
}

#[derive(Debug, Clone, Default)]
pub struct DistributedRunInfo {
    pub comm_init_info: CommInitInfo,
    pub world_rank_id: u32,
    pub world_num_ranks: u32,
    pub machine_device_id: u32,
    // Optional parameter in the case we want to run several ranks per process in different threads.
    // `machine_device_id` will be treated as a first device id, and the process will reserve
    // `machine_device_id`, `machine_device_id`+1, .., `machine_device_id`+`num_ranks_per_process`-1
    // devices (or cores depending on a platform).
    pub num_ranks_per_process: Option<u32>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct XLARunner {
    hardware: XLAHardware,
    distributed_run_info: Option<DistributedRunInfo>,
    // This flag is used to determine whether there were initialized procedures done, that may
    // need destructors as well. This flag being false will not prevent model from running,
    // because for some platforms the initialization should be done only once per process, while
    // having multiple XLA Runners in different threads.
    initialized: bool,
}

#[derive(Debug, Clone)]
// TODO In the future when we merge backends I imagine this will just be a Tensor.
// But currently the Tensor type in this xla crate does not contain any data.
pub enum Output {
    Bool(Rc<Vec<bool>>),
    Float32(Rc<Vec<f32>>),
}

#[derive(Debug, Clone)]
pub struct XLARunResults {
    pub output: Vec<Output>,
    /// The debug_ir human-readable reprsentation of the XLA HLO
    pub debug_ir: Option<String>,
    /// This is the graph exececution time without any compilation time, tensor allocation time, or
    /// output copying time.
    pub runtime: Duration,
}

fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

impl XLARunner {
    pub fn new_non_initialized(hardware: XLAHardware) -> Self {
        XLARunner {
            hardware,
            distributed_run_info: None,
            initialized: false,
        }
    }

    pub fn new(hardware: XLAHardware) -> Self {
        Self::new_distributed(hardware, None)
    }

    /// This inits the Neuron NRT library for Trainium if the trn feature is enabled.
    /// Note that NRT should be initialized only once per process. If nrt_close is called,
    /// subsequent calls to nrt_init will fail. This is a known issue with the current version of NRT.
    pub fn new_distributed(
        hardware: XLAHardware,
        distributed_run_info: Option<DistributedRunInfo>,
    ) -> Self {
        if distributed_run_info.is_some() {
            assert_eq!(
                hardware,
                XLAHardware::TRN,
                "Distributed xla runs for non TRN are not yet supported"
            );
        }
        if hardware == XLAHardware::TRN {
            if let Some(distributed_run_info) = distributed_run_info.as_ref() {
                let comm_init_info = distributed_run_info.comm_init_info.clone();
                if let Some(ip_address_of_rank0) = comm_init_info.ip_address_of_rank0 {
                    let num_ranks_per_process =
                        distributed_run_info.num_ranks_per_process.unwrap_or(1);
                    if distributed_run_info.world_rank_id % num_ranks_per_process != 0 {
                        // We only want to initialize once per process.
                        return XLARunner {
                            hardware,
                            distributed_run_info: Some(distributed_run_info.clone()),
                            initialized: false,
                        };
                    }

                    // According to
                    // <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/nrt-api-guide.html#initialization-configuration-and-teardown>
                    // these enviromental vars should be set before nrt_init() for collective ops.
                    env::set_var(
                        "NEURON_RT_ROOT_COMM_ID",
                        format!(
                            "{}:{}",
                            ip_address_of_rank0,
                            comm_init_info.port_of_rank0.unwrap()
                        ),
                    );
                    env::set_var("NEURON_RT_NUM_CORES", format!("{}", num_ranks_per_process));
                    let visible_cores = if num_ranks_per_process == 1 {
                        format!("{}", distributed_run_info.machine_device_id)
                    } else {
                        format!(
                            "{}-{}",
                            distributed_run_info.machine_device_id,
                            distributed_run_info.machine_device_id + num_ranks_per_process - 1
                        )
                    };
                    env::set_var("NEURON_RT_VISIBLE_CORES", visible_cores);
                }
            }
            {
                let result = unsafe {
                    nrt::nrt_init(
                        nrt::nrt_framework_type_t_NRT_FRAMEWORK_TYPE_NO_FW,
                        std::ptr::null() as *const i8,
                        std::ptr::null() as *const i8,
                    )
                };
                assert_eq!(result, nrt::NRT_STATUS_NRT_SUCCESS);
            }
        }
        XLARunner {
            hardware,
            distributed_run_info,
            initialized: true,
        }
    }

    pub fn run(
        &self,
        xla_hlo_pb_path: &str,
        run_name: &str,
        input: XLARunInput,
        benchmark: bool,
    ) -> Result<XLARunResults, String> {
        match self.hardware {
            XLAHardware::TRN => self.run_trn(xla_hlo_pb_path, run_name, input, benchmark),
        }
    }

    fn load_model_trn(
        &self,
        neff_data: &Vec<u8>,
        model: *mut *mut nrt::nrt_model_t,
    ) -> nrt::NRT_STATUS {
        unsafe {
            if self.distributed_run_info.is_none() {
                nrt::nrt_load(
                    neff_data.as_ptr() as *const _,
                    neff_data.len(),
                    0, // neuron core index to start from
                    1, // number of neuron cores to allocate the model to
                    model,
                )
            } else {
                let distributed_run_info = self.distributed_run_info.clone().unwrap();
                let num_ranks_per_process = distributed_run_info.num_ranks_per_process.unwrap_or(1);
                println!(
                    "Calling nrt_load_collectives with parameters: 
                neff_bytes: {:?}, 
                size: {}, 
                start_nc: {}, 
                nc_count: {}, 
                g_device_id: {}, 
                g_device_count: {}, 
                model: {:?}",
                    neff_data.as_ptr() as *const _,
                    neff_data.len(),
                    (distributed_run_info.world_rank_id % num_ranks_per_process) as i32,
                    // Number of neuron cores to allocate the model rank to. We always use 1 core per rank.
                    1,
                    distributed_run_info.world_rank_id,
                    distributed_run_info.world_num_ranks,
                    model,
                );
                nrt::nrt_load_collectives(
                    neff_data.as_ptr() as *const _,
                    neff_data.len(),
                    // Neuron core index to start from; since we set visible cores in NEURON_RT_VISIBLE_CORES
                    // it should be 0 if we run one rank per process.
                    (distributed_run_info.world_rank_id % num_ranks_per_process) as i32,
                    // Number of neuron cores to allocate the model rank to. We always use 1 core per rank.
                    1,
                    distributed_run_info.world_rank_id,
                    distributed_run_info.world_num_ranks,
                    model,
                )
            }
        }
    }

    #[allow(unused_variables)]
    /// This compiles the XLA HLO into a NEFF.
    /// See [AWS Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/neuronx-cc/api-reference-guide/neuron-compiler-cli-reference-guide.html).
    fn run_trn(
        &self,
        xla_hlo_pb_path: &str,
        run_name: &str,
        input: XLARunInput,
        benchmark: bool,
    ) -> Result<XLARunResults, String> {
        {
            let mut args: Vec<&str> = vec![
                "compile",
                xla_hlo_pb_path,
                "--framework",
                "XLA",
                "--target",
                "trn1",
                "--model-type",
                "transformer",
                // Keep precision for correctness testing
                #[cfg(debug_assertions)]
                "--auto-cast",
                #[cfg(debug_assertions)]
                "none",
            ];

            // Read the proto and hash it, don't re-compile if we've already compiled this graph.
            let mut file = File::open(xla_hlo_pb_path).expect("Unable to open HLO file.");
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)
                .expect("Failed to read HLO file.");
            let jax_proto = HloModuleProto::decode(&buffer[..]).unwrap();
            let hash_string = format!("{}{:?}{:?}", run_name, jax_proto, args);
            let hash = calculate_hash(&hash_string);
            let neff_path = format!("/tmp/{}_{}.neff", run_name, hash);
            if Path::new(&neff_path).exists() {
                println!(
                    "Found an exisitng compiled NEFF {}, skipping compilation",
                    neff_path
                );
            } else {
                args.push("--output");
                args.push(neff_path.as_str());
                println!("Compiling NEFF {}...", neff_path);
                let start = std::time::Instant::now();
                run_local_script("neuronx-cc", None, &args);
                println!("run_trn: Took {:?} to compile the NEFF.", start.elapsed());
            }

            // Read NEFF file into a byte vector
            let mut neff_file = File::open(neff_path.clone())
                .unwrap_or_else(|_| panic!("Unable to open NEFF file {}", neff_path));
            let mut neff_data: Vec<u8> = Vec::new();
            neff_file
                .read_to_end(&mut neff_data)
                .expect("Unable to read NEFF file");
            // Load the model
            let mut model: *mut nrt::nrt_model_t = std::ptr::null_mut();
            assert_eq!(model, std::ptr::null_mut());
            assert!(model.is_null());
            let result = self.load_model_trn(&neff_data, &mut model);
            assert_ne!(model, std::ptr::null_mut());
            assert!(!model.is_null());
            assert_eq!(result, nrt::NRT_STATUS_NRT_SUCCESS);

            // Allocate input and ouptut tensors
            let mut tensor_info_array: *mut nrt::nrt_tensor_info_array_t = std::ptr::null_mut();
            assert_eq!(tensor_info_array, std::ptr::null_mut());
            assert!(tensor_info_array.is_null());
            let result = unsafe {
                nrt::nrt_get_model_tensor_info(
                    model,
                    &mut tensor_info_array as *mut *mut nrt::nrt_tensor_info_array_t,
                )
            };
            assert_eq!(result, nrt::NRT_STATUS_NRT_SUCCESS);
            let tensor_info_array =
                std::ptr::NonNull::new(tensor_info_array).expect("Error: null tensor_info_array");

            let mut num_ranks_per_process = 1;
            let mut world_rank_id = 0;
            if let Some(distributed_run_info) = &self.distributed_run_info {
                world_rank_id = distributed_run_info.world_rank_id;
                num_ranks_per_process = distributed_run_info.num_ranks_per_process.unwrap_or(1);
            }
            let num_ranks_per_process = num_ranks_per_process;
            let world_rank_id = world_rank_id;

            let nrt_inputs = allocate_tensors(
                tensor_info_array,
                nrt::nrt_tensor_usage_NRT_TENSOR_USAGE_INPUT,
                (world_rank_id % num_ranks_per_process) as i32,
            );
            let nrt_inputs = nrt_inputs.expect("Error allocating input tensors");

            let outputs = allocate_tensors(
                tensor_info_array,
                nrt::nrt_tensor_usage_NRT_TENSOR_USAGE_OUTPUT,
                (world_rank_id % num_ranks_per_process) as i32,
            );
            let outputs = outputs.expect("Error allocating output tensors");

            // Note that even if input parameters are not initialized, it will
            // still run and it will still produce values.
            if !input.inputs.is_empty() {
                let result = load_tensor_values(
                    nrt_inputs,
                    tensor_info_array,
                    nrt::nrt_tensor_usage_NRT_TENSOR_USAGE_INPUT,
                    input.inputs,
                );
                result.expect("Error loading input tensor values");
            }

            // Run it
            let (result, runtime) = {
                if benchmark {
                    println!("Warming up...");
                    unsafe {
                        nrt::nrt_execute_repeat(model, nrt_inputs.as_ptr(), outputs.as_ptr(), 10)
                    };
                    println!("Done warning up");
                    println!("Benchmarking...");
                    let start = std::time::Instant::now();
                    let result = unsafe {
                        nrt::nrt_execute_repeat(model, nrt_inputs.as_ptr(), outputs.as_ptr(), 20)
                    };
                    let runtime = start.elapsed();
                    println!("Done benchmarking");
                    (result, runtime)
                } else {
                    let start = std::time::Instant::now();
                    let result =
                        unsafe { nrt::nrt_execute(model, nrt_inputs.as_ptr(), outputs.as_ptr()) };
                    let runtime = start.elapsed();
                    (result, runtime)
                }
            };
            assert_eq!(
                result,
                nrt::NRT_STATUS_NRT_SUCCESS,
                "nrt_execute failed to run model {}",
                run_name
            );

            // TODO: Instead of saving the outputs to file, get them in a Vec<Vec<f32>>
            // Saving outputs to files
            let result = unsafe {
                iterate_tensors(
                    outputs.as_ptr(),
                    tensor_info_array.as_ptr(),
                    nrt::nrt_tensor_usage_NRT_TENSOR_USAGE_OUTPUT,
                    handler_save_outputs,
                    std::ptr::null_mut(),
                )
            };
            let result = result.expect("Error saving output tensors");
            assert_eq!(result.0, nrt::NRT_STATUS_NRT_SUCCESS);
            let output = result.1;

            unsafe {
                nrt::nrt_destroy_tensor_set(&mut nrt_inputs.as_ptr());
                nrt::nrt_destroy_tensor_set(&mut outputs.as_ptr());
                nrt::nrt_free_model_tensor_info(tensor_info_array.as_ptr());
            };
            //let output = Vec::new();
            return Ok(XLARunResults {
                output,
                debug_ir: None,
                runtime,
            });
        }
        // This will be seen as unreachable code when --feature trn is enabled
        #[allow(unreachable_code)]
        Err("TRN feature is not enabled.".to_string())
    }
}

impl Drop for XLARunner {
    fn drop(&mut self) {
        if !self.initialized {
            return;
        }
        if self.hardware == XLAHardware::TRN {
            unsafe {
                nrt::nrt_close();
            }
        }
    }
}
