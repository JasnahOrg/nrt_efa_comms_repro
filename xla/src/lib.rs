pub mod bindings;

pub mod trn;
pub mod xla_runner;
pub mod xla_utils;

pub mod xla {
    #![allow(rustdoc::invalid_html_tags)]
    include!(concat!(env!("OUT_DIR"), "/xla.rs"));
}

#[cfg(test)]
mod tests {
    // Local
    use crate::xla_runner::{
        CommInitInfo, DistributedRunInfo, XLAHardware, XLARunInput, XLARunner,
    };

    fn test_comm(global_rank: u32) {
        let run_name = "test_comm";
        let proto_path = "rust_hlo_run_rank_0.pb";

        let comm_info = CommInitInfo {
            ip_address_of_rank0: Some("172.31.63.174".to_string()),
            port_of_rank0: Some(62128),
        };
        let distributed_info = DistributedRunInfo {
            comm_init_info: comm_info,
            world_rank_id: global_rank,
            world_num_ranks: 2,
            machine_device_id: 0,
            num_ranks_per_process: Some(1),
        };
        let runner = XLARunner::new_distributed(XLAHardware::TRN, Some(distributed_info));
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let input_names = &["x"];
        let input_shapes: Vec<Vec<u64>> = vec![vec![4]];
        let rust_output = runner.run(
            proto_path,
            run_name,
            XLARunInput::new(input_names, vec![x], input_shapes),
            false,
        );
        println!("rust_output: {:?}", rust_output);
    }

    #[test]
    fn test_instance_0() {
        test_comm(0);
    }

    #[test]
    fn test_instance_1() {
        test_comm(1);
    }
}
