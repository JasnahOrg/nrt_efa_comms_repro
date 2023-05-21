### Setup
- Setup two trn1.32xlarge instances with EFA enabled: `./start_efa_instances.sh --trn --n 2`
- Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- Make rust available in the current shell: `source "$HOME/.cargo/env"`
- Install clang
    - `sudo apt-get update`
    - `sudo apt-get install -y libclang-dev`
- Make NRT findable at compile time:
```
export CPATH=/opt/aws/neuron/include:$CPATH
export LIBRARY_PATH=/opt/aws/neuron/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/aws/neuron/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
```

### Test
- Create SSH key: `ssh-keygen -t ed25519 -C "your_email@example.com"`
- Add the SSH public key to ~/.ssh/authorized_keys on both instances so that nccom-test can ssh into both instances.
- Verify that EFA comms work: `NEURON_RT_ROOT_COMM_ID=172.31.63.174:62128 nccom-test -N 2 -r 64 --minbytes 100kb --maxbytes 1mb --stepfactor 10 --datatype fp32 --check allg --hosts 172.31.63.174 172.31.56.143`

### Reproduce
- `cargo test test_dummy_comm --features trn -- --show-output --nocapture`
```
running 1 test
Found an exisitng compiled NEFF /tmp/run_rank_1_7680805176524177539.neff, skipping compilation
2023-May-21 01:40:03.0080 93703:93705 [0] nccl_net_ofi_init:1415 CCOM WARN NET/OFI aws-ofi-nccl initialization failed
2023-May-21 01:40:03.0080 93703:93705 [0] init.cc:99 CCOM WARN OFI plugin initNet() failed is EFA enabled?
2023-May-21 01:40:15.0924 93703:93705 [0] transport.cc:36 CCOM WARN No transport found between devices 0 and 0. Possible replica group misconfiguration
2023-May-21 01:40:15.0924 93703:93705 [0] init.cc:655 CCOM WARN Unable to run multi-instance workload.  Ofi plugin is not installed or EFA is not enabled
2023-May-21 01:40:15.0924 93703:93705 ERROR   ENC:ncclInitGlobalComm                      failed neuronInitGlobalComm request to NCCL
2023-May-21 01:40:15.0924 93703:93705 ERROR   ENC:enc_init_global_comm                    [nec_dev 0] failed to create global comm
2023-May-21 01:40:15.0924 93703:93705 ERROR   NRT:nrt_load_collectives                    failed to create global communicator, global_device_id=1, global_device_count=2, ROOT_COMM_ID=172.31.63.174:62128)
thread '<unnamed>' panicked at 'assertion failed: `(left != right)`
  left: `0x0`,
 right: `0x0`', xla/src/xla_runner.rs:333:13
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
thread 'xla::tests::test_dummy_comm' panicked at 'called `Result::unwrap()` on an `Err` value: Any { .. }', comm/src/xla.rs:394:27
test xla::tests::test_dummy_comm ... FAILED
```
