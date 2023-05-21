### Setup
- Open start_efa_instances.sh and set the values `KEYNAME`, `SUBNET`, and `SECURITY_GROUP`
- Create two trn1.32xlarge instances with EFA enabled: `./start_efa_instances.sh --trn1 --n 2`
- SSH into each of the two instances and set them up:
- Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`. Choose Default installation.
- Make rust available in the current shell: `source "$HOME/.cargo/env"`
- Install clang, which is used to generate the Rust to C NRT lib FFI
    - `sudo apt-get update`
    - `sudo apt-get install -y libclang-dev`
- Install neuron compiler: `pip3 install neuronx-cc`
- Make NRT findable at compile time, add these to ~/.bashrc:
```
export CPATH=/opt/aws/neuron/include:$CPATH
export LIBRARY_PATH=/opt/aws/neuron/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/aws/neuron/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
```
- Close your shell and re-connect for `neuronx-cc` to become available: `which neuronx-cc`.

### Baseline Test
- Create an SSH key on your first instance only: `ssh-keygen -t ed25519 -C "your_email@example.com"`, leave the passphrase empty.
- `cat ~/.ssh/id_ed25519.pub` and add the SSH public key to ~/.ssh/authorized_keys on both instances so that nccom-test can ssh into both instances.
- Verify that you can ssh from your first instance into each of your two instances via their private IPs: `ssh 172.31.58.215` and `ssh 172.31.51.0`
- Verify that EFA comms work: `NEURON_RT_ROOT_COMM_ID=172.31.58.215:62128 nccom-test -N 2 -r 64 --minbytes 100kb --maxbytes 1mb --stepfactor 10 --datatype fp32 --check allr --hosts 172.31.58.215 172.31.51.0`
```
    size(B)    count(elems)    type    time(us)    algbw(GB/s)    busbw(GB/s)
     102400           25600    fp32         398           0.24           0.47
    1024000          256000    fp32         469           2.03           4.00
Avg bus bandwidth:	2.2375GB/s
```

### Reproduce
- Clone this repo onto each instance: `git clone https://github.com/JasnahOrg/nrt_efa_comms_repro.git`
- Open the file xla/src/lib.rs and change the line `ip_address_of_rank0: Some("172.31.58.215".to_string()),` to reflect the private IP of your rank 0 trn instance. This should be the same as the value used above for `NEURON_RT_ROOT_COMM_ID`.
- On the first instance, run: `cargo test test_instance_0 -- --show-output --nocapture`. After build and the test starts with `running 1 test`, you should see it hang, waiting for the other rank to come online.
- Now run this on the second instance: `cargo test test_instance_1 -- --show-output --nocapture`. You'll see this output on both machines:
```
running 1 test
Compiling NEFF /tmp/test_comm_7179921044148955493.neff...
Running command 'neuronx-cc compile rust_hlo_run_rank_0.pb --framework XLA --target trn1 --model-type transformer --auto-cast none --output /tmp/test_comm_7179921044148955493.neff'
Successfully executed script "neuronx-cc" "compile" "rust_hlo_run_rank_0.pb" "--framework" "XLA" "--target" "trn1" "--model-type" "transformer" "--auto-cast" "none" "--output" "/tmp/test_comm_7179921044148955493.neff"
run_trn: Took 1.184781638s to compile the NEFF.
Calling nrt_load_collectives with parameters:
                neff_bytes: 0x7faaec02c6f0,
                size: 8115,
                start_nc: 0,
                nc_count: 1,
                g_device_id: 1,
                g_device_count: 2,
                model: 0x7faaf26cf958
2023-May-21 14:27:33.0030 102203:102204 [0] nccl_net_ofi_init:1415 CCOM WARN NET/OFI aws-ofi-nccl initialization failed
2023-May-21 14:27:33.0030 102203:102204 [0] init.cc:99 CCOM WARN OFI plugin initNet() failed is EFA enabled?
2023-May-21 14:27:33.0045 102203:102204 [0] transport.cc:36 CCOM WARN No transport found between devices 0 and 0. Possible replica group misconfiguration
2023-May-21 14:27:33.0045 102203:102204 [0] init.cc:655 CCOM WARN Unable to run multi-instance workload.  Ofi plugin is not installed or EFA is not enabled
2023-May-21 14:27:33.0045 102203:102204 ERROR   ENC:ncclInitGlobalComm                      failed neuronInitGlobalComm request to NCCL
2023-May-21 14:27:33.0045 102203:102204 ERROR   ENC:enc_init_global_comm                    [nec_dev 0] failed to create global comm
2023-May-21 14:27:33.0045 102203:102204 ERROR   NRT:nrt_load_collectives                    failed to create global communicator, global_device_id=1, global_device_count=2, ROOT_COMM_ID=172.31.63.174:62128)
thread 'tests::test_instance_1' panicked at 'assertion failed: `(left != right)`
  left: `0x0`,
 right: `0x0`', xla/src/xla_runner.rs:316:13
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
test tests::test_instance_1 ... FAILED
```
- The only difference between the tests on instance 0 and instance 1 are the global rank assigned to the instance's device.
- The above commands compile to NEFF and run the graph xla/rust_hlo_run_rank_0.pb, which does a single all reduce op with a single input. The debug representation of the .pb:
```
HloModule xla_computation_ordered_wrapper, entry_computation_layout={(f32[4]{0})->(f32[4]{0})}

region_0.6 (Arg_0.3: f32[], Arg_1.3: f32[]) -> f32[] {
  Arg_0.3 = f32[] parameter(0)
  Arg_1.3 = f32[] parameter(1)
  ROOT add.5 = f32[] add(Arg_0.3, Arg_1.3)
}

ENTRY main.9 (Arg_0.1: f32[4]) -> (f32[4]) {
  Arg_0.1 = f32[4]{0} parameter(0)
  all-reduce.2 = f32[4]{0} all-reduce(Arg_0.1), replica_groups={}, to_apply=region_0.6
  ROOT tuple.7 = (f32[4]{0}) tuple(all-reduce.2)
}
```
