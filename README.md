# README

<<<<<<< HEAD
This code accompanies Private Analytics via Streaming, Sketching, and Silently Verifiable Proofs.

=======
>>>>>>> bfc8c8430962ee77cf9e36a57ac10bd52027f256
## WARNING: This is not production-ready code.

This is software for a research prototype. Please
do *NOT* use this code in production.

We have tested this code with:
>  rustc 1.71.0 (8ede3aae2 2023-07-12)

## Getting started - Heavy Hitters

In one shell, start the first server with the following. 

```
$ cargo run --release --package server-hh -- --config configs/hh-server-alice.json
```

In a different shell on the same machine, start the second server with the folloiwng
```
$ cargo run --release --package server-hh -- --config configs/hh-server-bob.json
```

Now, the servers should be ready to process client requests. In a third shell, run the following command to start a metaclient with the following

```
$ cargo run --release --package client-hh -- --config configs/hh-client.json
```

## Heavy Hitters server config:

* `client_port`: what port to listen to clients on.
* `num_clients`: how many clients to expect.
* `is_bob`: Set this to true for Bob, and false for Alice.
* `mpc_addr`: if I'm Alice, this is the port number to expose to the other server. If I'm Bob, this is a complete address, including port number of Alice. 
* `num_mpc_sockets`: how many sockets to use during my mpc with my peer.
* `recovery_threshold`: if a certain string appears more than a `recovery_threshold` fraction of the time, then we want to recover it as a heavy hitter.
* `batch_size`: only used for streaming. This is how many client submissions the server will hold in memory at once.
* `num_bad_clients`: How many malicious clients to expect. Malicious clients will still be identified if this number is off, but having a good estimate will make group testing more efficient.
* `log_level`: How verbose the output should be. `debug` is the most verbose, and `none` only outputs the heavy hitters.

Note: To run in streaming configuration, run with the `streaming` feature enabled. 

The client config is very similar, except: 
* `alice`: alice's ip:port 
* `bob`: bob's ip:port

## Getting Started - Prio3 / batched prio3

To start the first server, run the following.
```
$ cargo run --release --package server-batch-prio3 -- --config  configs/prio3-server-alice.json
```

To start the second server, run the following in a separate terminal.
```
$ cargo run --release --package server-batch-prio3 -- --config  configs/prio3-server-bob.json
```

Now, the servers should be ready to process client requests. In a third shell, run the following to start a meta-client.

```
$ cargo run --release --package client-batch-prio3 -- --config configs/prio3-client.json
```

## Prio3 server config:

Very similar to hh config. Some big differences: 
* No more `recovery_threshold` 
* `chunk_size`: Prio3 chunk size, for vector sum. See paper for details 
* `vec_size`: Prio3 vec size, for vector sum task.
* `agg_fn`: Which aggregation to use with Prio. `sv` for Vector Sum, `hs` for histogram, and `av` for average.
<<<<<<< HEAD
* `single_tag`: if true, then we perform group testing over all client submissions at once. If false, then we split client keys into `NUM_CORES` chunks, and do group testing over them in parallel. Enabling `single_tag` reduces server-server communication at the cost of higher runtime. The results in the paper were run with this parameter set to false.
=======
* `single_tag`: if true, then we perform group testing over all client submissions at once. if false, then we split client keys into `NUM_CORES` chunks, and do group testing over them in parallel. Enabling `single_tag` reduces server-server communication at the cost of higher runtime.   
>>>>>>> bfc8c8430962ee77cf9e36a57ac10bd52027f256

Note: to run base prio3, replace `server-batch-prio3` and `client-batch-prio3` with `server-base-prio3` and `client-base-prio3` respectively
