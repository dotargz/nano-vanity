use byteorder::ByteOrder;
use byteorder::NativeEndian;
use ocl;
use ocl::builders::DeviceSpecifier;
use ocl::builders::ProgramBuilder;
use ocl::core::CommandQueueProperties;
use ocl::core::ProfilingInfo;
use ocl::core::ProfilingInfoResult;
use ocl::flags::MemFlags;
use ocl::Buffer;
use ocl::Event;
use ocl::Platform;
use ocl::ProQue;
use ocl::Queue;
use ocl::Result;
use std::cmp;
use std::time::Instant;

use derivation::GenerateKeyType;
use gpu::GpuOptions;

// 256 is a common NVIDIA warp-multiple default (8 warps) for good occupancy.
const NVIDIA_DEFAULT_LOCAL_WORK_SIZE: usize = 256;
const NVIDIA_DEFAULT_BATCH_SIZE: usize = 32;
const AMD_DEFAULT_LOCAL_WORK_SIZE: usize = 256;
const AMD_DEFAULT_BATCH_SIZE: usize = 16;
const DEFAULT_BATCH_SIZE: usize = 8;
const BATCH_SIZE_ARG_INDEX: u32 = 7;

fn align_global_work_size(global_work_size: usize, local_work_size: usize) -> usize {
    if global_work_size % local_work_size == 0 {
        global_work_size
    } else {
        ((global_work_size + local_work_size - 1) / local_work_size) * local_work_size
    }
}

fn default_batch_size(vendor_lower: &str) -> usize {
    if vendor_lower.contains("nvidia") {
        NVIDIA_DEFAULT_BATCH_SIZE
    } else if vendor_lower.contains("amd") || vendor_lower.contains("advanced micro devices") {
        AMD_DEFAULT_BATCH_SIZE
    } else {
        DEFAULT_BATCH_SIZE
    }
}

fn candidate_batch_sizes(vendor_lower: &str) -> Vec<usize> {
    if vendor_lower.contains("nvidia") {
        vec![8, 16, 32, 64]
    } else if vendor_lower.contains("amd") || vendor_lower.contains("advanced micro devices") {
        vec![4, 8, 16, 32]
    } else {
        vec![2, 4, 8, 16]
    }
}

fn candidate_local_work_sizes(
    vendor_lower: &str,
    max_wg_size: Option<usize>,
) -> Vec<usize> {
    let mut candidates = if vendor_lower.contains("nvidia") {
        vec![64, 128, 256, 512]
    } else if vendor_lower.contains("amd") || vendor_lower.contains("advanced micro devices") {
        vec![64, 128, 256, 512]
    } else {
        vec![32, 64, 128, 256]
    };
    if let Some(max) = max_wg_size {
        candidates.retain(|&size| size <= max);
    }
    candidates.sort();
    candidates.dedup();
    candidates
}

fn event_duration_ms(event: &Event) -> Result<f64> {
    let start = match event.profiling_info(ProfilingInfo::Start)? {
        ProfilingInfoResult::Start(value) => value,
        _ => return Err("Unexpected profiling start result".into()),
    };
    let end = match event.profiling_info(ProfilingInfo::End)? {
        ProfilingInfoResult::End(value) => value,
        _ => return Err("Unexpected profiling end result".into()),
    };
    Ok((end.saturating_sub(start)) as f64 / 1_000_000.0)
}

fn benchmark_kernel(
    kernel: &ocl::Kernel,
    result: &Buffer<u64>,
    queue: &Queue,
    global_work_size: usize,
    local_work_size: Option<usize>,
    batch_size: usize,
) -> Result<f64> {
    kernel.set_arg(BATCH_SIZE_ARG_INDEX, batch_size as u32)?;
    result.write(&[!0u64] as &[u64]).enq()?;
    let mut cmd = kernel.cmd().global_work_size(global_work_size);
    if let Some(local_work_size) = local_work_size {
        cmd = cmd.local_work_size(local_work_size);
    }
    let start = Instant::now();
    unsafe {
        cmd.enq()?;
    }
    let mut buf = [0u64];
    result.read(&mut buf as &mut [u64]).enq()?;
    queue.finish()?;
    Ok(start.elapsed().as_secs_f64())
}

pub struct Gpu {
    kernel: ocl::Kernel,
    result: Buffer<u64>,
    key_root: Buffer<u8>,
    queue: Queue,
    global_work_size: usize,
    local_work_size: Option<usize>,
    batch_size: usize,
    profile: bool,
}

impl Gpu {
    pub fn new(opts: GpuOptions) -> Result<Gpu> {
        let mut prog_bldr = ProgramBuilder::new();
        let namespace_qualifier = if cfg!(feature = "apple") {
            "#define NAMESPACE_QUALIFIER __private\n"
        } else {
            "#if defined(__opencl_c_generic_address_space) || (defined(__OPENCL_C_VERSION__) && (__OPENCL_C_VERSION__ >= 200))\n\
#define NAMESPACE_QUALIFIER __generic\n\
#else\n\
#define NAMESPACE_QUALIFIER __private\n\
#endif\n"
        };
        prog_bldr
            .source(namespace_qualifier)
            .src(include_str!("opencl/blake2b.cl"))
            .src(include_str!("opencl/curve25519-constants.cl"))
            .src(include_str!("opencl/curve25519-constants2.cl"))
            .src(include_str!("opencl/curve25519.cl"))
            .src(include_str!("opencl/entry.cl"));
        let platforms = Platform::list();
        if platforms.len() == 0 {
            return Err("No OpenCL platforms exist (check your drivers and OpenCL setup)".into());
        }
        if opts.platform_idx >= platforms.len() {
            return Err(format!(
                "Platform index {} too large (max {})",
                opts.platform_idx,
                platforms.len() - 1
            )
            .into());
        }
        let mut pro_que_builder = ProQue::builder();
        pro_que_builder
            .prog_bldr(prog_bldr)
            .platform(platforms[opts.platform_idx])
            .device(DeviceSpecifier::Indices(vec![opts.device_idx]))
            .dims(1);
        if opts.profile || opts.autotune {
            pro_que_builder.queue_properties(CommandQueueProperties::new().profiling());
        }
        let mut pro_que = pro_que_builder.build()?;
        let queue = pro_que.queue().clone();

        let device = pro_que.device();
        let vendor = device.vendor()?;
        let vendor_lower = vendor.to_lowercase();
        let name = device.name()?;
        eprintln!("Initializing GPU {} {}", vendor, name);
        let max_wg_size = device.max_wg_size().ok();
        let mut global_work_size = opts.global_work_size.unwrap_or(opts.threads);
        let mut local_work_size = opts.local_work_size;
        if local_work_size.is_none() {
            if vendor_lower.contains("nvidia") {
                if let Some(max_wg_size) = max_wg_size {
                    let candidate = cmp::min(NVIDIA_DEFAULT_LOCAL_WORK_SIZE, max_wg_size);
                    if candidate > 0 {
                        local_work_size = Some(candidate);
                    }
                }
            } else if vendor_lower.contains("amd")
                || vendor_lower.contains("advanced micro devices")
            {
                if let Some(max_wg_size) = max_wg_size {
                    let candidate = cmp::min(AMD_DEFAULT_LOCAL_WORK_SIZE, max_wg_size);
                    if candidate > 0 {
                        local_work_size = Some(candidate);
                    }
                }
            }
        }
        let mut batch_size = opts.batch_size.unwrap_or(default_batch_size(&vendor_lower));
        if batch_size == 0 {
            return Err("GPU batch size must be greater than zero".into());
        }
        if let Some(local_work_size) = local_work_size {
            if global_work_size % local_work_size != 0 {
                let aligned = align_global_work_size(global_work_size, local_work_size);
                eprintln!(
                    "Adjusting global work size from {} to {} to match local work size {}",
                    global_work_size, aligned, local_work_size
                );
                global_work_size = aligned;
            }
        }

        let result = pro_que
            .buffer_builder::<u64>()
            .flags(MemFlags::new().write_only())
            .build()?;
        pro_que.set_dims(64);
        let key_root = pro_que
            .buffer_builder::<u8>()
            .flags(MemFlags::new().read_only().host_write_only())
            .build()?;
        pro_que.set_dims(opts.matcher.prefix_len());
        let req = pro_que
            .buffer_builder::<u8>()
            .flags(MemFlags::new().read_only().host_write_only())
            .build()?;
        let mask = pro_que
            .buffer_builder::<u8>()
            .flags(MemFlags::new().read_only().host_write_only())
            .build()?;
        pro_que.set_dims(32);
        let public_offset = pro_que
            .buffer_builder::<u8>()
            .flags(MemFlags::new().read_only().host_write_only())
            .build()?;
        pro_que.set_dims(1);

        req.write(opts.matcher.req()).enq()?;
        mask.write(opts.matcher.mask()).enq()?;
        result.write(&[!0u64] as &[u64]).enq()?;
        key_root.write(&[0u8; 32] as &[u8]).enq()?;
        let gen_key_type_code: u8 = match opts.generate_key_type {
            GenerateKeyType::PrivateKey => 0,
            GenerateKeyType::Seed => 1,
            GenerateKeyType::ExtendedPrivateKey(offset) => {
                let compressed = offset.compress();
                public_offset.write(compressed.as_bytes() as &[u8]).enq()?;
                2
            }
        };

        let kernel = {
            let mut kernel_builder = pro_que.kernel_builder("generate_pubkey");
            kernel_builder
                .arg(&result)
                .arg(&key_root)
                .arg(&req)
                .arg(&mask)
                .arg(opts.matcher.prefix_len() as u8)
                .arg(gen_key_type_code)
                .arg(&public_offset)
                .arg(batch_size as u32);
            kernel_builder.build()?
        };

        if opts.autotune {
            let mut global_candidates = vec![global_work_size];
            if opts.global_work_size.is_none() {
                if global_work_size > 1 {
                    global_candidates.push(global_work_size / 2);
                }
                global_candidates.push(global_work_size.saturating_mul(2));
            }
            global_candidates.retain(|&value| value > 0);
            global_candidates.sort();
            global_candidates.dedup();

            let mut local_candidates = Vec::new();
            if opts.local_work_size.is_some() {
                local_candidates.push(local_work_size);
            } else {
                local_candidates.push(None);
                for size in candidate_local_work_sizes(&vendor_lower, max_wg_size) {
                    local_candidates.push(Some(size));
                }
            }
            if local_candidates.is_empty() {
                local_candidates.push(None);
            }

            let mut batch_candidates = if opts.batch_size.is_some() {
                vec![batch_size]
            } else {
                candidate_batch_sizes(&vendor_lower)
            };
            batch_candidates.retain(|&value| value > 0);
            batch_candidates.sort();
            batch_candidates.dedup();

            let mut best_throughput = 0.0;
            let mut best_global = global_work_size;
            let mut best_local = local_work_size;
            let mut best_batch = batch_size;
            for global_candidate in global_candidates {
                for &local_candidate in &local_candidates {
                    let aligned_global = match local_candidate {
                        Some(local_value) => align_global_work_size(global_candidate, local_value),
                        None => global_candidate,
                    };
                    for &batch_candidate in &batch_candidates {
                        let duration = benchmark_kernel(
                            &kernel,
                            &result,
                            &queue,
                            aligned_global,
                            local_candidate,
                            batch_candidate,
                        )?;
                        let attempts =
                            (aligned_global as f64) * (batch_candidate as f64);
                        let throughput = attempts / duration.max(1e-9);
                        if throughput > best_throughput {
                            best_throughput = throughput;
                            best_global = aligned_global;
                            best_local = local_candidate;
                            best_batch = batch_candidate;
                        }
                    }
                }
            }
            global_work_size = best_global;
            local_work_size = best_local;
            batch_size = best_batch;
            kernel.set_arg(BATCH_SIZE_ARG_INDEX, batch_size as u32)?;
            result.write(&[!0u64] as &[u64]).enq()?;
            eprintln!(
                "Autotune selected global work size {}, local work size {:?}, batch size {}",
                global_work_size, local_work_size, batch_size
            );
        }

        Ok(Gpu {
            kernel,
            result,
            key_root,
            queue,
            global_work_size,
            local_work_size,
            batch_size,
            profile: opts.profile,
        })
    }

    pub fn compute(&mut self, out: &mut [u8], key_root: &[u8]) -> Result<bool> {
        self.key_root.write(key_root).enq()?;
        debug_assert!(out.iter().all(|&b| b == 0));
        debug_assert!({
            let mut result = [0u64];
            self.result.read(&mut result as &mut [u64]).enq()?;
            result == [!0u64]
        });

        unsafe {
            self.kernel.enq()?;
        }

        let mut buf = [0u64];
        self.result.read(&mut buf as &mut [u64]).enq()?;
        let thread = buf[0];
        let success = thread != !0u64;
        if success {
            self.result.write(&[!0u64] as &[u64]).enq()?;
            let base = NativeEndian::read_u64(key_root);
            NativeEndian::write_u64(out, base.wrapping_add(thread));
            out[8..].copy_from_slice(&key_root[8..]);
        }
        Ok(success)
    }

    pub fn global_work_size(&self) -> usize {
        self.global_work_size
    }
}
