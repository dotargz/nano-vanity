use byteorder::ByteOrder;
use byteorder::NativeEndian;
use ocl;
use ocl::builders::DeviceSpecifier;
use ocl::builders::ProgramBuilder;
use ocl::flags::MemFlags;
use ocl::Buffer;
use ocl::Platform;
use ocl::ProQue;
use ocl::Result;
use std::cmp;

use derivation::GenerateKeyType;
use gpu::GpuOptions;

// 256 is a common NVIDIA warp-multiple default (8 warps) for good occupancy.
const NVIDIA_DEFAULT_LOCAL_WORK_SIZE: usize = 256;
const MAX_PREFIX_LEN: usize = 37;
const PUBLIC_OFFSET_LEN: usize = 32;

pub struct Gpu {
    kernel: ocl::Kernel,
    result: Buffer<u64>,
    key_root: Buffer<u8>,
    global_work_size: usize,
    iterations: usize,
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
        let mut pro_que = ProQue::builder()
            .prog_bldr(prog_bldr)
            .platform(platforms[opts.platform_idx])
            .device(DeviceSpecifier::Indices(vec![opts.device_idx]))
            .dims(1)
            .build()?;

        let device = pro_que.device();
        let vendor = device.vendor()?;
        let name = device.name()?;
        eprintln!("Initializing GPU {} {}", vendor, name);
        let mut global_work_size = opts.global_work_size.unwrap_or(opts.threads);
        let mut local_work_size = opts.local_work_size;
        let mut iterations = opts.iterations.unwrap_or_else(|| {
            if vendor.to_lowercase().contains("nvidia") {
                4
            } else {
                1
            }
        });
        if iterations == 0 {
            iterations = 1;
        }
        if local_work_size.is_none() && vendor.to_lowercase().contains("nvidia") {
            if let Ok(max_wg_size) = device.max_wg_size() {
                let candidate = cmp::min(NVIDIA_DEFAULT_LOCAL_WORK_SIZE, max_wg_size);
                if candidate > 0 {
                    local_work_size = Some(candidate);
                }
            }
        }
        if let Some(local_work_size) = local_work_size {
            if global_work_size % local_work_size != 0 {
                let aligned =
                    ((global_work_size + local_work_size - 1) / local_work_size) * local_work_size;
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
        pro_que.set_dims(MAX_PREFIX_LEN);
        let req_buffer = pro_que
            .buffer_builder::<u8>()
            .flags(MemFlags::new().read_only().host_write_only())
            .build()?;
        let mask_buffer = pro_que
            .buffer_builder::<u8>()
            .flags(MemFlags::new().read_only().host_write_only())
            .build()?;
        pro_que.set_dims(PUBLIC_OFFSET_LEN);
        let public_offset = pro_que
            .buffer_builder::<u8>()
            .flags(MemFlags::new().read_only().host_write_only())
            .build()?;
        pro_que.set_dims(1);

        let mut req_padded = vec![0u8; MAX_PREFIX_LEN];
        let mut mask_padded = vec![0u8; MAX_PREFIX_LEN];
        let req_slice = opts.matcher.req();
        let mask_slice = opts.matcher.mask();
        let prefix_len = cmp::min(
            cmp::min(req_slice.len(), mask_slice.len()),
            MAX_PREFIX_LEN,
        );
        req_padded[..prefix_len].copy_from_slice(&req_slice[..prefix_len]);
        mask_padded[..prefix_len].copy_from_slice(&mask_slice[..prefix_len]);
        req_buffer.write(&req_padded).enq()?;
        mask_buffer.write(&mask_padded).enq()?;
        result.write(&[!0u64] as &[u64]).enq()?;
        let kernel_name = match opts.generate_key_type {
            GenerateKeyType::PrivateKey => "generate_pubkey_private",
            GenerateKeyType::Seed => "generate_pubkey_seed",
            GenerateKeyType::ExtendedPrivateKey(offset) => {
                let compressed = offset.compress();
                public_offset
                    .write(compressed.as_bytes() as &[u8])
                    .enq()?;
                "generate_pubkey_extended"
            }
        };
        if let GenerateKeyType::ExtendedPrivateKey(_) = opts.generate_key_type {
        } else {
            public_offset
                .write(&[0u8; PUBLIC_OFFSET_LEN] as &[u8])
                .enq()?;
        }

        let kernel = {
            let mut kernel_builder = pro_que.kernel_builder(kernel_name);
            kernel_builder
                .global_work_size(global_work_size)
                .arg(&result)
                .arg(&key_root)
                .arg(&req_buffer)
                .arg(&mask_buffer)
                .arg(prefix_len as u8)
                .arg(iterations as u32)
                .arg(&public_offset);
            if let Some(local_work_size) = local_work_size {
                kernel_builder.local_work_size(local_work_size);
            }
            kernel_builder.build()?
        };

        Ok(Gpu {
            kernel,
            result,
            key_root,
            global_work_size,
            iterations,
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
        let offset = buf[0];
        let success = offset != !0u64;
        if success {
            self.result.write(&[!0u64] as &[u64]).enq()?;
            let base = NativeEndian::read_u64(key_root);
            NativeEndian::write_u64(out, base.wrapping_add(offset));
            out[8..].copy_from_slice(&key_root[8..]);
        }
        Ok(success)
    }

    pub fn work_per_call(&self) -> usize {
        self.global_work_size.saturating_mul(self.iterations)
    }
}
