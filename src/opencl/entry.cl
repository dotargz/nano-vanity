inline void generate_checksum (uchar checksum[5], const uchar pubkey[32]) {
	// For some reason, this doesn't work when put in generate_pubkey.
	blake2b_state state;
	blake2b_init (&state, 5);
	blake2b_update (&state, (__private uchar *) pubkey, 32);
	blake2b_final (&state, (__private uchar *) checksum, 5);
}

#define MAX_PREFIX_LEN 37

__kernel void generate_pubkey (__global unsigned long *result, __global const uchar *key_root, __constant const uchar *pub_req, __constant const uchar *pub_mask, uchar prefix_len, uchar generate_key_type, __constant const uchar *public_offset, uint batch_size) {
	if (batch_size == 0) {
		return;
	}
	uchar key[32];
	uchar16 key_chunk0 = vload16(0, key_root);
	uchar16 key_chunk1 = vload16(1, key_root);
	vstore16(key_chunk0, 0, key);
	vstore16(key_chunk1, 1, key);
	__local uchar req_local[MAX_PREFIX_LEN];
	__local uchar mask_local[MAX_PREFIX_LEN];
	__local ge25519 public_offset_curvepoint;
	size_t local_id = get_local_id(0);
	if (local_id == 0) {
		for (uchar i = 0; i < prefix_len; i++) {
			req_local[i] = pub_req[i];
			mask_local[i] = pub_mask[i];
		}
		if (generate_key_type == 2) {
			uchar public_offset_copy[32];
			for (size_t i = 0; i < 32; i++) {
				public_offset_copy[i] = public_offset[i];
			}
			ge25519_unpack_vartime(&public_offset_curvepoint, public_offset_copy);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
#if defined(__ENDIAN_BIG__)
	ulong key_base = ((ulong)key[0] << 56)
		| ((ulong)key[1] << 48)
		| ((ulong)key[2] << 40)
		| ((ulong)key[3] << 32)
		| ((ulong)key[4] << 24)
		| ((ulong)key[5] << 16)
		| ((ulong)key[6] << 8)
		| ((ulong)key[7]);
#else
	ulong key_base = ((ulong)key[0])
		| ((ulong)key[1] << 8)
		| ((ulong)key[2] << 16)
		| ((ulong)key[3] << 24)
		| ((ulong)key[4] << 32)
		| ((ulong)key[5] << 40)
		| ((ulong)key[6] << 48)
		| ((ulong)key[7] << 56);
#endif
	ulong const global_size = (ulong)get_global_size(0);
	ulong const thread_base = (ulong)get_global_id(0);
	for (uint batch = 0; batch < batch_size; batch++) {
		if (*result != ~0UL) {
			return;
		}
		ulong thread = thread_base + ((ulong)batch * global_size);
		ulong base = key_base + thread;
#if defined(__ENDIAN_BIG__)
		key[7] = (uchar)(base);
		key[6] = (uchar)(base >> 8);
		key[5] = (uchar)(base >> 16);
		key[4] = (uchar)(base >> 24);
		key[3] = (uchar)(base >> 32);
		key[2] = (uchar)(base >> 40);
		key[1] = (uchar)(base >> 48);
		key[0] = (uchar)(base >> 56);
#else
		key[0] = (uchar)(base);
		key[1] = (uchar)(base >> 8);
		key[2] = (uchar)(base >> 16);
		key[3] = (uchar)(base >> 24);
		key[4] = (uchar)(base >> 32);
		key[5] = (uchar)(base >> 40);
		key[6] = (uchar)(base >> 48);
		key[7] = (uchar)(base >> 56);
#endif
		uchar seed_key[32];
		const uchar *scalar_key = key;
		if (generate_key_type == 1) {
			// seed
			blake2b_state keystate;
			blake2b_init (&keystate, sizeof (seed_key));
			blake2b_update (&keystate, key, sizeof (key));
			uint32_t idx = 0;
			blake2b_update (&keystate, (uchar *) &idx, 4);
			blake2b_final (&keystate, seed_key, sizeof (seed_key));
			scalar_key = seed_key;
		}
		blake2b_state state;
		bignum256modm a;
		ge25519 ALIGN(16) A;
		if (generate_key_type != 2) {
			// key is an ed25519 private key
			uchar hash[64];
			blake2b_init (&state, sizeof (hash));
			blake2b_update (&state, (uchar *) scalar_key, 32);
			blake2b_final (&state, hash, sizeof (hash));
			hash[0] &= 248;
			hash[31] &= 127;
			hash[31] |= 64;
			expand256_modm(a, hash, 32);
		} else {
			// key is a scalar
			expand256_modm(a, (uchar *) scalar_key, 32);
		}
		ge25519_scalarmult_base_niels(&A, a);
		if (generate_key_type == 2) {
			ge25519_add(&A, &A, &public_offset_curvepoint);
		}
		uchar pubkey[32];
		ge25519_pack(pubkey, &A);
		uchar pubkey_prefix_len = prefix_len;
		if (pubkey_prefix_len > 32) {
			pubkey_prefix_len = 32;
		}
		for (uchar i = 0; i < pubkey_prefix_len; i++) {
			if ((pubkey[i] & mask_local[i]) != req_local[i]) {
				goto next_batch;
			}
		}
		if (prefix_len > 32) {
			uchar checksum[5];
			generate_checksum (checksum, pubkey);
			for (uchar i = 32; i < prefix_len; i++) {
				if ((checksum[4 - (i - 32)] & mask_local[i]) != req_local[i]) {
					goto next_batch;
				}
			}
		}
		*result = thread;
		return;
next_batch:
		continue;
	}
}
