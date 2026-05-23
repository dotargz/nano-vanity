inline void generate_checksum (uchar checksum[5], const uchar pubkey[32]) {
	// For some reason, this doesn't work when put in generate_pubkey.
	blake2b_state state;
	blake2b_init (&state, 5);
	blake2b_update (&state, (__private uchar *) pubkey, 32);
	blake2b_final (&state, (__private uchar *) checksum, 5);
}

#define MAX_PREFIX_LEN 37

inline bool prefix_matches(
	const uchar pubkey[32],
	const uchar *req_local,
	const uchar *mask_local,
	uchar bounded_prefix_len,
	uchar pubkey_prefix_len,
	bool needs_checksum
) {
	for (uchar i = 0; i < pubkey_prefix_len; i++) {
		if ((pubkey[i] & mask_local[i]) != req_local[i]) {
			return false;
		}
	}
	if (needs_checksum) {
		uchar checksum[5];
		generate_checksum (checksum, pubkey);
		for (uchar i = 32; i < bounded_prefix_len; i++) {
			if ((checksum[4 - (i - 32)] & mask_local[i]) != req_local[i]) {
				return false;
			}
		}
	}
	return true;
}

__kernel void generate_pubkey_private (__global unsigned long *result, __global const uchar *key_root, __constant const uchar *pub_req, __constant const uchar *pub_mask, uchar prefix_len, uint iterations) {
	ulong const thread = (ulong)get_global_id (0);
	uchar seed[32];
	uchar req_local[MAX_PREFIX_LEN];
	uchar mask_local[MAX_PREFIX_LEN];
	uchar16 key_chunk0 = vload16(0, key_root);
	uchar16 key_chunk1 = vload16(1, key_root);
	vstore16(key_chunk0, 0, seed);
	vstore16(key_chunk1, 1, seed);
#if defined(__ENDIAN_BIG__)
	ulong base_seed = ((ulong)seed[0] << 56)
		| ((ulong)seed[1] << 48)
		| ((ulong)seed[2] << 40)
		| ((ulong)seed[3] << 32)
		| ((ulong)seed[4] << 24)
		| ((ulong)seed[5] << 16)
		| ((ulong)seed[6] << 8)
		| ((ulong)seed[7]);
#else
	ulong base_seed = ((ulong)seed[0])
		| ((ulong)seed[1] << 8)
		| ((ulong)seed[2] << 16)
		| ((ulong)seed[3] << 24)
		| ((ulong)seed[4] << 32)
		| ((ulong)seed[5] << 40)
		| ((ulong)seed[6] << 48)
		| ((ulong)seed[7] << 56);
#endif
	ulong stride = (ulong)get_global_size(0);
	ulong offset = thread;
	uchar bounded_prefix_len = prefix_len;
	if (bounded_prefix_len > MAX_PREFIX_LEN) {
		bounded_prefix_len = MAX_PREFIX_LEN;
	}
	uchar pubkey_prefix_len = bounded_prefix_len;
	if (pubkey_prefix_len > 32) {
		pubkey_prefix_len = 32;
	}
	const bool needs_checksum = bounded_prefix_len > 32;
	for (uchar i = 0; i < bounded_prefix_len; i++) {
		req_local[i] = pub_req[i];
		mask_local[i] = pub_mask[i];
	}
	for (uint iter = 0; iter < iterations; iter++) {
		ulong base = base_seed + offset;
#if defined(__ENDIAN_BIG__)
		seed[7] = (uchar)(base);
		seed[6] = (uchar)(base >> 8);
		seed[5] = (uchar)(base >> 16);
		seed[4] = (uchar)(base >> 24);
		seed[3] = (uchar)(base >> 32);
		seed[2] = (uchar)(base >> 40);
		seed[1] = (uchar)(base >> 48);
		seed[0] = (uchar)(base >> 56);
#else
		seed[0] = (uchar)(base);
		seed[1] = (uchar)(base >> 8);
		seed[2] = (uchar)(base >> 16);
		seed[3] = (uchar)(base >> 24);
		seed[4] = (uchar)(base >> 32);
		seed[5] = (uchar)(base >> 40);
		seed[6] = (uchar)(base >> 48);
		seed[7] = (uchar)(base >> 56);
#endif
		blake2b_state state;
		bignum256modm a;
		ge25519 ALIGN(16) A;
		// key is an ed25519 private key
		uchar hash[64];
		blake2b_init (&state, sizeof (hash));
		blake2b_update (&state, seed, 32);
		blake2b_final (&state, hash, sizeof (hash));
		hash[0] &= 248;
		hash[31] &= 127;
		hash[31] |= 64;
		expand256_modm(a, hash, 32);
		ge25519_scalarmult_base_niels(&A, a);
		uchar pubkey[32];
		ge25519_pack(pubkey, &A);
		if (prefix_matches(pubkey, req_local, mask_local, bounded_prefix_len, pubkey_prefix_len, needs_checksum)) {
			*result = offset;
			return;
		}
		offset += stride;
	}
}

__kernel void generate_pubkey_seed (__global unsigned long *result, __global const uchar *key_root, __constant const uchar *pub_req, __constant const uchar *pub_mask, uchar prefix_len, uint iterations) {
	ulong const thread = (ulong)get_global_id (0);
	uchar seed[32];
	uchar key[32];
	uchar req_local[MAX_PREFIX_LEN];
	uchar mask_local[MAX_PREFIX_LEN];
	uchar16 key_chunk0 = vload16(0, key_root);
	uchar16 key_chunk1 = vload16(1, key_root);
	vstore16(key_chunk0, 0, seed);
	vstore16(key_chunk1, 1, seed);
#if defined(__ENDIAN_BIG__)
	ulong base_seed = ((ulong)seed[0] << 56)
		| ((ulong)seed[1] << 48)
		| ((ulong)seed[2] << 40)
		| ((ulong)seed[3] << 32)
		| ((ulong)seed[4] << 24)
		| ((ulong)seed[5] << 16)
		| ((ulong)seed[6] << 8)
		| ((ulong)seed[7]);
#else
	ulong base_seed = ((ulong)seed[0])
		| ((ulong)seed[1] << 8)
		| ((ulong)seed[2] << 16)
		| ((ulong)seed[3] << 24)
		| ((ulong)seed[4] << 32)
		| ((ulong)seed[5] << 40)
		| ((ulong)seed[6] << 48)
		| ((ulong)seed[7] << 56);
#endif
	ulong stride = (ulong)get_global_size(0);
	ulong offset = thread;
	uchar bounded_prefix_len = prefix_len;
	if (bounded_prefix_len > MAX_PREFIX_LEN) {
		bounded_prefix_len = MAX_PREFIX_LEN;
	}
	uchar pubkey_prefix_len = bounded_prefix_len;
	if (pubkey_prefix_len > 32) {
		pubkey_prefix_len = 32;
	}
	const bool needs_checksum = bounded_prefix_len > 32;
	for (uchar i = 0; i < bounded_prefix_len; i++) {
		req_local[i] = pub_req[i];
		mask_local[i] = pub_mask[i];
	}
	const uint32_t seed_address_index = 0;
	for (uint iter = 0; iter < iterations; iter++) {
		ulong base = base_seed + offset;
#if defined(__ENDIAN_BIG__)
		seed[7] = (uchar)(base);
		seed[6] = (uchar)(base >> 8);
		seed[5] = (uchar)(base >> 16);
		seed[4] = (uchar)(base >> 24);
		seed[3] = (uchar)(base >> 32);
		seed[2] = (uchar)(base >> 40);
		seed[1] = (uchar)(base >> 48);
		seed[0] = (uchar)(base >> 56);
#else
		seed[0] = (uchar)(base);
		seed[1] = (uchar)(base >> 8);
		seed[2] = (uchar)(base >> 16);
		seed[3] = (uchar)(base >> 24);
		seed[4] = (uchar)(base >> 32);
		seed[5] = (uchar)(base >> 40);
		seed[6] = (uchar)(base >> 48);
		seed[7] = (uchar)(base >> 56);
#endif
		blake2b_state keystate;
		blake2b_init (&keystate, sizeof (seed));
		blake2b_update (&keystate, seed, sizeof (seed));
		blake2b_update (&keystate, (uchar *) &seed_address_index, 4);
		blake2b_final (&keystate, key, sizeof (key));
		blake2b_state state;
		bignum256modm a;
		ge25519 ALIGN(16) A;
		uchar hash[64];
		blake2b_init (&state, sizeof (hash));
		blake2b_update (&state, key, 32);
		blake2b_final (&state, hash, sizeof (hash));
		hash[0] &= 248;
		hash[31] &= 127;
		hash[31] |= 64;
		expand256_modm(a, hash, 32);
		ge25519_scalarmult_base_niels(&A, a);
		uchar pubkey[32];
		ge25519_pack(pubkey, &A);
		if (prefix_matches(pubkey, req_local, mask_local, bounded_prefix_len, pubkey_prefix_len, needs_checksum)) {
			*result = offset;
			return;
		}
		offset += stride;
	}
}

__kernel void generate_pubkey_extended (__global unsigned long *result, __global const uchar *key_root, __constant const uchar *pub_req, __constant const uchar *pub_mask, uchar prefix_len, uint iterations, __constant const uchar *public_offset) {
	ulong const thread = (ulong)get_global_id (0);
	uchar seed[32];
	uchar req_local[MAX_PREFIX_LEN];
	uchar mask_local[MAX_PREFIX_LEN];
	uchar16 key_chunk0 = vload16(0, key_root);
	uchar16 key_chunk1 = vload16(1, key_root);
	vstore16(key_chunk0, 0, seed);
	vstore16(key_chunk1, 1, seed);
#if defined(__ENDIAN_BIG__)
	ulong base_seed = ((ulong)seed[0] << 56)
		| ((ulong)seed[1] << 48)
		| ((ulong)seed[2] << 40)
		| ((ulong)seed[3] << 32)
		| ((ulong)seed[4] << 24)
		| ((ulong)seed[5] << 16)
		| ((ulong)seed[6] << 8)
		| ((ulong)seed[7]);
#else
	ulong base_seed = ((ulong)seed[0])
		| ((ulong)seed[1] << 8)
		| ((ulong)seed[2] << 16)
		| ((ulong)seed[3] << 24)
		| ((ulong)seed[4] << 32)
		| ((ulong)seed[5] << 40)
		| ((ulong)seed[6] << 48)
		| ((ulong)seed[7] << 56);
#endif
	ulong stride = (ulong)get_global_size(0);
	ulong offset = thread;
	uchar bounded_prefix_len = prefix_len;
	if (bounded_prefix_len > MAX_PREFIX_LEN) {
		bounded_prefix_len = MAX_PREFIX_LEN;
	}
	uchar pubkey_prefix_len = bounded_prefix_len;
	if (pubkey_prefix_len > 32) {
		pubkey_prefix_len = 32;
	}
	const bool needs_checksum = bounded_prefix_len > 32;
	for (uchar i = 0; i < bounded_prefix_len; i++) {
		req_local[i] = pub_req[i];
		mask_local[i] = pub_mask[i];
	}
	ge25519 ALIGN(16) public_offset_curvepoint;
	uchar public_offset_copy[32];
	uchar16 offset_chunk0 = vload16(0, public_offset);
	uchar16 offset_chunk1 = vload16(1, public_offset);
	vstore16(offset_chunk0, 0, public_offset_copy);
	vstore16(offset_chunk1, 1, public_offset_copy);
	ge25519_unpack_vartime(&public_offset_curvepoint, public_offset_copy);
	for (uint iter = 0; iter < iterations; iter++) {
		ulong base = base_seed + offset;
#if defined(__ENDIAN_BIG__)
		seed[7] = (uchar)(base);
		seed[6] = (uchar)(base >> 8);
		seed[5] = (uchar)(base >> 16);
		seed[4] = (uchar)(base >> 24);
		seed[3] = (uchar)(base >> 32);
		seed[2] = (uchar)(base >> 40);
		seed[1] = (uchar)(base >> 48);
		seed[0] = (uchar)(base >> 56);
#else
		seed[0] = (uchar)(base);
		seed[1] = (uchar)(base >> 8);
		seed[2] = (uchar)(base >> 16);
		seed[3] = (uchar)(base >> 24);
		seed[4] = (uchar)(base >> 32);
		seed[5] = (uchar)(base >> 40);
		seed[6] = (uchar)(base >> 48);
		seed[7] = (uchar)(base >> 56);
#endif
		bignum256modm a;
		ge25519 ALIGN(16) A;
		expand256_modm(a, seed, 32);
		ge25519_scalarmult_base_niels(&A, a);
		ge25519_add(&A, &A, &public_offset_curvepoint);
		uchar pubkey[32];
		ge25519_pack(pubkey, &A);
		if (prefix_matches(pubkey, req_local, mask_local, bounded_prefix_len, pubkey_prefix_len, needs_checksum)) {
			*result = offset;
			return;
		}
		offset += stride;
	}
}
