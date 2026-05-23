inline void generate_checksum (uchar checksum[5], const uchar pubkey[32]) {
	// For some reason, this doesn't work when put in generate_pubkey.
	blake2b_state state;
	blake2b_init (&state, 5);
	blake2b_update (&state, (__private uchar *) pubkey, 32);
	blake2b_final (&state, (__private uchar *) checksum, 5);
}

__kernel void generate_pubkey (__global unsigned long *result, __global const uchar *key_root, __global const uchar *pub_req, __global const uchar *pub_mask, uchar prefix_len, uchar generate_key_type, __global const uchar *public_offset) {
	ulong const thread = (ulong)get_global_id (0);
	uchar key[32];
	uchar16 key_chunk0 = vload16(0, key_root);
	uchar16 key_chunk1 = vload16(1, key_root);
	vstore16(key_chunk0, 0, key);
	vstore16(key_chunk1, 1, key);
#if defined(__ENDIAN_LITTLE__)
	ulong base = ((ulong)key[0])
		| ((ulong)key[1] << 8)
		| ((ulong)key[2] << 16)
		| ((ulong)key[3] << 24)
		| ((ulong)key[4] << 32)
		| ((ulong)key[5] << 40)
		| ((ulong)key[6] << 48)
		| ((ulong)key[7] << 56);
	base += thread;
	key[0] = (uchar)(base);
	key[1] = (uchar)(base >> 8);
	key[2] = (uchar)(base >> 16);
	key[3] = (uchar)(base >> 24);
	key[4] = (uchar)(base >> 32);
	key[5] = (uchar)(base >> 40);
	key[6] = (uchar)(base >> 48);
	key[7] = (uchar)(base >> 56);
#else
	ulong base = ((ulong)key[0] << 56)
		| ((ulong)key[1] << 48)
		| ((ulong)key[2] << 40)
		| ((ulong)key[3] << 32)
		| ((ulong)key[4] << 24)
		| ((ulong)key[5] << 16)
		| ((ulong)key[6] << 8)
		| ((ulong)key[7]);
	base += thread;
	key[7] = (uchar)(base);
	key[6] = (uchar)(base >> 8);
	key[5] = (uchar)(base >> 16);
	key[4] = (uchar)(base >> 24);
	key[3] = (uchar)(base >> 32);
	key[2] = (uchar)(base >> 40);
	key[1] = (uchar)(base >> 48);
	key[0] = (uchar)(base >> 56);
#endif
	if (generate_key_type == 1) {
		// seed
		blake2b_state keystate;
		blake2b_init (&keystate, sizeof (key));
		blake2b_update (&keystate, key, sizeof (key));
		uint32_t idx = 0;
		blake2b_update (&keystate, (uchar *) &idx, 4);
		blake2b_final (&keystate, key, sizeof (key));
	}
	blake2b_state state;
	bignum256modm a;
	ge25519 ALIGN(16) A;
	if (generate_key_type != 2) {
		// key is an ed25519 private key
		uchar hash[64];
		blake2b_init (&state, sizeof (hash));
		blake2b_update (&state, key, 32);
		blake2b_final (&state, hash, sizeof (hash));
		hash[0] &= 248;
		hash[31] &= 127;
		hash[31] |= 64;
		expand256_modm(a, hash, 32);
	} else {
		// key is a scalar
		expand256_modm(a, key, 32);
	}
	ge25519_scalarmult_base_niels(&A, a);
	if (generate_key_type == 2) {
		uchar public_offset_copy[32];
		for (size_t i = 0; i < 32; i++) {
			public_offset_copy[i] = public_offset[i];
		}
		ge25519 ALIGN(16) public_offset_curvepoint;
		ge25519_unpack_vartime(&public_offset_curvepoint, public_offset_copy);
		ge25519_add(&A, &A, &public_offset_curvepoint);
	}
	uchar pubkey[32];
	ge25519_pack(pubkey, &A);
	uchar pubkey_prefix_len = prefix_len;
	if (pubkey_prefix_len > 32) {
		pubkey_prefix_len = 32;
	}
	for (uchar i = 0; i < pubkey_prefix_len; i++) {
		if ((pubkey[i] & pub_mask[i]) != pub_req[i]) {
			return;
		}
	}
	if (prefix_len > 32) {
		uchar checksum[5];
		generate_checksum (checksum, pubkey);
		for (uchar i = 32; i < prefix_len; i++) {
			if ((checksum[4 - (i - 32)] & pub_mask[i]) != pub_req[i]) {
				return;
			}
		}
	}
	*result = thread;
}
