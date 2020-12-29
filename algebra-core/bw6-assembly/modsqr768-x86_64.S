// void modsqr768(const uint64_t a[12], const uint64_t m[13], uint64_t c[12])

// m[12] contains the least significant word of the negated inverse of the modulus mod 2^768

#ifdef _WIN64
#	define x	%rcx
#	define m	%rdx
#	define z	%r8

#	define pl	%rdi
#	define ph	%rsi

#	define z4	%r8
#else
#	define x	%rdi
#	define m	%rsi
#	define z	%rdx

#	define pl	%rcx
#	define ph	%r8

#	define z4	%rsi
#endif

#define z0	pl
#define z1	%rax
#define z2	%rbx
#define z3	%rbp

#define z5	%r9
#define z6	%r10
#define z7	%r11
#define z8	%r12
#define z9	%r13
#define z10	%r14
#define z11	%r15

#define l0	 0*8(x)
#define l1	 1*8(x)
#define l2	 2*8(x)
#define l3	 3*8(x)
#define l4	 4*8(x)
#define l5	 5*8(x)

#define h0	 6*8(x)
#define h1	 7*8(x)
#define h2	 8*8(x)
#define h3	 9*8(x)
#define h4	10*8(x)
#define h5	11*8(x)

#define c0	 5*8(%rsp)
#define c1	 4*8(%rsp)
#define c2	 3*8(%rsp)
#define c3	 2*8(%rsp)
#define c4	 1*8(%rsp)
#define c5	 0*8(%rsp)

#ifdef __APPLE__
#define sqr768 _sqr768
#endif

.global modsqr768

.text
.p2align 6,,63

modsqr768:
	// Preserve m, c, and callee-save registers

#ifdef _WIN64
	mov	%rdi,	$1*8(%rsp)
	mov	%rsi,	$2*8(%rsp)
#endif
	push	%r15
	push	%r14
	push	%r13
	push	%r12
	push	%rbp
	push	%rbx

	push	z
	push	m

	//// Karatsuba Squaring ////

	// Compute xl^2, save to stack

	mov	l0, %rdx;	mulx	l5, z5, z6

/*	mov	l0, %rdx;*/	mulx	l4, z4, ph;			add	ph, z5
	mov	l1, %rdx;	mulx	l5, pl, z7;	adcq	pl, z6;	adcq	$0, z7

	mov	l0, %rdx;	mulx	l3, z3, ph;			add	ph, z4
	mov	l1, %rdx;	mulx	l4, pl, ph;	adcq	pl, z5;	adcq	ph, z6
	mov	l2, %rdx;	mulx	l5, pl, z8;	adcq	pl, z7;	adcq	$0, z8

/*	mov	l2, %rdx;*/	mulx	l0, z2, ph;			add	ph, z3
	mov	l1, %rdx;	mulx	l3, pl, ph;	adcq	pl, z4;	adcq	ph, z5
	mov	l2, %rdx;	mulx	l4, pl, ph;	adcq	pl, z6;	adcq	ph, z7
	mov	l3, %rdx;	mulx	l5, pl, z9;	adcq	pl, z8;	adcq	$0, z9

	mov	l0, %rdx;	mulx	l1, z1, ph;			add	ph, z2
	mov	l1, %rdx;	mulx	l2, pl, ph;	adcq	pl, z3;	adcq	ph, z4
	mov	l2, %rdx;	mulx	l3, pl, ph;	adcq	pl, z5;	adcq	ph, z6
	mov	l3, %rdx;	mulx	l4, pl, ph;	adcq	pl, z7;	adcq	ph, z8
	mov	l4, %rdx;	mulx	l5, pl, z10;	adcq	pl, z9;	adcq	$0, z10

	// Double

	xor	z11, z11
	add	z1, z1
	adcq	z2, z2
	adcq	z3, z3
	adcq	z4, z4
	adcq	z5, z5
	adcq	z6, z6
	adcq	z7, z7
	adcq	z8, z8
	adcq	z9, z9
	adcq	z10, z10
	adcq	z11, z11

	// Main diagonal

	mov	l0, %rdx;	mulx	%rdx, z0, ph;	push	z0;		add	ph, z1;		push z1
	mov	l1, %rdx;	mulx	%rdx, pl, ph;	adcq	pl, z2;		adcq	ph, z3;		push z2
	mov	l2, %rdx;	mulx	%rdx, pl, ph;	adcq	pl, z4;		adcq	ph, z5;		push z3
	mov	l3, %rdx;	mulx	%rdx, pl, ph;	adcq	pl, z6;		adcq	ph, z7;		push z4
	mov	l4, %rdx;	mulx	%rdx, pl, ph;	adcq	pl, z8;		adcq	ph, z9;		push z5
	mov	l5, %rdx;	mulx	%rdx, pl, ph;	adcq	pl, z10;	adcq	ph, z11;	push z6

	// Compute xh^2, save to stack

	mov	h0, %rdx;	mulx	h5, z5, z6;	push	z7

/*	mov	h0, %rdx;*/	mulx	h4, z4, ph;			add	ph, z5;	push	z8
	mov	h1, %rdx;	mulx	h5, pl, z7;	adcq	pl, z6;	adcq	$0, z7;	push	z9

	mov	h0, %rdx;	mulx	h3, z3, ph;			add	ph, z4;	push	z10
	mov	h1, %rdx;	mulx	h4, pl, ph;	adcq	pl, z5;	adcq	ph, z6;	push	z11
	mov	h2, %rdx;	mulx	h5, pl, z8;	adcq	pl, z7;	adcq	$0, z8

/*	mov	h2, %rdx;*/	mulx	h0, z2, ph;			add	ph, z3
	mov	h1, %rdx;	mulx	h3, pl, ph;	adcq	pl, z4;	adcq	ph, z5
	mov	h2, %rdx;	mulx	h4, pl, ph;	adcq	pl, z6;	adcq	ph, z7
	mov	h3, %rdx;	mulx	h5, pl, z9;	adcq	pl, z8;	adcq	$0, z9

	mov	h0, %rdx;	mulx	h1, z1, ph;			add	ph, z2
	mov	h1, %rdx;	mulx	h2, pl, ph;	adcq	pl, z3;	adcq	ph, z4
	mov	h2, %rdx;	mulx	h3, pl, ph;	adcq	pl, z5;	adcq	ph, z6
	mov	h3, %rdx;	mulx	h4, pl, ph;	adcq	pl, z7;	adcq	ph, z8
	mov	h4, %rdx;	mulx	h5, pl, z10;	adcq	pl, z9;	adcq	$0, z10

	// Double

	xor	z11, z11
	add	z1, z1
	adcq	z2, z2
	adcq	z3, z3
	adcq	z4, z4
	adcq	z5, z5
	adcq	z6, z6
	adcq	z7, z7
	adcq	z8, z8
	adcq	z9, z9
	adcq	z10, z10
	adcq	z11, z11

	// Main diagonal

	mov	h0, %rdx;	mulx	%rdx, z0, ph;	push	z0;		add	ph, z1;		push z1
	mov	h1, %rdx;	mulx	%rdx, pl, ph;	adcq	pl, z2;		adcq	ph, z3;		push z2
	mov	h2, %rdx;	mulx	%rdx, pl, ph;	adcq	pl, z4;		adcq	ph, z5;		push z3
	mov	h3, %rdx;	mulx	%rdx, pl, ph;	adcq	pl, z6;		adcq	ph, z7;		push z4
	mov	h4, %rdx;	mulx	%rdx, pl, ph;	adcq	pl, z8;		adcq	ph, z9;		push z5
	mov	h5, %rdx;	mulx	%rdx, pl, ph;	adcq	pl, z10;	adcq	ph, z11;	push z6

	// Compute z = (xl-xh)

	xor	z6, z6
	mov	l0, z0;	sub	h0, z0;	push	z7
	mov	l1, z1;	sbb	h1, z1;	push	z8
	mov	l2, z2;	sbb	h2, z2;	push	z9
	mov	l3, z3;	sbb	h3, z3;	push	z10
	mov	l4, z4;	sbb	h4, z4;	push	z11
	mov	l5, z5;	sbb	h5, z5
			sbb	$0, z6	// -1 iff z<0, else 0

	// Compute z = |z|, save to stack

	xor	z6, z0
	xor	z6, z1
	xor	z6, z2
	xor	z6, z3
	xor	z6, z4
	xor	z6, z5

	sub	z6, z0;	push	z0
	sbb	z6, z1;	push	z1
	sbb	z6, z2;	push	z2
	sbb	z6, z3;	push	z3
	sbb	z6, z4;	push	z4
	sbb	z6, z5;	push	z5

	// Compute z^2

	mov	z0, %rdx;	mulx	z5, z5, z6

/*	mov	z0, %rdx;*/	mulx	z4, z4, ph;			add	ph, z5
	mov	z1, %rdx;	mulx	c5, pl, z7;	adcq	pl, z6;	adcq	$0, z7

	mov	c0, %rdx;	mulx	z3, z3, ph;			add	ph, z4
	mov	z1, %rdx;	mulx	c4, pl, ph;	adcq	pl, z5;	adcq	ph, z6
	mov	z2, %rdx;	mulx	c5, pl, z8;	adcq	pl, z7;	adcq	$0, z8

/*	mov	c2, %rdx;*/	mulx	c0, z2, ph;			add	ph, z3
	mov	z1, %rdx;	mulx	c3, pl, ph;	adcq	pl, z4;	adcq	ph, z5
	mov	c2, %rdx;	mulx	c4, pl, ph;	adcq	pl, z6;	adcq	ph, z7
	mov	c3, %rdx;	mulx	c5, pl, z9;	adcq	pl, z8;	adcq	$0, z9

	mov	c0, %rdx;	mulx	z1, z1, ph;			add	ph, z2
	mov	c1, %rdx;	mulx	c2, pl, ph;	adcq	pl, z3;	adcq	ph, z4
	mov	c2, %rdx;	mulx	c3, pl, ph;	adcq	pl, z5;	adcq	ph, z6
	mov	c3, %rdx;	mulx	c4, pl, ph;	adcq	pl, z7;	adcq	ph, z8
	mov	c4, %rdx;	mulx	c5, pl, z10;	adcq	pl, z9;	adcq	$0, z10

	// Double

	xor	z11, z11
	add	z1, z1
	adcq	z2, z2
	adcq	z3, z3
	adcq	z4, z4
	adcq	z5, z5
	adcq	z6, z6
	adcq	z7, z7
	adcq	z8, z8
	adcq	z9, z9
	adcq	z10, z10
	adcq	z11, z11

	// Main diagonal

	mov	c0, %rdx;	mulx	%rdx,  x, ph;				add	ph, z1
	mov	c1, %rdx;	mulx	%rdx, pl, ph;	adcq	pl, z2;		adcq	ph, z3
	mov	c2, %rdx;	mulx	%rdx, pl, ph;	adcq	pl, z4;		adcq	ph, z5
	mov	c3, %rdx;	mulx	%rdx, pl, ph;	adcq	pl, z6;		adcq	ph, z7
	mov	c4, %rdx;	mulx	%rdx, pl, ph;	adcq	pl, z8;		adcq	ph, z9
	mov	c5, %rdx;	mulx	%rdx, pl, ph;	adcq	pl, z10;	adcq	ph, z11

#undef c0
#undef c1
#undef c2
#undef c3
#undef c4
#undef c5

	// Let rdx point in the middle of xl^2 and xh^2, at the lsw of xh^2
	// These are stored on the stack in big-endian word order

	lea	(5+12)*8(%rsp), %rdx

	// Compute xh^2-z^2
	// Notes:
	//   pl and ph are available
	//   z0 is in x, the rest of z^2 is in z11:...:z1
	//   there is no reverse subtract for integers
	//   we need to preserve xh^2 and xl^2 on the stack for now

#define u0	ph
#define u1	x
#define u2	z1
#define u3	z2
#define u4	z3
#define u5	z4
#define u6	z5
#define u7	z6
#define u8	z7
#define u9	z8
#define u10	z9
#define u11	z10
#define u12	z11

#define v0	pl
#define v1	u0
#define v2	u1
#define v3	u2
#define v4	u3
#define v5	u4
#define v6	u5
#define v7	u6
#define v8	u7
#define v9	u8
#define v10	u9
#define v11	u10
#define v12	u11

	mov	(12 -0)*8(%rdx),  u0;	sub	  x,  u0
	mov	(12 -1)*8(%rdx),  u1;	sbb	 z1,  u1
	mov	(12 -2)*8(%rdx),  u2;	sbb	 z2,  u2
	mov	(12 -3)*8(%rdx),  u3;	sbb	 z3,  u3
	mov	(12 -4)*8(%rdx),  u4;	sbb	 z4,  u4
	mov	(12 -5)*8(%rdx),  u5;	sbb	 z5,  u5
	mov	(12 -6)*8(%rdx),  u6;	sbb	 z6,  u6
	mov	(12 -7)*8(%rdx),  u7;	sbb	 z7,  u7
	mov	(12 -8)*8(%rdx),  u8;	sbb	 z8,  u8
	mov	(12 -9)*8(%rdx),  u9;	sbb	 z9,  u9
	mov	(12-10)*8(%rdx), u10;	sbb	z10, u10
	mov	(12-11)*8(%rdx), u11;	sbb	z11, u11
	mov	             $0, u12;	sbb	u12, u12

	mov	(0 -0)*8(%rdx),  v0;	add	 u0,  v0
	mov	(0 -1)*8(%rdx),  v1;	adcq	 u1,  v1
	mov	(0 -2)*8(%rdx),  v2;	adcq	 u2,  v2
	mov	(0 -3)*8(%rdx),  v3;	adcq	 u3,  v3
	mov	(0 -4)*8(%rdx),  v4;	adcq	 u4,  v4
	mov	(0 -5)*8(%rdx),  v5;	adcq	 u5,  v5
	mov	(0 -6)*8(%rdx),  v6;	adcq	 u6,  v6
	mov	(0 -7)*8(%rdx),  v7;	adcq	 u7,  v7
	mov	(0 -8)*8(%rdx),  v8;	adcq	 u8,  v8
	mov	(0 -9)*8(%rdx),  v9;	adcq	 u9,  v9
	mov	(0-10)*8(%rdx), v10;	adcq	u10, v10
	mov	(0-11)*8(%rdx), v11;	adcq	u11, v11
	mov		    $0, v12;	adcq	u12, v12

	// z11 is no longer in use. restore m into it

#undef m
#define m	z11

	mov	(6+12+12)*8(%rsp), m

	// Make space for m[] on top of the stack:
	// reuse 6 words, allocate 7 more

	sub	$7*8, %rsp

	// Add to the middle half of
	// xh^2:xl^2 to complete the
	// Karatsuba multiplication	// Load m[]

	add	 v0, (6 -0)*8(%rdx);	mov	 0*8(m),  v0
	adcq	 v1, (6 -1)*8(%rdx);	mov	 1*8(m),  v1
	adcq	 v2, (6 -2)*8(%rdx);	mov	 2*8(m),  v2
	adcq	 v3, (6 -3)*8(%rdx);	mov	 3*8(m),  v3
	adcq	 v4, (6 -4)*8(%rdx);	mov	 4*8(m),  v4
	adcq	 v5, (6 -5)*8(%rdx);	mov	 5*8(m),  v5
	adcq	 v6, (6 -6)*8(%rdx);	mov	 6*8(m),  v6
	adcq	 v7, (6 -7)*8(%rdx);	mov	 7*8(m),  v7
	adcq	 v8, (6 -8)*8(%rdx);	mov	 8*8(m),  v8
	adcq	 v9, (6 -9)*8(%rdx);	mov	 9*8(m),  v9
	adcq	v10, (6-10)*8(%rdx);	mov	10*8(m), v10
	adcq	v11, (6-11)*8(%rdx);	mov	11*8(m), v11
	adcq	v12, (6-12)*8(%rdx)
	adcq	 $0, (6-13)*8(%rdx)
	adcq	 $0, (6-14)*8(%rdx)
	adcq	 $0, (6-15)*8(%rdx)
	adcq	 $0, (6-16)*8(%rdx)
	adcq	 $0, (6-17)*8(%rdx);	mov	12*8(m), %rdx

	// Prepare defines to suit the reduction code

#define l	z11
#define h	v12

#define t0	v0
#define t1	v1
#define t2	v2
#define t3	v3
#define t4	v4
#define t5	v5
#define t6	v6
#define t7	v7
#define t8	v8
#define t9	v9
#define t10	v10
#define t11	v11
#define t12	t0
#define t13	t1
#define t14	t2
#define t15	t3
#define t16	t4
#define t17	t5
#define t18	t6
#define t19	t7
#define t20	t8
#define t21	t9
#define t22	t10
#define t23	t11
#define t24	h

#define m0	0*8(%rsp)
#define m1	1*8(%rsp)
#define m2	2*8(%rsp)
#define m3	3*8(%rsp)
#define m4	4*8(%rsp)
#define m5	5*8(%rsp)
#define m6	6*8(%rsp)
#define m7	7*8(%rsp)
#define m8	8*8(%rsp)
#define m9	9*8(%rsp)
#define m10	10*8(%rsp)
#define m11	11*8(%rsp)
#define inv	12*8(%rsp)

	// Store m[] on stack	// Load low half of x^2
	mov	  v0,  m0;	mov	(12+24 -0)*8(%rsp),  t0
	mov	  v1,  m1;	mov	(12+24 -1)*8(%rsp),  t1
	mov	  v2,  m2;	mov	(12+24 -2)*8(%rsp),  t2
	mov	  v3,  m3;	mov	(12+24 -3)*8(%rsp),  t3
	mov	  v4,  m4;	mov	(12+24 -4)*8(%rsp),  t4
	mov	  v5,  m5;	mov	(12+24 -5)*8(%rsp),  t5
	mov	  v6,  m6;	mov	(12+24 -6)*8(%rsp),  t6
	mov	  v7,  m7;	mov	(12+24 -7)*8(%rsp),  t7
	mov	  v8,  m8;	mov	(12+24 -8)*8(%rsp),  t8
	mov	  v9,  m9;	mov	(12+24 -9)*8(%rsp),  t9
	mov	 v10, m10;	mov	(12+24-10)*8(%rsp), t10
	mov	 v11, m11;	mov	(12+24-11)*8(%rsp), t11
	mov	%rdx, inv;

	//// Reduction (Operand Scanning) ////

// 0

	//mov	inv, %rdx
	mulx	t0, %rdx, h
	xor	h, h

	mulx	m1, l, h;	adox	l, t1;	adox	h, t2
	mulx	m0, l, h;	adcx	l, t0;	adcx	h, t1
	mulx	m3, l, h;	adox	l, t3;	adox	h, t4
	mulx	m2, l, h;	adcx	l, t2;	adcx	h, t3
	mulx	m5, l, h;	adox	l, t5;	adox	h, t6
	mulx	m4, l, h;	adcx	l, t4;	adcx	h, t5
	mulx	m7, l, h;	adox	l, t7;	adox	h, t8
	mulx	m6, l, h;	adcx	l, t6;	adcx	h, t7
	mulx	m9, l, h;	adox	l, t9;	adox	h, t10
	mulx	m8, l, h;	adcx	l, t8;	adcx	h, t9
	mulx	m11, l, h;	adox	l, t11;	adox	h, t12
	mulx	m10, l, h

// 1

	mov	inv, %rdx;	adcx	l, t10;	adcx	h, t11;	adcq	$0, t12
	mulx	t1, %rdx, h
	xor	h, h

	mulx	m1, l, h;	adox	l, t2;	adox	h, t3
	mulx	m0, l, h;	adcx	l, t1;	adcx	h, t2
	mulx	m3, l, h;	adox	l, t4;	adox	h, t5
	mulx	m2, l, h;	adcx	l, t3;	adcx	h, t4
	mulx	m5, l, h;	adox	l, t6;	adox	h, t7
	mulx	m4, l, h;	adcx	l, t5;	adcx	h, t6
	mulx	m7, l, h;	adox	l, t8;	adox	h, t9
	mulx	m6, l, h;	adcx	l, t7;	adcx	h, t8
	mulx	m9, l, h;	adox	l, t10;	adox	h, t11
	mulx	m8, l, h;	adcx	l, t9;	adcx	h, t10
	mulx	m11, l, h;	adox	l, t12;	adox	h, t13
	mulx	m10, l, h

// 2

	mov	inv, %rdx;	adcx	l, t11;	adcx	h, t12;	adcq	$0, t13
	mulx	t2, %rdx, h
	xor	h, h

	mulx	m1, l, h;	adox	l, t3;	adox	h, t4
	mulx	m0, l, h;	adcx	l, t2;	adcx	h, t3
	mulx	m3, l, h;	adox	l, t5;	adox	h, t6
	mulx	m2, l, h;	adcx	l, t4;	adcx	h, t5
	mulx	m5, l, h;	adox	l, t7;	adox	h, t8
	mulx	m4, l, h;	adcx	l, t6;	adcx	h, t7
	mulx	m7, l, h;	adox	l, t9;	adox	h, t10
	mulx	m6, l, h;	adcx	l, t8;	adcx	h, t9
	mulx	m9, l, h;	adox	l, t11;	adox	h, t12
	mulx	m8, l, h;	adcx	l, t10;	adcx	h, t11
	mulx	m11, l, h;	adox	l, t13;	adox	h, t14
	mulx	m10, l, h

// 3

	mov	inv, %rdx;	adcx	l, t12;	adcx	h, t13;	adcq	$0, t14
	mulx	t3, %rdx, h
	xor	h, h

	mulx	m1, l, h;	adox	l, t4;	adox	h, t5
	mulx	m0, l, h;	adcx	l, t3;	adcx	h, t4
	mulx	m3, l, h;	adox	l, t6;	adox	h, t7
	mulx	m2, l, h;	adcx	l, t5;	adcx	h, t6
	mulx	m5, l, h;	adox	l, t8;	adox	h, t9
	mulx	m4, l, h;	adcx	l, t7;	adcx	h, t8
	mulx	m7, l, h;	adox	l, t10;	adox	h, t11
	mulx	m6, l, h;	adcx	l, t9;	adcx	h, t10
	mulx	m9, l, h;	adox	l, t12;	adox	h, t13
	mulx	m8, l, h;	adcx	l, t11;	adcx	h, t12
	mulx	m11, l, h;	adox	l, t14;	adox	h, t15
	mulx	m10, l, h

// 4

	mov	inv, %rdx;	adcx	l, t13;	adcx	h, t14;	adcq	$0, t15
	mulx	t4, %rdx, h
	xor	h, h

	mulx	m1, l, h;	adox	l, t5;	adox	h, t6
	mulx	m0, l, h;	adcx	l, t4;	adcx	h, t5
	mulx	m3, l, h;	adox	l, t7;	adox	h, t8
	mulx	m2, l, h;	adcx	l, t6;	adcx	h, t7
	mulx	m5, l, h;	adox	l, t9;	adox	h, t10
	mulx	m4, l, h;	adcx	l, t8;	adcx	h, t9
	mulx	m7, l, h;	adox	l, t11;	adox	h, t12
	mulx	m6, l, h;	adcx	l, t10;	adcx	h, t11
	mulx	m9, l, h;	adox	l, t13;	adox	h, t14
	mulx	m8, l, h;	adcx	l, t12;	adcx	h, t13
	mulx	m11, l, h;	adox	l, t15;	adox	h, t16
	mulx	m10, l, h

// 5

	mov	inv, %rdx;	adcx	l, t14;	adcx	h, t15;	adcq	$0, t16
	mulx	t5, %rdx, h
	xor	h, h

	mulx	m1, l, h;	adox	l, t6;	adox	h, t7
	mulx	m0, l, h;	adcx	l, t5;	adcx	h, t6
	mulx	m3, l, h;	adox	l, t8;	adox	h, t9
	mulx	m2, l, h;	adcx	l, t7;	adcx	h, t8
	mulx	m5, l, h;	adox	l, t10;	adox	h, t11
	mulx	m4, l, h;	adcx	l, t9;	adcx	h, t10
	mulx	m7, l, h;	adox	l, t12;	adox	h, t13
	mulx	m6, l, h;	adcx	l, t11;	adcx	h, t12
	mulx	m9, l, h;	adox	l, t14;	adox	h, t15
	mulx	m8, l, h;	adcx	l, t13;	adcx	h, t14
	mulx	m11, l, h;	adox	l, t16;	adox	h, t17
	mulx	m10, l, h

// 6

	mov	inv, %rdx;	adcx	l, t15;	adcx	h, t16;	adcq	$0, t17
	mulx	t6, %rdx, h
	xor	h, h

	mulx	m1, l, h;	adox	l, t7;	adox	h, t8
	mulx	m0, l, h;	adcx	l, t6;	adcx	h, t7
	mulx	m3, l, h;	adox	l, t9;	adox	h, t10
	mulx	m2, l, h;	adcx	l, t8;	adcx	h, t9
	mulx	m5, l, h;	adox	l, t11;	adox	h, t12
	mulx	m4, l, h;	adcx	l, t10;	adcx	h, t11
	mulx	m7, l, h;	adox	l, t13;	adox	h, t14
	mulx	m6, l, h;	adcx	l, t12;	adcx	h, t13
	mulx	m9, l, h;	adox	l, t15;	adox	h, t16
	mulx	m8, l, h;	adcx	l, t14;	adcx	h, t15
	mulx	m11, l, h;	adox	l, t17;	adox	h, t18
	mulx	m10, l, h

// 7

	mov	inv, %rdx;	adcx	l, t16;	adcx	h, t17;	adcq	$0, t18
	mulx	t7, %rdx, h
	xor	h, h

	mulx	m1, l, h;	adox	l, t8;	adox	h, t9
	mulx	m0, l, h;	adcx	l, t7;	adcx	h, t8
	mulx	m3, l, h;	adox	l, t10;	adox	h, t11
	mulx	m2, l, h;	adcx	l, t9;	adcx	h, t10
	mulx	m5, l, h;	adox	l, t12;	adox	h, t13
	mulx	m4, l, h;	adcx	l, t11;	adcx	h, t12
	mulx	m7, l, h;	adox	l, t14;	adox	h, t15
	mulx	m6, l, h;	adcx	l, t13;	adcx	h, t14
	mulx	m9, l, h;	adox	l, t16;	adox	h, t17
	mulx	m8, l, h;	adcx	l, t15;	adcx	h, t16
	mulx	m11, l, h;	adox	l, t18;	adox	h, t19
	mulx	m10, l, h

// 8

	mov	inv, %rdx;	adcx	l, t17;	adcx	h, t18;	adcq	$0, t19
	mulx	t8, %rdx, h
	xor	h, h

	mulx	m1, l, h;	adox	l, t9;	adox	h, t10
	mulx	m0, l, h;	adcx	l, t8;	adcx	h, t9
	mulx	m3, l, h;	adox	l, t11;	adox	h, t12
	mulx	m2, l, h;	adcx	l, t10;	adcx	h, t11
	mulx	m5, l, h;	adox	l, t13;	adox	h, t14
	mulx	m4, l, h;	adcx	l, t12;	adcx	h, t13
	mulx	m7, l, h;	adox	l, t15;	adox	h, t16
	mulx	m6, l, h;	adcx	l, t14;	adcx	h, t15
	mulx	m9, l, h;	adox	l, t17;	adox	h, t18
	mulx	m8, l, h;	adcx	l, t16;	adcx	h, t17
	mulx	m11, l, h;	adox	l, t19;	adox	h, t20
	mulx	m10, l, h

// 9

	mov	inv, %rdx;	adcx	l, t18;	adcx	h, t19;	adcq	$0, t20
	mulx	t9, %rdx, h
	xor	h, h

	mulx	m1, l, h;	adox	l, t10;	adox	h, t11
	mulx	m0, l, h;	adcx	l, t9;	adcx	h, t10
	mulx	m3, l, h;	adox	l, t12;	adox	h, t13
	mulx	m2, l, h;	adcx	l, t11;	adcx	h, t12
	mulx	m5, l, h;	adox	l, t14;	adox	h, t15
	mulx	m4, l, h;	adcx	l, t13;	adcx	h, t14
	mulx	m7, l, h;	adox	l, t16;	adox	h, t17
	mulx	m6, l, h;	adcx	l, t15;	adcx	h, t16
	mulx	m9, l, h;	adox	l, t18;	adox	h, t19
	mulx	m8, l, h;	adcx	l, t17;	adcx	h, t18
	mulx	m11, l, h;	adox	l, t20;	adox	h, t21
	mulx	m10, l, h

// 10

	mov	inv, %rdx;	adcx	l, t19;	adcx	h, t20;	adcq	$0, t21
	mulx	t10, %rdx, h
	xor	h, h

	mulx	m1, l, h;	adox	l, t11;	adox	h, t12
	mulx	m0, l, h;	adcx	l, t10;	adcx	h, t11
	mulx	m3, l, h;	adox	l, t13;	adox	h, t14
	mulx	m2, l, h;	adcx	l, t12;	adcx	h, t13
	mulx	m5, l, h;	adox	l, t15;	adox	h, t16
	mulx	m4, l, h;	adcx	l, t14;	adcx	h, t15
	mulx	m7, l, h;	adox	l, t17;	adox	h, t18
	mulx	m6, l, h;	adcx	l, t16;	adcx	h, t17
	mulx	m9, l, h;	adox	l, t19;	adox	h, t20
	mulx	m8, l, h;	adcx	l, t18;	adcx	h, t19
	mulx	m11, l, h;	adox	l, t21;	adox	h, t22
	mulx	m10, l, h

// 11

	mov	inv, %rdx;	adcx	l, t20;	adcx	h, t21;	adcq	$0, t22
	mulx	t11, %rdx, h
	xor	h, h

	mulx	m1, l, h;	adox	l, t12;	adox	h, t13
	mulx	m0, l, h;	adcx	l, t11;	adcx	h, t12
	mulx	m3, l, h;	adox	l, t14;	adox	h, t15
	mulx	m2, l, h;	adcx	l, t13;	adcx	h, t14
	mulx	m5, l, h;	adox	l, t16;	adox	h, t17
	mulx	m4, l, h;	adcx	l, t15;	adcx	h, t16
	mulx	m7, l, h;	adox	l, t18;	adox	h, t19
	mulx	m6, l, h;	adcx	l, t17;	adcx	h, t18
	mulx	m9, l, h;	adox	l, t20;	adox	h, t21
	mulx	m8, l, h;	adcx	l, t19;	adcx	h, t20
	mulx	m11, l, h;	adox	l, t22;	adox	h, t23
	mulx	m10, l, h;	adcx	l, t21;	adcx	h, t22;	adcq	$0, t23

				// l = original stack pointer
	xor	t24, t24;	lea	8*(13+12+12+2+6)(%rsp), l

#define c0	36*8(%rsp)
#define c1	35*8(%rsp)
#define c2	34*8(%rsp)
#define c3	33*8(%rsp)
#define c4	32*8(%rsp)
#define c5	31*8(%rsp)
#define c6	30*8(%rsp)
#define c7	29*8(%rsp)
#define c8	28*8(%rsp)
#define c9	27*8(%rsp)
#define c10	26*8(%rsp)
#define c11	25*8(%rsp)
#define c12	24*8(%rsp)
#define c13	23*8(%rsp)
#define c14	22*8(%rsp)
#define c15	21*8(%rsp)
#define c16	20*8(%rsp)
#define c17	19*8(%rsp)
#define c18	18*8(%rsp)
#define c19	17*8(%rsp)
#define c20	16*8(%rsp)
#define c21	15*8(%rsp)
#define c22	14*8(%rsp)
#define c23	13*8(%rsp)

	add	c12, t12
	adcq	c13, t13
	adcq	c14, t14
	adcq	c15, t15
	adcq	c16, t16
	adcq	c17, t17
	adcq	c18, t18
	adcq	c19, t19
	adcq	c20, t20
	adcq	c21, t21
	adcq	c22, t22
	adcq	c23, t23	// Restore c into rdx
	adcq	$0, t24;	mov	-7*8(l), %rdx

	// Conditional subtraction of m

	mov	t12, c0;	sub	m0, t12
	mov	t13, c1;	sbb	m1, t13
	mov	t14, c2;	sbb	m2, t14
	mov	t15, c3;	sbb	m3, t15
	mov	t16, c4;	sbb	m4, t16
	mov	t17, c5;	sbb	m5, t17
	mov	t18, c6;	sbb	m6, t18
	mov	t19, c7;	sbb	m7, t19
	mov	t20, c8;	sbb	m8, t20
	mov	t21, c9;	sbb	m9, t21
	mov	t22, c10;	sbb	m10, t22
	mov	t23, c11;	sbb	m11, t23
				sbb	$0, t24
	cmovc	c0, t12
	cmovc	c1, t13
	cmovc	c2, t14
	cmovc	c3, t15
	cmovc	c4, t16
	cmovc	c5, t17
	cmovc	c6, t18
	cmovc	c7, t19
	cmovc	c8, t20
	cmovc	c9, t21
	cmovc	c10, t22	// Deallocate stack
	cmovc	c11, t23;	add	$(13+12+12+2)*8, %rsp

	mov	t12,  0*8(%rdx)
	mov	t13,  1*8(%rdx)
	mov	t14,  2*8(%rdx)
	mov	t15,  3*8(%rdx)
	mov	t16,  4*8(%rdx)
	mov	t17,  5*8(%rdx)
	mov	t18,  6*8(%rdx)
	mov	t19,  7*8(%rdx)
	mov	t20,  8*8(%rdx)
	mov	t21,  9*8(%rdx)
	mov	t22, 10*8(%rdx)
	mov	t23, 11*8(%rdx)

	// Restore callee-saves

#ifdef _WIN64
	mov	$7*8(%rsp), %rdi
	mov	$8*8(%rsp), %rsi
#endif
	pop	%rbx
	pop	%rbp
	pop	%r12
	pop	%r13
	pop	%r14
	pop	%r15

	ret
