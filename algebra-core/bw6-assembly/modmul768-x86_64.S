// void modmul768(const uint64_t x[12], const uint64_t y[12], const uint64_t m[13], uint64_t z[12])

// m[12] contains the least significant word of the negated inverse of the modulus mod 2^768

#ifdef _WIN64
#	define x	%rcx
#	define y	%rdx
#	define m	%r8
#	define z	%r9
#	define pl	%rsi
#	define ph	%rdi
#	define y5	%rax
#	define y4	m
#else
#	define x	%rdi
#	define y	%rsi
#	define m	%rdx
#	define z	%rcx
#	define pl	%r8
#	define ph	%r9
#	define y5	y
#	define y4	%rax
#endif

#define y3	z
#define y2	x
#define y1	%rbx
#define y0	%rbp
#define y6	y0
#define y7	y1
#define y8	y2
#define y9	y3
#define ya	y4
#define yb	y5

#define t0	%r10
#define t1	%r11
#define t2	%r12
#define t3	%r13
#define t4	%r14
#define t5	%r15
#define t6	t0
#define t7	t1
#define t8	t2
#define t9	t3
#define ta	t4
#define tb	t5

#define _0	(%rsp)

#define x0	0x1*8(%rsp)
#define x1	0x2*8(%rsp)
#define x2	0x3*8(%rsp)
#define x3	0x4*8(%rsp)
#define x4	0x5*8(%rsp)
#define x5	0x6*8(%rsp)
#define x6	0x7*8(%rsp)
#define x7	0x8*8(%rsp)
#define x8	0x9*8(%rsp)
#define x9	0xa*8(%rsp)
#define xa	0xb*8(%rsp)
#define xb	0xc*8(%rsp)

#define zl0	0x10*8(%rsp)
#define zl1	0x11*8(%rsp)
#define zl2	0x12*8(%rsp)
#define zl3	0x13*8(%rsp)
#define zl4	0x14*8(%rsp)
#define zl5	0x15*8(%rsp)
#define zl6	0x16*8(%rsp)
#define zl7	0x17*8(%rsp)
#define zl8	0x18*8(%rsp)
#define zl9	0x19*8(%rsp)
#define zla	0x1a*8(%rsp)
#define zlb	0x1b*8(%rsp)
#define zh0	0x1c*8(%rsp)
#define zh1	0x1d*8(%rsp)
#define zh2	0x1e*8(%rsp)
#define zh3	0x1f*8(%rsp)
#define zh4	0x20*8(%rsp)
#define zh5	0x21*8(%rsp)
#define zh6	0x22*8(%rsp)
#define zh7	0x23*8(%rsp)
#define zh8	0x24*8(%rsp)
#define zh9	0x25*8(%rsp)
#define zha	0x26*8(%rsp)
#define zhb	0x27*8(%rsp)

// regs:
//	1	rsp
//	1	rdx,	x[i], (xl-xh)[i]
//	2	pl, ph
//	6	t0-tb
//	6	y[5:0], (yh-yl)[]

// stack:
//	 0	_0
//	 1-12	x[], (xl-xh)[], m[]
//	13	y, inv
//	14	m
//	15	z
//	16-39	z[] (x*y before reduction)
//	40-45	callee-save registers

.text

#ifdef __APPLE__
#define modmul768 _modmul768
#endif

.global modmul768

.p2align 6,,63
modmul768:

// Save registers

#ifdef _WIN64
mov	%rsi, 1*8(%rsp)
	mov	%rdi, 2*8(%rsp)
#endif
push	%r15
        push	%r14
        push	%r13
        push	%r12
        push	%rbx
        push	%rbp

// Allocate stack

sub	$40*8, %rsp

// Save y,m,z on the stack

mov	y, 13*8(%rsp)
mov	m, 14*8(%rsp)
mov	z, 15*8(%rsp)

// copy x to the stack

mov	0xb*8(x), %rax;	mov	%rax, xb
mov	0xa*8(x), %rax;	mov	%rax, xa
mov	0x9*8(x), %rax;	mov	%rax, x9
mov	0x8*8(x), %rax;	mov	%rax, x8
mov	0x7*8(x), %rax;	mov	%rax, x7
mov	0x6*8(x), %rax;	mov	%rax, x6
mov	0x5*8(x), %rax;	mov	%rax, x5
mov	0x4*8(x), %rax;	mov	%rax, x4
mov	0x3*8(x), %rax;	mov	%rax, x3
mov	0x2*8(x), %rax;	mov	%rax, x2
mov	0x1*8(x), %rax;	mov	%rax, x1
mov	0x0*8(x), %rax;	mov	%rax, x0

// xl * yl

// copy yl to registers

mov	0x0*8(y), y0
        mov	0x1*8(y), y1
        mov	0x2*8(y), y2	// overwrites x (no longer needed)
mov	0x3*8(y), y3	// overwrites z (restore later)
mov	0x4*8(y), y4
        mov	0x5*8(y), y5	// may overwrite y (restore later)

mov	x0, %rdx

        mulx	y0, t0, t1;	mov	t0, zl0
mulx	y2, t2, t3;	xor	t0, t0
        mulx	y1, pl, ph;	adcx	pl, t1;	adcx	ph, t2
        mulx	y4, t4, t5;	mov	t0, _0
mulx	y3, pl, ph;	adcx	pl, t3;	adcx	ph, t4
        mulx	y5, pl, ph;	adcx	pl, t5;	adcx	ph, t6	// t6 = t0

mov	x1, %rdx

        mulx	y1, pl, ph;	adox	pl, t2;	adox	ph, t3
        mulx	y0, pl, ph;	adcx	pl, t1;	adcx	ph, t2;	mov	t1, zl1
mulx	y3, pl, ph;	adox	pl, t4;	adox	ph, t5
        mulx	y2, pl, ph;	adcx	pl, t3;	adcx	ph, t4
        mulx	y5, pl, t7;	adox	pl, t6;	adox	_0, t7
        mulx	y4, pl, ph;	adcx	pl, t5;	adcx	ph, t6;	adcx	_0, t7

        mov	x2, %rdx

        mulx	y1, pl, ph;	adox	pl, t3;	adox	ph, t4
        mulx	y0, pl, ph;	adcx	pl, t2;	adcx	ph, t3;	mov	t2, zl2
mulx	y3, pl, ph;	adox	pl, t5;	adox	ph, t6
        mulx	y2, pl, ph;	adcx	pl, t4;	adcx	ph, t5
        mulx	y5, pl, t8;	adox	pl, t7;	adox	_0, t8
        mulx	y4, pl, ph;	adcx	pl, t6;	adcx	ph, t7;	adcx	_0, t8

        mov	x3, %rdx

        mulx	y1, pl, ph;	adox	pl, t4;	adox	ph, t5
        mulx	y0, pl, ph;	adcx	pl, t3;	adcx	ph, t4;	mov	t3, zl3
mulx	y3, pl, ph;	adox	pl, t6;	adox	ph, t7
        mulx	y2, pl, ph;	adcx	pl, t5;	adcx	ph, t6
        mulx	y5, pl, t9;	adox	pl, t8;	adox	_0, t9
        mulx	y4, pl, ph;	adcx	pl, t7;	adcx	ph, t8;	adcx	_0, t9

        mov	x4, %rdx

        mulx	y1, pl, ph;	adox	pl, t5;	adox	ph, t6
        mulx	y0, pl, ph;	adcx	pl, t4;	adcx	ph, t5;	mov	t4, zl4
mulx	y3, pl, ph;	adox	pl, t7;	adox	ph, t8
        mulx	y2, pl, ph;	adcx	pl, t6;	adcx	ph, t7
        mulx	y5, pl, ta;	adox	pl, t9;	adox	_0, ta
        mulx	y4, pl, ph;	adcx	pl, t8;	adcx	ph, t9;	adcx	_0, ta

        mov	x5, %rdx

        mulx	y1, pl, ph;	adox	pl, t6;	adox	ph, t7
        mulx	y0, pl, ph;	adcx	pl, t5;	adcx	ph, t6;	mov	t5, zl5
mulx	y3, pl, ph;	adox	pl, t8;	adox	ph, t9
        mulx	y2, pl, ph;	adcx	pl, t7;	adcx	ph, t8
        mulx	y5, pl, tb;	adox	pl, ta;	adox	_0, tb
        mulx	y4, pl, ph;	adcx	pl, t9;	adcx	ph, ta;	adc	_0, tb

// restore y in pl

mov	13*8(%rsp), pl

// xh * yh

// store t, copy yh to registers

mov	t6, zl6;	mov	0x6*8(pl), y6
        mov	t7, zl7;	mov	0x7*8(pl), y7
        mov	t8, zl8;	mov	0x8*8(pl), y8
        mov	t9, zl9;	mov	0x9*8(pl), y9
        mov	ta, zla;	mov	0xa*8(pl), ya
        mov	tb, zlb;	mov	0xb*8(pl), yb

        mov	x6, %rdx

        mulx	y6, t0, t1;	mov	t0, zh0
mulx	y8, t2, t3;	xor	t0, t0
        mulx	y7, pl, ph;	adcx	pl, t1;	adcx	ph, t2
        mulx	ya, t4, t5;	mov	t0, _0
mulx	y9, pl, ph;	adcx	pl, t3;	adcx	ph, t4
        mulx	yb, pl, ph;	adcx	pl, t5;	adcx	ph, t6	// t6 = t0

mov	x7, %rdx

        mulx	y7, pl, ph;	adox	pl, t2;	adox	ph, t3
        mulx	y6, pl, ph;	adcx	pl, t1;	adcx	ph, t2;	mov	t1, zh1
mulx	y9, pl, ph;	adox	pl, t4;	adox	ph, t5
        mulx	y8, pl, ph;	adcx	pl, t3;	adcx	ph, t4
        mulx	yb, pl, t7;	adox	pl, t6;	adox	_0, t7
        mulx	ya, pl, ph;	adcx	pl, t5;	adcx	ph, t6;	adcx	_0, t7

        mov	x8, %rdx

        mulx	y7, pl, ph;	adox	pl, t3;	adox	ph, t4
        mulx	y6, pl, ph;	adcx	pl, t2;	adcx	ph, t3;	mov	t2, zh2
mulx	y9, pl, ph;	adox	pl, t5;	adox	ph, t6
        mulx	y8, pl, ph;	adcx	pl, t4;	adcx	ph, t5
        mulx	yb, pl, t8;	adox	pl, t7;	adox	_0, t8
        mulx	ya, pl, ph;	adcx	pl, t6;	adcx	ph, t7;	adcx	_0, t8

        mov	x9, %rdx

        mulx	y7, pl, ph;	adox	pl, t4;	adox	ph, t5
        mulx	y6, pl, ph;	adcx	pl, t3;	adcx	ph, t4;	mov	t3, zh3
mulx	y9, pl, ph;	adox	pl, t6;	adox	ph, t7
        mulx	y8, pl, ph;	adcx	pl, t5;	adcx	ph, t6
        mulx	yb, pl, t9;	adox	pl, t8;	adox	_0, t9
        mulx	ya, pl, ph;	adcx	pl, t7;	adcx	ph, t8;	adcx	_0, t9

        mov	xa, %rdx

        mulx	y7, pl, ph;	adox	pl, t5;	adox	ph, t6
        mulx	y6, pl, ph;	adcx	pl, t4;	adcx	ph, t5;	mov	t4, zh4
mulx	y9, pl, ph;	adox	pl, t7;	adox	ph, t8
        mulx	y8, pl, ph;	adcx	pl, t6;	adcx	ph, t7
        mulx	yb, pl, ta;	adox	pl, t9;	adox	_0, ta
        mulx	ya, pl, ph;	adcx	pl, t8;	adcx	ph, t9;	adcx	_0, ta

        mov	xb, %rdx

        mulx	y7, pl, ph;	adox	pl, t6;	adox	ph, t7
        mulx	y6, pl, ph;	adcx	pl, t5;	adcx	ph, t6;	mov	t5, zh5
mulx	y9, pl, ph;	adox	pl, t8;	adox	ph, t9
        mulx	y8, pl, ph;	adcx	pl, t7;	adcx	ph, t8
        mulx	yb, pl, tb;	adox	pl, ta;	adox	_0, tb
        mulx	ya, pl, ph;	adcx	pl, t9;	adcx	ph, ta;	adc	_0, tb

// restore y in pl

mov	13*8(%rsp), pl

// store t		//  compute yh-yl	// load xl in rdx, t1-t5

mov	t6, zh6;	sub	0x0*8(pl), y6;	mov	x0, %rdx
        mov	t7, zh7;	sbb	0x1*8(pl), y7;	mov	x1, t1
        mov	t8, zh8;	sbb	0x2*8(pl), y8;	mov	x2, t2
        mov	t9, zh9;	sbb	0x3*8(pl), y9;	mov	x3, t3
        mov	ta, zha;	sbb	0x4*8(pl), ya;	mov	x4, t4
        mov	tb, zhb;	sbb	0x5*8(pl), yb;	mov	x5, t5
        sbb	pl, pl	// -1 if yl>yh, else 0

// compute |yh-yl|

xor	pl, y6;		xor	pl, y7;		xor	pl, y8
xor	pl, y9;		xor	pl, ya;		xor	pl, yb

        sub	pl, y6;		sbb	pl, y7;		sbb	pl, y8
        sbb	pl, y9;		sbb	pl, ya;		sbb	pl, yb

// compute xl-xh

sub	x6, %rdx
        sbb	x7, t1
        sbb	x8, t2
        sbb	x9, t3
        sbb	xa, t4
        sbb	xb, t5
        sbb	ph, ph	// -1 if xh>xl, else 0

// compute |xl-xh|

xor	ph, %rdx;	xor	ph, t1;		xor	ph, t2
xor	ph, t3;		xor	ph, t4;		xor	ph, t5

        sub	ph, %rdx	// store t1-t5 in x1-x5
sbb	ph, t1;		mov	t1, x1
sbb	ph, t2;		mov	t2, x2
sbb	ph, t3;		mov	t3, x3
sbb	ph, t4;		mov	t4, x4
sbb	ph, t5;		mov	t5, x5

// compute whether we will need to negate |yh-yl|*|xl-xh|, store in x0

xor	pl, ph;		mov	ph, x0

// compute |yh-yl|*|xl-xh|, store low half in x6-xb

mulx	y0, t0, t1;	mov	t0, x6
mulx	y2, t2, t3;	xor	t0, t0
        mulx	y1, pl, ph;	adcx	pl, t1;	adcx	ph, t2
        mulx	y4, t4, t5;	mov	t0, _0
mulx	y3, pl, ph;	adcx	pl, t3;	adcx	ph, t4
        mulx	y5, pl, ph;	adcx	pl, t5;	adcx	ph, t6	// t6 = t0

mov	x1, %rdx

        mulx	y1, pl, ph;	adox	pl, t2;	adox	ph, t3
        mulx	y0, pl, ph;	adcx	pl, t1;	adcx	ph, t2;	mov	t1, x7
mulx	y3, pl, ph;	adox	pl, t4;	adox	ph, t5
        mulx	y2, pl, ph;	adcx	pl, t3;	adcx	ph, t4
        mulx	y5, pl, t7;	adox	pl, t6;	adox	_0, t7
        mulx	y4, pl, ph;	adcx	pl, t5;	adcx	ph, t6;	adcx	_0, t7

        mov	x2, %rdx

        mulx	y1, pl, ph;	adox	pl, t3;	adox	ph, t4
        mulx	y0, pl, ph;	adcx	pl, t2;	adcx	ph, t3;	mov	t2, x8
mulx	y3, pl, ph;	adox	pl, t5;	adox	ph, t6
        mulx	y2, pl, ph;	adcx	pl, t4;	adcx	ph, t5
        mulx	y5, pl, t8;	adox	pl, t7;	adox	_0, t8
        mulx	y4, pl, ph;	adcx	pl, t6;	adcx	ph, t7;	adcx	_0, t8

        mov	x3, %rdx

        mulx	y1, pl, ph;	adox	pl, t4;	adox	ph, t5
        mulx	y0, pl, ph;	adcx	pl, t3;	adcx	ph, t4;	mov	t3, x9
mulx	y3, pl, ph;	adox	pl, t6;	adox	ph, t7
        mulx	y2, pl, ph;	adcx	pl, t5;	adcx	ph, t6
        mulx	y5, pl, t9;	adox	pl, t8;	adox	_0, t9
        mulx	y4, pl, ph;	adcx	pl, t7;	adcx	ph, t8;	adcx	_0, t9

        mov	x4, %rdx

        mulx	y1, pl, ph;	adox	pl, t5;	adox	ph, t6
        mulx	y0, pl, ph;	adcx	pl, t4;	adcx	ph, t5;	mov	t4, xa
mulx	y3, pl, ph;	adox	pl, t7;	adox	ph, t8
        mulx	y2, pl, ph;	adcx	pl, t6;	adcx	ph, t7
        mulx	y5, pl, ta;	adox	pl, t9;	adox	_0, ta
        mulx	y4, pl, ph;	adcx	pl, t8;	adcx	ph, t9;	adcx	_0, ta

        mov	x5, %rdx

        mulx	y1, pl, ph;	adox	pl, t6;	adox	ph, t7
        mulx	y0, pl, ph;	adcx	pl, t5;	adcx	ph, t6;	mov	t5, xb
mulx	y3, pl, ph;	adox	pl, t8;	adox	ph, t9
        mulx	y2, pl, ph;	adcx	pl, t7;	adcx	ph, t8
        mulx	y5, pl, tb;	adox	pl, ta;	adox	_0, tb
        mulx	y4, pl, ph;	adcx	pl, t9;	adcx	ph, ta;	adc	_0, tb

        mov	x0, pl

// reload low half into y0-y5

mov	x6, y0
        mov	x7, y1
        mov	x8, y2
        mov	x9, y3
        mov	xa, y4
        mov	xb, y5

// compute (yh-yl)(xl-xh)

xor	pl, y0
xor	pl, y1
xor	pl, y2
xor	pl, y3
xor	pl, y4
xor	pl, y5
xor	pl, t6
xor	pl, t7
xor	pl, t8
xor	pl, t9
xor	pl, ta
xor	pl, tb

        sub	pl, y0
        sbb	pl, y1
        sbb	pl, y2
        sbb	pl, y3
        sbb	pl, y4
        sbb	pl, y5
        sbb	pl, t6
        sbb	pl, t7
        sbb	pl, t8
        sbb	pl, t9
        sbb	pl, ta
        sbb	pl, tb

// preserve borrow

sbb	pl, pl

// add xl*yl

add	zl0, y0
        adc	zl1, y1
        adc	zl2, y2
        adc	zl3, y3
        adc	zl4, y4
        adc	zl5, y5
        adc	zl6, t6
        adc	zl7, t7
        adc	zl8, t8
        adc	zl9, t9
        adc	zla, ta
        adc	zlb, tb
        adc	$0, pl

// add xh*yh

add	zh0, y0
        adc	zh1, y1
        adc	zh2, y2
        adc	zh3, y3
        adc	zh4, y4
        adc	zh5, y5
        adc	zh6, t6
        adc	zh7, t7
        adc	zh8, t8
        adc	zh9, t9
        adc	zha, ta
        adc	zhb, tb
        adc	$0, pl

// add to high three quarters of zh:zl

add	zl6, y0
        adc	zl7, y1
        adc	zl8, y2
        adc	zl9, y3
        adc	zla, y4
        adc	zlb, y5

        adc	t6, zh0
adc	t7, zh1
adc	t8, zh2
adc	t9, zh3
adc	ta, zh4
adc	tb, zh5
adc	pl, zh6
adcq	$0, zh7
adcq	$0, zh8
adcq	$0, zh9
adcq	$0, zha
adcq	$0, zhb

// Define new register and stack location roles to suit the reduction code

#define	m0	x0
#define	m1	x1
#define	m2	x2
#define	m3	x3
#define	m4	x4
#define	m5	x5
#define	m6	x6
#define	m7	x7
#define	m8	x8
#define	m9	x9
#define	m10	xa
#define	m11	xb

#define inv	_0

#define	l	pl
#define	h	ph

#undef t6
#undef t7
#undef t8
#undef t9
#undef ta
#undef tb

#define t6	y0
#define t7	y1
#define t8	y2
#define t9	y3
#define t10	y4
#define t11	y5
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

#define z0	zl0
#define z1	zl1
#define z2	zl2
#define z3	zl3
#define z4	zl4
#define z5	zl5
#define z6	zl6
#define z7	zl7
#define z8	zl8
#define z9	zl9
#define z10	zla
#define z11	zlb
#define z12	zh0
#define z13	zh1
#define z14	zh2
#define z15	zh3
#define z16	zh4
#define z17	zh5
#define z18	zh6
#define z19	zh7
#define z20	zh8
#define z21	zh9
#define z22	zha
#define z23	zhb

// Load low half of zl

mov	zl0, t0
        mov	zl1, t1
        mov	zl2, t2
        mov	zl3, t3
        mov	zl4, t4
        mov	zl5, t5

// Copy m to stack

mov	14*8(%rsp), pl
        mov	0x0*8(pl), %rdx;	mov %rdx, m0
mov	0x1*8(pl), %rdx;	mov %rdx, m1
mov	0x2*8(pl), %rdx;	mov %rdx, m2
mov	0x3*8(pl), %rdx;	mov %rdx, m3
mov	0x4*8(pl), %rdx;	mov %rdx, m4
mov	0x5*8(pl), %rdx;	mov %rdx, m5
mov	0x6*8(pl), %rdx;	mov %rdx, m6
mov	0x7*8(pl), %rdx;	mov %rdx, m7
mov	0x8*8(pl), %rdx;	mov %rdx, m8
mov	0x9*8(pl), %rdx;	mov %rdx, m9
mov	0xa*8(pl), %rdx;	mov %rdx, m10
mov	0xb*8(pl), %rdx;	mov %rdx, m11
mov	0xc*8(pl), %rdx;	mov %rdx, inv

////////////////////////////////////////////////////////////////
// Reduction
////////////////////////////////////////////////////////////////

// z0

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

// z1

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

// z2

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

// z3

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

// z4

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

// z5

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

// z6

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

// z7

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

// z8

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

// z9

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

// z10

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

// z11

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

xor	t24, t24;	lea	46*8(%rsp), pl

        add	z12, t12
        adc	z13, t13
        adc	z14, t14
        adc	z15, t15
        adc	z16, t16
        adc	z17, t17
        adc	z18, t18
        adc	z19, t19
        adc	z20, t20
        adc	z21, t21
        adc	z22, t22
        adc	z23, t23
        adcq	$0, t24;	mov	15*8(%rsp), %rdx

// Conditional subtraction of m

mov	t12, z0;	sub	m0, t12
        mov	t13, z1;	sbb	m1, t13
        mov	t14, z2;	sbb	m2, t14
        mov	t15, z3;	sbb	m3, t15
        mov	t16, z4;	sbb	m4, t16
        mov	t17, z5;	sbb	m5, t17
        mov	t18, z6;	sbb	m6, t18
        mov	t19, z7;	sbb	m7, t19
        mov	t20, z8;	sbb	m8, t20
        mov	t21, z9;	sbb	m9, t21
        mov	t22, z10;	sbb	m10, t22
        mov	t23, z11;	sbb	m11, t23
        sbb	$0, t24

        cmovc	z0, t12
        cmovc	z1, t13
        cmovc	z2, t14
        cmovc	z3, t15
        cmovc	z4, t16
        cmovc	z5, t17
        cmovc	z6, t18
        cmovc	z7, t19
        cmovc	z8, t20
        cmovc	z9, t21
        cmovc	z10, t22
        cmovc	z11, t23

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

mov	-6*8(pl), %rbp
        mov	-5*8(pl), %rbx
        mov	-4*8(pl), %r12
        mov	-3*8(pl), %r13
        mov	-2*8(pl), %r14
        mov	-1*8(pl), %r15

        add	$46*8, %rsp

#ifdef _WIN64
mov	2*8(pl), %rdi
	mov	1*8(pl), %rsi	// pl may be rsi
#endif
        ret
