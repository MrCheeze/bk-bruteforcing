#include <stdio.h>

// These unions are necessary to put the constants in .rodata rather than .data.
// TODO: is it possible to remove them somehow?

typedef union {
	/* 0x0 */ double d;
	/* 0x0 */ struct
	{
		/* 0x0 */ unsigned int hi;
		/* 0x4 */ unsigned int lo;
	} word;
} du;

typedef union {
	/* 0x0 */ float f;
	/* 0x0 */ unsigned int i;
} fu;

static const du P[5] = {{1.0},
						{-0.16666659550427756},
						{0.008333066246082155},
						{-1.980960290193795E-4},
						{2.605780637968037E-6}};

static const du rpi = {0.3183098861837907};

static const du pihi = {
	3.1415926218032837};

static const du pilo = {
	3.178650954705639E-8};

static const fu zero = {0.0};

float sinf_bk(float x)
{
	double dx;  // double x
	double xsq; // x squared
	double poly;
	double dn;
	int n;
	double result;
	int ix; // int x
	int xpt;

	ix = *(int *)&x;
	xpt = (ix >> 22) & 0x1FF;

	if (xpt < 255)
	{
		dx = x;
		if (xpt >= 230)
		{
			xsq = dx * dx;

			poly = (((((P[4].d * xsq) + P[3].d) * xsq) + P[2].d) * xsq) + P[1].d;

			result = ((dx * xsq) * poly) + dx;

			return result;
		}
		else
		{
			return x;
		}
	}

	if (xpt < 310)
	{
		dx = x;

		dn = dx * rpi.d;

		if (dn >= 0)
		{
			n = dn + 0.5;
		}
		else
		{
			n = dn - 0.5;
		}

		dn = n;

		dx -= dn * pihi.d;
		dx -= dn * pilo.d;

		xsq = dx * dx;

		poly = (((((P[4].d * xsq) + P[3].d) * xsq) + P[2].d) * xsq) + P[1].d;

		result = ((dx * xsq) * poly) + dx;

		if ((n & 0x1) == 0)
		{
			return result;
		}
		else
		{
			return -(float)result;
		}
	}

	if (x != x)
	{
		return x;
	}

	return zero.f;
}

float cosf_bk(float x)
{
	double dx;  // double x
	double xsq; // x squared
	double poly;
	double dn;
	float xabs;
	int n;
	double result;
	int ix; // int x
	int xpt;
	ix = *(int *)&x;
	xpt = (ix >> 22) & 0x1FF;

	if (xpt < 310)
	{
		if (0 < x)
			xabs = x;
		else
			xabs = -x;
		dx = xabs;

		dn = dx * rpi.d + .5;
		if (0 <= dn)
		{

			n = dn + .5;
		}
		else
		{
			n = dn - .5;
		}
		dn = n;

		dx -= (dn - .5) * pihi.d;
		dx -= (dn - .5) * pilo.d;
		xsq = dx * dx;

		poly = (((((P[4].d * xsq) + P[3].d) * xsq) + P[2].d) * xsq) + P[1].d;

		result = ((dx * xsq) * poly) + dx;

		if ((n & 0x1) == 0)
		{
			return result;
		}
		else
		{
			return -(float)result;
		}
	}

	if (x != x)
	{
		return x;
	}

	return zero.f;
}

void main() {
	printf("%.20f\n", sinf_bk((0.0/180.0)*3.141592654));
	printf("%.20f\n", sinf_bk((90.0/180.0)*3.141592654));
	printf("%.20f\n", sinf_bk((180.0/180.0)*3.141592654));
	printf("%.20f\n", sinf_bk((270.0/180.0)*3.141592654));
	printf("%.20f\n", sinf_bk((360.0/180.0)*3.141592654));
	printf("\n");
	printf("%.20f\n", sinf_bk(3.14159265358979323846) / -6.60112888795e-08);
	printf("%.20f\n", sinf_bk(3.14159265358979323846) / -1.35214051511e-07);
	printf("%.20f\n", sinf_bk(3.14159265358979323846) / 1.86179693884e-08);
	printf("\n");
	printf("%.20f %.20f\n", sinf_bk(0.0*3.141592654), cosf_bk(0.0*3.141592654));
	printf("%.20f %.20f\n", sinf_bk(0.5*3.141592654), cosf_bk(0.5*3.141592654));
	printf("%.20f %.20f\n", sinf_bk(1.0*3.141592654), cosf_bk(1.0*3.141592654));
	printf("%.20f %.20f\n", sinf_bk(1.5*3.141592654), cosf_bk(1.5*3.141592654));
	printf("\n");
	printf("%.20f %.20f\n", sinf_bk(0.0*3.14159265358979323846), cosf_bk(0.0*3.14159265358979323846));
	printf("%.20f %.20f\n", sinf_bk(0.125*3.14159265358979323846), cosf_bk(0.125*3.14159265358979323846));
	printf("%.20f %.20f\n", sinf_bk(0.25*3.14159265358979323846), cosf_bk(0.25*3.14159265358979323846));
	printf("%.20f %.20f\n", sinf_bk(0.5*3.14159265358979323846), cosf_bk(0.5*3.14159265358979323846));
	printf("%.20f %.20f\n", sinf_bk(0.75*3.14159265358979323846), cosf_bk(0.75*3.14159265358979323846));
	printf("%.20f %.20f\n", sinf_bk(1.0*3.14159265358979323846), cosf_bk(1.0*3.14159265358979323846));
	printf("%.20f %.20f\n", sinf_bk(1.25*3.14159265358979323846), cosf_bk(1.25*3.14159265358979323846));
	printf("%.20f %.20f\n", sinf_bk(1.5*3.14159265358979323846), cosf_bk(1.5*3.14159265358979323846));
	printf("%.20f %.20f\n", sinf_bk(1.75*3.14159265358979323846), cosf_bk(1.75*3.14159265358979323846));
	printf("%.20f %.20f\n", sinf_bk(1.875*3.14159265358979323846), cosf_bk(1.875*3.14159265358979323846));
	printf("%.20f %.20f\n", sinf_bk(2.0*3.14159265358979323846), cosf_bk(2.0*3.14159265358979323846));
	printf("\n");
	printf("%.40f %.40f\n", 8181*3.14159265358979323846/20000, sinf_bk(8181*3.14159265358979323846/20000) * 65535.0);
}