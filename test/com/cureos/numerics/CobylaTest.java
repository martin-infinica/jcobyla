/*
 * jcobyla
 * 
 * The MIT License
 *
 * Copyright (c) 2012 Anders Gustafsson, Cureos AB.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files 
 * (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, 
 * publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 * 
 * Remarks:
 * 
 * The original Fortran 77 version of this code was by Michael Powell (M.J.D.Powell @ damtp.cam.ac.uk)
 * The Fortran 90 version was by Alan Miller (Alan.Miller @ vic.cmis.csiro.au). Latest revision - 30 October 1998
 */
package com.cureos.numerics;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;

import org.junit.Ignore;
import org.junit.Test;

import math.functions.Functions;

/**
 * Test class for COBYLA2 employing tests from Report DAMTP 1992/NA5.
 * 
 * @author Anders Gustafsson, Cureos AB.
 */
public class CobylaTest
{

	private static final MathContext mc = new MathContext(64, RoundingMode.HALF_EVEN);

	// FIELDS

	private BigDecimal rhobeg = new BigDecimal(0.5, mc);
	private BigDecimal rhoend = new BigDecimal(1.0e-6, mc);
	private int iprint = 1;
	private int maxfun = 3500;

	// TESTS

	private static BigDecimal convert(double value, MathContext mc)
	{
		return new BigDecimal(value, mc);
	}

	private static void assertArrayEquals(String message, double[] expected, BigDecimal[] actual, double delta)
	{
		assertEquals(message, expected.length, actual.length);
		for (int i = 0; i < expected.length; ++i)
		{
			assertEquals(message + " " + i, expected[i], actual[i].doubleValue(), delta);
		}
	}

	private BigDecimal square(BigDecimal v, MathContext mc)
	{
		return v.multiply(v, mc);
	}

	private BigDecimal pow(BigDecimal v, long exponent, MathContext mc)
	{
		return Functions.pow(v, exponent, mc);
	}

	/**
	 * Minimization of a simple quadratic function of two variables.
	 */
	@Test
	public void test01FindMinimum()
	{
		System.out.format("%nOutput from test problem 1 (Simple quadratic)%n");
		Calcfc calcfc = new Calcfc()
		{
			@Override
			public BigDecimal compute(int n, int m, BigDecimal[] x, BigDecimal[] con, MathContext mc)
			{
				return BigDecimal.TEN.multiply(square(x[0].add(BigDecimal.ONE, mc), mc), mc)
						.add(square(x[1], mc), mc);
			}
		};
		BigDecimal[] x = { BigDecimal.ONE, BigDecimal.ONE };
		CobylaExitStatus result = Cobyla.findMinimum(calcfc, mc, 2, 0, x, rhobeg,
				rhoend, iprint, maxfun);
		assertArrayEquals(null, new double[] { -1.0, 0.0 }, x, 1.0e-5);
		assertSame(CobylaExitStatus.Normal, result);
	}

	/**
	 * Easy two dimensional minimization in unit circle.
	 */
	@Test
	public void test02FindMinimum()
	{
		System.out
				.format("%nOutput from test problem 2 (2D unit circle calculation)%n");
		Calcfc calcfc = new Calcfc()
		{
			@Override
			public BigDecimal compute(int n, int m, BigDecimal[] x, BigDecimal[] con, MathContext mc)
			{
				con[0] = BigDecimal.ONE.subtract(square(x[0], mc), mc).subtract(square(x[1], mc), mc);
				return x[0].multiply(x[1], mc);
			}
		};
		BigDecimal[] x = { BigDecimal.ONE, BigDecimal.ONE };
		CobylaExitStatus result = Cobyla.findMinimum(calcfc, mc, 2, 1, x, rhobeg,
				rhoend, iprint, maxfun);
		assertArrayEquals(null, new double[] { Math.sqrt(0.5), -Math.sqrt(0.5) }, x,
				1.0e-5);
		assertSame(CobylaExitStatus.Normal, result);
	}

	/**
	 * Easy three dimensional minimization in ellipsoid.
	 */
	@Test
	public void test03FindMinimum()
	{
		System.out
				.format("%nOutput from test problem 3 (3D ellipsoid calculation)%n");
		Calcfc calcfc = new Calcfc()
		{
			@Override
			public BigDecimal compute(int n, int m, BigDecimal[] x, BigDecimal[] con, MathContext mc)
			{
				con[0] = BigDecimal.ONE
						.subtract(square(x[0], mc), mc)
						.subtract(convert(2.0, mc).multiply(square(x[1], mc), mc), mc)
						.subtract(convert(3.0, mc).multiply(square(x[2], mc), mc), mc);
				return x[0].multiply(x[1], mc).multiply(x[2], mc);
			}
		};
		BigDecimal[] x = { BigDecimal.ONE, BigDecimal.ONE, BigDecimal.ONE };
		CobylaExitStatus result = Cobyla.findMinimum(calcfc, mc, 3, 1, x, rhobeg,
				rhoend, iprint, maxfun);
		assertArrayEquals(null,
				new double[] { 1.0 / Math.sqrt(3.0), 1.0 / Math.sqrt(6.0), -1.0 / 3.0 },
				x, 1.0e-5);
		assertSame(CobylaExitStatus.Normal, result);
	}

	/**
	 * Weak version of Rosenbrock's problem.
	 */
	@Test
	public void test04FindMinimum()
	{
		System.out.format("%nOutput from test problem 4 (Weak Rosenbrock)%n");
		Calcfc calcfc = new Calcfc()
		{
			@Override
			public BigDecimal compute(int n, int m, BigDecimal[] x, BigDecimal[] con, MathContext mc)
			{
				return square(square(x[0], mc).subtract(x[1], mc), mc)
						.add(square(BigDecimal.ONE.add(x[0], mc), mc), mc);
			}
		};
		BigDecimal[] x = { BigDecimal.ONE, BigDecimal.ONE };
		CobylaExitStatus result = Cobyla.findMinimum(calcfc, mc, 2, 0, x, rhobeg,
				rhoend, iprint, maxfun);
		assertArrayEquals(null, new double[] { -1.0, 1.0 }, x, 1.0e-4);
		assertSame(CobylaExitStatus.Normal, result);
	}

	/**
	 * Intermediate version of Rosenbrock's problem.
	 */
	@Test
	public void test05FindMinimum()
	{
		System.out
				.format("%nOutput from test problem 5 (Intermediate Rosenbrock)%n");
		Calcfc calcfc = new Calcfc()
		{
			@Override
			public BigDecimal compute(int n, int m, BigDecimal[] x, BigDecimal[] con, MathContext mc)
			{
				return BigDecimal.TEN.multiply(square(square(x[0], mc).subtract(x[1], mc), mc), mc)
						.add(square(BigDecimal.ONE.add(x[0], mc), mc), mc);
			}
		};
		BigDecimal[] x = { BigDecimal.ONE, BigDecimal.ONE };
		CobylaExitStatus result = Cobyla.findMinimum(calcfc, mc, 2, 0, x, rhobeg,
				rhoend, iprint, maxfun);
		assertArrayEquals(null, new double[] { -1.0, 1.0 }, x, 3.0e-4);
		assertSame(CobylaExitStatus.Normal, result);
	}

	/**
	 * This problem is taken from Fletcher's book Practical Methods of
	 * Optimization and has the equation number (9.1.15).
	 */
	@Test
	public void test06FindMinimum()
	{
		System.out.format(
				"%nOutput from test problem 6 (Equation (9.1.15) in Fletcher's book)%n");
		Calcfc calcfc = new Calcfc()
		{
			@Override
			public BigDecimal compute(int n, int m, BigDecimal[] x, BigDecimal[] con, MathContext mc)
			{
				con[0] = x[1].subtract(square(x[0], mc), mc);
				con[1] = BigDecimal.ONE.subtract(square(x[0], mc), mc).subtract(square(x[1], mc), mc);
				return x[0].add(x[1], mc).negate();
			}
		};
		BigDecimal[] x = { BigDecimal.ONE, BigDecimal.ONE };
		CobylaExitStatus result = Cobyla.findMinimum(calcfc, mc, 2, 2, x, rhobeg,
				rhoend, iprint, maxfun);
		assertArrayEquals(null, new double[] { Math.sqrt(0.5), Math.sqrt(0.5) }, x,
				1.0e-5);
		assertSame(CobylaExitStatus.Normal, result);
	}

	/**
	 * This problem is taken from Fletcher's book Practical Methods of
	 * Optimization and has the equation number (14.4.2).
	 */
	@Test
	public void test07FindMinimum()
	{
		System.out.format(
				"%nOutput from test problem 7 (Equation (14.4.2) in Fletcher)%n");
		Calcfc calcfc = new Calcfc()
		{
			@Override
			public BigDecimal compute(int n, int m, BigDecimal[] x, BigDecimal[] con, MathContext mc)
			{
				con[0] = convert(5.0, mc).multiply(x[0], mc).subtract(x[1], mc).add(x[2], mc);
				con[1] = x[2]
						.subtract(square(x[0], mc), mc)
						.subtract(square(x[1], mc), mc)
						.subtract(convert(4.0, mc).multiply(x[1], mc), mc);
				con[2] = x[2]
						.subtract(convert(5.0, mc).multiply(x[0], mc), mc)
						.subtract(x[1], mc);
				return x[2];
			}
		};
		BigDecimal[] x = { BigDecimal.ONE, BigDecimal.ONE, BigDecimal.ONE };
		CobylaExitStatus result = Cobyla.findMinimum(calcfc, mc, 3, 3, x, rhobeg,
				rhoend, iprint, maxfun);
		assertArrayEquals(null, new double[] { 0.0, -3.0, -3.0 }, x, 1.0e-5);
		assertSame(CobylaExitStatus.Normal, result);
	}

	/**
	 * This problem is taken from page 66 of Hock and Schittkowski's book Test
	 * Examples for Nonlinear Programming Codes. It is their test problem Number
	 * 43, and has the name Rosen-Suzuki.
	 */
	@Test
	public void test08FindMinimum()
	{
		System.out.format("%nOutput from test problem 8 (Rosen-Suzuki)%n");
		Calcfc calcfc = new Calcfc()
		{
			@Override
			public BigDecimal compute(int n, int m, BigDecimal[] x, BigDecimal[] con, MathContext mc)
			{
				con[0] = convert(8.0, mc)
						.subtract(square(x[0], mc), mc)
						.subtract(square(x[1], mc), mc)
						.subtract(square(x[2], mc), mc)
						.subtract(square(x[3], mc), mc)
						.subtract(x[0], mc).add(x[1], mc).subtract(x[2], mc).add(x[3], mc);
				con[1] = convert(10.0, mc)
						.subtract(square(x[0], mc), mc)
						.subtract(square(x[1], mc).multiply(convert(2.0, mc), mc), mc)
						.subtract(square(x[2], mc), mc)
						.subtract(square(x[3], mc).multiply(convert(2.0, mc), mc), mc)
						.add(x[0], mc).add(x[3], mc);
				con[2] = convert(5.0, mc)
						.subtract(square(x[0], mc).multiply(convert(2.0, mc), mc), mc)
						.subtract(square(x[1], mc), mc)
						.subtract(square(x[2], mc), mc)
						.subtract(x[0].multiply(convert(2.0, mc), mc), mc).add(x[1], mc).add(x[3], mc);
				return square(x[0], mc)
						.add(square(x[1], mc), mc)
						.add(square(x[2], mc).multiply(convert(2.0, mc), mc), mc)
						.add(square(x[3], mc), mc)
						.subtract(x[0].multiply(convert(5.0, mc), mc), mc)
						.subtract(x[1].multiply(convert(5.0, mc), mc), mc)
						.subtract(x[2].multiply(convert(21.0, mc), mc), mc)
						.add(x[3].multiply(convert(7.0, mc), mc), mc);
			}
		};
		BigDecimal[] x = { BigDecimal.ONE, BigDecimal.ONE, BigDecimal.ONE, BigDecimal.ONE };
		CobylaExitStatus result = Cobyla.findMinimum(calcfc, mc, 4, 3, x, rhobeg,
				rhoend, iprint, maxfun);
		assertArrayEquals(null, new double[] { 0.0, 1.0, 2.0, -1.0 }, x, 1.0e-5);
		assertSame(CobylaExitStatus.Normal, result);
	}

	/**
	 * This problem is taken from page 111 of Hock and Schittkowski's book Test
	 * Examples for Nonlinear Programming Codes. It is their test problem Number
	 * 100.
	 */
	@Test
	public void test09FindMinimum()
	{
		System.out
				.format("%nOutput from test problem 9 (Hock and Schittkowski 100)%n");
		Calcfc calcfc = new Calcfc()
		{
			@Override
			public BigDecimal compute(int n, int m, BigDecimal[] x, BigDecimal[] con, MathContext mc)
			{
				con[0] = convert(127.0, mc)
						.subtract(square(x[0], mc).multiply(convert(2.0, mc), mc), mc)
						.subtract(pow(x[1], 4, mc).multiply(convert(3.0, mc), mc), mc)
						.subtract(x[2], mc)
						.subtract(square(x[3], mc).multiply(convert(4.0, mc), mc), mc)
						.subtract(x[4].multiply(convert(5.0, mc), mc), mc);
				con[1] = convert(282.0, mc)
						.subtract(x[0].multiply(convert(7.0, mc), mc), mc)
						.subtract(x[1].multiply(convert(3.0, mc), mc), mc)
						.subtract(square(x[2].multiply(BigDecimal.TEN, mc), mc), mc)
						.subtract(x[3], mc).add(x[4], mc);
				con[2] = convert(196.0, mc)
						.subtract(x[0].multiply(convert(23.0,  mc), mc))
						.subtract(square(x[1], mc), mc)
						.subtract(square(x[5], mc).multiply(convert(6.0, mc), mc), mc)
						.add(x[6].multiply(convert(8.0, mc), mc), mc);
				con[3] = square(x[0], mc).multiply(convert(-4.0, mc), mc)
						.subtract(square(x[1], mc), mc)
						.add(x[0].multiply(x[1], mc).multiply(convert(3.0, mc), mc), mc)
						.subtract(square(x[2], mc).multiply(convert(2.0, mc), mc), mc)
						.subtract(x[5].multiply(convert(5.0, mc), mc), mc)
						.add(x[6].multiply(convert(11.0, mc), mc), mc);
				return square(x[0].subtract(BigDecimal.TEN, mc), mc)
						.add(square(x[1].subtract(convert(12.0, mc), mc), mc).multiply(convert(5.0, mc), mc), mc)
						.add(pow(x[2], 4, mc), mc)
						.add(square(x[3].subtract(convert(11.0, mc), mc), mc).multiply(convert(3.0, mc), mc), mc)
						.add(pow(x[4], 6, mc).multiply(BigDecimal.TEN, mc), mc)
						.add(square(x[5], mc).multiply(convert(7.0, mc), mc), mc)
						.add(pow(x[6], 4, mc), mc)
						.subtract(x[5].multiply(x[6], mc).multiply(convert(4.0, mc), mc), mc)
						.subtract(x[5].multiply(BigDecimal.TEN, mc), mc)
						.subtract(x[6].multiply(convert(8.0, mc), mc), mc);
			}
		};
		BigDecimal[] x = { BigDecimal.ONE, BigDecimal.ONE, BigDecimal.ONE, BigDecimal.ONE, BigDecimal.ONE, BigDecimal.ONE, BigDecimal.ONE };
		CobylaExitStatus result = Cobyla.findMinimum(calcfc, mc, 7, 4, x, rhobeg,
				rhoend, iprint, maxfun);
		assertArrayEquals(null, new double[] { 2.330499, 1.951372, -0.4775414,
				4.365726, -0.624487, 1.038131, 1.594227 }, x, 1.0e-5);
		assertSame(CobylaExitStatus.Normal, result);
	}

	/**
	 * This problem is taken from page 415 of Luenberger's book Applied Nonlinear
	 * Programming. It is to maximize the area of a hexagon of unit diameter.
	 */
	@Test
	@Ignore
	// FIXME
	// This test fails after switching to BigDecimal, the other ones work.
	// Investigations in progress.
	public void test10FindMinimum()
	{
		System.out.format("%nOutput from test problem 10 (Hexagon area)%n");
		Calcfc calcfc = new Calcfc()
		{
			@Override
			public BigDecimal compute(int n, int m, BigDecimal[] x, BigDecimal[] con, MathContext mc)
			{
				con[0] = BigDecimal.ONE
						.subtract(square(x[2], mc), mc)
						.subtract(square(x[3], mc), mc);
				con[1] = BigDecimal.ONE
						.subtract(square(x[8], mc), mc);
				con[2] = BigDecimal.ONE
						.subtract(square(x[4], mc), mc)
						.subtract(square(x[5], mc), mc);
				con[3] = BigDecimal.ONE
						.subtract(square(x[0], mc), mc)
						.subtract(square(x[1].subtract(x[8], mc), mc), mc);
				con[4] = BigDecimal.ONE
						.subtract(square(x[0].subtract(x[4], mc), mc), mc)
						.subtract(square(x[1].subtract(x[5], mc), mc), mc);
				con[5] = BigDecimal.ONE
						.subtract(square(x[0].subtract(x[6], mc), mc), mc)
						.subtract(square(x[1].subtract(x[7], mc), mc), mc);
				con[6] = BigDecimal.ONE
						.subtract(square(x[2].subtract(x[4], mc), mc), mc)
						.subtract(square(x[3].subtract(x[5], mc), mc), mc);
				con[7] = BigDecimal.ONE
						.subtract(square(x[2].subtract(x[6], mc), mc), mc)
						.subtract(square(x[3].subtract(x[7], mc), mc), mc);
				con[8] = BigDecimal.ONE
						.subtract(square(x[6], mc), mc)
						.subtract(square(x[7].subtract(x[8], mc), mc), mc);
				con[9] = x[0].multiply(x[3], mc).subtract(x[1].multiply(x[2], mc), mc);
				con[10] = x[2].multiply(x[8], mc);
				con[11] = x[4].multiply(x[8], mc).negate();
				con[12] = x[4].multiply(x[7], mc).subtract(x[5].multiply(x[6], mc), mc);
				con[13] = x[8];
				return x[0].multiply(x[3], mc)
						.subtract(x[1].multiply(x[2], mc), mc)
						.add(x[2].multiply(x[8], mc), mc)
						.subtract(x[4].multiply(x[8], mc), mc)
						.add(x[4].multiply(x[7], mc), mc)
						.subtract(x[5].multiply(x[6], mc), mc)
						.multiply(convert(-0.5, mc), mc);
			}
		};
		BigDecimal[] x = { BigDecimal.ONE, BigDecimal.ONE, BigDecimal.ONE, BigDecimal.ONE, BigDecimal.ONE, BigDecimal.ONE, BigDecimal.ONE, BigDecimal.ONE, BigDecimal.ONE };
		CobylaExitStatus result = Cobyla.findMinimum(calcfc, mc, 9, 14, x, rhobeg,
				rhoend, iprint, maxfun);
		assertArrayEquals(null,
				new double[] {
						x[0].doubleValue(), x[1].doubleValue(), x[2].doubleValue(), x[3].doubleValue(),
						x[0].doubleValue(), x[1].doubleValue(), x[2].doubleValue(), x[3].doubleValue(), 0.0 }, x,
				1.0e-4);
		assertSame(CobylaExitStatus.Normal, result);
	}

}
