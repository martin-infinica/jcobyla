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

import java.math.BigDecimal;
import java.math.MathContext;
import java.util.Arrays;

/**
 * Constrained Optimization BY Linear Approximation in Java.
 * 
 * COBYLA2 is an implementation of Powell’s nonlinear derivative–free
 * constrained optimization that uses a linear approximation approach. The
 * algorithm is a sequential trust–region algorithm that employs linear
 * approximations to the objective and constraint functions, where the
 * approximations are formed by linear interpolation at n + 1 points in the
 * space of the variables and tries to maintain a regular–shaped simplex over
 * iterations.
 * 
 * It solves nonsmooth NLP with a moderate number of variables (about 100).
 * Inequality constraints only.
 * 
 * The initial point X is taken as one vertex of the initial simplex with zero
 * being another, so, X should not be entered as the zero vector.
 * 
 * @author Anders Gustafsson, Cureos AB.
 */
public class Cobyla
{
	/**
	 * Minimizes the objective function F with respect to a set of inequality
	 * constraints CON, and returns the optimal variable array. F and CON may be
	 * non-linear, and should preferably be smooth.
	 * 
	 * @param calcfc
	 *          Interface implementation for calculating objective function and
	 *          constraints.
	 * @param n
	 *          Number of variables.
	 * @param m
	 *          Number of constraints.
	 * @param x
	 *          On input initial values of the variables (zero-based array). On
	 *          output optimal values of the variables obtained in the COBYLA
	 *          minimization.
	 * @param rhobeg
	 *          Initial size of the simplex.
	 * @param rhoend
	 *          Final value of the simplex.
	 * @param iprint
	 *          Print level, 0 &lt;= iprint &lt;= 3, where 0 provides no output
	 *          and 3 provides full output to the console.
	 * @param maxfun
	 *          Maximum number of function evaluations before terminating.
	 * @return Exit status of the COBYLA2 optimization.
	 */
	public static CobylaExitStatus findMinimum(final Calcfc calcfc, MathContext mc, int n, int m,
			BigDecimal[] x, BigDecimal rhobeg, BigDecimal rhoend, int iprint, int maxfun)
	{
		// This subroutine minimizes an objective function F(X) subject to M
		// inequality constraints on X, where X is a vector of variables that has
		// N components. The algorithm employs linear approximations to the
		// objective and constraint functions, the approximations being formed by
		// linear interpolation at N+1 points in the space of the variables.
		// We regard these interpolation points as vertices of a simplex. The
		// parameter RHO controls the size of the simplex and it is reduced
		// automatically from RHOBEG to RHOEND. For each RHO the subroutine tries
		// to achieve a good vector of variables for the current size, and then
		// RHO is reduced until the value RHOEND is reached. Therefore RHOBEG and
		// RHOEND should be set to reasonable initial changes to and the required
		// accuracy in the variables respectively, but this accuracy should be
		// viewed as a subject for experimentation because it is not guaranteed.
		// The subroutine has an advantage over many of its competitors, however,
		// which is that it treats each constraint individually when calculating
		// a change to the variables, instead of lumping the constraints together
		// into a single penalty function. The name of the subroutine is derived
		// from the phrase Constrained Optimization BY Linear Approximations.

		// The user must set the values of N, M, RHOBEG and RHOEND, and must
		// provide an initial vector of variables in X. Further, the value of
		// IPRINT should be set to 0, 1, 2 or 3, which controls the amount of
		// printing during the calculation. Specifically, there is no output if
		// IPRINT=0 and there is output only at the end of the calculation if
		// IPRINT=1. Otherwise each new value of RHO and SIGMA is printed.
		// Further, the vector of variables and some function information are
		// given either when RHO is reduced or when each new value of F(X) is
		// computed in the cases IPRINT=2 or IPRINT=3 respectively. Here SIGMA
		// is a penalty parameter, it being assumed that a change to X is an
		// improvement if it reduces the merit function
		// F(X)+SIGMA*MAX(0.0, - C1(X), - C2(X),..., - CM(X)),
		// where C1,C2,...,CM denote the constraint functions that should become
		// nonnegative eventually, at least to the precision of RHOEND. In the
		// printed output the displayed term that is multiplied by SIGMA is
		// called MAXCV, which stands for 'MAXimum Constraint Violation'. The
		// argument ITERS is an integer variable that must be set by the user to a
		// limit on the number of calls of CALCFC, the purpose of this routine being
		// given below. The value of ITERS will be altered to the number of calls
		// of CALCFC that are made.

		// In order to define the objective and constraint functions, we require
		// a subroutine that has the name and arguments
		// SUBROUTINE CALCFC (N,M,X,F,CON)
		// DIMENSION X(:),CON(:) .
		// The values of N and M are fixed and have been defined already, while
		// X is now the current vector of variables. The subroutine should return
		// the objective and constraint functions at X in F and CON(1),CON(2),
		// ...,CON(M). Note that we are trying to adjust X so that F(X) is as
		// small as possible subject to the constraint functions being nonnegative.

		// Local variables
		int mpp = m + 2;

		// Internal base-1 X array
		BigDecimal[] iox = new BigDecimal[n + 1];
		System.arraycopy(x, 0, iox, 1, n);

		// Internal representation of the objective and constraints calculation
		// method,
		// accounting for that X and CON arrays in the cobylb method are base-1
		// arrays.
		Calcfc fcalcfc = new Calcfc()
		{
			/**
			 *
			 * @param n
			 *          the value of n
			 * @param m
			 *          the value of m
			 * @param x
			 *          the values of x (input)
			 * @param con
			 *          the values of con (output)
			 * @return the double
			 */
			@Override
			public BigDecimal compute(int n, int m, BigDecimal[] x, BigDecimal[] con, MathContext mc)
			{
				BigDecimal[] ix = new BigDecimal[n];
				System.arraycopy(x, 1, ix, 0, n);
				BigDecimal[] ocon = new BigDecimal[m];
				BigDecimal f = calcfc.compute(n, m, ix, ocon, mc);
				System.arraycopy(ocon, 0, con, 1, m);
				return f;
			}
		};

		CobylaExitStatus status = cobylb(fcalcfc, mc, n, m, mpp, iox, rhobeg, rhoend,
				iprint, maxfun);
		System.arraycopy(iox, 1, x, 0, n);

		return status;
	}

	private static BigDecimal convert(double value, MathContext mc)
	{
		try
		{
			return new BigDecimal(value, mc);
		}
		catch(NumberFormatException ex)
		{
			ex.printStackTrace();
			throw ex;
		}
	}

	private static BigDecimal epsilon(MathContext mc)
	{
		// The original epsilon was 1.0E-6, which worked for double precision.
		// For BigDecimal, the epsilon is adapted to the precision in use.
		return BigDecimal.ONE.movePointLeft(mc.getPrecision() / 2);
	}

	private static BigDecimal square(BigDecimal v, MathContext mc)
	{
		return v.multiply(v, mc);
	}

	private static void fill(BigDecimal[] row)
	{
		Arrays.fill(row, BigDecimal.ZERO);
	}

	private static void fill(BigDecimal[][] matrix)
	{
		for (BigDecimal[] row: matrix)
		{
			fill(row);
		}
	}

	private static final double alpha = 0.25;
	private static final double beta = 2.1;
	private static final double gamma = 0.5;
	private static final double delta = 1.1;
	private static final double stepSize = 0.1;

	private static CobylaExitStatus cobylb(Calcfc calcfc, MathContext mc, int n, int m, int mpp,
			BigDecimal[] x, BigDecimal rhobeg, BigDecimal rhoend, int iprint, int maxfun)
	{
		// N.B. Arguments CON, SIM, SIMI, DATMAT, A, VSIG, VETA, SIGBAR, DX, W &
		// IACT
		// have been removed.

		// Set the initial values of some parameters. The last column of SIM holds
		// the optimal vertex of the current simplex, and the preceding N columns
		// hold the displacements from the optimal vertex to the other vertices.
		// Further, SIMI holds the inverse of the matrix that is contained in the
		// first N columns of SIM.

		// Local variables

		CobylaExitStatus status;

		final BigDecimal alpha = convert(Cobyla.alpha, mc);
		final BigDecimal beta = convert(Cobyla.beta, mc);
		final BigDecimal gamma = convert(Cobyla.gamma, mc);
		final BigDecimal delta = convert(Cobyla.delta, mc);

		BigDecimal f = BigDecimal.ZERO;
		BigDecimal resmax = BigDecimal.ZERO;
		BigDecimal total;

		int np = n + 1;
		int mp = m + 1;
		BigDecimal rho = rhobeg;
		BigDecimal parmu = BigDecimal.ZERO;

		boolean iflag = false;
		boolean ifull = false;
		BigDecimal parsig = BigDecimal.ZERO;
		BigDecimal prerec = BigDecimal.ZERO;
		BigDecimal prerem = BigDecimal.ZERO;

		final BigDecimal[] con = new BigDecimal[1 + mpp];
		final BigDecimal[][] sim = new BigDecimal[1 + n][1 + np];
		final BigDecimal[][] simi = new BigDecimal[1 + n][1 + n];
		final BigDecimal[][] datmat = new BigDecimal[1 + mpp][1 + np];
		final BigDecimal[][] a = new BigDecimal[1 + n][1 + mp];
		final BigDecimal[] vsig = new BigDecimal[1 + n];
		final BigDecimal[] veta = new BigDecimal[1 + n];
		final BigDecimal[] sigbar = new BigDecimal[1 + n];
		final BigDecimal[] dx = new BigDecimal[1 + n];
		final BigDecimal[] w = new BigDecimal[1 + n];

		// The original double arrays where instantiated with 0.0 values.
		// The BigDecimal arrays are instantiated with nulls, so we need to fill in zeros for the same behaviour.
		fill(con);
		fill(sim);
		fill(simi);
		fill(datmat);
		fill(a);
		fill(vsig);
		fill(veta);
		fill(sigbar);
		fill(dx);
		fill(w);

		if (iprint >= 2)
		{
			System.out.format(
					"%nThe initial value of RHO is %s and PARMU is set to zero.%n",
					rho.toString());
		}

		int nfvals = 0;
		BigDecimal temp = BigDecimal.ONE.divide(rho, mc);

		for(int i = 1; i <= n; ++i)
		{
			sim[i][np] = x[i];
			sim[i][i] = rho;
			simi[i][i] = temp;
		}

		int jdrop = np;
		boolean ibrnch = false;

		// Make the next call of the user-supplied subroutine CALCFC. These
		// instructions are also used for calling CALCFC during the iterations of
		// the algorithm.

		L_40: do
		{
			if (nfvals >= maxfun && nfvals > 0)
			{
				status = CobylaExitStatus.MaxIterationsReached;
				break L_40;
			}

			++nfvals;

			f = calcfc.compute(n, m, x, con, mc);
			resmax = BigDecimal.ZERO;
			for(int k = 1; k <= m; ++k)
			{
				resmax = resmax.max(con[k].negate());
			}

			if (nfvals == iprint - 1 || iprint == 3)
			{
				printIterationResult(nfvals, f, resmax, x, n, calcfc);
			}

			con[mp] = f;
			con[mpp] = resmax;

			// Set the recently calculated function values in a column of DATMAT. This
			// array has a column for each vertex of the current simplex, the entries
			// of
			// each column being the values of the constraint functions (if any)
			// followed by the objective function and the greatest constraint
			// violation
			// at the vertex.

			boolean skipVertexIdent = true;
			if (!ibrnch)
			{
				skipVertexIdent = false;

				for(int i = 1; i <= mpp; ++i)
				{
					datmat[i][jdrop] = con[i];
				}

				if (nfvals <= np)
				{
					// Exchange the new vertex of the initial simplex with the optimal
					// vertex if
					// necessary. Then, if the initial simplex is not complete, pick its
					// next
					// vertex and calculate the function values there.

					if (jdrop <= n)
					{
						if (datmat[mp][np].compareTo(f) <= 0)
						{
							x[jdrop] = sim[jdrop][np];
						}
						else
						{
							sim[jdrop][np] = x[jdrop];
							for(int k = 1; k <= mpp; ++k)
							{
								datmat[k][jdrop] = datmat[k][np];
								datmat[k][np] = con[k];
							}
							for(int k = 1; k <= jdrop; ++k)
							{
								sim[jdrop][k] = rho.negate();
								temp = BigDecimal.ZERO;
								for(int i = k; i <= jdrop; ++i)
								{
									temp = temp.subtract(simi[i][k], mc);
								}
								simi[jdrop][k] = temp;
							}
						}
					}
					if (nfvals <= n)
					{
						jdrop = nfvals;
//						x[jdrop] += rho;
						x[jdrop] = x[jdrop].add(rho, mc);
						continue L_40;
					}
				}

				ibrnch = true;
			}

			L_140: do
			{
				L_550: do
				{
					if (!skipVertexIdent)
					{
						// Identify the optimal vertex of the current simplex.

						BigDecimal phimin = datmat[mp][np].add(parmu.multiply(datmat[mpp][np], mc), mc);
						int nbest = np;

						for(int j = 1; j <= n; ++j)
						{
							temp = datmat[mp][j].add(parmu.multiply(datmat[mpp][j], mc), mc);
							if (temp.compareTo(phimin) < 0)
							{
								nbest = j;
								phimin = temp;
							}
							else if (temp.compareTo(phimin) == 0 && parmu.signum() == 0
									&& datmat[mpp][j].compareTo(datmat[mpp][nbest]) < 0)
							{
								nbest = j;
							}
						}

						// Switch the best vertex into pole position if it is not there
						// already,
						// and also update SIM, SIMI and DATMAT.

						if (nbest <= n)
						{
							for(int i = 1; i <= mpp; ++i)
							{
								temp = datmat[i][np];
								datmat[i][np] = datmat[i][nbest];
								datmat[i][nbest] = temp;
							}
							for(int i = 1; i <= n; ++i)
							{
								temp = sim[i][nbest];
								sim[i][nbest] = BigDecimal.ZERO;
								sim[i][np] = sim[i][np].add(temp, mc);

								BigDecimal tempa = BigDecimal.ZERO;
								for(int k = 1; k <= n; ++k)
								{
									sim[i][k] = sim[i][k].subtract(temp, mc);
									tempa = tempa.subtract(simi[k][i], mc);
								}
								simi[nbest][i] = tempa;
							}
						}

						// Make an error return if SIGI is a poor approximation to the
						// inverse of
						// the leading N by N submatrix of SIG.

						BigDecimal error = BigDecimal.ZERO;
						for(int i = 1; i <= n; ++i)
						{
							for(int j = 1; j <= n; ++j)
							{
								temp = dotProduct(part(row(simi, i), 1, n), part(col(sim, j), 1, n), mc);
								if (i == j)
								{
									temp = temp.subtract(BigDecimal.ONE, mc);
								}
								error = error.max(temp.abs());
							}
						}
						if (error.compareTo(convert(0.1, mc)) > 0)
						{
							status = CobylaExitStatus.DivergingRoundingErrors;
							break L_40;
						}

						// Calculate the coefficients of the linear approximations to the
						// objective
						// and constraint functions, placing minus the objective function
						// gradient
						// after the constraint gradients in the array A. The vector W is
						// used for
						// working space.

						for(int k = 1; k <= mp; ++k)
						{
							con[k] = datmat[k][np].negate();
							for(int j = 1; j <= n; ++j)
							{
								w[j] = datmat[k][j].add(con[k], mc);
							}

							for(int i = 1; i <= n; ++i)
							{
								BigDecimal dp = dotProduct(part(w, 1, n), part(col(simi, i), 1, n), mc);
								a[i][k] = (k == mp ? dp.negate() : dp);
							}
						}

						// Calculate the values of sigma and eta, and set IFLAG = 0 if the
						// current
						// simplex is not acceptable.

						iflag = true;
						parsig = alpha.multiply(rho, mc);
						BigDecimal pareta = beta.multiply(rho, mc);

						for(int j = 1; j <= n; ++j)
						{
							BigDecimal wsig = BigDecimal.ZERO;
							for(int k = 1; k <= n; ++k)
							{
								wsig = wsig.add(square(simi[j][k], mc), mc);
							}
							BigDecimal weta = BigDecimal.ZERO;
							for(int k = 1; k <= n; ++k)
							{
								weta = weta.add(square(sim[k][j], mc), mc);
							}
							vsig[j] = BigDecimal.ONE.divide(wsig.sqrt(mc), mc);
							veta[j] = weta.sqrt(mc);
							if (vsig[j].compareTo(parsig) < 0 || veta[j].compareTo(pareta) > 0)
							{
								iflag = false;
							}
						}

						// If a new vertex is needed to improve acceptability, then decide
						// which
						// vertex to drop from the simplex.

						if (!ibrnch && !iflag)
						{
							jdrop = 0;
							temp = pareta;
							for(int j = 1; j <= n; ++j)
							{
								if (veta[j].compareTo(temp) > 0)
								{
									jdrop = j;
									temp = veta[j];
								}
							}
							if (jdrop == 0)
							{
								for(int j = 1; j <= n; ++j)
								{
									if (vsig[j].compareTo(temp) < 0)
									{
										jdrop = j;
										temp = vsig[j];
									}
								}
							}

							// Calculate the step to the new vertex and its sign.

							temp = gamma.multiply(rho, mc).multiply(vsig[jdrop], mc);
							for(int k = 1; k <= n; ++k)
							{
								dx[k] = temp.multiply(simi[jdrop][k], mc);
							}
							BigDecimal cvmaxp = BigDecimal.ZERO;
							BigDecimal cvmaxm = BigDecimal.ZERO;

							total = BigDecimal.ZERO;
							for(int k = 1; k <= mp; ++k)
							{
								total = dotProduct(part(col(a, k), 1, n), part(dx, 1, n), mc);
								if (k < mp)
								{
									temp = datmat[k][np];
									cvmaxp = cvmaxp.max(total.negate().subtract(temp, mc));
									cvmaxm = cvmaxm.max(total.subtract(temp, mc));
								}
							}
							boolean dxneg = parmu.multiply(cvmaxp.subtract(cvmaxm, mc), mc).compareTo(total.add(total, mc)) > 0;

							// Update the elements of SIM and SIMI, and set the next X.

							temp = BigDecimal.ZERO;
							for(int i = 1; i <= n; ++i)
							{
								if (dxneg)
								{
									dx[i] = dx[i].negate();
								}
								sim[i][jdrop] = dx[i];
								temp = temp.add(simi[jdrop][i].multiply(dx[i], mc), mc);
							}
							for(int k = 1; k <= n; ++k)
							{
								simi[jdrop][k] = simi[jdrop][k].divide(temp, mc);
							}

							for(int j = 1; j <= n; ++j)
							{
								if (j != jdrop)
								{
									temp = dotProduct(part(row(simi, j), 1, n), part(dx, 1, n), mc);
									for(int k = 1; k <= n; ++k)
									{
										simi[j][k] = simi[j][k].subtract(temp.multiply(simi[jdrop][k], mc), mc);
									}
								}
								x[j] = sim[j][np].add(dx[j], mc);
							}
							continue L_40;
						}

						// Calculate DX = x(*)-x(0).
						// Branch if the length of DX is less than 0.5*RHO.

						ifull = trstlp(n, m, a, con, rho, dx, mc);
						if (!ifull)
						{
							temp = BigDecimal.ZERO;
							for(int k = 1; k <= n; ++k)
							{
								temp = temp.add(square(dx[k], mc), mc);
							}
							if (temp.multiply(convert(4.0, mc), mc).compareTo(square(rho, mc)) < 0)
							{
								ibrnch = true;
								break L_550;
							}
						}

						// Predict the change to F and the new maximum constraint violation
						// if the
						// variables are altered from x(0) to x(0) + DX.

						total = BigDecimal.ZERO;
						BigDecimal resnew = BigDecimal.ZERO;
						con[mp] = BigDecimal.ZERO;
						for(int k = 1; k <= mp; ++k)
						{
							total = con[k].subtract(dotProduct(part(col(a, k), 1, n), part(dx, 1, n), mc), mc);
							if (k < mp)
							{
								resnew = resnew.max(total);
							}
						}

						// Increase PARMU if necessary and branch back if this change alters
						// the
						// optimal vertex. Otherwise PREREM and PREREC will be set to the
						// predicted
						// reductions in the merit function and the maximum constraint
						// violation
						// respectively.

						prerec = datmat[mpp][np].subtract(resnew, mc);
						BigDecimal barmu = prerec.signum() > 0 ? total.divide(prerec, mc) : BigDecimal.ZERO;
						if (parmu.compareTo(convert(1.5, mc).multiply(barmu, mc)) < 0)
						{
							parmu = convert(2.0, mc).multiply(barmu, mc);
							if (iprint >= 2)
							{
								System.out.format("%nIncrease in PARMU to %s%n", parmu.toString());
							}
							BigDecimal phi = datmat[mp][np].add(parmu.multiply(datmat[mpp][np], mc), mc);
							for(int j = 1; j <= n; ++j)
							{
								temp = datmat[mp][j].add(parmu.multiply(datmat[mpp][j], mc), mc);
								if (temp.compareTo(phi) < 0 || (temp.compareTo(phi) == 0 && parmu.signum() == 0
										&& datmat[mpp][j].compareTo(datmat[mpp][np]) < 0))
								{
									continue L_140;
								}
							}
						}
						prerem = parmu.multiply(prerec, mc).subtract(total, mc);

						// Calculate the constraint and objective functions at x(*).
						// Then find the actual reduction in the merit function.

						for(int k = 1; k <= n; ++k)
						{
							x[k] = sim[k][np].add(dx[k], mc);
						}
						ibrnch = true;
						continue L_40;
					}

					skipVertexIdent = false;
					BigDecimal vmold = datmat[mp][np].add(parmu.multiply(datmat[mpp][np], mc), mc);
					BigDecimal vmnew = f.add(parmu.multiply(resmax, mc), mc);
					BigDecimal trured = vmold.subtract(vmnew, mc);
					if (parmu.signum() == 0 && f.compareTo(datmat[mp][np]) == 0)
					{
						prerem = prerec;
						trured = datmat[mpp][np].subtract(resmax, mc);
					}

					// Begin the operations that decide whether x(*) should replace one of
					// the
					// vertices of the current simplex, the change being mandatory if
					// TRURED is
					// positive. Firstly, JDROP is set to the index of the vertex that is
					// to be
					// replaced.

					BigDecimal ratio = trured.signum() <= 0 ? BigDecimal.ONE : BigDecimal.ZERO;
					jdrop = 0;
					for(int j = 1; j <= n; ++j)
					{
						temp = dotProduct(part(row(simi, j), 1, n), part(dx, 1, n), mc).abs();
						if (temp.compareTo(ratio) > 0)
						{
							jdrop = j;
							ratio = temp;
						}
						sigbar[j] = temp.multiply(vsig[j], mc);
					}

					// Calculate the value of ell.

					BigDecimal edgmax = delta.multiply(rho, mc);
					int l = 0;
					for(int j = 1; j <= n; ++j)
					{
						if (sigbar[j].compareTo(parsig) >= 0 || sigbar[j].compareTo(vsig[j]) >= 0)
						{
							temp = veta[j];
							if (trured.signum() > 0)
							{
								temp = BigDecimal.ZERO;
								for(int k = 1; k <= n; ++k)
								{
									temp = temp.add(square(dx[k].subtract(sim[k][j], mc), mc), mc);
								}
								temp = temp.sqrt(mc);
							}
							if (temp.compareTo(edgmax) > 0)
							{
								l = j;
								edgmax = temp;
							}
						}
					}
					if (l > 0)
						jdrop = l;

					if (jdrop != 0)
					{
						// Revise the simplex by updating the elements of SIM, SIMI and
						// DATMAT.

						temp = BigDecimal.ZERO;
						for(int i = 1; i <= n; ++i)
						{
							sim[i][jdrop] = dx[i];
							temp = temp.add(simi[jdrop][i].multiply(dx[i], mc), mc);
						}
						for(int k = 1; k <= n; ++k)
						{
							simi[jdrop][k] = simi[jdrop][k].divide(temp, mc);
						}
						for(int j = 1; j <= n; ++j)
						{
							if (j != jdrop)
							{
								temp = dotProduct(part(row(simi, j), 1, n), part(dx, 1, n), mc);
								for(int k = 1; k <= n; ++k)
								{
									simi[j][k] = simi[j][k].subtract(temp.multiply(simi[jdrop][k], mc), mc);
								}
							}
						}
						for(int k = 1; k <= mpp; ++k)
						{
							datmat[k][jdrop] = con[k];
						}

						// Branch back for further iterations with the current RHO.

						if (trured.signum() > 0 && trured.compareTo(convert(0.1, mc).multiply(prerem, mc)) >= 0)
						{
							continue L_140;
						}
					}
				}
				while(false);

				if (!iflag)
				{
					ibrnch = false;
					continue L_140;
				}

				if (rho.compareTo(rhoend) <= 0)
				{
					status = CobylaExitStatus.Normal;
					break L_40;
				}

				// Otherwise reduce RHO if it is not at its least value and reset PARMU.

				BigDecimal cmin = BigDecimal.ZERO, cmax = BigDecimal.ZERO;

				rho = rho.multiply(convert(0.5, mc), mc);
				if (rho.compareTo(convert(1.5, mc).multiply(rhoend, mc)) <= 0)
				{
					rho = rhoend;
				}
				if (parmu.signum() > 0)
				{
					BigDecimal denom = BigDecimal.ZERO;
					for(int k = 1; k <= mp; ++k)
					{
						cmin = datmat[k][np];
						cmax = cmin;
						for(int i = 1; i <= n; ++i)
						{
							cmin = cmin.min(datmat[k][i]);
							cmax = cmax.max(datmat[k][i]);
						}
						if (k <= m && cmin.compareTo(convert(0.5, mc).multiply(cmax, mc)) < 0)
						{
							temp = cmax.max(BigDecimal.ZERO).subtract(cmin, mc);
							denom = denom.signum() <= 0 ? temp : denom.min(temp);
						}
					}
					if (denom.signum() == 0)
					{
						parmu = BigDecimal.ZERO;
					}
					else if (cmax.subtract(cmin, mc).compareTo(parmu.multiply(denom, mc)) < 0)
					{
						parmu = cmax.subtract(cmin, mc).divide(denom, mc);
					}
				}
				if (iprint >= 2)
				{
					System.out.format(
							"%nReduction in RHO to %s  and PARMU = %s%n", rho.toString(),
							parmu.toString());
				}
				if (iprint == 2)
				{
					printIterationResult(nfvals, datmat[mp][np], datmat[mpp][np],
							col(sim, np), n, calcfc);
				}

			}
			while(true);
		}
		while(true);

		switch(status)
		{
			case Normal:
				if (iprint >= 1)
				{
					System.out.format("%nNormal return from subroutine COBYLA%n");
				}
				if (ifull)
				{
					if (iprint >= 1)
					{
						printIterationResult(nfvals, f, resmax, x, n, calcfc);
					}
					return status;
				}
				break;
			case MaxIterationsReached:
				if (iprint >= 1)
				{
					System.out.format(
							"%nReturn from subroutine COBYLA because the MAXFUN limit has been reached.%n");
				}
				break;
			case DivergingRoundingErrors:
				if (iprint >= 1)
				{
					System.out.format(
							"%nReturn from subroutine COBYLA because rounding errors are becoming damaging.%n");
				}
				break;
		}

		for(int k = 1; k <= n; ++k)
		{
			x[k] = sim[k][np];
		}
		if (iprint >= 1)
		{
			f = calcfc.compute(n, m, x, con, mc);
			resmax = datmat[mpp][np];
			printIterationResult(nfvals, f, resmax, x, n, calcfc);
		}

		return status;
	}

	private static boolean trstlp(int n, int m, BigDecimal[][] a, BigDecimal[] b,
			BigDecimal rho, BigDecimal[] dx, MathContext mc)
	{
		// N.B. Arguments Z, ZDOTA, VMULTC, SDIRN, DXNEW, VMULTD & IACT have been
		// removed.

		// This subroutine calculates an N-component vector DX by applying the
		// following two stages. In the first stage, DX is set to the shortest
		// vector that minimizes the greatest violation of the constraints
		// A(1,K)*DX(1)+A(2,K)*DX(2)+...+A(N,K)*DX(N) .GE. B(K), K = 2,3,...,M,
		// subject to the Euclidean length of DX being at most RHO. If its length is
		// strictly less than RHO, then we use the resultant freedom in DX to
		// minimize the objective function
		// -A(1,M+1)*DX(1) - A(2,M+1)*DX(2) - ... - A(N,M+1)*DX(N)
		// subject to no increase in any greatest constraint violation. This
		// notation allows the gradient of the objective function to be regarded as
		// the gradient of a constraint. Therefore the two stages are distinguished
		// by MCON .EQ. M and MCON .GT. M respectively. It is possible that a
		// degeneracy may prevent DX from attaining the target length RHO. Then the
		// value IFULL = 0 would be set, but usually IFULL = 1 on return.

		// In general NACT is the number of constraints in the active set and
		// IACT(1),...,IACT(NACT) are their indices, while the remainder of IACT
		// contains a permutation of the remaining constraint indices. Further, Z
		// is an orthogonal matrix whose first NACT columns can be regarded as the
		// result of Gram-Schmidt applied to the active constraint gradients. For
		// J = 1,2,...,NACT, the number ZDOTA(J) is the scalar product of the J-th
		// column of Z with the gradient of the J-th active constraint. DX is the
		// current vector of variables and here the residuals of the active
		// constraints should be zero. Further, the active constraints have
		// nonnegative Lagrange multipliers that are held at the beginning of
		// VMULTC. The remainder of this vector holds the residuals of the inactive
		// constraints at DX, the ordering of the components of VMULTC being in
		// agreement with the permutation of the indices of the constraints that is
		// in IACT. All these residuals are nonnegative, which is achieved by the
		// shift RESMAX that makes the least residual zero.

		// Initialize Z and some other variables. The value of RESMAX will be
		// appropriate to DX = 0, while ICON will be the index of a most violated
		// constraint if RESMAX is positive. Usually during the first stage the
		// vector SDIRN gives a search direction that reduces all the active
		// constraint violations by one simultaneously.

		// Local variables

		BigDecimal temp;

		int nactx = 0;
		BigDecimal resold = BigDecimal.ZERO;

		BigDecimal[][] z = new BigDecimal[1 + n][1 + n];
		BigDecimal[] zdota = new BigDecimal[2 + m];
		BigDecimal[] vmultc = new BigDecimal[2 + m];
		BigDecimal[] sdirn = new BigDecimal[1 + n];
		BigDecimal[] dxnew = new BigDecimal[1 + n];
		BigDecimal[] vmultd = new BigDecimal[2 + m];
		int[] iact = new int[2 + m];

		// The original double arrays where instantiated with 0.0 values.
		// The BigDecimal arrays are instantiated with nulls, so we need to fill in zeros for the same behaviour.
		fill(z);
		fill(zdota);
		fill(vmultc);
		fill(sdirn);
		fill(dxnew);
		fill(vmultd);

		int mcon = m;
		int nact = 0;
		for(int i = 1; i <= n; ++i)
		{
			z[i][i] = BigDecimal.ONE;
			dx[i] = BigDecimal.ZERO;
		}

		int icon = 0;
		BigDecimal resmax = BigDecimal.ZERO;
		if (m >= 1)
		{
			for(int k = 1; k <= m; ++k)
			{
				if (b[k].compareTo(resmax) > 0)
				{
					resmax = b[k];
					icon = k;
				}
			}
			for(int k = 1; k <= m; ++k)
			{
				iact[k] = k;
				vmultc[k] = resmax.subtract(b[k], mc);
			}
		}

		// End the current stage of the calculation if 3 consecutive iterations
		// have either failed to reduce the best calculated value of the objective
		// function or to increase the number of active constraints since the best
		// value was calculated. This strategy prevents cycling, but there is a
		// remote possibility that it will cause premature termination.

		boolean first = true;
		do
		{
			L_60: do
			{
				if (!first || (first && resmax.signum() == 0))
				{
					mcon = m + 1;
					icon = mcon;
					iact[mcon] = mcon;
					vmultc[mcon] = BigDecimal.ZERO;
				}
				first = false;

				BigDecimal optold = BigDecimal.ZERO;
				int icount = 0;

				BigDecimal step, stpful;

				L_70: do
				{
					BigDecimal optnew = mcon == m ? resmax
							: dotProduct(part(dx, 1, n), part(col(a, mcon), 1, n), mc).negate();

					if (icount == 0 || optnew.compareTo(optold) < 0)
					{
						optold = optnew;
						nactx = nact;
						icount = 3;
					}
					else if (nact > nactx)
					{
						nactx = nact;
						icount = 3;
					}
					else
					{
						--icount;
					}
					if (icount == 0)
					{
						break L_60;
					}

					// If ICON exceeds NACT, then we add the constraint with index
					// IACT(ICON) to
					// the active set. Apply Givens rotations so that the last N-NACT-1
					// columns
					// of Z are orthogonal to the gradient of the new constraint, a scalar
					// product being set to zero if its nonzero value could be due to
					// computer
					// rounding errors. The array DXNEW is used for working space.

					BigDecimal ratio;
					if (icon <= nact)
					{
						if (icon < nact)
						{
							// Delete the constraint that has the index IACT(ICON) from the
							// active set.

							int isave = iact[icon];
							BigDecimal vsave = vmultc[icon];
							int k = icon;
							do
							{
								int kp = k + 1;
								int kk = iact[kp];
								BigDecimal sp = dotProduct(part(col(z, k), 1, n), part(col(a, kk), 1, n), mc);
								temp = square(sp, mc).add(square(zdota[kp], mc), mc).sqrt(mc);
								BigDecimal alpha = zdota[kp].divide(temp, mc);
								BigDecimal beta = sp.divide(temp, mc);
								zdota[kp] = alpha.multiply(zdota[k], mc);
								zdota[k] = temp;
								for(int i = 1; i <= n; ++i)
								{
									temp = alpha.multiply(z[i][kp], mc).add(beta.multiply(z[i][k], mc), mc);
									z[i][kp] = alpha.multiply(z[i][k], mc).subtract(beta.multiply(z[i][kp], mc), mc);
									z[i][k] = temp;
								}
								iact[k] = kk;
								vmultc[k] = vmultc[kp];
								k = kp;
							}
							while(k < nact);

							iact[k] = isave;
							vmultc[k] = vsave;
						}
						--nact;

						// If stage one is in progress, then set SDIRN to the direction of
						// the next
						// change to the current vector of variables.

						if (mcon > m)
						{
							// Pick the next search direction of stage two.

							temp = BigDecimal.ONE.divide(zdota[nact], mc);
							for(int k = 1; k <= n; ++k)
							{
								sdirn[k] = temp.multiply(z[k][nact], mc);
							}
						}
						else
						{
							temp = dotProduct(part(sdirn, 1, n), part(col(z, nact + 1), 1, n), mc);
							for(int k = 1; k <= n; ++k)
							{
								sdirn[k] = sdirn[k].subtract(temp.multiply(z[k][nact + 1], mc), mc);
							}
						}
					}
					else
					{
						int kk = iact[icon];
						for(int k = 1; k <= n; ++k)
						{
							dxnew[k] = a[k][kk];
						}
						BigDecimal tot = BigDecimal.ZERO;

						{
							int k = n;
							while(k > nact)
							{
								BigDecimal sp = BigDecimal.ZERO;
								BigDecimal spabs = BigDecimal.ZERO;
								for(int i = 1; i <= n; ++i)
								{
									temp = z[i][k].multiply(dxnew[i], mc);
									sp = sp.add(temp, mc);
									spabs = spabs.add(temp.abs(), mc);
								}
								BigDecimal delta = convert(stepSize, mc).multiply(sp.abs(), mc);
								BigDecimal acca = spabs.add(delta, mc);
								BigDecimal accb = acca.add(delta, mc);
								if (spabs.compareTo(acca) >= 0 || acca.compareTo(accb) >= 0)
								{
									sp = BigDecimal.ZERO;
								}
								if (tot.signum() == 0)
								{
									tot = sp;
								}
								else
								{
									int kp = k + 1;
									temp = square(sp, mc).add(square(tot, mc), mc).sqrt(mc);
									BigDecimal alpha = sp.divide(temp, mc);
									BigDecimal beta = tot.divide(temp, mc);
									tot = temp;
									for(int i = 1; i <= n; ++i)
									{
										temp = alpha.multiply(z[i][k], mc).add(beta.multiply(z[i][kp], mc), mc);
										z[i][kp] = alpha.multiply(z[i][kp], mc).subtract(beta.multiply(z[i][k], mc), mc);
										z[i][k] = temp;
									}
								}
								--k;
							}
						}

						if (tot.signum() == 0)
						{
							// The next instruction is reached if a deletion has to be made
							// from the
							// active set in order to make room for the new active constraint,
							// because
							// the new constraint gradient is a linear combination of the
							// gradients of
							// the old active constraints. Set the elements of VMULTD to the
							// multipliers
							// of the linear combination. Further, set IOUT to the index of
							// the
							// constraint to be deleted, but branch if no suitable index can
							// be found.

							ratio = BigDecimal.ONE.negate();
							{
								int k = nact;
								do
								{
									BigDecimal zdotv = BigDecimal.ZERO;
									BigDecimal zdvabs = BigDecimal.ZERO;

									for(int i = 1; i <= n; ++i)
									{
										temp = z[i][k].multiply(dxnew[i], mc);
										zdotv = zdotv.add(temp, mc);
										zdvabs = zdvabs.add(temp.abs(), mc);
									}
									BigDecimal delta = convert(stepSize, mc).multiply(zdotv.abs(), mc);
									BigDecimal acca = zdvabs.add(delta, mc);
									BigDecimal accb = acca.add(delta, mc);
									if (zdvabs.compareTo(acca) < 0 && acca.compareTo(accb) < 0)
									{
										temp = zdotv.divide(zdota[k], mc);
										if (temp.signum() > 0 && iact[k] <= m)
										{
											BigDecimal tempa = vmultc[k].divide(temp, mc);
											if (ratio.signum() < 0 || tempa.compareTo(ratio) < 0)
											{
												ratio = tempa;
											}
										}

										if (k >= 2)
										{
											int kw = iact[k];
											for(int i = 1; i <= n; ++i)
											{
												dxnew[i] = dxnew[i].subtract(temp.multiply(a[i][kw], mc), mc);
											}
										}
										vmultd[k] = temp;
									}
									else
									{
										vmultd[k] = BigDecimal.ZERO;
									}
								}
								while(--k > 0);
							}
							if (ratio.signum() < 0)
							{
								break L_60;
							}

							// Revise the Lagrange multipliers and reorder the active
							// constraints so
							// that the one to be replaced is at the end of the list. Also
							// calculate the
							// new value of ZDOTA(NACT) and branch if it is not acceptable.

							for(int k = 1; k <= nact; ++k)
							{
								vmultc[k] = BigDecimal.ZERO.max(vmultc[k].subtract(ratio.multiply(vmultd[k], mc), mc));
							}
							if (icon < nact)
							{
								int isave = iact[icon];
								BigDecimal vsave = vmultc[icon];
								int k = icon;
								do
								{
									int kp = k + 1;
									int kw = iact[kp];
									BigDecimal sp = dotProduct(part(col(z, k), 1, n), part(col(a, kw), 1, n), mc);
									temp = square(sp, mc).add(square(zdota[kp], mc), mc).sqrt(mc);
									BigDecimal alpha = zdota[kp].divide(temp, mc);
									BigDecimal beta = sp.divide(temp, mc);
									zdota[kp] = alpha.multiply(zdota[k], mc);
									zdota[k] = temp;
									for(int i = 1; i <= n; ++i)
									{
										temp = alpha.multiply(z[i][kp], mc).add(beta.multiply(z[i][k], mc), mc);
										z[i][kp] = alpha.multiply(z[i][k], mc).subtract(beta.multiply(z[i][kp], mc), mc);
										z[i][k] = temp;
									}
									iact[k] = kw;
									vmultc[k] = vmultc[kp];
									k = kp;
								}
								while(k < nact);
								iact[k] = isave;
								vmultc[k] = vsave;
							}
							temp = dotProduct(part(col(z, nact), 1, n), part(col(a, kk), 1, n), mc);
							if (temp.signum() == 0)
							{
								break L_60;
							}
							zdota[nact] = temp;
							vmultc[icon] = BigDecimal.ZERO;
							vmultc[nact] = ratio;
						}
						else
						{
							// Add the new constraint if this can be done without a deletion
							// from the
							// active set.

							++nact;
							zdota[nact] = tot;
							vmultc[icon] = vmultc[nact];
							vmultc[nact] = BigDecimal.ZERO;
						}

						// Update IACT and ensure that the objective function continues to
						// be
						// treated as the last active constraint when MCON>M.

						iact[icon] = iact[nact];
						iact[nact] = kk;
						if (mcon > m && kk != mcon)
						{
							int k = nact - 1;
							BigDecimal sp = dotProduct(part(col(z, k), 1, n), part(col(a, kk), 1, n), mc);
							temp = square(sp, mc).add(square(zdota[nact], mc), mc).sqrt(mc);
							BigDecimal alpha = zdota[nact].divide(temp, mc);
							BigDecimal beta = sp.divide(temp, mc);
							zdota[nact] = alpha.multiply(zdota[k], mc);
							zdota[k] = temp;
							for(int i = 1; i <= n; ++i)
							{
								temp = alpha.multiply(z[i][nact], mc).add(beta.multiply(z[i][k], mc), mc);
								z[i][nact] = alpha.multiply(z[i][k], mc).subtract(beta.multiply(z[i][nact], mc), mc);
								z[i][k] = temp;
							}
							iact[nact] = iact[k];
							iact[k] = kk;
							temp = vmultc[k];
							vmultc[k] = vmultc[nact];
							vmultc[nact] = temp;
						}

						// If stage one is in progress, then set SDIRN to the direction of
						// the next
						// change to the current vector of variables.

						if (mcon > m)
						{
							// Pick the next search direction of stage two.

							temp = BigDecimal.ONE.divide(zdota[nact], mc);
							for(int k = 1; k <= n; ++k)
							{
								sdirn[k] = temp.multiply(z[k][nact], mc);
							}
						}
						else
						{
							kk = iact[nact];
							temp = dotProduct(part(sdirn, 1, n), part(col(a, kk), 1, n), mc).subtract(BigDecimal.ONE, mc)
									.divide(zdota[nact], mc);
							for(int k = 1; k <= n; ++k)
							{
								sdirn[k] = sdirn[k].subtract(temp.multiply(z[k][nact], mc), mc);
							}
						}
					}

					// Calculate the step to the boundary of the trust region or take the step
					// that reduces RESMAX to zero. The two statements below that include the
					// factor 1.0E-6 prevent some harmless underflows that occurred in a test
					// calculation. Further, we skip the step if it could be zero within a
					// reasonable tolerance for computer rounding errors.

					// Replaced constant factor 1.0E-6 by a custom epsilon depending on the precision.

					BigDecimal dd = square(rho, mc);
					BigDecimal sd = BigDecimal.ZERO;
					BigDecimal ss = BigDecimal.ZERO;
					for(int i = 1; i <= n; ++i)
					{
						if (dx[i].abs().compareTo(epsilon(mc).multiply(rho, mc)) >= 0)
						{
							dd = dd.subtract(square(dx[i], mc), mc);
						}
						sd = sd.add(dx[i].multiply(sdirn[i], mc), mc);
						ss = ss.add(sdirn[i].multiply(sdirn[i], mc), mc);
					}
					if (dd.signum() <= 0)
					{
						break L_60;
					}
					temp = ss.multiply(dd, mc).sqrt(mc);
					if (sd.abs().compareTo(epsilon(mc).multiply(temp, mc)) >= 0)
					{
						temp = ss.multiply(dd, mc).add(square(sd, mc), mc).sqrt(mc);
					}
					stpful = dd.divide(temp.add(sd, mc), mc);
					step = stpful;
					if (mcon == m)
					{
						BigDecimal delta = convert(stepSize, mc).multiply(resmax, mc);
						BigDecimal acca = step.add(delta, mc);
						BigDecimal accb = acca.add(delta, mc);
						if (step.compareTo(acca) >= 0 || acca.compareTo(accb) >= 0)
						{
							break L_70;
						}
						step = step.min(resmax);
					}

					// Set DXNEW to the new variables if STEP is the steplength, and
					// reduce
					// RESMAX to the corresponding maximum residual if stage one is being
					// done.
					// Because DXNEW will be changed during the calculation of some
					// Lagrange
					// multipliers, it will be restored to the following value later.

					for(int k = 1; k <= n; ++k)
					{
						dxnew[k] = dx[k].add(step.multiply(sdirn[k], mc), mc);
					}
					if (mcon == m)
					{
						resold = resmax;
						resmax = BigDecimal.ZERO;
						for(int k = 1; k <= nact; ++k)
						{
							int kk = iact[k];
							temp = b[kk].subtract(dotProduct(part(col(a, kk), 1, n), part(dxnew, 1, n), mc), mc);
							resmax = resmax.max(temp);
						}
					}

					// Set VMULTD to the VMULTC vector that would occur if DX became
					// DXNEW. A
					// device is included to force VMULTD(K) = 0.0 if deviations from this
					// value
					// can be attributed to computer rounding errors. First calculate the
					// new
					// Lagrange multipliers.

					{
						int k = nact;
						do
						{
							BigDecimal zdotw = BigDecimal.ZERO;
							BigDecimal zdwabs = BigDecimal.ZERO;
							for(int i = 1; i <= n; ++i)
							{
								temp = z[i][k].multiply(dxnew[i], mc);
								zdotw = zdotw.add(temp, mc);
								zdwabs = zdwabs.add(temp.abs(), mc);
							}
							BigDecimal delta = convert(stepSize, mc).multiply(zdotw.abs(), mc);
							BigDecimal acca = zdwabs.add(delta, mc);
							BigDecimal accb = acca.add(delta, mc);
							if (zdwabs.compareTo(acca) >= 0 || acca.compareTo(accb) >= 0)
							{
								zdotw = BigDecimal.ZERO;
								// do not divide by zdota[k] in this case - it may be zero
								vmultd[k] = BigDecimal.ZERO;
							}
							else
							{
								vmultd[k] = zdotw.divide(zdota[k], mc);
							}
							if (k >= 2)
							{
								int kk = iact[k];
								for(int i = 1; i <= n; ++i)
								{
									dxnew[i] = dxnew[i].subtract(vmultd[k].multiply(a[i][kk], mc), mc);
								}
							}
						}
						while(k-- >= 2);
						if (mcon > m)
						{
							vmultd[nact] = vmultd[nact].max(BigDecimal.ZERO);
						}
					}

					// Complete VMULTC by finding the new constraint residuals.

					for(int k = 1; k <= n; ++k)
					{
						dxnew[k] = dx[k].add(step.multiply(sdirn[k], mc), mc);
					}
					if (mcon > nact)
					{
						int kl = nact + 1;
						for(int k = kl; k <= mcon; ++k)
						{
							int kk = iact[k];
							BigDecimal total = resmax.subtract(b[kk], mc);
							BigDecimal sumabs = resmax.add(b[kk].abs(), mc);
							for(int i = 1; i <= n; ++i)
							{
								temp = a[i][kk].multiply(dxnew[i], mc);
								total = total.add(temp, mc);
								sumabs = sumabs.add(temp.abs(), mc);
							}
							BigDecimal delta = convert(stepSize, mc).multiply(total.abs(), mc);
							BigDecimal acca = sumabs.add(delta, mc);
							BigDecimal accb = acca.add(delta, mc);
							if (sumabs.compareTo(acca) >= 0 || acca.compareTo(accb) >= 0)
							{
								total = BigDecimal.ZERO;
							}
							vmultd[k] = total;
						}
					}

					// Calculate the fraction of the step from DX to DXNEW that will be
					// taken.

					ratio = BigDecimal.ONE;
					icon = 0;
					for(int k = 1; k <= mcon; ++k)
					{
						if (vmultd[k].signum() < 0)
						{
							temp = vmultc[k].divide(vmultc[k].subtract(vmultd[k], mc), mc);
							if (temp.compareTo(ratio) < 0)
							{
								ratio = temp;
								icon = k;
							}
						}
					}

					// Update DX, VMULTC and RESMAX.

					temp = BigDecimal.ONE.subtract(ratio, mc);
					for(int k = 1; k <= n; ++k)
					{
						dx[k] = temp.multiply(dx[k], mc).add(ratio.multiply(dxnew[k], mc), mc);
					}
					for(int k = 1; k <= mcon; ++k)
					{
						vmultc[k] = BigDecimal.ZERO.max(temp.multiply(vmultc[k], mc).add(ratio.multiply(vmultd[k], mc), mc));
					}
					if (mcon == m)
					{
						resmax = resmax.subtract(resold, mc).multiply(ratio, mc).add(resold, mc);
					}

					// If the full step is not acceptable then begin another iteration.
					// Otherwise switch to stage two or end the calculation.

				}
				while(icon > 0);

				if (step == stpful)
				{
					return true;
				}

			}
			while(true);

			// We employ any freedom that may be available to reduce the objective
			// function before returning a DX whose length is less than RHO.

		}
		while(mcon == m);

		return false;
	}

	private static void printIterationResult(int nfvals, BigDecimal f, BigDecimal resmax,
			BigDecimal[] x, int n, Calcfc fn)
	{
		System.out.format("%nNFVALS = %1$5d   F = %2$s    MAXCV = %3$s%n",
				nfvals, f.toString(), resmax.toString());
		System.out.format("X = %s%n", format(part(x, 1, n)));
	}

	private static BigDecimal[] row(BigDecimal[][] src, int rowidx)
	{
		return src[rowidx];
	}

	private static BigDecimal[] col(BigDecimal[][] src, int colidx)
	{
		int rows = src.length;
		BigDecimal[] dest = new BigDecimal[rows];
		for(int row = 0; row < rows; ++row)
		{
			dest[row] = src[row][colidx];
		}
		return dest;
	}

	private static BigDecimal[] part(BigDecimal[] src, int from, int to)
	{
		BigDecimal[] dest = new BigDecimal[to - from + 1];
		System.arraycopy(src, from, dest, 0, dest.length);
		return dest;
	}

	private static String format(BigDecimal[] x)
	{
		String fmt = "";
		for(int i = 0; i < x.length; ++i)
		{
			fmt = fmt + String.format("%s ", x[i].toString());
		}
		return fmt;
	}

	private static BigDecimal dotProduct(BigDecimal[] lhs, BigDecimal[] rhs, MathContext mc)
	{
		BigDecimal sum = BigDecimal.ZERO;
		for(int i = 0; i < lhs.length; ++i)
		{
			sum = sum.add(lhs[i].multiply(rhs[i], mc), mc);
		}
		return sum;
	}
}
