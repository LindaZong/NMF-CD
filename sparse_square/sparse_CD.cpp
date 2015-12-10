// 
// Invocation form within Matlab:
// [W, H] = sparse_CD(V, k, maxiter, Winit, Hinit, type)
// 
// input arguments: 
// 		V: data matrix (n by m matrix)
// 	    k: target rank (integer)
// 	    maxiter: maximum number of iterations (integer)
// 	    Winit: initial value for W (n by k matrix)
// 	    Hinit: initial value for H (k by m matrix)
// 	    type: 0 or 1
// 	        0: Greedy Coordinate Descent (recommended)
// 	        1: Cyclic Coordinate Descent 
//
//
// output arguments:
//      W: n by k output matrix
//      H: k by m output matrix
//

#include "math.h"
#include "mex.h" 
#include <time.h>
#include <set>
#include <map>
#include <utility>
#include <algorithm>
#include <iostream>
#define MAXN 47240
#define MAXM 583000
#define MAXK 50
#define MAXITER 10000

void exit_with_help()
{
	mexPrintf(
			"\n"
 "[W, H] = sparse_CD(V, k, maxiter, Winit, Hinit, type)\n"
 " input arguments: \n"
 "      V: data matrix (n by m matrix)\n"
 "      k: target rank (integer)\n"
 "      maxiter: maximum number of iterations (integer)\n"
 "      Winit: initial value for W (n by k matrix)\n"
 "      Hinit: initial value for H (k by m matrix)\n"
 "      type: 0 or 1\n"
 "          0: Greedy Coordinate Descent (recommended)\n"
 "          1: Cyclic Coordinate Descent \n"
 "\n"
 " output arguments:\n"
 "      W: n by k output matrix\n"
 "      H: k by m output matrix\n"
 "\n"
	);
}

static void fake_answer(mxArray *plhs[])
{
	plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(0, 0, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(0, 0, mxREAL);
	plhs[3] = mxCreateDoubleMatrix(0, 0, mxREAL);
	plhs[4] = mxCreateDoubleMatrix(0, 0, mxREAL);
}


struct mat {
	double *values;
	int *row_index;
	int *col_begin;
	int n;
	int m;
};

double **MatrixAlloc(int a, int b)
{
	double **ptr = (double **)malloc(a*sizeof(double *));
	for ( int i=0; i<a ; i++ )
		ptr[i] = (double *)malloc(b*sizeof(double));
	return ptr;
}

void MatrixFree(double **ptr, int a, int b)
{
	for ( int i=0; i<a ; i++ )
		free(ptr[i]);
	free(ptr);
}

double timelist[MAXITER], difflist[MAXITER], projlist[MAXITER];

double obj(struct mat *V, int n, int m, int k,double **W, double **H)
{
	int i, j, r, q;
	double total = 0, v;

	for ( j=0 ; j<V->m ; j++ )
	{
		int begin = V->col_begin[j], end = V->col_begin[j+1];
		int now = 0;
		for ( q=begin ; q<end ; q++ )
		{
			i = V->row_index[q];
			// compute elements with Vij==0
			for ( ; now<i ; now++ )
			{
				v=0;
				for (r=0 ; r<k ; r++)
					v -= W[now][r]*H[r][j];
				total += v*v;
			}
			
			// compute element with Vij!=0
			v = V->values[q];
			for (r=0 ; r<k ; r++ )
				v -= W[i][r]*H[r][j];
			total = total + v*v;
			now = i+1;
		}
		for ( ; now<n ; now++ )
		{
			v=0;
			for (r=0 ; r<k ; r++)
				v -= W[now][r]*H[r][j];
			total += v*v;
		}
	}
	return total/2;
}

double proj(int n,int m,int k, double **W, double **H, double **GW, double **GH, double **VH, double **HH, double **WV, double **WW)
{
	double pg = 0;
	int count = 0;
	int i, j,r;

	// Compute GW and GH
	for ( i=0 ; i<n ; i++ )
		for ( j=0 ; j<k ; j++ )
		{
			GW[i][j] = (-1)*VH[i][j];
			for ( r=0 ; r<k ; r++ )
				GW[i][j] += W[i][r]*HH[r][j];
		}

	for ( i=0 ; i<k ; i++ )
		for ( j=0 ; j<m ; j++ )
		{
			GH[i][j] = (-1)*WV[i][j];
			for ( r=0 ; r<k ; r++ )
				GH[i][j] += WW[i][r]*H[r][j];
		}

	for ( i=0 ; i<n ; i++ )
		for ( j=0 ; j<k ; j++ )
		{
			if ( W[i][j] == 0)
			{
				if ( GW[i][j]<0)
					pg += GW[i][j]*GW[i][j];
			}
			else
				pg += GW[i][j]*GW[i][j];
		}
	for ( i=0 ; i<k ; i++ )
		for ( j=0 ; j<m ; j++ )
		{
			if ( H[i][j] == 0)
			{
				if ( GH[i][j]<0)
					pg += GH[i][j]*GH[i][j];
			}
			else
				pg += GH[i][j]*GH[i][j];
		}
	pg = sqrt(pg);

	return pg;
}

void fasthals(struct mat *V, struct mat *Vt, int n, int m, int k, int maxiter, double **W, double **H)
{
	int i,j,r,q;
	double **WV = MatrixAlloc(k,m);
	double **VH = MatrixAlloc(n,k);
	double **WW = MatrixAlloc(k,k);
	double **HH = MatrixAlloc(k,k);
	double **GW = MatrixAlloc(n,k);
	double **GH = MatrixAlloc(k,m);


	double totaltime = 0, atotal = 0;
	for ( int iter=0 ; iter<maxiter ; iter++ )
	{
		double begin = clock();

		// H's updates
		// Compute V'*W
		for ( i=0 ; i<m ; i++ )
			for ( j=0 ; j<k ; j++ )
				WV[j][i] =0 ;

		for ( i=0 ; i<k ; i++ )
			for ( j=0 ; j<n ; j++ )
				{
					double g=W[j][i];
					int begin = Vt->col_begin[j], end = Vt->col_begin[j+1];
					for ( q=begin ; q<end ; q++ )
						WV[i][Vt->row_index[q]] += g*(Vt->values[q]);
				}

		// Compute W'*W
		for ( i=0 ; i<k ; i++ )
			for ( j=0 ; j<k ; j++ )
			{
				WW[i][j] = 0;
				for ( r=0 ; r<n ; r++ )
					WW[i][j] += W[r][i]*W[r][j];
			}
	
		// Cyclic updatesw
		for ( j=0; j<k ; j++ )
		{
			for ( r=0 ; r<m ; r++ )
			{
				double tmp = 0;
				double gg = 1e-10;
				if (WW[j][j]>gg)
					gg = WW[j][j];
				for ( i=0 ; i<k ; i++ )
					tmp -= WW[j][i]*H[i][r];
				tmp += WV[j][r];
				tmp += gg*H[j][r];
				H[j][r] = tmp/gg;
				if ( H[j][r] < 0 )
					H[j][r] = 0;
			}
		}
		
		// W's updates
		// Compute V*H'
		for ( i=0 ; i<n ; i++ )
			for ( j=0 ; j<k ; j++ )
				VH[i][j] = 0;

		for ( i=0 ; i<k ; i++ )
			for ( j=0 ; j<m ; j++ )
				{
					double g = H[i][j];
					int begin = V->col_begin[j], end = V->col_begin[j+1];
					for ( q=begin ; q<end ; q++ )
						VH[V->row_index[q]][i] += g*(V->values[q]);
				}

		// Compute H*H'
		for ( i=0 ; i<k ; i++ )
			for ( j=0 ; j<k ; j++ )
			{
				HH[i][j] = 0;
				for ( r=0 ; r<m ; r++ )
					HH[i][j] += H[i][r]*H[j][r];
			}

		// Cyclic updates
		for ( j=0 ; j<k ; j++ )
		{
			for (r=0 ; r<n ; r++ )
			{
				double tmp = 0;
				double gg = 1e-10;
				if ( gg < HH[j][j])
					gg = HH[j][j];
				for ( i=0 ; i<k ; i++ )
					tmp -= W[r][i]*HH[i][j];
				tmp += VH[r][j];
				tmp += gg*W[r][j];
				W[r][j] = tmp/gg;
				if ( W[r][j] < 0)
					W[r][j] = 0;
			}
		}
		totaltime += clock()-begin;
		if (iter%10==0)
		{
			printf("Training CCD iter %d\n", iter);
			fflush(stdout);
		}

/*
		if (iter%10==0)
		{
			timelist[iter/10] = totaltime/CLOCKS_PER_SEC;
			difflist[iter/10] = obj(V,n,m,k,W,H);
			projlist[iter/10] = proj(n,m,k,W,H,GW,GH,VH,HH,WV,WW);
			printf("FastHals iter: %d   objective value: %lf\n", iter, difflist[iter/10]);
		}
		*/
	}
}

void GCD(struct mat *V, struct mat *Vt, int n, int m, int k, int maxiter, double **W, double **H)
{
	int i,j,p,q,r;
	double maxw_v[MAXN], maxh_v[MAXM];
	int maxw_ind[MAXN], maxh_ind[MAXM];
	int winner, hinner;

	double **VH = MatrixAlloc(n,k);
	double **WV = MatrixAlloc(k,m);
	double **HH = MatrixAlloc(k,k);
	double **WW = MatrixAlloc(k,k);
	double **Hnew = MatrixAlloc(k,m), **GH = MatrixAlloc(k,m), **HD = MatrixAlloc(k,m), **SH = MatrixAlloc(k,m);
	double **Wnew = MatrixAlloc(n,k), **GW = MatrixAlloc(n,k), **WD = MatrixAlloc(n,k), **SW = MatrixAlloc(n,k);

	double tol = 0.001;

	// VH=V*H'
	for ( i=0  ; i<n ; i++ )
		for ( j=0 ; j<k ; j++ )
		{
			VH[i][j] = 0;
			int begin = Vt->col_begin[i], end = Vt->col_begin[i+1];
			for ( q=begin ; q<end ; q++ )
				VH[i][j] += (Vt->values[q])*H[j][Vt->row_index[q]];
		}
	
	// WV=W'*V
	for ( i=0 ; i<k ; i++ )
		for ( j=0 ; j<m ; j++ )
		{
			WV[i][j] = 0;
			int begin = V->col_begin[j], end = V->col_begin[j+1];
			for ( q=begin ; q<end ; q++ )
				WV[i][j] += W[V->row_index[q]][i]*(V->values[q]);
		}

	// HH = H*H'
	for ( i=0 ; i<k ; i++ )
		for (j=0 ; j<k ; j++ )
		{
			HH[i][j] = 0;
			for ( r=0 ; r<m ; r++ )
				HH[i][j] += H[i][r]*H[j][r];
		}

	// WW = W'*W
	for ( i=0 ; i<k ; i++ )
		for ( j=0 ; j<k ; j++ )
		{
			WW[i][j] = 0;
			for ( r=0 ; r<n ; r++ )
				WW[i][j] += W[r][i]*W[r][j];
		}

	// Initial Hnew to be zero
	for ( i=0 ; i<k ; i++ )
		for ( j=0 ; j<m ; j++ )
			Hnew[i][j] = 0;


	double ss, diffobj;
	double totaltime=0;
	double atotal = 0, abegin = 0;
	double btotal = 0, bbegin = 0;
	for ( int iter = 0 ; iter<maxiter ; iter++ )
	{
		double begin = clock();
		double init = 0;

		/// W updates
		///
		for ( i=0 ; i<n ; i++ )
			for ( j=0 ; j<k ; j++ )
				Wnew[i][j] = 0;

		// Update VH
		for ( i=0 ; i<k ; i++ )
			for ( j=0 ; j<m ; j++ )
				if (Hnew[i][j] !=0)
				{
					double g = Hnew[i][j];
					int begin = V->col_begin[j], end = V->col_begin[j+1];
					for ( q=begin ; q<end ; q++ )
						VH[V->row_index[q]][i] += g*(V->values[q]);
				}

		// Compute GW
		for ( i=0 ; i<n ; i++ )
			for ( j=0 ; j<k ; j++ )
			{
				GW[i][j] = (-1)*VH[i][j];
				for ( r=0 ; r<k ; r++ )
					GW[i][j] += W[i][r]*HH[r][j];
			}

		// Compute Initial GW, WD, SW
		init = 0;
		for ( i=0 ; i<n ; i++ )
		{
			maxw_v[i] = 0;
			maxw_ind[i] = -1;
			for ( j=0 ; j<k ; j++ )
			{
				double s = GW[i][j]/HH[j][j];
				s = W[i][j]-s;
				if ( s< 0)
					s=0;
				s = s-W[i][j];
				SW[i][j] = s;
				double diffobj = (-1)*s*GW[i][j]-0.5*HH[j][j]*s*s;
				WD[i][j] = diffobj;
				if (diffobj > maxw_v[i])
				{
					maxw_v[i] = diffobj;
					maxw_ind[i] = j;
				}
			}
			if ( maxw_v[i]>init)
				init = maxw_v[i];
		}

		// W's Coordinate updates
		int totalinner =0;
		for ( int p=0 ; p<n ; p++)
		{
			for ( winner = 0 ; winner <m ; winner++)
			{
				double pv=maxw_v[p];
				int q = maxw_ind[p];
				if ( q==-1 )
					break;
				double s =SW[p][q];

				if ( pv< init*tol)
					break;
				for ( i=0 ; i<k ; i++)
				{
					WW[q][i] = WW[q][i]+s*W[p][i];
					WW[i][q] = WW[q][i];
				}
				WW[q][q] = WW[q][q] + s*s+s*W[p][q];
	
				Wnew[p][q] += s;
				W[p][q] = W[p][q] + s;
	
				maxw_v[p] = 0;
				maxw_ind[p]=-1;
				for ( i=0 ; i<k ; i++ )
				{
					GW[p][i] = GW[p][i] + s*HH[q][i];
					double ss = W[p][i]-GW[p][i]/HH[i][i];
					if (ss < 0)
						ss=0;
					ss = ss-W[p][i];
					SW[p][i] = ss;
					double diffobj = (-1)*(ss*GW[p][i]+0.5*HH[i][i]*ss*ss);
					if ( diffobj > maxw_v[p]) 
					{
						maxw_v[p] = diffobj;
						maxw_ind[p] = i;
					}
				}
			}
			totalinner += winner;
		}


		/// H updates
		///

		// Compute WV 
		for ( i=0 ; i<k ; i++ )
			for ( j=0 ; j<n ; j++ )
				if (Wnew[j][i] != 0)
				{
					double g=Wnew[j][i];
					int begin = Vt->col_begin[j], end = Vt->col_begin[j+1];
					for ( q=begin ; q<end ; q++ )
						WV[i][Vt->row_index[q]] += g*(Vt->values[q]);
				}

		for ( i=0 ; i<k ; i++ )
			for ( j=0 ; j<m ; j++ )
				Hnew[i][j] = 0;

		// Compute GH
		for ( i=0 ; i<k ; i++ )
			for ( j=0 ; j<m ; j++ )
			{
				GH[i][j] = (-1)*WV[i][j];
				for ( r=0 ; r<k ; r++ )
					GH[i][j] += WW[i][r]*H[r][j];
			}

		// Compute Initial GH, HD, SH
		init = 0;
		for ( i=0 ; i<m ; i++ )
		{
			maxh_v[i] = 0;
			maxh_ind[i] = -1;
			for ( j=0 ; j<k ; j++ )
			{
				double s = GH[j][i]/WW[j][j];
				s = H[j][i]-s;
				if ( s< 0)
					s=0;
				s = s-H[j][i];
				SH[j][i] = s;
				double diffobj = (-1)*s*GH[j][i]-0.5*WW[j][j]*s*s;
				HD[j][i] = diffobj;
				if (diffobj > maxh_v[i])
				{
					maxh_v[i] = diffobj;
					maxh_ind[i] = j;
				}
			}
			if ( maxh_v[i] > init )
				init = maxh_v[i];
		}

		// H's coordinate updates
		totalinner = 0;
		for (int q=0 ; q<m ; q++ )
		{
			for ( hinner = 0 ; hinner <n ; hinner++)
			{
				double pv=maxh_v[q];
				int p = maxh_ind[q];
				if ( p ==-1 )
					break;
				double s = SH[p][q];
	
				if (pv < init*tol)
					break;
				for ( i=0 ; i<k ; i++)
				{
					HH[p][i] = HH[p][i]+s*H[i][q];
					HH[i][p] = HH[p][i];
				}
				HH[p][p] = HH[p][p] + s*s+s*H[p][q];

				Hnew[p][q] += s;
				H[p][q] = H[p][q] + s;

				maxh_v[q] = 0;
				maxh_ind[q]=-1;
				double tmpg, tmpw, tmph;
				for ( i=0 ; i<k ; i++ )
				{
					GH[i][q] = GH[i][q] + s*WW[i][p];
					ss = H[i][q]-GH[i][q]/WW[i][i];
					if (ss < 0)
						ss=0;
					ss = ss-H[i][q];
					SH[i][q] = ss;
					diffobj = ss*(GH[i][q]+0.5*WW[i][i]*ss);
					diffobj *= (-1);
					if ( diffobj > maxh_v[q]) 
					{
						maxh_v[q] = diffobj;
						maxh_ind[q] = i;
					}
				}
			}
			totalinner += hinner;
		}

		totaltime += clock()-begin;
		if (iter%10==0)
		{
			printf("Training GCD iter %d\n", iter);
			fflush(stdout);
		}
/*		if (iter%10==0)
		{
			timelist[iter/10] = totaltime/CLOCKS_PER_SEC;
			difflist[iter/10] = obj(V,n,m,k,W,H);
			printf("GCD iter: %d   objective value: %lf\n", iter, difflist[iter/10]);
			projlist[iter/10] = proj(n,m,k,W,H,GW,GH,VH,HH,WV,WW);
		}*/
	}
}

struct mat *read_matrix(const mxArray *matrix)
{
	int num_elements = mxGetNzmax(matrix);
	double *elements;
	mwIndex *ir,*jc;
	elements = mxGetPr(matrix);
	struct mat *V = (struct mat *)malloc(sizeof(struct mat));
	V->n = mxGetM(matrix);
	V->m = mxGetN(matrix);
	ir = mxGetIr(matrix);
	jc = mxGetJc(matrix);
	V->values = mxGetPr(matrix);
	V->row_index = (int *)malloc(sizeof(int)*num_elements);
	V->col_begin = (int *)malloc(sizeof(int)*(V->m+1));
	for ( int i=0 ; i<num_elements ; i++ )
		V->row_index[i] = (int)ir[i];
	for ( int i=0 ; i<=V->m ; i++ )
		V->col_begin[i] = (int)jc[i];
	return V;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	const mxArray *xData, *wData,*hData, *tmpData;
	double *xValues;
	int i,j,g;
	int rowLen, colLen;
	double avg;

	struct mat *Vmat, *Vt;
	int n,m, k, maxiter;
	double **W, **H;

	if ( nrhs != 6 )
	{
		exit_with_help();
		return;
	}

	if ( !mxIsSparse(prhs[0]))
	{
		printf("Input matrix should be sparse!!\n");
		return ;
	}
	else
	{
		Vmat = read_matrix(prhs[0]);
		n = Vmat->n;
		m = Vmat->m;
		mxArray *rhs[1], *lhs[1];
		rhs[0] = mxDuplicateArray(prhs[0]);
		if ( mexCallMATLAB(1,lhs,1,rhs,"transpose"))
		{
			printf("Error: cannot transpose training instance matrix\n");
			return ;
		}
		Vt = read_matrix(lhs[0]);
	}

	tmpData = prhs[1];
	xValues = mxGetPr(tmpData);
	k = xValues[0];
	tmpData = prhs[2];
	xValues = mxGetPr(tmpData);
	maxiter = xValues[0];

	wData = prhs[3];
	xValues = mxGetPr(wData);
	rowLen = mxGetN(wData);
	colLen = mxGetM(wData);
	W = MatrixAlloc(n,k);
	for (i=0 ; i<rowLen ; i++ )
		for (j=0 ; j<colLen ; j++ )
			W[j][i] = xValues[i*colLen+j];

	hData = prhs[4];
	xValues = mxGetPr(hData);
	rowLen = mxGetN(hData);
	colLen = mxGetM(hData);
	H = MatrixAlloc(k,m);
	for (i=0 ; i<rowLen ; i++ )
		for (j=0 ; j<colLen ; j++ )
			H[j][i] = xValues[i*colLen+j];
	tmpData = prhs[5];
	xValues = mxGetPr(tmpData);
	int type = xValues[0];

	for ( i=0 ; i<maxiter ; i++ )
	{
		difflist[i] = 0;
		timelist[i] = 0;
		projlist[i] = 0;
	}
	if (type == 1)
		GCD(Vmat,Vt, n,m, k,maxiter,W,H);
	else
		fasthals(Vmat,Vt,n,m,k,maxiter,W,H);

	plhs[0] = mxCreateDoubleMatrix(n, k, mxREAL);
	double *outW = mxGetPr(plhs[0]);
	for (i=0, g=0 ; i<k ; i++ )
		for (j=0 ; j<n ; j++, g++ )
			outW[g] = W[j][i];
	plhs[1] = mxCreateDoubleMatrix(k,m, mxREAL);
	double *outH = mxGetPr(plhs[1]);
	for (i=0, g=0 ; i<m ; i++ )
		for (j=0 ; j<k ; j++, g++ )
			outH[g] = H[j][i];

	MatrixFree(W,n,k);
	MatrixFree(H,k,m);
	return;
}



