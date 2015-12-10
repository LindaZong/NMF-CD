function [W H objGCD timeGCD] = NMF_GCD(V, k, maxiter, Winit, Hinit, trace)
% Nonnegative Matrix Factorization (NMF) via Greedy Coordinate Descent
%
% Usage: [W H objGCD timeGCD] = NMF_GCD(V, k, maxiter, Winit, Hinit, trace)
%
% This software solve the following least squares NMF problem:
%
%	min_{W,H} ||V-WH||_F^2     s.t. W>=0, H>=0
%
% input: 
%		V: the input n by m dense matrix
%		k: the specified rank
%		maxiter: maximum number of iterations
%		Winit: initial of W (n by k dense matrix)
%		Hinit: initial of H (k by m dense matrix)
%		trace: 1: compute objective value per iteration.
%			   0: do not compute objective value per iteration. (default)
%
% output: 
%		NMF_GCD will output nonnegative matrices W, H, such that WH is an approximation of V
%		W: n by k dense matrix
%		H: k by m dense matrix
%		objGCD: objective values. 
%		timeGCD: time taken by GCD. 
%

n = size(V,1);
m = size(V,2);
W = Winit;
H = Hinit;

%% Stopping tolerance for subproblems
tol = 0.01;

total = 0;

fprintf('Start running NMF_GCD with trace=%g\n', trace);
for iter = 1:maxiter
	begin = cputime;


	% update variables of H
	WV = W'*V;
	WW = W'*W;
	GH = -(WV-WW*H);
	
	Hnew = doiter(GH, WW,H,tol, k^2); % Coordinate descent updates for H
	H = Hnew;

	% update variables of W
	VH = V*H';
	HH = H*H'; % Hessian of each row of W
	
	GW = -(VH - W*HH); % gradient of W

	Wnew = doiter(GW', HH, W',tol, k^2); % Coordinate descent updates for W%
	W = Wnew';


	total = total + cputime - begin;
	if trace==1
		obj =  0.5*norm(V-W*H,'fro')^2;
		objGCD(iter) = obj;
		timeGCD(iter) = total;
		if mod(iter,10)==0
			fprintf('Iteration %g, objective value %g\n', iter, obj);
		end
	end
end
if trace==0
	obj =  0.5*norm(V-W*H,'fro')^2
	objGCD = obj;
	timeGCD = total;
end
fprintf('Finished NMF_GCD with trace=%g, final objective value %g\n', trace,obj);

