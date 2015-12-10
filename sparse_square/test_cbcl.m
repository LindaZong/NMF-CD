load ../data/cbcl;

V = sparse(V);
n = size(V,1);
m = size(V,2);
k = 10;
Winit = rand(n,k);
Hinit = rand(k,m);
maxiter = 50;
type = 1; %% GCD

[W H] = sparse_CD(V, k, maxiter, Winit, Hinit, type);
