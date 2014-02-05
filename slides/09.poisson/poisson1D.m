function e = poisson1D(N, x0, x1, f, e)
  A = spalloc(N-1,N-1,3*(N-1));
  h = (x1-x0)/N;
  A = spdiags( 2*ones(N-1,1), 0,A) + ...
      spdiags(-1*ones(N-1,1),-1,A) + ...
      spdiags(-1*ones(N-1,1), 1,A);
  fgrid = linspace(x0, x1, N+1);
  g = h*h*feval(f, fgrid(2:end-1));
  u = [0; A\g; 0];
  clf;
  plot(fgrid, u);
  legend('finite difference solution');
  set(gca,'fontsize',18);

  ers = feval(e, fgrid);
  e = norm(ers-u,'inf');
